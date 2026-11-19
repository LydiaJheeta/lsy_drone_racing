"""Experiment A (diagnose the delay failure) and B (T_traj x delay x k_end grid).

Run from project/ with the ROS PYTHONPATH removed (see README):

    cd project
    env -u PYTHONPATH pixi run python grid.py --exp=A                     # creates results/grid_<ts>/
    env -u PYTHONPATH pixi run python grid.py --exp=B --out_dir=results/grid_<ts>   # same folder

Every experiment writes to its own subfolder (`expA/`, `expB/`) of the grid folder; an existing
subfolder is never reused or overwritten.
"""

from __future__ import annotations

import lsy_drone_racing  # noqa: F401  (must be imported before scipy)

import csv
import functools
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from crazyflow.dynamics import Dynamics, load_params  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from common import PROJECT_DIR, RESULTS_DIR, EpisodeResult, Tee, TAKEOFF_Z  # noqa: E402
from common import load_cfg, load_controller_cls, run_episode  # noqa: E402
from run_eval import FIELDS, _build_env, _fmt, _mean, failure_group, make_row  # noqa: E402

T_VALUES = (18.75, 16.0)
D_VALUES = (0, 1, 2)
K_VALUES = (1.0, 0.85, 0.7, 0.55)
BAND = (0.3, 0.7)  # the success-rate band the experiment is looking for
GRID_FIELDS = [*FIELDS, "delay", "k_end"]


def hover_thrust(cfg: Any) -> float:
    """Total hover thrust in N, computed like attitude_mpc.py does (mass * g)."""
    p = load_params(Dynamics.so_rpy, cfg.sim.drone)
    return float(p["mass"] * -p["gravity_vec"][-1])


# --------------------------------------------------------------------------------------------
# Experiment A
# --------------------------------------------------------------------------------------------


def actual_rpy(state: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw (xyz Euler, rad) from the logged 13-vector states (quat at [3:7])."""
    return R.from_quat(state[:, 3:7]).as_euler("xyz")


def window_amp(t: np.ndarray, x: np.ndarray, t0: float, t1: float) -> float:
    """Max |x| over t0 <= t <= t1 (NaN if the window is empty)."""
    m = (t >= t0) & (t <= t1)
    return float(np.abs(x[m]).max()) if m.any() else float("nan")


def episode_metrics(ep: dict[str, Any], res: EpisodeResult) -> dict[str, Any]:
    """Numbers used to decide between a take-off failure and a growing oscillation."""
    t, st = ep["t"], ep["state"]
    z, rpy = st[:, 2], actual_rpy(st)
    up = np.flatnonzero(z > TAKEOFF_Z)
    t_up = float(t[up[0]]) if up.size else float("nan")
    t_end = float(t[-1])
    roll_pitch = np.maximum(np.abs(rpy[:, 0]), np.abs(rpy[:, 1]))
    return {
        "t_end_s": t_end,
        "t_takeoff_s": t_up,
        "z_max": float(z.max()),
        "z_at_end": float(z[-1]),
        # peak |roll|,|pitch| right after lift-off vs in the last second before the end
        "tilt_peak_first_0.5s_after_takeoff": (
            window_amp(t, roll_pitch, t_up, t_up + 0.5) if up.size else float("nan")
        ),
        "tilt_peak_last_0.5s": window_amp(t, roll_pitch, t_end - 0.5, t_end),
        "tilt_peak_overall": float(roll_pitch.max()),
        "success": res.success, "n_gates": res.n_gates, "cause": res.cause,
        "min_z_airborne": res.min_z_airborne,
    }


def plot_episode(ep: dict[str, Any], mpc_cmd: np.ndarray, title: str, path: Path,
                 hover: float) -> None:
    """z(t); roll and pitch (MPC command, applied command, actual); thrust (MPC, applied)."""
    t, st, applied = ep["t"], ep["state"], ep["action"]
    rpy = actual_rpy(st)
    n = min(len(t), len(mpc_cmd))
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    ax[0].plot(t, st[:, 2], "k", label="z (actual)")
    ax[0].axhline(TAKEOFF_Z, color="gray", ls=":", label=f"take-off threshold {TAKEOFF_Z} m")
    ax[0].set_ylabel("z [m]")
    for a, i, name in ((ax[1], 0, "roll"), (ax[2], 1, "pitch")):
        a.plot(t[:n], mpc_cmd[:n, i], "C0", label=f"{name} MPC command")
        a.plot(t, applied[:, i], "C1--", label=f"{name} applied (after delay)")
        a.plot(t, rpy[:, i], "C3", lw=1, label=f"{name} actual")
        a.set_ylabel(f"{name} [rad]")
    ax[3].plot(t[:n], mpc_cmd[:n, 3], "C0", label="thrust MPC command")
    ax[3].plot(t, applied[:, 3], "C1--", label="thrust applied")
    ax[3].axhline(hover, color="gray", ls=":", label=f"hover {hover:.3f} N")
    ax[3].set_ylabel("total thrust [N]")
    ax[3].set_xlabel("t [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(loc="upper right", fontsize=8)
    ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def exp_a(out: Path, seeds: tuple[int, ...] = (0, 1, 2), T: float = 18.75, delay: int = 2,
          k_end: float = 1.0, fills: tuple[str, ...] = ("first", "hover"),
          config: str = "att_level0.toml", subdir: str = "expA") -> None:
    """Delay-only runs with the logger on; one PNG per (fill, seed) plus a metrics CSV."""
    sub = out / subdir
    sub.mkdir(parents=True, exist_ok=False)
    cfg = load_cfg(config)
    mpc = load_controller_cls(PROJECT_DIR / "my_mpc.py")
    hov = hover_thrust(cfg)
    rows = []
    for fill in fills:
        env, lg = _build_env(cfg, "wrapped", sub / "logs",
                             dict(k_end=k_end, delay=delay, T_end=T, fill=fill, hover_thrust=hov))
        lg.tag = f"A_{fill}"
        for seed in seeds:
            cmds: list[np.ndarray] = []

            def factory(obs: Any, info: Any, config_: Any, _c: list = cmds) -> Any:
                c = mpc(obs, info, config_, T_traj=T)
                orig = c.compute_control

                def rec(o: Any, i: Any = None) -> np.ndarray:
                    u = orig(o, i)
                    _c.append(np.array(u, dtype=float))
                    return u

                c.compute_control = rec
                return c

            res = run_episode(env, factory, cfg, seed)
            lg.flush()
            ep = lg.last_episode
            m = {"fill": fill, "seed": seed, "T_traj": T, "delay": delay, "k_end": k_end,
                 **episode_metrics(ep, res)}
            rows.append(m)
            title = (f"seed {seed}, T_traj={T:g}, d={delay}, k_end={k_end}, fill={fill} -> "
                     f"{'SUCCESS' if res.success else res.cause} at {m['t_end_s']:.2f}s, "
                     f"gates={res.n_gates}")
            plot_episode(ep, np.asarray(cmds), title, sub / f"A_{fill}_seed{seed}.png", hov)
            print(f"[A] fill={fill} seed={seed} ok={res.success} gates={res.n_gates} "
                  f"t_end={m['t_end_s']:.2f}s cause={res.cause} t_takeoff={m['t_takeoff_s']:.2f} "
                  f"zmax={m['z_max']:.2f} tilt_first={m['tilt_peak_first_0.5s_after_takeoff']:.3f} "
                  f"tilt_last={m['tilt_peak_last_0.5s']:.3f}", flush=True)
        env.close()
    write_csv(sub / "expA_metrics.csv", rows)


# --------------------------------------------------------------------------------------------
# Experiment B
# --------------------------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows to CSV (no-op for empty input)."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def in_band(rate: float, band: tuple[float, float] = BAND) -> bool:
    """True if a success rate lies in the 30-70% band (inclusive)."""
    return band[0] - 1e-9 <= rate <= band[1] + 1e-9


def summarize_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (T_traj, delay, k_end), paired against the nominal (d=0, k_end=1) at the same T."""
    out = []
    conds = sorted({(r["T_traj"], r["delay"], r["k_end"]) for r in rows},
                   key=lambda c: (-c[0], c[1], -c[2]))
    for T, d, k in conds:
        rs = [r for r in rows if (r["T_traj"], r["delay"], r["k_end"]) == (T, d, k)]
        nom_ok = {r["seed"] for r in rows
                  if (r["T_traj"], r["delay"], r["k_end"]) == (T, 0, 1.0) and r["success"]}
        ok = [r for r in rs if r["success"]]
        bad = [r for r in rs if not r["success"]]
        paired = [r for r in rs if r["seed"] in nom_ok]
        groups: dict[str, int] = {}
        for r in bad:
            groups[r["group"]] = groups.get(r["group"], 0) + 1
        zs = [r["min_z_airborne"] for r in rs if not math.isnan(r["min_z_airborne"])]
        out.append({
            "T_traj": T, "delay": d, "k_end": k, "n": len(rs), "n_success": len(ok),
            "success_rate": len(ok) / len(rs),
            "in_band_30_70": in_band(len(ok) / len(rs)),
            "n_nominal_ok_seeds": len(paired),
            "success_on_nominal_ok": sum(r["success"] for r in paired),
            "success_rate_on_nominal_ok": (sum(r["success"] for r in paired) / len(paired))
            if paired else float("nan"),
            "lap_time_mean_s": _mean([r["time_s"] for r in ok]),
            "gates_mean_all": _mean([r["n_gates"] for r in rs]),
            "gates_mean_failed": _mean([r["n_gates"] for r in bad]),
            "failure_causes": ", ".join(f"{k_}:{v}" for k_, v in sorted(groups.items())) or "-",
            "min_z_airborne_min": min(zs) if zs else float("nan"),
            "min_z_airborne_mean": _mean(zs),
        })
    return out


def grid_table(cond: list[dict[str, Any]], T: float, field: str = "success_rate",
               ks: tuple[float, ...] = K_VALUES, ds: tuple[int, ...] = D_VALUES) -> str:
    """Markdown table for one T: rows = delay d, columns = k_end. `*` marks the 30-70% band."""
    by = {(c["delay"], c["k_end"]): c for c in cond if c["T_traj"] == T}
    head = "| d \\ k_end | " + " | ".join(f"{k:g}" for k in ks) + " |\n|" + "---|" * (len(ks) + 1)
    lines = [head]
    for d in ds:
        cells = []
        for k in ks:
            c = by.get((d, k))
            if c is None:
                cells.append("n/a")
                continue
            v = c[field]
            if field == "success_rate":
                cells.append(f"{v:.0%} ({c['n_success']}/{c['n']})" + (" *" if c["in_band_30_70"] else ""))
            elif field == "success_rate_on_nominal_ok":
                cells.append("-" if math.isnan(v) else
                             f"{v:.0%} ({c['success_on_nominal_ok']}/{c['n_nominal_ok_seeds']})"
                             + (" *" if in_band(v) else ""))
            else:
                cells.append(_fmt(v))
        lines.append(f"| d={d} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def exp_b(out: Path, seeds: range = range(20), ts: tuple[float, ...] = T_VALUES,
          ds: tuple[int, ...] = D_VALUES, ks: tuple[float, ...] = K_VALUES,
          config: str = "att_level0.toml", log_episodes: bool = False) -> None:
    """Run every (T_traj, d, k_end) on identical seeds and write tables + CSVs."""
    sub = out / "expB"
    sub.mkdir(parents=True, exist_ok=False)
    cfg = load_cfg(config)
    mpc = load_controller_cls(PROJECT_DIR / "my_mpc.py")
    env, lg = _build_env(cfg, "wrapped", (sub / "logs") if log_episodes else None,
                         dict(k_end=1.0, delay=0, T_end=ts[0]))  # set per block below
    f = open(sub / "episodes.csv", "w", newline="")
    w = csv.DictWriter(f, fieldnames=GRID_FIELDS)
    w.writeheader()
    rows: list[dict[str, Any]] = []
    for T in ts:
        for d in ds:
            for k in ks:
                env.set_params(k_end=k, delay=d, T_end=T)  # k(t) must reach k_end within the lap
                lg.tag = f"B_T{T:g}_d{d}_k{k:g}"
                for seed in seeds:
                    res = run_episode(env, functools.partial(mpc, T_traj=T), cfg, seed)
                    row = {**make_row("grid", T, f"d{d}_k{k:g}", res), "delay": d, "k_end": k}
                    rows.append(row)
                    w.writerow(row)
                    f.flush()
                    print(f"[B] T={T:g} d={d} k={k:g} seed={seed:2d} ok={res.success} "
                          f"gates={res.n_gates} t={res.time_s:.2f}s group={row['group']} "
                          f"minz_air={res.min_z_airborne:.3f}", flush=True)
    f.close()
    env.close()
    cond = summarize_grid(rows)
    write_csv(sub / "conditions.csv", cond)
    write_csv(sub / "failures.csv", [r for r in rows if not r["success"]])
    md = []
    for T in ts:
        md.append(f"### T_traj = {T:g}  (success rate; `*` = within 30-70%)\n\n"
                  + grid_table(cond, T, "success_rate"))
        md.append(f"### T_traj = {T:g}  (success on seeds where nominal d=0,k=1 succeeded)\n\n"
                  + grid_table(cond, T, "success_rate_on_nominal_ok"))
        md.append(f"### T_traj = {T:g}  (mean lap time of successes [s])\n\n"
                  + grid_table(cond, T, "lap_time_mean_s"))
        md.append(f"### T_traj = {T:g}  (mean gates passed, all episodes)\n\n"
                  + grid_table(cond, T, "gates_mean_all"))
        md.append(f"### T_traj = {T:g}  (min z after take-off over all episodes [m])\n\n"
                  + grid_table(cond, T, "min_z_airborne_min"))
    md.append("### Failure causes per condition\n\n| T_traj | d | k_end | causes |\n|---|---|---|---|\n"
              + "\n".join(f"| {c['T_traj']:g} | {c['delay']} | {c['k_end']:g} | "
                          f"{c['failure_causes']} |" for c in cond))
    band = [c for c in cond if c["in_band_30_70"]]
    md.append("### Conditions in the 30-70% band\n\n" + (
        "\n".join(f"- T={c['T_traj']:g}, d={c['delay']}, k_end={c['k_end']:g}: "
                  f"{c['success_rate']:.0%} ({c['n_success']}/{c['n']})" for c in band)
        if band else "(none)"))
    (sub / "tables.md").write_text("\n\n".join(md) + "\n")
    print("\n" + "\n\n".join(md), flush=True)


def main(exp: str = "A", out_dir: str | None = None) -> None:
    """Run experiment A or B into results/grid_<timestamp>/ (or into an existing `out_dir`)."""
    exp = exp.upper()
    if exp not in ("A", "B"):
        raise ValueError("exp must be 'A' or 'B'")
    out = Path(out_dir) if out_dir else RESULTS_DIR / f"grid_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"exp{exp}").exists():
        raise FileExistsError(f"{out / f'exp{exp}'} already exists; refusing to overwrite")
    sys.stdout = Tee(sys.stdout, open(out / f"run_{exp}.log", "x"))
    print(f"Writing results to {out / f'exp{exp}'}", flush=True)
    (out / f"params_{exp}.json").write_text(json.dumps(
        dict(exp=exp, T=T_VALUES, d=D_VALUES, k=K_VALUES, seeds="0-19 (B), 0,1,2 (A)"), indent=2))
    (exp_a if exp == "A" else exp_b)(out)


if __name__ == "__main__":
    fire.Fire(main)
