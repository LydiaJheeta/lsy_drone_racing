"""Round 3: experiment C (is the delay fragility a tuning artefact?) and D (thrust drop at speed).

Run from project/ with the ROS PYTHONPATH removed (see README):

    cd project
    env -u PYTHONPATH pixi run python round3.py --exp=C          # creates results/round3_<ts>/
    env -u PYTHONPATH pixi run python round3.py --exp=D --out_dir=results/round3_<ts>

Experiment C -- level 0, T_traj = 18.75, k_end = 1.0 (no thrust drop), seeds 0-19,
    delay d in {0, 1, 2} steps x `u_weight_scale` in {1, 3, 10, 30}. `u_weight_scale` multiplies
    the whole input-weight block R = diag([1, 1, 1, 50]) of the MPC cost, i.e. every term that
    penalises a command (see my_mpc.py). d=0 at the same scale is the control of each column.

Experiment D -- level 0, d = 0, u_weight_scale = 1, seeds 0-19,
    T_traj in {16, 14, 12} x k_end in {1.0, 0.9, 0.8, 0.7, 0.6}, with T_end = T_traj so that k(t)
    reaches k_end at the end of the reference (the old code kept T_end at 18 s, so a fast lap never
    saw the full drop). k_end = 1.0 is the per-T nominal baseline and is run in the same block list.

Both experiments report, per condition: success rate, success restricted to the seeds where that
condition's baseline succeeded, mean lap time of the successes, failure causes, and two saturation
measures of the MPC's OWN command (before the wrapper touches it):

* `tilt_sat_pct`: share of steps with max(|roll_cmd|, |pitch_cmd|) >= 0.49 rad (the MPC's own
  bound is 0.5 rad), i.e. how much of the flight is spent on the attitude constraint.
* `thrust_above_eff_pct`: share of steps with thrust_cmd > thrust_max * k(t). The MPC does not know
  about the degradation, so its command u is applied as u * k(t); the most the degraded actuator can
  deliver is thrust_max * k(t) = 0.8 N * k(t). Steps above that are requests that cannot be met.
  k is evaluated at the time the command is issued, which equals the time it is applied for d = 0.
"""

from __future__ import annotations

import lsy_drone_racing  # noqa: F401  (must be imported before scipy)

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Iterable, Sequence  # noqa: E402

import fire  # noqa: E402
import numpy as np  # noqa: E402

from common import PROJECT_DIR, RESULTS_DIR, Tee, load_cfg  # noqa: E402
from common import load_controller_cls, run_episode  # noqa: E402
from grid import write_csv  # noqa: E402
from run_eval import FIELDS, _build_env, _fmt, _mean, make_row  # noqa: E402
from wrapper import THRUST_IDX  # noqa: E402

C_T = 18.75
C_DELAYS = (0, 1, 2)
C_SCALES = (1.0, 3.0, 10.0, 30.0)
D_DELAY = 0
D_SCALE = 1.0
D_TS = (16.0, 14.0, 12.0)
D_KS = (1.0, 0.9, 0.8, 0.7, 0.6)
N_SEEDS = 20
TILT_SAT = 0.49  # rad; the MPC's own roll/pitch bound is 0.5

BLOCK_KEYS = ("T_traj", "delay", "k_end", "u_weight_scale")
CMD_FIELDS = [
    "n_cmd_steps", "tilt_sat_steps", "tilt_cmd_max",
    "thrust_above_eff_steps", "thrust_cmd_max", "thrust_cmd_mean",
]
EPISODE_FIELDS = [*FIELDS, "delay", "k_end", "u_weight_scale", *CMD_FIELDS]


# --------------------------------------------------------------------------------------------
# per-episode command recording and metrics
# --------------------------------------------------------------------------------------------


def recording_factory(ctrl_cls: Any, store: list, **kwargs: Any) -> Callable:
    """Controller factory that appends every command the MPC returns to `store`.

    `run_episode` only sees the command *before* the wrapper, so the recorded values are the MPC's
    own requests: that is what the saturation metrics are about.
    """

    def factory(obs: Any, info: Any, config: Any) -> Any:
        controller = ctrl_cls(obs, info, config, **kwargs)
        inner = controller.compute_control

        def recording_compute_control(o: Any, i: Any = None) -> Any:
            u = inner(o, i)
            store.append(np.asarray(u, dtype=float).copy())
            return u

        controller.compute_control = recording_compute_control
        return controller

    return factory


def cmd_metrics(
    cmds: Sequence[np.ndarray], k_of_t: Callable[[float], float], thrust_max: float, freq: float
) -> dict[str, Any]:
    """Saturation counts over the MPC's own commands of one episode (see the module docstring)."""
    nan = float("nan")
    if len(cmds) == 0:
        return dict(n_cmd_steps=0, tilt_sat_steps=0, tilt_cmd_max=nan,
                    thrust_above_eff_steps=0, thrust_cmd_max=nan, thrust_cmd_mean=nan)
    a = np.asarray(cmds, dtype=float)
    tilt = np.maximum(np.abs(a[:, 0]), np.abs(a[:, 1]))
    k = np.array([k_of_t(n / freq) for n in range(len(a))])
    return {
        "n_cmd_steps": int(len(a)),
        "tilt_sat_steps": int((tilt >= TILT_SAT).sum()),
        "tilt_cmd_max": float(tilt.max()),
        "thrust_above_eff_steps": int((a[:, THRUST_IDX] > thrust_max * k).sum()),
        "thrust_cmd_max": float(a[:, THRUST_IDX].max()),
        "thrust_cmd_mean": float(a[:, THRUST_IDX].mean()),
    }


# --------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------


def _pct(num: Iterable[int], den: Iterable[int]) -> float:
    """100 * sum(num) / sum(den) in percent, NaN for an empty denominator."""
    d = sum(den)
    return 100.0 * sum(num) / d if d else float("nan")


def summarize_conditions(
    rows: list[dict[str, Any]], keys: Sequence[str], baseline_of: Callable[[tuple], tuple]
) -> list[dict[str, Any]]:
    """One summary row per distinct value of `keys`.

    `baseline_of(key)` names the condition a key is compared against (its own control). The
    `*_on_baseline_ok` columns restrict the success rate to the seeds on which that baseline
    succeeded, so a column cannot look bad merely because its baseline already failed.
    """
    def key_of(r: dict[str, Any]) -> tuple:
        return tuple(r[k] for k in keys)

    out = []
    for key in sorted({key_of(r) for r in rows}):
        rs = [r for r in rows if key_of(r) == key]
        base = baseline_of(key)
        base_ok = {r["seed"] for r in rows if key_of(r) == base and r["success"]}
        ok = [r for r in rs if r["success"]]
        bad = [r for r in rs if not r["success"]]
        paired = [r for r in rs if r["seed"] in base_ok]
        n_paired_ok = sum(bool(r["success"]) for r in paired)
        groups: dict[str, int] = {}
        for r in bad:
            groups[r["group"]] = groups.get(r["group"], 0) + 1
        zs = [r["min_z_airborne"] for r in rs if not math.isnan(float(r["min_z_airborne"]))]
        out.append({
            **dict(zip(keys, key)),
            "n": len(rs),
            "n_success": len(ok),
            "success_rate": len(ok) / len(rs) if rs else float("nan"),
            "n_baseline_ok_seeds": len(paired),
            "success_on_baseline_ok": n_paired_ok,
            "success_rate_on_baseline_ok": n_paired_ok / len(paired) if paired else float("nan"),
            "lap_time_mean_s": _mean([r["time_s"] for r in ok]),
            "lap_time_std_s": float(np.std([r["time_s"] for r in ok])) if ok else float("nan"),
            "gates_mean_all": _mean([r["n_gates"] for r in rs]),
            "gates_mean_failed": _mean([r["n_gates"] for r in bad]),
            "failure_causes": ", ".join(f"{k}:{v}" for k, v in sorted(groups.items())) or "-",
            "tilt_sat_pct_all": _pct([r["tilt_sat_steps"] for r in rs],
                                     [r["n_cmd_steps"] for r in rs]),
            "tilt_sat_pct_success": _pct([r["tilt_sat_steps"] for r in ok],
                                         [r["n_cmd_steps"] for r in ok]),
            "thrust_above_eff_pct_all": _pct([r["thrust_above_eff_steps"] for r in rs],
                                             [r["n_cmd_steps"] for r in rs]),
            "thrust_above_eff_pct_success": _pct([r["thrust_above_eff_steps"] for r in ok],
                                                 [r["n_cmd_steps"] for r in ok]),
            "thrust_cmd_max": max([r["thrust_cmd_max"] for r in rs], default=float("nan")),
            "min_z_airborne_min": min(zs) if zs else float("nan"),
        })
    return out


def cell_success(c: dict[str, Any]) -> str:
    """`rate (n_success/n)`."""
    return f"{c['success_rate']:.0%} ({c['n_success']}/{c['n']})"


def cell_success_on_baseline(c: dict[str, Any]) -> str:
    """`rate (n/n)` restricted to the seeds where the condition's baseline succeeded."""
    v = c["success_rate_on_baseline_ok"]
    if math.isnan(v):
        return "-"
    return f"{v:.0%} ({c['success_on_baseline_ok']}/{c['n_baseline_ok_seeds']})"


def cell_field(field: str, pct: bool = False) -> Callable[[dict[str, Any]], str]:
    """Renderer for a plain numeric column."""
    def cell(c: dict[str, Any]) -> str:
        v = c[field]
        if isinstance(v, float) and math.isnan(v):
            return "-"
        return f"{v:.1f}%" if pct else _fmt(v)

    return cell


def pivot_table(
    cond: list[dict[str, Any]],
    row_key: str,
    col_key: str,
    row_vals: Sequence[float],
    col_vals: Sequence[float],
    cell: Callable[[dict[str, Any]], str],
) -> str:
    """Markdown table with one row per `row_vals` and one column per `col_vals`."""
    by = {(c[row_key], c[col_key]): c for c in cond}
    lines = [
        f"| {row_key} \\ {col_key} | " + " | ".join(f"{v:g}" for v in col_vals) + " |",
        "|" + "---|" * (len(col_vals) + 1),
    ]
    for rv in row_vals:
        cells = [cell(by[(rv, cv)]) if (rv, cv) in by else "n/a" for cv in col_vals]
        lines.append(f"| {rv:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tables_markdown(
    cond: list[dict[str, Any]], row_key: str, col_key: str,
    row_vals: Sequence[float], col_vals: Sequence[float], title: str,
) -> str:
    """The full set of tables for one experiment."""
    specs = [
        ("success rate", cell_success),
        ("success on the seeds where the baseline succeeded", cell_success_on_baseline),
        ("mean lap time of the successes [s]", cell_field("lap_time_mean_s")),
        ("steps with |roll| or |pitch| command >= 0.49 rad, all episodes",
         cell_field("tilt_sat_pct_all", pct=True)),
        ("steps with thrust command > 0.8 N * k(t), all episodes",
         cell_field("thrust_above_eff_pct_all", pct=True)),
        ("mean gates passed, all episodes", cell_field("gates_mean_all")),
        ("failure causes", lambda c: c["failure_causes"]),
    ]
    md = [f"## {title}"]
    for name, cell in specs:
        md.append(f"### {name}\n\n"
                  + pivot_table(cond, row_key, col_key, row_vals, col_vals, cell))
    return "\n\n".join(md)


# --------------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------------


def run_blocks(
    sub: Path, stage: str, blocks: list[dict[str, Any]], seeds: Sequence[int],
    cfg: Any, ctrl_cls: Any,
) -> list[dict[str, Any]]:
    """Run every block on the same seeds; stream one row per episode to `episodes.csv`."""
    env, lg = _build_env(cfg, "wrapped", None,
                         dict(k_end=1.0, delay=0, T_end=blocks[0]["T_traj"]))
    thrust_max = float(np.asarray(env.action_space.high)[THRUST_IDX])
    rows: list[dict[str, Any]] = []
    with open(sub / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        for b in blocks:
            T, d, k, s = (b[key] for key in BLOCK_KEYS)
            # T_end = T_traj of THIS block: otherwise a fast lap ends before k(t) reaches k_end.
            env.set_params(k_end=k, delay=d, T_end=T)
            lg.tag = f"{stage}_T{T:g}_d{d}_k{k:g}_w{s:g}"
            for seed in seeds:
                cmds: list[np.ndarray] = []
                factory = recording_factory(ctrl_cls, cmds, T_traj=T, u_weight_scale=s)
                res = run_episode(env, factory, cfg, seed)
                row = {
                    **make_row(stage, T, f"d{d}_k{k:g}_w{s:g}", res),
                    "delay": d, "k_end": k, "u_weight_scale": s,
                    **cmd_metrics(cmds, env.thrust_scale, thrust_max, cfg.env.freq),
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()
                tilt_pct = 100.0 * row["tilt_sat_steps"] / max(row["n_cmd_steps"], 1)
                thr_pct = 100.0 * row["thrust_above_eff_steps"] / max(row["n_cmd_steps"], 1)
                print(f"[{stage}] T={T:g} d={d} k={k:g} w={s:g} seed={seed:2d} "
                      f"ok={res.success} gates={res.n_gates} t={res.time_s:.2f}s "
                      f"group={row['group']} tilt_sat={tilt_pct:.1f}% "
                      f"thr_over={thr_pct:.1f}% minz_air={res.min_z_airborne:.3f}", flush=True)
    env.close()
    return rows


def exp_c(out: Path, seeds: Sequence[int], cfg: Any, ctrl_cls: Any,
          ds: Sequence[int] = C_DELAYS, scales: Sequence[float] = C_SCALES,
          T: float = C_T, k_end: float = 1.0) -> None:
    """Delay x input-weight-scale grid at the nominal reference duration."""
    sub = out / "expC"
    sub.mkdir(parents=True, exist_ok=False)
    blocks = [dict(T_traj=float(T), delay=int(d), k_end=float(k_end), u_weight_scale=float(s))
              for s in scales for d in ds]
    rows = run_blocks(sub, "expC", blocks, seeds, cfg, ctrl_cls)
    cond = summarize_conditions(rows, ("delay", "u_weight_scale"), lambda key: (0, key[1]))
    write_csv(sub / "conditions.csv", cond)
    write_csv(sub / "failures.csv", [r for r in rows if not r["success"]])
    md = tables_markdown(
        cond, "delay", "u_weight_scale", [int(d) for d in ds], [float(s) for s in scales],
        f"Experiment C -- T_traj={T:g}, k_end={k_end:g}, level 0, seeds "
        f"{min(seeds)}-{max(seeds)}. Baseline of each column: d=0 at the same scale.",
    )
    (sub / "tables.md").write_text(md + "\n")
    print("\n" + md, flush=True)


def exp_d(out: Path, seeds: Sequence[int], cfg: Any, ctrl_cls: Any,
          ts: Sequence[float] = D_TS, ks: Sequence[float] = D_KS,
          delay: int = D_DELAY, scale: float = D_SCALE) -> None:
    """T_traj x k_end grid with T_end = T_traj, no delay, original weights."""
    sub = out / "expD"
    sub.mkdir(parents=True, exist_ok=False)
    blocks = [dict(T_traj=float(T), delay=int(delay), k_end=float(k), u_weight_scale=float(scale))
              for T in ts for k in ks]
    rows = run_blocks(sub, "expD", blocks, seeds, cfg, ctrl_cls)
    cond = summarize_conditions(rows, ("T_traj", "k_end"), lambda key: (key[0], 1.0))
    write_csv(sub / "conditions.csv", cond)
    write_csv(sub / "failures.csv", [r for r in rows if not r["success"]])
    md = tables_markdown(
        cond, "T_traj", "k_end", [float(t) for t in ts], [float(k) for k in ks],
        f"Experiment D -- d={delay}, u_weight_scale={scale:g}, T_end=T_traj, level 0, seeds "
        f"{min(seeds)}-{max(seeds)}. Baseline of each row: k_end=1.0 at the same T_traj.",
    )
    (sub / "tables.md").write_text(md + "\n")
    print("\n" + md, flush=True)


def main(exp: str = "C", out_dir: str | None = None, n_seeds: int = N_SEEDS,
         config: str = "att_level0.toml", controller: str | None = None) -> None:
    """Run experiment C and/or D into results/round3_<timestamp>/ (or an existing `out_dir`)."""
    exp = exp.upper()
    if exp not in ("C", "D", "CD"):
        raise ValueError("exp must be 'C', 'D' or 'CD'")
    out = Path(out_dir) if out_dir else RESULTS_DIR / f"round3_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    for e in exp:
        if (out / f"exp{e}").exists():
            raise FileExistsError(f"{out / f'exp{e}'} already exists; refusing to overwrite")
    sys.stdout = Tee(sys.stdout, open(out / f"run_{exp}.log", "x"))
    ctrl_path = controller or str(PROJECT_DIR / "my_mpc.py")
    seeds = list(range(n_seeds))
    (out / f"params_{exp}.json").write_text(json.dumps(dict(
        exp=exp, seeds=f"{seeds[0]}-{seeds[-1]}", config=config,
        controller=Path(ctrl_path).name, tilt_sat_rad=TILT_SAT,
        C=dict(T_traj=C_T, delays=C_DELAYS, u_weight_scales=C_SCALES, k_end=1.0),
        D=dict(ts=D_TS, ks=D_KS, delay=D_DELAY, u_weight_scale=D_SCALE, T_end="= T_traj"),
    ), indent=2))
    print(f"Writing results to {out} (exp {exp}, seeds {seeds[0]}-{seeds[-1]})", flush=True)
    cfg = load_cfg(config)
    ctrl_cls = load_controller_cls(ctrl_path)
    if "C" in exp:
        exp_c(out, seeds, cfg, ctrl_cls)
    if "D" in exp:
        exp_d(out, seeds, cfg, ctrl_cls)


if __name__ == "__main__":
    fire.Fire(main)
