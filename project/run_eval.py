"""Step 4: paired evaluation of my_mpc over T_traj, nominal vs. thrust-degradation + delay wrapper.

Run from project/ (acados writes c_generated_code/ into the working directory):

    cd project && pixi run python run_eval.py

Stage 1 (sweep): every T_traj in `ts`, conditions {nominal, wrapped}, seeds 0..n_sweep-1. The SAME
    seeds are used in every condition (paired comparison).
Stage 2 (final): smallest T_traj whose nominal success rate is >= `threshold`; ONLY that T_traj is
    run again (nominal and wrapped) with seeds 0..n_final-1.

Both conditions use config level 0 (att_level0.toml). "nominal" has no wrapper; "wrapped" has
ThrustDegradeDelay(k_end, T_end, delay). Success = n_gates_passed == 5. Lap time = steps / 50 and is
averaged over successful runs only. Failures are listed separately. Results go to a NEW folder
`results/eval_<timestamp>/` (never overwritten).
"""

from __future__ import annotations

import lsy_drone_racing  # noqa: F401  (must be imported before scipy)

import csv
import functools
import json
import math
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import numpy as np

from common import PROJECT_DIR, RESULTS_DIR, EpisodeResult, Tee, load_cfg, load_controller_cls
from common import make_env, run_episode
from logger import EpisodeLogger
from wrapper import ThrustDegradeDelay

SWEEP_TS = (18.75, 16.0, 14.0, 12.0, 10.0)
FALL_Z = 0.05  # m; below this (after take-off) a timeout is counted as a fall to the floor
CONDITIONS = ("nominal", "wrapped")
FIELDS = ["stage", "T_traj", "condition", *[f.name for f in fields(EpisodeResult)], "group"]


def failure_group(r: EpisodeResult) -> str:
    """Coarse outcome: success | collision | fall | timeout | out_of_bounds.

    The env does not terminate on floor contact, so a crash to the floor ends as a timeout or as
    the controller's trajectory end. Those are reclassified as `fall` from the z statistics.
    """
    if r.success:
        return "success"
    if r.cause in ("gate", "obstacle", "contact_unknown"):
        return "collision"
    if r.cause == "floor":
        return "fall"
    if r.cause == "out_of_bounds":
        return "out_of_bounds"
    low_after_takeoff = not math.isnan(r.min_z_airborne) and r.min_z_airborne < FALL_Z
    return "fall" if (r.final_z < FALL_Z or low_after_takeoff) else "timeout"


def make_row(stage: str, T: float, condition: str, r: EpisodeResult) -> dict[str, Any]:
    """One CSV row for an episode."""
    return {"stage": stage, "T_traj": T, "condition": condition, **r.as_row(),
            "group": failure_group(r)}


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (T_traj, condition): success rate, lap time (successes only), gates, failure causes."""
    out = []
    keys = sorted({(r["T_traj"], r["condition"]) for r in rows},
                  key=lambda k: (-k[0], CONDITIONS.index(k[1])))
    for T, cond in keys:
        rs = [r for r in rows if r["T_traj"] == T and r["condition"] == cond]
        ok = [r for r in rs if r["success"]]
        bad = [r for r in rs if not r["success"]]
        groups: dict[str, int] = {}
        for r in bad:
            groups[r["group"]] = groups.get(r["group"], 0) + 1
        out.append({
            "T_traj": T, "condition": cond, "n": len(rs), "n_success": len(ok),
            "success_rate": len(ok) / len(rs),
            "lap_time_mean_s": _mean([r["time_s"] for r in ok]),
            "lap_time_std_s": float(np.std([r["time_s"] for r in ok])) if ok else float("nan"),
            "gates_mean_all": _mean([r["n_gates"] for r in rs]),
            "gates_mean_failed": _mean([r["n_gates"] for r in bad]),
            "failure_causes": ", ".join(f"{k}:{v}" for k, v in sorted(groups.items())) or "-",
        })
    return out


def choose_T(sweep_summary: list[dict[str, Any]], threshold: float = 0.9) -> float | None:
    """Smallest T_traj whose nominal success rate is >= threshold (None if there is none)."""
    ok = [s["T_traj"] for s in sweep_summary
          if s["condition"] == "nominal" and s["success_rate"] >= threshold - 1e-9]
    return min(ok) if ok else None


def paired_table(rows: list[dict[str, Any]], T: float) -> dict[str, Any]:
    """2x2 paired outcome counts over identical seeds, plus the paired lap-time difference."""
    by = {c: {r["seed"]: r for r in rows if r["T_traj"] == T and r["condition"] == c}
          for c in CONDITIONS}
    seeds = sorted(set(by["nominal"]) & set(by["wrapped"]))
    n_ok = [by["nominal"][s]["success"] for s in seeds]
    w_ok = [by["wrapped"][s]["success"] for s in seeds]
    both = [s for s, a, b in zip(seeds, n_ok, w_ok) if a and b]
    return {
        "n_pairs": len(seeds),
        "both_success": len(both),
        "nominal_only": sum(a and not b for a, b in zip(n_ok, w_ok)),
        "wrapped_only": sum(b and not a for a, b in zip(n_ok, w_ok)),
        "both_fail": sum(not a and not b for a, b in zip(n_ok, w_ok)),
        "lap_diff_mean_s": _mean([by["wrapped"][s]["time_s"] - by["nominal"][s]["time_s"]
                                  for s in both]),
    }


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return "-" if math.isnan(x) else (f"{x:.3f}" if abs(x) < 1 else f"{x:.2f}")
    return str(x)


def summary_markdown(summary: list[dict[str, Any]]) -> str:
    """Markdown table of `summarize` output."""
    cols = ["T_traj", "condition", "n", "n_success", "success_rate", "lap_time_mean_s",
            "lap_time_std_s", "gates_mean_all", "gates_mean_failed", "failure_causes"]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(_fmt(s[c]) for c in cols) + " |" for s in summary)


def failures_markdown(rows: list[dict[str, Any]]) -> str:
    """Markdown list of failed episodes (reported separately from lap times)."""
    cols = ["T_traj", "condition", "seed", "n_gates", "time_s", "group", "cause", "min_z_airborne",
            "final_z", "detail"]
    bad = [r for r in rows if not r["success"]]
    if not bad:
        return "(no failures)"
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in bad)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows to CSV (header from the first row)."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _build_env(cfg: Any, condition: str, log_dir: Path | None, wrapper_kw: dict) -> tuple:
    """Env with the logger innermost and (for `wrapped`) the wrapper outside it."""
    holder: dict[str, EpisodeLogger] = {}

    def wrap(e: Any) -> Any:
        lg = EpisodeLogger(e, log_dir, freq=cfg.env.freq)
        holder["lg"] = lg
        if condition == "wrapped":
            return ThrustDegradeDelay(lg, freq=cfg.env.freq, **wrapper_kw)
        return lg

    return make_env(cfg, wrapper=wrap), holder["lg"]


def run_block(stage: str, T: float, condition: str, seeds: range, cfg: Any, env: Any,
              lg: EpisodeLogger, ctrl_cls: Any, sink: Any) -> list[dict[str, Any]]:
    """Run all `seeds` for one (T_traj, condition); append each finished row to `sink`."""
    lg.tag = f"{stage}_T{T:g}_{condition}"
    rows = []
    for seed in seeds:
        res = run_episode(env, functools.partial(ctrl_cls, T_traj=T), cfg, seed)
        row = make_row(stage, T, condition, res)
        rows.append(row)
        sink(row)
        print(f"[{stage}] T={T:g} {condition:7s} seed={seed:3d} ok={res.success} "
              f"gates={res.n_gates} t={res.time_s:.2f}s group={row['group']} "
              f"minz_air={res.min_z_airborne:.3f}", flush=True)
    return rows


def run_eval(
    out_dir: str | None = None,
    ts: tuple[float, ...] = SWEEP_TS,
    n_sweep: int = 20,
    n_final: int = 50,
    threshold: float = 0.9,
    k_end: float = 0.85,
    T_end: float = 18.0,
    delay: int = 2,
    config: str = "att_level0.toml",
    controller: str = str(PROJECT_DIR / "my_mpc.py"),
    log_episodes: bool = True,
) -> None:
    """Run the sweep, pick T_traj, run the final comparison, and write tables + CSVs."""
    ts = (float(ts),) if isinstance(ts, (int, float)) else tuple(float(t) for t in ts)
    out = Path(out_dir) if out_dir else RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)  # never reuse or overwrite an existing folder
    sys.stdout = Tee(sys.stdout, open(out / "run.log", "w"))
    params = dict(ts=ts, n_sweep=n_sweep, n_final=n_final, threshold=threshold, k_end=k_end,
                  T_end=T_end, delay=delay, config=config, controller=controller)
    (out / "params.json").write_text(json.dumps(params, indent=2))
    print(f"Writing results to {out}\nparams: {params}", flush=True)

    cfg = load_cfg(config)
    ctrl_cls = load_controller_cls(controller)
    wrapper_kw = dict(k_end=k_end, T_end=T_end, delay=delay)
    log_dir = out / "episode_logs" if log_episodes else None
    envs = {c: _build_env(cfg, c, log_dir, wrapper_kw) for c in CONDITIONS}

    csv_file = open(out / "episodes.csv", "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    writer.writeheader()

    def sink(row: dict[str, Any]) -> None:
        writer.writerow(row)
        csv_file.flush()

    # ---- Stage 1: sweep over T_traj, identical seeds in every condition
    sweep_rows: list[dict[str, Any]] = []
    for T in ts:
        for cond in CONDITIONS:
            env, lg = envs[cond]
            sweep_rows += run_block("sweep", T, cond, range(n_sweep), cfg, env, lg, ctrl_cls, sink)
    sweep = summarize(sweep_rows)
    write_csv(out / "summary_sweep.csv", sweep)
    write_csv(out / "failures_sweep.csv", [r for r in sweep_rows if not r["success"]])
    md = (f"## Sweep (level 0, seeds 0-{n_sweep - 1}, same seeds in all rows)\n\n"
          f"{summary_markdown(sweep)}\n\n### Failures (sweep)\n\n{failures_markdown(sweep_rows)}\n")
    (out / "results.md").write_text(md)
    print("\n" + md, flush=True)

    # ---- Stage 2: smallest T_traj with nominal success >= threshold, seeds 0..n_final-1
    T_star = choose_T(sweep, threshold)
    if T_star is None:
        msg = f"No T_traj reached nominal success >= {threshold:.0%}; final stage NOT run."
        print(msg, flush=True)
        (out / "results.md").write_text(md + f"\n{msg}\n")
        csv_file.close()
        return
    print(f"\nSelected T_traj = {T_star:g} (smallest with nominal >= {threshold:.0%})", flush=True)
    final_rows: list[dict[str, Any]] = []
    for cond in CONDITIONS:
        env, lg = envs[cond]
        final_rows += run_block("final", T_star, cond, range(n_final), cfg, env, lg, ctrl_cls, sink)
    final = summarize(final_rows)
    pair = paired_table(final_rows, T_star)
    write_csv(out / "summary_final.csv", final)
    write_csv(out / "failures_final.csv", [r for r in final_rows if not r["success"]])
    write_csv(out / "paired_final.csv", [pair])
    md2 = (f"\n## Final (T_traj={T_star:g}, seeds 0-{n_final - 1}, paired)\n\n"
           f"{summary_markdown(final)}\n\n### Paired outcomes (same seeds)\n\n"
           + "\n".join(f"- {k}: {_fmt(v)}" for k, v in pair.items())
           + f"\n\n### Failures (final)\n\n{failures_markdown(final_rows)}\n")
    (out / "results.md").write_text(md + md2)
    print(md2, flush=True)
    csv_file.close()
    for env, _ in envs.values():
        env.close()


if __name__ == "__main__":
    fire.Fire(run_eval)
