"""Step 0: run the attitude MPC N times per level with fixed seeds and diagnose every failure.

Run from project/ (acados writes c_generated_code/ into the working directory):

    cd project && pixi run python diagnose.py --levels 0 1 --n_runs 20
"""

from __future__ import annotations

import lsy_drone_racing  # noqa: F401  (must be imported before scipy)

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from common import PROJECT_DIR, RESULTS_DIR, EpisodeResult, Tee, load_cfg, load_controller_cls
from common import make_env, run_episode

REPO = PROJECT_DIR.parent
logger = logging.getLogger("diagnose")


def diagnose(
    levels: tuple[int, ...] = (0, 1),
    n_runs: int = 20,
    seed0: int = 0,
    controller: str = "attitude_mpc.py",
    repeat_failures: bool = True,
    out_dir: str | None = None,
) -> None:
    """Run `n_runs` seeded episodes per level; print a per-failure report; save CSV + traces."""
    levels = (levels,) if isinstance(levels, int) else tuple(levels)
    out = Path(out_dir) if out_dir else RESULTS_DIR / f"step0_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)  # never reuse or overwrite an existing folder
    sys.stdout = Tee(sys.stdout, open(out / "step0.log", "w"))  # log lives in the results folder
    print(f"Writing results to {out}", flush=True)
    ctrl_cls = load_controller_cls(REPO / "lsy_drone_racing" / "control" / controller)
    for level in levels:
        cfg = load_cfg(f"att_level{level}.toml")
        env = make_env(cfg)
        results: list[EpisodeResult] = []
        traces: dict[int, np.ndarray] = {}
        for seed in range(seed0, seed0 + n_runs):
            trace: list = []
            res = run_episode(env, ctrl_cls, cfg, seed, keep_trace=trace)
            results.append(res)
            if not res.success:
                traces[seed] = np.asarray(trace)
            print(f"L{level} seed={seed:3d} ok={res.success} gates={res.n_gates}/{res.n_gates_total} "
                  f"t={res.time_s:.2f}s cause={res.cause} {res.detail}", flush=True)
        fails = [r for r in results if not r.success]
        print(f"\n== Level {level}: {n_runs - len(fails)}/{n_runs} successes ==")
        for r in fails:
            print(f"  FAIL seed={r.seed} t={r.time_s:.2f}s gates={r.n_gates} cause={r.cause} "
                  f"({r.detail}) impact_pos={np.round(r.last_pos, 3).tolist()}")
        if repeat_failures and fails:
            print("  -- reproducibility: re-running failing seeds --")
            for r in fails:
                r2 = run_episode(env, ctrl_cls, cfg, r.seed)
                same = (r2.steps, r2.n_gates, r2.cause) == (r.steps, r.n_gates, r.cause)
                print(f"  seed={r.seed}: rerun steps={r2.steps} gates={r2.n_gates} "
                      f"cause={r2.cause} identical={same}")
        with open(out / f"step0_level{level}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].as_row()))
            w.writeheader()
            w.writerows(r.as_row() for r in results)
        if traces:
            np.savez(out / f"step0_level{level}_fail_traces.npz",
                     **{f"seed{s}": t for s, t in traces.items()})
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(diagnose)
