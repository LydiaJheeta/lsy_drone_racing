"""Throughput benchmark: vectorized env steps/s on CPU with random actions, no training.

Measured pipelines (`--pipelines`):

* `race_state`     -- `VecDroneRaceEnv` (`DroneRacing-v0` vector entry point), `control_mode="state"`
                      (`config/level0.toml`, the upstream config).
* `race_attitude`  -- the same env with `control_mode="attitude"` (`project/config/att_level0.toml`).
* `rl_train`       -- `rl.env.make_train_env`: the phase-1 PPO stack (`GateProgressRaceEnv` +
                      JaxToNumpy + `VecAttitudeAction`), random actions in [-1,1]^3.
* `rl_train_perturb` -- the same stack with `rl.vec_perturb.VecThrustDegradeDelay` inserted
                      (per-world delay 0-2 steps, k_end in [0.6, 1.0], T_end=18.75 s), to measure
                      what the perturbation wrapper costs.
* `train_rl`       -- the env stack that `lsy_drone_racing/control/train_rl.py:470` builds
                      (`RandTrajEnv` + NormalizeActions + StackObs + AngleReward + ActionPenalty +
                      FlattenJaxObservation + JaxToTorch). Needs torch/wandb, i.e. the `tests`
                      pixi env (feature `rl`), and two in-process compatibility shims, because
                      train_rl.py does not run against crazyflow 0.4.0 as committed (see
                      `_patch_train_rl_for_crazyflow_040`). Nothing outside this file is modified.

Actions are drawn once from the action space with a fixed seed and replayed from a ring buffer, so
action sampling is not part of the measured loop. Two timings per configuration:

* `blocked`:   `jax.block_until_ready` after every step -- what a policy in the loop sees, because
               it has to read the observation before it can produce the next action.
* `pipelined`: block only after the last step, i.e. JAX dispatch is allowed to run ahead. Upper
               bound, only reachable if nothing reads the observations.

Compile time (first step, masked autoreset path) is excluded from both; it is reported separately.

Usage (from project/):
    env -u PYTHONPATH pixi run -e tests --frozen python bench_rl_env.py
    env -u PYTHONPATH pixi run python bench_rl_env.py --pipelines="[race_state,race_attitude]"
"""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# lsy_drone_racing (-> crazyflow) must be imported before scipy: it sets SCIPY_ARRAY_API.
import lsy_drone_racing  # noqa: F401  # isort: skip
from common import RESULTS_DIR, load_cfg  # noqa: E402  # isort: skip

import fire  # noqa: E402
import gymnasium  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

PIPELINES = ("race_state", "race_attitude", "train_rl", "rl_train", "rl_train_perturb")
N_ACTION_BUFFER = 64  # number of pre-sampled random action batches, replayed cyclically


@dataclass
class BenchResult:
    """One (pipeline, num_envs) measurement."""

    pipeline: str
    num_envs: int
    n_steps: int
    build_s: float  # env construction (includes mjx model build)
    compile_s: float  # first step + masked reset compile, excluded from the timings below
    warmup_s: float
    blocked_s: float
    pipelined_s: float
    env_steps_per_s_blocked: float  # num_envs * n_steps / blocked_s
    env_steps_per_s_pipelined: float
    vec_steps_per_s_blocked: float  # n_steps / blocked_s
    ms_per_vec_step_blocked: float
    done_fraction: float  # fraction of (step, env) pairs with terminated | truncated
    action_dim: int
    obs_dim: int | None = None
    notes: str = ""
    error: str = ""
    per_step_ms: list[float] = field(default_factory=list)  # blocked loop, for percentiles


# region env builders


def _build_race_env(num_envs: int, config_name: str) -> tuple[Any, np.ndarray]:
    """Build VecDroneRaceEnv exactly as benchmarks/sim.py:27-38 does, on CPU."""
    config = load_cfg(config_name)
    env = gymnasium.make_vec(
        config.env.id,
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
        device="cpu",
    )
    env.action_space.seed(0)
    actions = np.stack([env.action_space.sample() for _ in range(N_ACTION_BUFFER)])
    return env, actions


def _patch_train_rl_for_crazyflow_040() -> list[str]:
    """Make `train_rl.make_envs` runnable against crazyflow 0.4.0, in this process only.

    As committed, `train_rl.py` fails on crazyflow 0.4.0 for two independent reasons:

    1. `train_rl.py:196` inserts the dynamics disturbance before the sim step stage `"integrate"`,
       but the stage is called `"integration"` in crazyflow 0.4.0 (`race_core.py:674` uses the new
       name). With any config that sets `env.disturbances.dynamics` (all of level0-3) the
       constructor raises `KeyError: No stage named 'integrate'`.
    2. The reset-randomization callbacks built by `RandTrajEnv.build_reset_randomization_fn`
       (`train_rl.py:318`, `:325`) take `(data, mask)`, but crazyflow 0.4.0 calls every function in
       the reset pipeline as `fn(data, default_data, mask)` (`crazyflow/sim/sim.py:441`). The first
       `reset()` raises `TypeError: ... takes 2 positional arguments but 3 were given`.

    Both shims are monkeypatches on the already-imported module/class, so no repo file is touched.
    They change the call signature only, not the simulated dynamics.

    Returns:
        Human-readable list of the applied shims, recorded in the result notes.
    """
    from crazyflow.sim.pipeline import insert_fn_before as _insert_fn_before
    from lsy_drone_racing.control import train_rl

    notes = []

    def insert_fn_before_shim(pipeline: Any, name: str, fn: Any, **kwargs: Any) -> Any:
        return _insert_fn_before(pipeline, "integration" if name == "integrate" else name, fn)

    insert_fn_before_shim._bench_patched = True
    if not getattr(train_rl.insert_fn_before, "_bench_patched", False):
        train_rl.insert_fn_before = insert_fn_before_shim
        notes.append("shim: step stage 'integrate' -> 'integration'")

    orig = train_rl.RandTrajEnv.build_reset_randomization_fn
    if not getattr(orig, "_bench_patched", False):

        def build_reset_randomization_fn(self: Any, dynamics: Any) -> Any:
            fn = orig(self, dynamics)
            return lambda data, _default_data, mask: fn(data, mask)

        build_reset_randomization_fn._bench_patched = True
        train_rl.RandTrajEnv.build_reset_randomization_fn = build_reset_randomization_fn
        notes.append("shim: reset_randomization(data, mask) -> (data, default_data, mask)")
    return notes


def _build_train_rl_env(num_envs: int) -> tuple[Any, Any, str]:
    """Build the env stack of lsy_drone_racing/control/train_rl.py:470 (make_envs), on CPU."""
    import torch
    from lsy_drone_racing.control.train_rl import Args, make_envs

    notes = _patch_train_rl_for_crazyflow_040()

    args = Args.create()  # defaults, only the reward/obs coefficients are used by make_envs
    coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    env = make_envs(
        config="level0.toml",
        num_envs=num_envs,
        jax_device="cpu",
        torch_device=torch.device("cpu"),
        coefs=coefs,
    )
    g = torch.Generator().manual_seed(0)
    actions = (
        torch.rand((N_ACTION_BUFFER, num_envs, 4), generator=g, dtype=torch.float32) * 2.0 - 1.0
    )
    return env, actions, "; ".join(notes)


def _build_rl_train(num_envs: int, perturb: dict[str, Any] | None) -> tuple[Any, np.ndarray, str]:
    """The phase-1 PPO stack from `rl.env.make_train_env` (3D action in [-1,1])."""
    from rl.env import ACT_DIM, make_train_env

    env = make_train_env(num_envs, seed=0, perturb=perturb)
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1.0, 1.0, size=(N_ACTION_BUFFER, num_envs, ACT_DIM)).astype(np.float32)
    note = "att_level0.toml, gate-progress reward"
    return env, actions, note + ("" if perturb is None else f", perturb={perturb}")


PERTURB = {"T_end": 18.75, "k_end_range": (0.6, 1.0), "delay_range": (0, 2), "seed": 0}


def _build(pipeline: str, num_envs: int) -> tuple[Any, Any, str]:
    match pipeline:
        case "race_state":
            cfg = Path(__file__).parents[1] / "config" / "level0.toml"
            return (*_build_race_env(num_envs, cfg), "config/level0.toml, control_mode=state")
        case "race_attitude":
            return (
                *_build_race_env(num_envs, "att_level0.toml"),
                "project/config/att_level0.toml, control_mode=attitude",
            )
        case "train_rl":
            return _build_train_rl_env(num_envs)
        case "rl_train":
            return _build_rl_train(num_envs, None)
        case "rl_train_perturb":
            return _build_rl_train(num_envs, PERTURB)
        case _:
            raise ValueError(f"Unknown pipeline: {pipeline}. Choose from {PIPELINES}")


# region measurement


def _sync(obs: Any) -> None:
    """Force the step to be finished (JAX dispatch is asynchronous)."""
    if hasattr(obs, "numel"):  # torch tensor (train_rl pipeline, after JaxToTorch)
        float(obs.reshape(-1)[0])
        return
    jax.block_until_ready(obs)


def _is_race_env(env: Any) -> bool:
    return hasattr(env.unwrapped, "data") and hasattr(env.unwrapped.data, "marked_for_reset")


def benchmark(pipeline: str, num_envs: int, n_steps: int, n_warmup: int) -> BenchResult:
    """Measure steps/s for one pipeline at one vectorization width."""
    t0 = time.perf_counter()
    env, actions, notes = _build(pipeline, num_envs)
    build_s = time.perf_counter() - t0

    # Compile: first reset + step, and the masked reset used by autoreset (benchmarks/sim.py:40-48)
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=0)
    _sync(obs)
    obs, *_ = env.step(actions[0])
    _sync(obs)
    if _is_race_env(env):
        mask = env.unwrapped.data.marked_for_reset.at[0].set(True)
        jax.block_until_ready(env.unwrapped._reset(env.unwrapped.data, mask=mask))
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n_warmup):
        obs, *_ = env.step(actions[i % N_ACTION_BUFFER])
        _sync(obs)
    warmup_s = time.perf_counter() - t0

    # Timed loop 1: block after every step (a policy has to read the obs before acting)
    n_done = 0
    per_step_ms: list[float] = []
    t_start = time.perf_counter()
    for i in range(n_steps):
        t_step = time.perf_counter()
        obs, _, terminated, truncated, _ = env.step(actions[i % N_ACTION_BUFFER])
        _sync(obs)
        per_step_ms.append((time.perf_counter() - t_step) * 1e3)
        n_done += int(np.asarray(terminated).sum() + np.asarray(truncated).sum())
    blocked_s = time.perf_counter() - t_start

    # Timed loop 2: block only at the end (JAX may run ahead; upper bound)
    t_start = time.perf_counter()
    for i in range(n_steps):
        obs, *_ = env.step(actions[i % N_ACTION_BUFFER])
    _sync(obs)
    pipelined_s = time.perf_counter() - t_start

    obs_dim = None
    if hasattr(env, "single_observation_space") and hasattr(env.single_observation_space, "shape"):
        shape = env.single_observation_space.shape
        obs_dim = int(np.prod(shape)) if shape else None
    action_dim = int(np.prod(np.shape(actions)[2:]))
    env.close()

    total = num_envs * n_steps
    return BenchResult(
        pipeline=pipeline,
        num_envs=num_envs,
        n_steps=n_steps,
        build_s=build_s,
        compile_s=compile_s,
        warmup_s=warmup_s,
        blocked_s=blocked_s,
        pipelined_s=pipelined_s,
        env_steps_per_s_blocked=total / blocked_s,
        env_steps_per_s_pipelined=total / pipelined_s,
        vec_steps_per_s_blocked=n_steps / blocked_s,
        ms_per_vec_step_blocked=blocked_s / n_steps * 1e3,
        done_fraction=n_done / total,
        action_dim=action_dim,
        obs_dim=obs_dim,
        notes=notes,
        per_step_ms=per_step_ms,
    )


# region main


def _machine_load() -> dict[str, Any]:
    """Load average and which other known jobs run, recorded with every measurement."""
    import os
    import subprocess

    try:
        procs = subprocess.run(
            ["pgrep", "-af", "python"], capture_output=True, text=True, timeout=10
        ).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        procs = []
    tags = sorted({t for t in ("round4", "round3", "grid", "run_eval", "ppo", "pytest")
                   if any(t in p for p in procs)})
    return {
        "loadavg": [round(x, 2) for x in os.getloadavg()],
        "n_cpus": os.cpu_count(),
        "n_python_procs": len(procs),
        "other_jobs": tags,
    }


def main(
    pipelines: str | list[str] = ("race_state", "race_attitude"),
    num_envs: list[int] = (1, 16, 64, 256),
    n_steps: int = 200,
    n_warmup: int = 20,
    out_dir: str | None = None,
) -> None:
    """Run the throughput benchmark and write a CSV + JSON into a new results folder."""
    if isinstance(pipelines, str):
        pipelines = [pipelines]
    pipelines, num_envs = list(pipelines), [int(n) for n in num_envs]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir else RESULTS_DIR / f"rlbench_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    results: list[BenchResult] = []
    for pipeline in pipelines:
        for n in num_envs:
            print(f"\n=== {pipeline} | num_envs={n} | {n_steps} steps ===", flush=True)
            try:
                r = benchmark(pipeline, n, n_steps, n_warmup)
            except Exception as e:  # noqa: BLE001 - a failing configuration must not kill the run
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                r = BenchResult(
                    pipeline=pipeline,
                    num_envs=n,
                    n_steps=n_steps,
                    build_s=float("nan"),
                    compile_s=float("nan"),
                    warmup_s=float("nan"),
                    blocked_s=float("nan"),
                    pipelined_s=float("nan"),
                    env_steps_per_s_blocked=float("nan"),
                    env_steps_per_s_pipelined=float("nan"),
                    vec_steps_per_s_blocked=float("nan"),
                    ms_per_vec_step_blocked=float("nan"),
                    done_fraction=float("nan"),
                    action_dim=-1,
                    error=f"{type(e).__name__}: {e}",
                )
            results.append(r)
            if not r.error:
                p50, p99 = np.percentile(r.per_step_ms, [50, 99])
                print(
                    f"  build {r.build_s:6.2f}s | compile {r.compile_s:6.2f}s | "
                    f"{r.ms_per_vec_step_blocked:8.3f} ms/vec-step (p50 {p50:.3f}, p99 {p99:.3f})",
                    flush=True,
                )
                print(
                    f"  steps/s blocked {r.env_steps_per_s_blocked:10.1f} | "
                    f"pipelined {r.env_steps_per_s_pipelined:10.1f} | "
                    f"done frac {r.done_fraction:.4f}",
                    flush=True,
                )

    cols = [k for k in asdict(results[0]) if k != "per_step_ms"]
    with (out / "throughput.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in asdict(r).items() if k != "per_step_ms"})
    meta = {
        "timestamp": ts,
        "machine_load": _machine_load(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "gymnasium": gymnasium.__version__,
        "n_steps": n_steps,
        "n_warmup": n_warmup,
        "pipelines": pipelines,
        "num_envs": num_envs,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))

    print(f"\n{'pipeline':<14}{'N':>5}{'ms/step':>10}{'steps/s':>12}{'steps/s(pipe)':>15}{'done':>8}")
    for r in results:
        if r.error:
            print(f"{r.pipeline:<14}{r.num_envs:>5}  {r.error}")
            continue
        print(
            f"{r.pipeline:<14}{r.num_envs:>5}{r.ms_per_vec_step_blocked:>10.3f}"
            f"{r.env_steps_per_s_blocked:>12.1f}{r.env_steps_per_s_pipelined:>15.1f}"
            f"{r.done_fraction:>8.4f}"
        )
    print(f"\nWrote {out}/throughput.csv")


if __name__ == "__main__":
    fire.Fire(main)
