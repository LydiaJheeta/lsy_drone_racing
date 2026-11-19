"""Evaluate a PPO checkpoint with the SAME single-env pipeline as the MPC evaluations.

Uses `common.make_env` / `common.run_episode` (so: `DroneRaceEnv` via `gymnasium.make`, seeds
passed to `env.reset(seed=...)`, success = `n_gates_passed == 5`, lap time = `steps / 50`, failure
cause from the replay diagnosis) and writes `episodes.csv` with the field list of
`run_eval.FIELDS`, plus `summary.csv` from `run_eval.summarize`.

The single env is bit-identical to `VecDroneRaceEnv` with `num_envs=1` for the same seed and
actions (verified: max |dpos| = 0 over 60 steps); it only differs in `autoreset=False` and in the
array shapes. So a policy trained on the vector env can be evaluated here directly.

Usage:
    cd project
    env -u PYTHONPATH pixi run -e tests --frozen python -m rl.eval_policy \
        --ckpt=results/ppo_<ts>/ckpt_final.pt
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

# lsy_drone_racing (-> crazyflow) must be imported before scipy.
import lsy_drone_racing  # noqa: F401  # isort: skip

import fire  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from common import RESULTS_DIR, make_env, run_episode  # noqa: E402
from lsy_drone_racing.control.controller import Controller  # noqa: E402
from rl.env import ACT_DIM, attitude_limits, load_rl_config, map_action, obs_from_single  # noqa: E402
from rl.ppo import load_agent  # noqa: E402
from run_eval import FIELDS, make_row, summarize, write_csv  # noqa: E402


class PolicyController(Controller):
    """Runs a trained PPO policy as a `Controller` (deterministic by default)."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: Any):
        """Load the checkpoint named in `config.rl.ckpt` and reset the last-action memory."""
        super().__init__(obs, info, config)
        import torch

        self.torch = torch
        self.agent = load_agent(config.rl["ckpt"], device="cpu")
        self.deterministic = bool(config.rl.get("deterministic", True))
        self.low, self.high = attitude_limits(config)
        self._last_action = np.zeros(ACT_DIM, dtype=np.float32)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the 4D attitude command `[roll, pitch, yaw=0, thrust]`."""
        flat = obs_from_single(obs, self._last_action)
        with self.torch.no_grad():
            action, *_ = self.agent.get_action_and_value(
                self.torch.as_tensor(np.array(flat, dtype=np.float32)).unsqueeze(0),
                deterministic=self.deterministic,
            )
        a = np.clip(np.asarray(action[0], dtype=np.float32), -1.0, 1.0)
        self._last_action = a
        return map_action(a, self.low, self.high)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Never finish early: a policy has no reference trajectory to run out of.

        `Controller.step_callback` returns `True` by default (`controller.py:95`, whose comment
        says the opposite), and `common.run_episode` stops the episode on a truthy return. Without
        this override every episode would end after one step with cause `trajectory_end`.
        """
        return False

    def episode_reset(self) -> None:
        """Forget the last action between episodes."""
        self._last_action = np.zeros(ACT_DIM, dtype=np.float32)


def evaluate(
    ckpt: str,
    config: str = "att_level0.toml",
    seeds: int = 20,
    seed0: int = 0,
    deterministic: bool = True,
    out_dir: str | None = None,
    stage: str = "ppo",
) -> Path:
    """Run `seeds` episodes with the policy and write `episodes.csv` + `summary.csv`."""
    cfg = load_rl_config(config)
    cfg.rl = {"ckpt": str(Path(ckpt).resolve()), "deterministic": deterministic}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir else RESULTS_DIR / f"ppo_eval_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg)
    rows: list[dict[str, Any]] = []
    try:
        for seed in range(seed0, seed0 + seeds):
            res = run_episode(env, PolicyController, cfg, seed)
            # T_traj is a reference-trajectory duration: not applicable to a policy,
            # so it is 0.0 (NaN would make run_eval.summarize group on NaN != NaN).
            rows.append(make_row(stage, 0.0, "nominal", res))
            print(
                f"  seed {seed:3d} success={res.success} gates={res.n_gates}/{res.n_gates_total} "
                f"t={res.time_s:5.2f}s cause={res.cause} {res.detail}",
                flush=True,
            )
    finally:
        env.close()

    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    write_csv(out / "summary.csv", summary)
    n_ok = sum(r["success"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} successes | checkpoint {ckpt}")
    for s in summary:
        print(
            f"  success_rate={s['success_rate']:.2f} lap_time_mean={s['lap_time_mean_s']:.2f}s "
            f"gates_mean={s['gates_mean_all']:.2f} causes: {s['failure_causes']}"
        )
    print(f"Wrote {out}/episodes.csv")
    return out


if __name__ == "__main__":
    fire.Fire({"evaluate": evaluate})
