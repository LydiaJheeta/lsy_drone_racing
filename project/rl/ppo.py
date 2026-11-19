"""PPO for the gate-progress racing env. Core copied from `train_rl.py:557-757` (cleanrl-style).

Differences from `lsy_drone_racing/control/train_rl.py`, all deliberate:

* environment: `rl.env.make_train_env` (`VecDroneRaceEnv` subclass, attitude, gate-progress
  reward) instead of `RandTrajEnv`; `train_rl.py` is not imported at all (it does not run against
  crazyflow 0.4.0, see `results/rlbench_20261113_151905/FINDINGS.md`);
* no wandb: per-update metrics go to `progress.csv`, checkpoints to `ckpt_*.pt`;
* `Args` defaults are the ones of `train_rl.py:47-108`, except `num_envs=256`, `num_steps=32`
  (-> `batch_size=8192`, `minibatch_size=1024`);
* the actor has 3 outputs ([roll, pitch, thrust]); yaw is always commanded 0, so
  `actor_logstd` starts at `[-1, -1, 1]` (`train_rl.py:532-534` minus the yaw entry).

The rollout treats the autoreset "dummy" transition (gymnasium `AutoresetMode.NEXT_STEP`) the way
cleanrl does: `dones[t]` holds the done flag of the *previous* step, so GAE zeroes the bootstrap
across an episode boundary. The reward of a dummy transition is 0 by construction (see
`rl/env.py`), and episode statistics skip those steps.

Usage (needs torch -> pixi env `tests`):
    cd project
    env -u PYTHONPATH pixi run -e tests --frozen python -m rl.ppo --total_timesteps=2_000_000
"""

from __future__ import annotations

import csv
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions.normal import Normal

from rl.env import ACT_DIM, OBS_DIM, make_train_env

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

METRICS = [
    "update", "global_step", "wall_s", "sps_loop", "sps_env", "n_episodes",
    "ep_reward_mean", "ep_reward_std", "ep_len_mean", "gates_mean",
    "frac_success", "frac_terminated_fail", "frac_timeout",
    "value_loss", "policy_loss", "entropy", "approx_kl", "clipfrac", "explained_var", "lr",
]


@dataclass
class Args:
    """Hyperparameters. Defaults are `train_rl.py:47-108` with num_envs=256, num_steps=32."""

    seed: int = 42
    torch_deterministic: bool = True
    total_timesteps: int = 2_000_000
    learning_rate: float = 1.5e-3
    num_envs: int = 256
    num_steps: int = 32
    anneal_lr: bool = True
    gamma: float = 0.94
    gae_lambda: float = 0.97
    num_minibatches: int = 8
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.26
    clip_vloss: bool = True
    ent_coef: float = 0.007
    vf_coef: float = 0.7
    max_grad_norm: float = 1.5
    target_kl: float | None = None
    config: str = "att_level0.toml"
    # filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    @staticmethod
    def create(**kwargs: Any) -> Args:
        """Create the arguments and the derived batch sizes (`train_rl.py:110-117`)."""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args


def set_seeds(seed: int) -> None:
    """Seed python, numpy and torch (`train_rl.py:459-466`)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal init (`train_rl.py:503-507`)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Tanh MLP actor-critic, 2x64, as in `train_rl.py:511-553` but with 3 action dimensions."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        """Build the critic and the actor."""
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01), nn.Tanh(),
        )
        # smaller initial std for roll/pitch than for thrust
        self.actor_logstd = nn.Parameter(torch.Tensor([[-1.0, -1.0, 1.0]]))

    def get_value(self, x: Tensor) -> Tensor:
        """Value estimate."""
        return self.critic(x)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample (or evaluate) an action and return log-prob, entropy and value."""
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean if deterministic else probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class EpisodeTracker:
    """Per-world episode return/length/outcome, skipping autoreset dummy transitions."""

    def __init__(self, num_envs: int, n_passes: int):
        """Create the accumulators."""
        self.n_passes = n_passes
        self.ret = np.zeros(num_envs, dtype=np.float64)
        self.len = np.zeros(num_envs, dtype=np.int64)
        self.episodes: list[dict[str, float]] = []

    def update(
        self,
        reward: np.ndarray,
        active: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        n_gates: np.ndarray,
    ) -> None:
        """Accumulate one step. `active` is False for dummy (post-autoreset) transitions."""
        self.ret += reward * active
        self.len += active
        done = (terminated | truncated) & active
        if not done.any():
            return
        for w in np.flatnonzero(done):
            gates = int(n_gates[w])
            success = bool(terminated[w] and gates >= self.n_passes)
            self.episodes.append({
                "ret": float(self.ret[w]),
                "len": int(self.len[w]),
                "gates": gates,
                "success": success,
                "terminated_fail": bool(terminated[w] and not success),
                "timeout": bool(truncated[w] and not terminated[w]),
            })
        self.ret[done] = 0.0
        self.len[done] = 0

    def drain(self) -> list[dict[str, float]]:
        """Return the episodes finished since the last call."""
        eps, self.episodes = self.episodes, []
        return eps


def _episode_metrics(eps: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate finished episodes (NaN when none finished in the window)."""
    if not eps:
        return {k: float("nan") for k in (
            "ep_reward_mean", "ep_reward_std", "ep_len_mean", "gates_mean",
            "frac_success", "frac_terminated_fail", "frac_timeout")} | {"n_episodes": 0}
    rets = np.array([e["ret"] for e in eps])
    return {
        "n_episodes": len(eps),
        "ep_reward_mean": float(rets.mean()),
        "ep_reward_std": float(rets.std()),
        "ep_len_mean": float(np.mean([e["len"] for e in eps])),
        "gates_mean": float(np.mean([e["gates"] for e in eps])),
        "frac_success": float(np.mean([e["success"] for e in eps])),
        "frac_terminated_fail": float(np.mean([e["terminated_fail"] for e in eps])),
        "frac_timeout": float(np.mean([e["timeout"] for e in eps])),
    }


def machine_load() -> dict[str, Any]:
    """Load average and which other jobs are running, for the record.

    Only script basenames are kept: this ends up in `meta.json`, which is tracked, so no absolute
    paths or user names go in.
    """
    import os

    try:
        procs = subprocess.run(
            ["pgrep", "-af", "python"], capture_output=True, text=True, timeout=10
        ).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        procs = []
    tags = sorted({t for t in ("round4", "round3", "grid", "run_eval", "pytest", "bench_rl_env")
                   if any(t in p for p in procs)})
    return {
        "loadavg": [round(x, 2) for x in os.getloadavg()],
        "n_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "n_python_procs": len(procs),
        "other_jobs": tags,
    }


def train(
    out_dir: str | None = None,
    ckpt_every: int = 50,
    log_every: int = 1,
    device: str = "cpu",
    **kwargs: Any,
) -> Path:
    """Train PPO on the gate-progress env and write metrics/checkpoints into a results folder."""
    args = Args.create(**kwargs)
    dev = torch.device(device)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir else RESULTS_DIR / f"ppo_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)

    envs = make_train_env(args.num_envs, config=args.config, seed=args.seed, device=device)
    n_passes = int(envs.unwrapped.n_passes)
    agent = Agent(OBS_DIM, ACT_DIM).to(dev)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    meta = {
        "timestamp": ts,
        "args": asdict(args),
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "n_passes": n_passes,
        "device": device,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "machine_load_at_start": machine_load(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ppo] {out} | {args.num_iterations} updates x {args.batch_size} steps "
          f"= {args.num_iterations * args.batch_size} timesteps", flush=True)
    print(f"[ppo] machine load at start: {meta['machine_load_at_start']}", flush=True)

    obs_buf = torch.zeros((args.num_steps, args.num_envs, OBS_DIM), device=dev)
    act_buf = torch.zeros((args.num_steps, args.num_envs, ACT_DIM), device=dev)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=dev)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=dev)
    dones = torch.zeros((args.num_steps, args.num_envs), device=dev)
    values = torch.zeros((args.num_steps, args.num_envs), device=dev)

    global_step = 0
    start_time = time.perf_counter()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(np.array(next_obs_np, dtype=np.float32), device=dev)
    next_done = torch.zeros(args.num_envs, device=dev)
    tracker = EpisodeTracker(args.num_envs, n_passes)

    csv_file = (out / "progress.csv").open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=METRICS)
    writer.writeheader()

    for iteration in range(1, args.num_iterations + 1):
        it_start = time.perf_counter()
        env_time = 0.0
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            act_buf[step] = action
            logprobs[step] = logprob

            active = (next_done == 0).cpu().numpy()
            t_env = time.perf_counter()
            next_obs_np, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy().astype(np.float32)
            )
            env_time += time.perf_counter() - t_env
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            reward = np.asarray(reward, dtype=np.float64)
            tracker.update(
                reward, active, terminated, truncated,
                np.asarray(info["n_gates_passed"], dtype=np.int64),
            )
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=dev)
            # np.array(..) copies: jax->numpy arrays are read-only and torch would warn.
            next_obs = torch.as_tensor(np.array(next_obs_np, dtype=np.float32), device=dev)
            next_done = torch.as_tensor(
                (terminated | truncated).astype(np.float32), device=dev
            )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs_buf.reshape((-1, OBS_DIM))
        b_logprobs = logprobs.reshape(-1)
        b_actions = act_buf.reshape((-1, ACT_DIM))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs: list[float] = []
        approx_kl = torch.tensor(0.0)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
        it_time = time.perf_counter() - it_start
        row = {
            "update": iteration,
            "global_step": global_step,
            "wall_s": round(time.perf_counter() - start_time, 2),
            "sps_loop": args.batch_size / it_time,  # env + policy + update
            "sps_env": args.batch_size / env_time,  # env.step only
            "value_loss": float(v_loss.item()),
            "policy_loss": float(pg_loss.item()),
            "entropy": float(entropy_loss.item()),
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(np.mean(clipfracs)),
            "explained_var": explained_var,
            "lr": optimizer.param_groups[0]["lr"],
            **_episode_metrics(tracker.drain()),
        }
        writer.writerow({k: row[k] for k in METRICS})
        csv_file.flush()
        if iteration % log_every == 0 or iteration == 1:
            print(
                f"[{iteration:4d}/{args.num_iterations}] step {global_step:>9d} "
                f"sps {row['sps_loop']:7.0f} (env {row['sps_env']:7.0f}) "
                f"eps {row['n_episodes']:4d} R {row['ep_reward_mean']:+7.2f} "
                f"gates {row['gates_mean']:4.2f} succ {row['frac_success']:.3f} "
                f"fail {row['frac_terminated_fail']:.3f} to {row['frac_timeout']:.3f}",
                flush=True,
            )
        if ckpt_every and iteration % ckpt_every == 0:
            _save(agent, args, out / f"ckpt_{iteration:05d}.pt", global_step)

    _save(agent, args, out / "ckpt_final.pt", global_step)
    csv_file.close()
    envs.close()
    meta["machine_load_at_end"] = machine_load()
    meta["wall_s"] = round(time.perf_counter() - start_time, 2)
    meta["global_step"] = global_step
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ppo] done in {meta['wall_s']:.1f}s -> {out}", flush=True)
    return out


def _save(agent: Agent, args: Args, path: Path, global_step: int) -> None:
    """Write a checkpoint that `eval_policy` can load without any other state."""
    torch.save(
        {
            "state_dict": agent.state_dict(),
            "obs_dim": agent.obs_dim,
            "act_dim": agent.act_dim,
            "args": asdict(args),
            "global_step": global_step,
        },
        path,
    )


def load_agent(path: str | Path, device: str = "cpu") -> Agent:
    """Load a checkpoint written by `_save` into an `Agent` in eval mode."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    agent = Agent(int(ckpt["obs_dim"]), int(ckpt["act_dim"])).to(device)
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent


if __name__ == "__main__":
    fire.Fire({"train": train})
