"""Vectorized command perturbation: per-world command delay and thrust degradation k(t).

Same semantics as `project/wrapper.py` (`ThrustDegradeDelay`), applied per world. For the command
issued at step `n` of a world's episode (`t = n / freq`):

    1. delay:   apply the command issued `d` steps ago. While the buffer is still filling (n < d)
                the world's FIRST command of the episode is applied (`fill="first"`).
    2. degrade: thrust *= k(t),  k(t) = 1 + (k_end - 1) * min(t / T_end, 1)
    3. clip:    to the 4D attitude action-space bounds (thrust limits checked AFTER scaling)

The perturbation is on the **command**, never on the plant.

`k_end` and `d` are drawn per world from a seeded `numpy.random.Generator` when that world starts an
episode: at `reset()` for every world, and at an autoreset for the worlds that restarted. The
wrapper detects autoresets from `info["dummy"]` of `GateProgressRaceEnv` (gymnasium
`AutoresetMode.NEXT_STEP`: the world is reset *inside* the step that follows its terminal step, so
the new episode's clock starts at the next step).

Degenerate ranges give fixed values: `k_end_range=(0.85, 0.85)`, `delay_range=(2, 2)` behaves
exactly like `ThrustDegradeDelay(k_end=0.85, delay=2)` on a single world (tested).

In phase 1 of the PPO work this wrapper is NOT in the training stack (`perturb=None`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.vector import VectorWrapper

THRUST_IDX = 3  # [roll, pitch, yaw, thrust]


class VecThrustDegradeDelay(VectorWrapper):
    """Per-world command delay and linear thrust degradation for a vectorized attitude env."""

    def __init__(
        self,
        env: Any,
        *,
        T_end: float,
        k_end_range: tuple[float, float] = (0.85, 0.85),
        delay_range: tuple[int, int] = (2, 2),
        freq: float = 50.0,
        seed: int = 0,
    ) -> None:
        """Create the wrapper.

        Args:
            env: Vector env with a 4-element attitude `Box` action space (numpy-facing).
            T_end: Seconds over which k falls linearly from 1.0 to `k_end`. REQUIRED and
                keyword-only, for the reason given in `project/wrapper.py`: it must be the
                reference duration of the experiment and no default fits every one.
            k_end_range: Inclusive range the per-world `k_end` is drawn from (uniform).
            delay_range: Inclusive integer range the per-world delay (in env steps) is drawn from.
            freq: Env step frequency in Hz (per-world time = that world's steps / freq).
            seed: Seed of the generator that draws `k_end` and the delay.
        """
        super().__init__(env)
        low = np.asarray(env.single_action_space.low, dtype=np.float64)
        high = np.asarray(env.single_action_space.high, dtype=np.float64)
        if low.shape != (4,):
            raise ValueError(f"expected a 4-element attitude action space, got {low.shape}")
        if T_end <= 0:
            raise ValueError("T_end must be > 0")
        if delay_range[0] < 0 or delay_range[1] < delay_range[0]:
            raise ValueError(f"invalid delay_range {delay_range}")
        if k_end_range[1] < k_end_range[0]:
            raise ValueError(f"invalid k_end_range {k_end_range}")
        self._low, self._high = low, high
        self.T_end, self.freq = float(T_end), float(freq)
        self.k_end_range = (float(k_end_range[0]), float(k_end_range[1]))
        self.delay_range = (int(delay_range[0]), int(delay_range[1]))
        self.seed_value = int(seed)
        self._rng = np.random.default_rng(self.seed_value)
        n = self.num_envs
        self._len = self.delay_range[1] + 1  # ring buffer length
        self._hist = np.zeros((self._len, n, 4), dtype=np.float64)
        self._steps = np.zeros(n, dtype=np.int64)  # per-world steps since that world's reset
        self._needs_fill = np.ones(n, dtype=bool)
        self.k_end = np.ones(n)
        self.delay = np.zeros(n, dtype=np.int64)
        self.last_applied_action: np.ndarray | None = None
        self._draw(np.ones(n, dtype=bool))

    def _draw(self, mask: np.ndarray) -> None:
        """Draw `k_end` and `delay` for the worlds in `mask`."""
        n = int(mask.sum())
        if n == 0:
            return
        self.k_end[mask] = self._rng.uniform(self.k_end_range[0], self.k_end_range[1], size=n)
        self.delay[mask] = self._rng.integers(
            self.delay_range[0], self.delay_range[1] + 1, size=n
        )

    def thrust_scale(self, t: np.ndarray) -> np.ndarray:
        """k(t) per world: 1.0 at t=0, `k_end` at `T_end`, linear in between, constant after."""
        frac = np.clip(np.asarray(t, dtype=np.float64) / self.T_end, 0.0, 1.0)
        return 1.0 + (self.k_end - 1.0) * frac

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Reset the env, the per-world clocks and buffers, and redraw `k_end`/`delay`."""
        self._rng = np.random.default_rng(self.seed_value if seed is None else seed)
        self._steps[:] = 0
        self._needs_fill[:] = True
        self._hist[:] = 0.0
        self._draw(np.ones(self.num_envs, dtype=bool))
        self.last_applied_action = None
        return self.env.reset(seed=seed, options=options)

    def step(self, actions: np.ndarray) -> tuple[Any, Any, Any, Any, dict]:
        """Delay, degrade and clip the commands per world, then step the wrapped env."""
        action = np.asarray(actions, dtype=np.float64)
        if action.shape != (self.num_envs, 4):
            raise ValueError(f"expected actions of shape {(self.num_envs, 4)}, got {action.shape}")
        # Pre-fill the whole ring buffer of worlds that just (re)started, so that the first
        # command of the episode is what gets applied while the buffer fills.
        if self._needs_fill.any():
            self._hist[:, self._needs_fill, :] = action[None, self._needs_fill, :]
            self._needs_fill[:] = False
        head = self._steps % self._len
        worlds = np.arange(self.num_envs)
        self._hist[head, worlds, :] = action
        applied = self._hist[(self._steps - self.delay) % self._len, worlds, :].copy()
        applied[:, THRUST_IDX] *= self.thrust_scale(self._steps / self.freq)
        applied = np.clip(applied, self._low, self._high)
        self.last_applied_action = applied
        self._steps += 1

        obs, reward, terminated, truncated, info = self.env.step(applied.astype(np.float32))
        # Worlds that were autoreset inside this step start a new episode at the next step.
        restarted = np.asarray(info.get("dummy", np.zeros(self.num_envs, dtype=bool))).astype(bool)
        if restarted.any():
            self._steps[restarted] = 0
            self._needs_fill[restarted] = True
            self._draw(restarted)
        return obs, reward, terminated, truncated, info
