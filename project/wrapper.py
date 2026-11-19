"""Gymnasium wrapper: linear thrust degradation k(t) plus a fixed command delay.

Attitude command layout (verified in crazyflow, see README): [roll, pitch, yaw, thrust], with
`thrust` the TOTAL collective thrust in Newtons at index 3 (= -1). Only that element is scaled.

Order of operations for the command applied at env step n (t = n / freq):
    1. delay:   take the command issued `delay` steps ago. While the buffer fills (n < delay) the
                fill command is applied: the controller's FIRST command (`fill="first"`, default)
                or a nominal hover command [0, 0, 0, hover_thrust] (`fill="hover"`)
    2. degrade: thrust *= k(t)  with  k(t) = 1 + (k_end - 1) * min(t / T_end, 1)
    3. clip:    to the env's action-space bounds (thrust limits are checked *after* scaling)

`T_end` must be set to the reference duration (`T_traj`) of the experiment being run, so that the
drone sees the whole degradation 1.0 -> k_end within one lap. With a fixed `T_end` (the old default
of 18 s) a fast lap (small `T_traj`) only ever saw part of the drop. No single default can be
right for every `T_traj`, so it is a REQUIRED keyword-only argument: `T_end=T_traj` at
construction, `set_params(T_end=...)` between the blocks of a sweep. Omitting it is a TypeError,
and it cannot be filled by a misplaced positional argument.

The wrapper contains no randomness. It accepts a `seed` that is forwarded to `env.reset`, so two runs
with the same seed and the same commands are identical.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray

THRUST_IDX = 3  # [roll, pitch, yaw, thrust]


class ThrustDegradeDelay(gymnasium.Wrapper):
    """Scale the collective thrust by k(t) and delay all commands by `delay` env steps."""

    def __init__(
        self,
        env: gymnasium.Env,
        *,
        T_end: float,
        k_end: float = 0.85,
        delay: int = 2,
        freq: float = 50.0,
        seed: int | None = None,
        fill: str = "first",
        hover_thrust: float | None = None,
    ) -> None:
        """Create the wrapper.

        Args:
            env: Env with a 4-element attitude `Box` action space.
            T_end: Time in seconds over which k falls linearly from 1.0 to `k_end`. REQUIRED and
                keyword-only: it must be the `T_traj` of the experiment, and no default can be
                right for every `T_traj`, so there is deliberately none to pass silently.
                `set_params` can change it between the blocks of a sweep.
            k_end: Thrust scale reached at `T_end` and kept afterwards.
            delay: Command delay in env steps (2 steps at 50 Hz = 40 ms).
            freq: Env step frequency in Hz (time = steps / freq).
            seed: Default seed used by `reset()` when none is passed.
            fill: What is applied during the first `delay` steps: "first" (repeat the first
                command) or "hover" (nominal hover command, needs `hover_thrust`).
            hover_thrust: Total hover thrust in N (mass * g) for `fill="hover"`.
        """
        super().__init__(env)
        space = env.action_space
        if space.shape != (4,):
            raise ValueError(f"Expected a 4-element attitude action space, got {space.shape}")
        if delay < 0 or T_end <= 0:
            raise ValueError("delay must be >= 0 and T_end > 0")
        if fill not in ("first", "hover") or (fill == "hover" and hover_thrust is None):
            raise ValueError("fill must be 'first' or 'hover' (the latter needs hover_thrust)")
        self.fill = fill
        self._fill_cmd = (
            None if hover_thrust is None else np.array([0.0, 0.0, 0.0, hover_thrust], dtype=float)
        )
        self.k_end, self.T_end, self.delay, self.freq = float(k_end), float(T_end), int(delay), freq
        self.seed = seed
        self._low, self._high = np.asarray(space.low, float), np.asarray(space.high, float)
        self._buffer: deque[NDArray] = deque(maxlen=self.delay + 1)
        self._n = 0
        self.last_applied_action: NDArray | None = None

    def set_params(
        self, k_end: float | None = None, delay: int | None = None, T_end: float | None = None
    ) -> None:
        """Change `k_end`, `delay` and/or `T_end`; takes effect from the next `reset()`.

        `T_end` must be the reference duration `T_traj` of the experiment that is about to run,
        otherwise a fast lap ends before k(t) has reached `k_end` (see the module docstring).
        """
        if k_end is not None:
            self.k_end = float(k_end)
        if delay is not None:
            if delay < 0:
                raise ValueError("delay must be >= 0")
            self.delay = int(delay)
        if T_end is not None:
            if T_end <= 0:
                raise ValueError("T_end must be > 0")
            self.T_end = float(T_end)

    def thrust_scale(self, t: float) -> float:
        """k(t): 1.0 at t=0, `k_end` at t=T_end, linear in between, constant after."""
        return 1.0 + (self.k_end - 1.0) * min(max(t, 0.0) / self.T_end, 1.0)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        """Reset the env and clear the delay buffer and clock."""
        self._buffer = deque(maxlen=self.delay + 1)
        self._n = 0
        self.last_applied_action = None
        return self.env.reset(seed=self.seed if seed is None else seed, options=options)

    def step(self, action: NDArray) -> tuple[Any, float, bool, bool, dict]:
        """Delay, degrade and clip `action`, then step the wrapped env."""
        action = np.asarray(action, dtype=float).copy()
        if self._n == 0:  # pre-fill so that buffer[0] is the fill command while n < delay
            fill = self._fill_cmd if self.fill == "hover" else action
            for _ in range(self.delay):
                self._buffer.append(fill.copy())
        self._buffer.append(action)
        applied = self._buffer[0].copy()  # command from `delay` steps ago (or the fill command)
        applied[THRUST_IDX] *= self.thrust_scale(self._n / self.freq)
        applied = np.clip(applied, self._low, self._high)
        self.last_applied_action = applied
        self._n += 1
        return self.env.step(applied)
