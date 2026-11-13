"""A tiny deterministic gymnasium env with the same action layout as the attitude env."""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces

LOW = np.array([-np.pi / 2] * 3 + [0.0854], dtype=np.float32)
HIGH = np.array([np.pi / 2] * 3 + [0.8], dtype=np.float32)


class FakeEnv(gymnasium.Env):
    """Records every action it receives; truncates after `n_steps` steps."""

    def __init__(self, n_steps: int = 5) -> None:
        self.action_space = spaces.Box(LOW, HIGH, dtype=np.float32)
        self.observation_space = spaces.Dict({})
        self.n_steps, self.received, self.reset_seeds, self.t = n_steps, [], [], 0

    def _obs(self) -> dict:
        t = float(self.t)
        return {
            "pos": np.array([t, 0.0, 1.0]), "quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "vel": np.array([1.0, 0.0, 0.0]), "ang_vel": np.zeros(3),
            "n_gates_passed": np.array(min(self.t // 2, 5)),
            "gate_sequence": np.array([0, 1, 2, 3, 1]),
        }

    def reset(self, *, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.received.append(np.array(action, dtype=float))
        self.t += 1
        return self._obs(), 0.0, False, self.t >= self.n_steps, {}
