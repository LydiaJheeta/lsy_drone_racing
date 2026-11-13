"""Per-episode .npz logger (innermost wrapper, so it records what the base env actually received).

Stack it directly on the base env and put any command-modifying wrapper *outside* it:

    env = ThrustDegradeDelay(EpisodeLogger(base_env, out_dir, freq), ...)

Saved arrays (T = number of env steps taken):
    t              (T,)    time of each step, n / freq
    state          (T, 13) state *before* the step: pos(3), quat(4, xyzw), vel(3), ang_vel(3)
    action         (T, 4)  command applied to the env at that step: [roll, pitch, yaw, thrust]
    next_gate_idx  (T,)    index into the gate order of the next gate to pass (= n_gates_passed)
    next_gate_id   (T,)    gate id of that next gate (clamped to the last entry after the lap)
    final_state    (13,)   observation after the last step (pos is -1 if the drone was disabled)
    final_n_gates, seed, freq, terminated, truncated, state_names, action_names
Together, `state[1:]` with `final_state` gives x_{t+1} for every step (SINDy targets).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray

STATE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz")
ACTION_NAMES = ("roll", "pitch", "yaw", "thrust")


def state_vector(obs: dict[str, NDArray]) -> NDArray:
    """Flatten the drone state of an observation into the 13-vector described above."""
    return np.concatenate([obs["pos"], obs["quat"], obs["vel"], obs["ang_vel"]]).astype(float)


class EpisodeLogger(gymnasium.Wrapper):
    """Record time, state, applied command and next-gate index per step; one .npz per episode."""

    def __init__(
        self, env: gymnasium.Env, out_dir: str | Path | None, freq: float = 50.0, tag: str = ""
    ) -> None:
        """Create the logger. `out_dir=None` keeps episodes in memory only (`last_episode`)."""
        super().__init__(env)
        self.out_dir = Path(out_dir) if out_dir is not None else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.freq, self.tag = freq, tag
        self.last_applied_action: NDArray | None = None
        self.last_episode: dict[str, Any] | None = None
        self.last_path: Path | None = None
        self._obs: dict | None = None
        self._seed: int | None = None
        self._rows: list[tuple[float, NDArray, NDArray, int, int]] = []
        self._term = self._trunc = False

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Flush the previous episode (if any) and start a new one."""
        self.flush()
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs, self._seed, self._rows = obs, seed, []
        self._term = self._trunc = False
        return obs, info

    def step(self, action: NDArray) -> tuple[Any, float, bool, bool, dict]:
        """Record (state, action, gate index) for this step, then step the env."""
        act = np.asarray(action, dtype=float).copy()
        obs0 = self._obs
        seq = obs0["gate_sequence"]
        idx = int(obs0["n_gates_passed"])
        gate_id = int(seq[min(idx, len(seq) - 1)])
        self._rows.append((len(self._rows) / self.freq, state_vector(obs0), act, idx, gate_id))
        self.last_applied_action = act
        obs, reward, term, trunc, info = self.env.step(action)
        self._obs, self._term, self._trunc = obs, bool(term), bool(trunc)
        if term or trunc:
            self.flush()
        return obs, reward, term, trunc, info

    def flush(self) -> None:
        """Write the buffered episode to disk (no-op if nothing is buffered)."""
        if not self._rows:
            return
        t, st, ac, gi, gid = zip(*self._rows)
        ep = dict(
            t=np.asarray(t), state=np.stack(st), action=np.stack(ac),
            next_gate_idx=np.asarray(gi), next_gate_id=np.asarray(gid),
            final_state=state_vector(self._obs), final_n_gates=int(self._obs["n_gates_passed"]),
            seed=-1 if self._seed is None else int(self._seed), freq=float(self.freq),
            terminated=self._term, truncated=self._trunc,
            state_names=np.asarray(STATE_NAMES), action_names=np.asarray(ACTION_NAMES),
        )
        self.last_episode, self._rows = ep, []
        if self.out_dir is not None:
            stem = f"{self.tag + '_' if self.tag else ''}seed{ep['seed']:03d}"
            path, k = self.out_dir / f"{stem}.npz", 1
            while path.exists():  # never overwrite an existing log
                path, k = self.out_dir / f"{stem}_{k}.npz", k + 1
            np.savez_compressed(path, **ep)
            self.last_path = path

    def close(self) -> None:
        """Flush and close the wrapped env."""
        self.flush()
        super().close()
