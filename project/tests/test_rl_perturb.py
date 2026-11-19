"""Tests for rl.vec_perturb: equivalence with the scalar wrapper, per-world draws, autoreset."""

from __future__ import annotations

import numpy as np
import pytest
from fakes import HIGH, LOW, FakeEnv
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from rl.vec_perturb import THRUST_IDX, VecThrustDegradeDelay
from wrapper import ThrustDegradeDelay

T_END = 18.75


class FakeVecEnv(VectorEnv):
    """Vectorized twin of `fakes.FakeEnv`: records commands, optional scripted autoresets."""

    def __init__(self, num_envs: int = 1, dummy_at: dict[int, np.ndarray] | None = None) -> None:
        """`dummy_at` maps a step index to the per-world autoreset mask reported in `info`."""
        self.num_envs = num_envs
        self.single_action_space = spaces.Box(LOW, HIGH, dtype=np.float32)
        self.single_observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(
            np.tile(LOW, (num_envs, 1)), np.tile(HIGH, (num_envs, 1)), dtype=np.float32
        )
        self.received: list[np.ndarray] = []
        self.dummy_at = dummy_at or {}
        self.t = 0

    def _obs(self) -> np.ndarray:
        return np.zeros((self.num_envs, 2), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # noqa: ANN001, ANN201, D102
        self.t = 0
        self.received = []
        return self._obs(), {}

    def step(self, action):  # noqa: ANN001, ANN201, D102
        self.received.append(np.array(action, dtype=float))
        dummy = self.dummy_at.get(self.t, np.zeros(self.num_envs, dtype=bool))
        self.t += 1
        z = np.zeros(self.num_envs, dtype=bool)
        return self._obs(), np.zeros(self.num_envs), z, z, {"dummy": np.asarray(dummy)}


def _cmds(n: int, thrust: float = 0.5) -> np.ndarray:
    return np.stack(
        [np.array([0.01 * i, 0.02 * i, 0.0, thrust], dtype=np.float64) for i in range(n)]
    )


def test_identity_when_k_end_is_one_and_no_delay():
    env = FakeVecEnv(3)
    w = VecThrustDegradeDelay(env, T_end=T_END, k_end_range=(1.0, 1.0), delay_range=(0, 0))
    w.reset(seed=0)
    for cmd in _cmds(10):
        batch = np.tile(cmd, (3, 1))
        w.step(batch)
        assert np.allclose(w.last_applied_action, batch), "must be the identity"


def test_matches_the_scalar_wrapper_for_one_world():
    """Same k_end, same delay, same command sequence -> same applied commands."""
    for k_end, delay in [(0.85, 2), (0.7, 0), (1.0, 3), (0.55, 1)]:
        vec = VecThrustDegradeDelay(
            FakeVecEnv(1), T_end=T_END, k_end_range=(k_end, k_end), delay_range=(delay, delay)
        )
        scalar = ThrustDegradeDelay(FakeEnv(n_steps=10**6), T_end=T_END, k_end=k_end, delay=delay)
        vec.reset(seed=0)
        scalar.reset(seed=0)
        for cmd in _cmds(40):
            vec.step(cmd[None, :])
            scalar.step(cmd)
            assert np.allclose(vec.last_applied_action[0], scalar.last_applied_action, atol=1e-12), (
                f"k_end={k_end} delay={delay}"
            )


def test_delay_and_degradation_values():
    w = VecThrustDegradeDelay(
        FakeVecEnv(1), T_end=10.0, k_end_range=(0.5, 0.5), delay_range=(2, 2), freq=50.0
    )
    w.reset(seed=0)
    cmds = _cmds(6)
    for cmd in cmds:
        w.step(cmd[None, :])
    # step 5 applies the command issued at step 3, scaled by k(5/50) = 1 - 0.5*0.1/10*... :
    k = 1.0 + (0.5 - 1.0) * min((5 / 50.0) / 10.0, 1.0)
    expected = cmds[3].copy()
    expected[THRUST_IDX] *= k
    assert np.allclose(w.last_applied_action[0], expected)
    assert w.thrust_scale(np.array([0.0]))[0] == pytest.approx(1.0)
    assert w.thrust_scale(np.array([10.0]))[0] == pytest.approx(0.5)
    assert w.thrust_scale(np.array([99.0]))[0] == pytest.approx(0.5), "constant after T_end"


def test_first_commands_are_held_while_the_buffer_fills():
    w = VecThrustDegradeDelay(
        FakeVecEnv(1), T_end=T_END, k_end_range=(1.0, 1.0), delay_range=(3, 3)
    )
    w.reset(seed=0)
    cmds = _cmds(5)
    applied = []
    for cmd in cmds:
        w.step(cmd[None, :])
        applied.append(w.last_applied_action[0].copy())
    for i in range(3):
        assert np.allclose(applied[i], cmds[0]), f"step {i} must repeat the first command"
    assert np.allclose(applied[3], cmds[0])  # command from 3 steps ago == first
    assert np.allclose(applied[4], cmds[1])


def test_per_world_draws_are_reproducible_and_vary():
    a = VecThrustDegradeDelay(
        FakeVecEnv(64), T_end=T_END, k_end_range=(0.6, 1.0), delay_range=(0, 3), seed=7
    )
    b = VecThrustDegradeDelay(
        FakeVecEnv(64), T_end=T_END, k_end_range=(0.6, 1.0), delay_range=(0, 3), seed=7
    )
    a.reset(seed=None)
    b.reset(seed=None)
    assert np.allclose(a.k_end, b.k_end) and np.array_equal(a.delay, b.delay)
    assert a.k_end.std() > 0.05 and len(np.unique(a.delay)) > 1
    assert a.k_end.min() >= 0.6 and a.k_end.max() <= 1.0
    assert a.delay.min() >= 0 and a.delay.max() <= 3


def test_autoreset_restarts_the_per_world_clock_and_redraws():
    """A world reported as `dummy` starts a fresh episode: k(t) goes back to 1.0."""
    n = 2
    dummy = {5: np.array([True, False])}
    w = VecThrustDegradeDelay(
        FakeVecEnv(n, dummy_at=dummy), T_end=1.0, k_end_range=(0.5, 0.5), delay_range=(0, 0)
    )
    w.reset(seed=0)
    cmd = np.tile(np.array([0.0, 0.0, 0.0, 0.5]), (n, 1))
    for _ in range(6):
        w.step(cmd)
    assert w._steps[0] == 0 and w._steps[1] == 6
    w.step(cmd)
    # world 0 is at t=0 again (k=1), world 1 at t=6/50 s with T_end=1 s
    assert w.last_applied_action[0, THRUST_IDX] == pytest.approx(0.5)
    k1 = 1.0 + (0.5 - 1.0) * (6 / 50.0)
    assert w.last_applied_action[1, THRUST_IDX] == pytest.approx(0.5 * k1)


def test_thrust_is_clipped_after_scaling():
    w = VecThrustDegradeDelay(
        FakeVecEnv(1), T_end=1.0, k_end_range=(0.01, 0.01), delay_range=(0, 0)
    )
    w.reset(seed=0)
    for _ in range(80):  # well past T_end, so k ~ 0.01
        w.step(np.array([[0.0, 0.0, 0.0, 0.2]]))
    assert w.last_applied_action[0, THRUST_IDX] == pytest.approx(float(LOW[THRUST_IDX]))


def test_rejects_a_non_attitude_action_space():
    env = FakeVecEnv(1)
    env.single_action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    with pytest.raises(ValueError, match="4-element"):
        VecThrustDegradeDelay(env, T_end=T_END)
