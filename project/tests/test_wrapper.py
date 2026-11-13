"""Tests for wrapper.ThrustDegradeDelay."""

from __future__ import annotations

import numpy as np
import pytest
from fakes import HIGH, LOW, FakeEnv

from wrapper import THRUST_IDX, ThrustDegradeDelay


def _cmd(i: int, thrust: float = 0.5) -> np.ndarray:
    return np.array([0.01 * i, 0.02 * i, 0.03 * i, thrust], dtype=np.float32)


def test_thrust_is_last_element_of_the_real_action_space():
    """The attitude layout is [roll, pitch, yaw, thrust] (race_core.build_action_space)."""
    from lsy_drone_racing.envs.race_core import build_action_space

    space = build_action_space("attitude", "cf21B_500", "first_principles")
    assert THRUST_IDX == 3
    assert space.low[THRUST_IDX] == pytest.approx(4 * 0.02136263, rel=1e-4)
    assert space.high[THRUST_IDX] == pytest.approx(0.8)
    assert np.allclose(space.low[:3], -np.pi / 2)


def test_k_at_zero_and_at_T_end():
    w = ThrustDegradeDelay(FakeEnv(), k_end=0.85, T_end=18.0)
    assert w.thrust_scale(0.0) == 1.0
    assert w.thrust_scale(18.0) == pytest.approx(0.85)
    assert w.thrust_scale(9.0) == pytest.approx(0.925)  # linear
    assert w.thrust_scale(30.0) == pytest.approx(0.85)  # constant afterwards


def test_scaling_hits_only_thrust_and_follows_k_of_t():
    env = FakeEnv(n_steps=100)
    w = ThrustDegradeDelay(env, k_end=0.5, T_end=1.0, delay=0, freq=50.0)
    w.reset(seed=0)
    for n in range(3):
        w.step(_cmd(n, 0.6))
    for n, got in enumerate(env.received):
        assert np.allclose(got[:3], _cmd(n)[:3])
        assert got[3] == pytest.approx(0.6 * (1 - 0.5 * n / 50), rel=1e-5)


def test_delay_is_d_steps_and_first_command_repeats():
    env = FakeEnv(n_steps=100)
    w = ThrustDegradeDelay(env, k_end=1.0, delay=2)
    w.reset(seed=0)
    cmds = [_cmd(i, 0.3 + 0.01 * i) for i in range(8)]
    for c in cmds:
        w.step(c)
    for n, got in enumerate(env.received):
        assert np.allclose(got, cmds[max(n - 2, 0)], atol=1e-6), n


def test_clip_to_env_thrust_limits_after_scaling():
    env = FakeEnv()
    w = ThrustDegradeDelay(env, k_end=1.0, delay=0)
    w.reset(seed=0)
    w.step(np.array([0, 0, 0, 5.0]))
    w.step(np.array([0, 0, 0, 0.001]))
    assert env.received[0][3] == pytest.approx(HIGH[3])
    assert env.received[1][3] == pytest.approx(LOW[3])


def test_same_seed_is_reproducible_and_seed_is_forwarded():
    def run(seed):
        env = FakeEnv(n_steps=50)
        w = ThrustDegradeDelay(env, k_end=0.85, delay=2, seed=seed)
        obs, _ = w.reset()  # default seed from the constructor
        out = [obs["pos"].copy()]
        for i in range(10):
            obs, *_ = w.step(_cmd(i, 0.4))
            out.append(obs["pos"].copy())
        return env.reset_seeds, np.array(env.received), np.array(out)

    a, b = run(7), run(7)
    assert a[0] == [7] and b[0] == [7]
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert ThrustDegradeDelay(FakeEnv(), seed=7).reset(seed=3) is not None  # explicit seed wins


def test_reset_clears_buffer_and_clock():
    env = FakeEnv(n_steps=100)
    w = ThrustDegradeDelay(env, k_end=0.5, T_end=1.0, delay=2, freq=50.0)
    w.reset(seed=0)
    for i in range(5):
        w.step(_cmd(i))
    w.reset(seed=1)
    w.step(_cmd(9, 0.6))
    assert np.allclose(env.received[-1][:3], _cmd(9)[:3])  # first command repeated again
    assert env.received[-1][3] == pytest.approx(0.6)  # k(0) = 1 again


def test_hover_fill_applies_hover_during_the_first_d_steps():
    env = FakeEnv(n_steps=100)
    w = ThrustDegradeDelay(env, k_end=1.0, delay=2, fill="hover", hover_thrust=0.4258)
    w.reset(seed=0)
    cmds = [_cmd(i, 0.7) for i in range(6)]
    for c in cmds:
        w.step(c)
    for n in (0, 1):  # fill phase
        assert np.allclose(env.received[n], [0, 0, 0, 0.4258], atol=1e-6)
    for n in range(2, 6):  # afterwards identical to the default behaviour
        assert np.allclose(env.received[n], cmds[n - 2], atol=1e-6)
    with pytest.raises(ValueError):
        ThrustDegradeDelay(FakeEnv(), fill="hover")  # hover needs hover_thrust


def test_set_params_applies_from_next_reset():
    env = FakeEnv(n_steps=100)
    w = ThrustDegradeDelay(env, k_end=1.0, delay=0)
    w.set_params(k_end=1.0, delay=3)
    w.reset(seed=0)
    cmds = [_cmd(i, 0.3 + 0.01 * i) for i in range(6)]
    for c in cmds:
        w.step(c)
    for n, got in enumerate(env.received):
        assert np.allclose(got, cmds[max(n - 3, 0)], atol=1e-6)
