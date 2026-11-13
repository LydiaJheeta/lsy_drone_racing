"""Tests for logger.EpisodeLogger."""

from __future__ import annotations

import numpy as np
from fakes import FakeEnv

from logger import EpisodeLogger
from wrapper import ThrustDegradeDelay


def _run(env, n=5, thrust=0.5, seed=3):
    env.reset(seed=seed)
    for i in range(n):
        env.step(np.array([0.1 * i, 0, 0, thrust]))


def test_saves_one_npz_per_episode_with_expected_arrays(tmp_path):
    base = FakeEnv(n_steps=5)
    lg = EpisodeLogger(base, tmp_path, freq=50.0, tag="x")
    _run(lg)
    d = np.load(tmp_path / "x_seed003.npz")
    assert d["t"].shape == (5,) and np.allclose(d["t"], np.arange(5) / 50)
    assert d["state"].shape == (5, 13) and d["action"].shape == (5, 4)
    assert d["final_state"].shape == (13,)
    assert list(d["next_gate_idx"]) == [0, 0, 1, 1, 2]  # FakeEnv passes a gate every 2 steps
    assert list(d["next_gate_id"]) == [0, 0, 1, 1, 2]
    assert d["truncated"] and not d["terminated"] and int(d["seed"]) == 3
    assert np.allclose(d["state"][:, 0], np.arange(5))  # state recorded *before* each step


def test_records_the_command_applied_to_the_env_after_the_wrapper(tmp_path):
    base = FakeEnv(n_steps=6)
    lg = EpisodeLogger(base, tmp_path, tag="w")
    env = ThrustDegradeDelay(lg, k_end=0.5, T_end=0.1, delay=2, freq=50.0)  # k falls fast
    env.reset(seed=1)
    cmds = [np.array([0.1 * i, 0, 0, 0.6]) for i in range(6)]
    for c in cmds:
        env.step(c)
    d = np.load(tmp_path / "w_seed001.npz")
    assert np.allclose(d["action"], np.array(base.received))  # what the base env got
    assert not np.allclose(d["action"][:, 3], 0.6)  # i.e. not the controller's thrust
    assert np.allclose(d["action"][3:, 0], [0.1, 0.2, 0.3])  # delayed by 2 steps


def test_flush_on_reset_and_no_overwrite(tmp_path):
    lg = EpisodeLogger(FakeEnv(n_steps=50), tmp_path, tag="p")
    _run(lg, n=2, seed=5)  # not finished: flushed by the next reset
    _run(lg, n=2, seed=5)
    lg.close()
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == ["p_seed005.npz", "p_seed005_1.npz"]


def test_in_memory_mode(tmp_path):
    lg = EpisodeLogger(FakeEnv(n_steps=3), None)
    _run(lg, n=3)
    assert lg.last_episode["action"].shape == (3, 4)
    assert not list(tmp_path.iterdir())
