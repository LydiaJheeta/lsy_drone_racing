"""Tests for common.py (pure helpers + episode runner on the real env)."""

from __future__ import annotations

import io

import numpy as np
import pytest

from common import Tee, applied_action, run_episode, z_stats, _ring_distance


def test_ring_distance():
    assert _ring_distance(np.array([0.0, 0.3, 0.0])) == pytest.approx(0.0)  # on the frame
    assert _ring_distance(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.2)  # centre of the opening
    assert _ring_distance(np.array([1.0, 0.3, 0.0])) == pytest.approx(1.0)  # 1 m in front


def test_z_stats():
    assert z_stats([0.01, 0.02, 0.5, 1.0, 0.03]) == (0.01, 0.03, 0.03)
    mn, air, fin = z_stats([0.01, 0.01, 0.02])  # never took off
    assert mn == 0.01 and np.isnan(air) and fin == 0.02


def test_applied_action_walks_outermost_first():
    class W:
        def __init__(self, env=None, a=None):
            self.env, self.last_applied_action = env, a

    inner, outer = W(None, 1), W(None, None)
    outer.env = inner
    assert applied_action(outer, 0) == 1
    assert applied_action(W(None, None), 7) == 7


def test_tee_duplicates_writes():
    a, b = io.StringIO(), io.StringIO()
    Tee(a, b).write("hi")
    assert a.getvalue() == b.getvalue() == "hi"


class _Const:
    """Controller returning a fixed attitude command; `finish_after` ends the episode."""

    def __init__(self, cmd, finish_after=None):
        self.cmd, self.finish_after, self.n = np.array(cmd, dtype=np.float32), finish_after, 0

    def compute_control(self, obs, info=None):
        return self.cmd

    def step_callback(self, *a, **k):
        self.n += 1
        return self.finish_after is not None and self.n >= self.finish_after

    def episode_callback(self): ...
    def episode_reset(self): ...


def test_run_episode_trajectory_end_is_reproducible(real_env):
    cfg, env = real_env
    mk = lambda *a: _Const([0, 0, 0, 0.0434 * 9.81], finish_after=20)  # noqa: E731
    r1, r2 = run_episode(env, mk, cfg, seed=4), run_episode(env, mk, cfg, seed=4)
    assert (r1.steps, r1.cause, r1.success, r1.n_gates) == (20, "trajectory_end", False, 0)
    assert r1.time_s == pytest.approx(20 / 50)  # steps / 50 (sim.py would report 19/50)
    assert (r1.final_z, r1.min_z) == (r2.final_z, r2.min_z)  # same seed, same result


def test_run_episode_diagnoses_out_of_bounds(real_env):
    """Full thrust straight up leaves the z<=2 m box; the replayed step must say so."""
    cfg, env = real_env
    res = run_episode(env, lambda *a: _Const([0, 0, 0, 0.8]), cfg, seed=0)
    assert not res.success and res.cause == "out_of_bounds"
    assert res.final_z > 2.0 and res.min_z_airborne > 0.1
