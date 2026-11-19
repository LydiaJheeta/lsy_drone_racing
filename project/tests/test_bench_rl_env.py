"""Tests for bench_rl_env (throughput benchmark of the vectorized envs)."""

from __future__ import annotations

import numpy as np
import pytest

import bench_rl_env as b

torch = pytest.importorskip("torch", reason="train_rl needs torch (pixi env 'tests', feature 'rl')")


def test_action_buffer_is_deterministic_and_inside_the_action_space():
    """Pre-sampled random actions must be reproducible, so timings compare like with like."""
    env1, actions1, notes = b._build("race_attitude", 2)
    env2, actions2 = b._build_race_env(2, "att_level0.toml")
    try:
        assert "attitude" in notes  # the config is recorded in the result row
        assert actions1.shape == (b.N_ACTION_BUFFER, 2, 4)
        assert np.array_equal(actions1, actions2)
        low, high = env1.single_action_space.low, env1.single_action_space.high
        assert np.all(actions1 >= low) and np.all(actions1 <= high)
    finally:
        env1.close()
        env2.close()


def test_sync_handles_both_jax_dicts_and_torch_tensors():
    """_sync must force completion for the racing env (jax dict) and train_rl (torch tensor)."""
    import jax.numpy as jp

    b._sync({"pos": jp.zeros((4, 3)), "vel": jp.ones((4, 3))})  # must not raise
    b._sync(torch.zeros((4, 7)))


def test_train_rl_patch_is_idempotent_and_fixes_both_incompatibilities():
    """train_rl.py does not run on crazyflow 0.4.0; the shims adapt the two call signatures."""
    from lsy_drone_racing.control import train_rl

    first = b._patch_train_rl_for_crazyflow_040()
    assert len(first) == 2, f"expected both shims on a fresh import, got {first}"
    assert b._patch_train_rl_for_crazyflow_040() == [], "patching twice must be a no-op"

    # Shim 1: the sim step stage is called "integration" in crazyflow 0.4.0, not "integrate".
    from crazyflow.sim import Sim

    sim = Sim(n_worlds=1, n_drones=1, device="cpu")
    stages = list(sim.step_pipeline)  # OrderedDict keyed by stage name
    assert "integration" in stages and "integrate" not in stages

    # Shim 2: reset-pipeline functions are called as fn(data, default_data, mask).
    from crazyflow.dynamics import Dynamics

    fn = train_rl.RandTrajEnv.build_reset_randomization_fn(
        object.__new__(train_rl.RandTrajEnv), Dynamics.first_principles
    )
    data = sim.data
    out = fn(data, sim.default_data, np.ones((1,), dtype=bool))
    assert float(out.states.rotor_vel.max()) == pytest.approx(10000.0)


def test_benchmark_returns_consistent_rates():
    """A tiny end-to-end run: the reported rates must match the measured wall times."""
    r = b.benchmark("race_attitude", num_envs=2, n_steps=4, n_warmup=1)
    assert not r.error
    assert r.action_dim == 4
    assert len(r.per_step_ms) == 4
    assert r.env_steps_per_s_blocked == pytest.approx(2 * 4 / r.blocked_s, rel=1e-6)
    assert r.vec_steps_per_s_blocked == pytest.approx(4 / r.blocked_s, rel=1e-6)
    assert r.ms_per_vec_step_blocked == pytest.approx(r.blocked_s / 4 * 1e3, rel=1e-6)
    assert 0.0 <= r.done_fraction <= 1.0


def test_unknown_pipeline_is_rejected():
    with pytest.raises(ValueError, match="Unknown pipeline"):
        b._build("does_not_exist", 1)
