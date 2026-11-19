"""Tests for rl.env: action mapping, observation, gate-progress reward."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from rl.env import (
    ACT_DIM,
    DPOS_SCALE,
    OBS_DIM,
    OMEGA_COEF,
    POS_SCALE,
    RP_LIMIT,
    R_FAIL,
    R_SUCCESS,
    GateProgressRaceEnv,
    _quat_to_matrix,
    attitude_limits,
    build_obs,
    gate_progress_reward,
    load_rl_config,
    make_train_env,
    map_action,
    obs_from_single,
    unmap_action,
)

CFG = load_rl_config("att_level0.toml")
LOW, HIGH = attitude_limits(CFG)
T_MIN, T_MAX = float(LOW[3]), float(HIGH[3])
ANG_VEL_SCALE = 10.0  # rl.env.ANG_VEL_SCALE, re-stated so a silent change breaks a test
REPO_LEVEL0 = str(Path(__file__).resolve().parents[2] / "config" / "level0.toml")


@pytest.fixture(scope="module")
def vec_env():
    """A 2-world training stack (built once; JIT compilation takes a few seconds)."""
    env = make_train_env(2, seed=0)
    yield env
    env.close()


# region action mapping


def test_action_mapping_hits_the_corners_of_the_command_box():
    cmd = map_action(np.array([[-1, -1, -1], [1, 1, 1], [0, 0, 0]], dtype=np.float32), LOW, HIGH)
    assert cmd.shape == (3, 4)
    assert np.allclose(cmd[0], [-RP_LIMIT, -RP_LIMIT, 0.0, T_MIN])
    assert np.allclose(cmd[1], [RP_LIMIT, RP_LIMIT, 0.0, T_MAX])
    assert np.allclose(cmd[2], [0.0, 0.0, 0.0, 0.5 * (T_MIN + T_MAX)])
    assert np.all(cmd[:, 2] == 0.0), "yaw must always be commanded zero"


def test_action_mapping_clips_unbounded_policy_samples():
    """The PPO policy samples from a Normal, so actions can leave [-1,1]."""
    cmd = map_action(np.array([[-5.0, 7.0, 3.0]], dtype=np.float32), LOW, HIGH)
    assert np.allclose(cmd[0], [-RP_LIMIT, RP_LIMIT, 0.0, T_MAX])
    assert np.all(cmd >= LOW) and np.all(cmd <= HIGH)


def test_action_mapping_roundtrip():
    a = np.linspace(-1, 1, 7, dtype=np.float32)[:, None].repeat(ACT_DIM, axis=1)
    assert np.allclose(unmap_action(map_action(a, LOW, HIGH), LOW, HIGH), a, atol=1e-6)


# region observation


def test_quat_to_matrix_matches_scipy():
    rng = np.random.default_rng(0)
    quat = rng.normal(size=(5, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    mine = np.asarray(_quat_to_matrix(quat)).reshape(5, 3, 3)
    assert np.allclose(mine, R.from_quat(quat).as_matrix(), atol=1e-6)


def test_obs_layout_at_reset(vec_env):
    """Known values at the level-0 start pose: scaled pos, identity rotation, zero last action."""
    obs, _ = vec_env.reset(seed=3)
    assert obs.shape == (vec_env.num_envs, OBS_DIM) and obs.dtype == np.float32
    assert np.isfinite(obs).all()
    core = vec_env.unwrapped
    start = np.asarray(core.config.env.track.drones[0]["pos"], dtype=np.float32)
    assert np.allclose(obs[0, 0:3], start / POS_SCALE, atol=1e-6)
    assert np.allclose(obs[0, 3:6], 0.0)  # at rest
    assert np.allclose(obs[0, 6:15], np.eye(3).ravel(), atol=1e-6)  # rpy = 0
    assert np.allclose(obs[0, 15:18], 0.0)
    assert np.allclose(obs[0, 38:41], 0.0)  # no action yet


def test_obs_gate_features_use_the_true_gates_and_the_signed_gate_x_axis(vec_env):
    obs, _ = vec_env.reset(seed=0)
    core = vec_env.unwrapped
    gates_pos = np.asarray(core.data.gates_pos)[0]
    gates_quat = np.asarray(core.data.gates_quat)[0]
    sequence = np.asarray(core.data.gate_sequence)
    direction = np.asarray(core.data.gate_sequence_direction)
    pos = obs[0, 0:3] * POS_SCALE
    for slot, (sl_d, sl_n) in enumerate([(slice(18, 21), slice(21, 24)), (slice(24, 27), slice(27, 30))]):
        gate_id = sequence[slot]
        expected_d = (gates_pos[gate_id] - pos) / DPOS_SCALE
        normal = R.from_quat(gates_quat[gate_id]).apply([1.0, 0.0, 0.0])
        if direction[slot] < 0:
            normal = -normal
        assert np.allclose(obs[0, sl_d], expected_d, atol=1e-5), f"gate slot {slot} offset"
        assert np.allclose(obs[0, sl_n], normal, atol=1e-5), f"gate slot {slot} normal"
        assert np.isclose(np.linalg.norm(obs[0, sl_n]), 1.0, atol=1e-5)


def test_level0_sensor_poses_equal_the_true_poses(vec_env):
    """Task D assumes sensor == nominal == true on level 0."""
    core = vec_env.unwrapped
    assert np.allclose(np.asarray(core.data.gates_pos), np.asarray(core.data.nominal_gates_pos))
    assert np.allclose(np.asarray(core.data.gates_quat), np.asarray(core.data.nominal_gates_quat))
    assert np.allclose(
        np.asarray(core.data.obstacles_pos), np.asarray(core.data.nominal_obstacles_pos)
    )


def test_obstacle_features_are_relative_xy(vec_env):
    obs, _ = vec_env.reset(seed=0)
    pos = obs[0, 0:3] * POS_SCALE
    obstacles = np.asarray(vec_env.unwrapped.data.obstacles_pos)[0]
    expected = ((obstacles[:, :2] - pos[:2]) / DPOS_SCALE).ravel()
    assert np.allclose(obs[0, 30:38], expected, atol=1e-5)


def test_obs_from_single_matches_the_batched_observation(vec_env):
    """The eval path must build exactly the training observation."""
    obs, _ = vec_env.reset(seed=11)
    core = vec_env.unwrapped
    dict_obs = {k: np.asarray(v)[0] for k, v in core.dict_obs().items()}
    single = obs_from_single(dict_obs, np.zeros(ACT_DIM, dtype=np.float32))
    assert np.allclose(single, obs[0], atol=1e-6)


# region reward


def _reward(p_prev, p_now, g, omega=(0.0, 0.0, 0.0), terminated=False, gates=0, dummy=False):
    out = gate_progress_reward(
        np.array([p_prev], dtype=np.float32),
        np.array([p_now], dtype=np.float32),
        np.array([g], dtype=np.float32),
        np.array([omega], dtype=np.float32),
        np.array([terminated]),
        np.array([gates]),
        5,
        np.array([dummy]),
    )
    return float(np.asarray(out)[0])


def test_progress_towards_the_gate_is_positive_and_equals_the_distance_gained():
    r = _reward((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert r == pytest.approx(0.3, abs=1e-6)
    assert _reward((0.3, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(-0.3, abs=1e-6)


def test_omega_penalty():
    r0 = _reward((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (1.0, 0.0, 0.0))
    r1 = _reward((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (1.0, 0.0, 0.0), omega=(3.0, 4.0, 0.0))
    assert r1 == pytest.approx(r0 - OMEGA_COEF * 5.0, abs=1e-6)


def test_terminal_steps_carry_only_the_terminal_term():
    """Progress is meaningless on a terminal step: the drone has been warped to (-1,-1,-1)."""
    assert _reward((0.5, 0, 0), (-1, -1, -1), (1, 0, 0), terminated=True, gates=5) == R_SUCCESS
    assert _reward((0.5, 0, 0), (-1, -1, -1), (1, 0, 0), terminated=True, gates=3) == R_FAIL
    # the huge apparent progress of the warp never enters the reward
    assert _reward((0.5, 0, 0), (-1, -1, -1), (1, 0, 0), terminated=True, gates=5, omega=(9, 9, 9)) == R_SUCCESS


def test_autoreset_dummy_transition_is_exactly_zero():
    assert _reward((0.5, 0, 0), (-1.5, 0.75, 0.01), (1, 0, 0), dummy=True) == 0.0
    assert _reward((0.5, 0, 0), (0.9, 0, 0), (1, 0, 0), terminated=True, gates=5, dummy=True) == 0.0


def test_the_step_that_passes_a_gate_uses_the_old_target(vec_env):
    """Fly through gate 0 in one step and check the reward against the PRE-step target gate.

    The drone is placed 0.1 m in front of the gate plane with a high velocity along the gate
    normal, so a single env step crosses the plane and `n_gates_passed` advances inside the step.
    The reward must still be measured against the gate the drone was flying towards; measuring
    against the new target would change it by ~1 m (asserted below), which is the jump the brief
    asks to avoid.
    """
    core = vec_env.unwrapped
    vec_env.reset(seed=0)
    gates_pos = np.asarray(core.data.gates_pos)[0]
    gates_quat = np.asarray(core.data.gates_quat)[0]
    sequence = np.asarray(core.data.gate_sequence)
    g_old, g_new = gates_pos[sequence[0]], gates_pos[sequence[1]]
    normal = R.from_quat(gates_quat[sequence[0]]).apply([1.0, 0.0, 0.0])

    p_prev = g_old - 0.1 * normal
    vel = 15.0 * normal
    sd = core.data.sim_data
    pos_all = np.asarray(sd.states.pos).copy()
    vel_all = np.asarray(sd.states.vel).copy()
    pos_all[0, 0], vel_all[0, 0] = p_prev, vel
    core.data = core.data.replace(
        sim_data=sd.replace(states=sd.states.replace(pos=pos_all, vel=vel_all))
    )
    n_before = int(np.asarray(core.data.n_gates_passed)[0, 0])

    hover = np.array([[0.0, 0.0, 0.0]] * vec_env.num_envs, dtype=np.float32)
    obs, r, term, trunc, info = vec_env.step(hover)
    assert int(info["n_gates_passed"][0]) == n_before + 1, "the step must pass gate 0"

    p_now = obs[0, 0:3] * POS_SCALE
    omega = obs[0, 15:18] * ANG_VEL_SCALE
    expected_old = (
        np.linalg.norm(g_old - p_prev) - np.linalg.norm(g_old - p_now)
        - OMEGA_COEF * np.linalg.norm(omega)
    )
    expected_new = (
        np.linalg.norm(g_new - p_prev) - np.linalg.norm(g_new - p_now)
        - OMEGA_COEF * np.linalg.norm(omega)
    )
    assert float(r[0]) == pytest.approx(expected_old, abs=2e-3)
    # Discriminating: with the post-step target the same step would be scored ~-0.004 instead of
    # ~-0.098 (measured), i.e. the assertion above would fail by a factor of ~20.
    assert abs(expected_new - expected_old) > 0.05, "the two targets must differ measurably"


# region env-level behaviour


def test_terminal_then_dummy_sequence(vec_env):
    """Crash one world on purpose: terminal step gives -10, the next step gives exactly 0."""
    vec_env.reset(seed=0)
    crash = np.array([[0.0, 0.0, -1.0]] * vec_env.num_envs, dtype=np.float32)  # min thrust: fall
    climb = np.array([[0.0, 0.6, 1.0]] * vec_env.num_envs, dtype=np.float32)  # leave the pad first
    for _ in range(12):
        vec_env.step(climb)
    for i in range(400):
        obs, r, term, trunc, info = vec_env.step(crash)
        if term.any():
            w = int(np.flatnonzero(term)[0])
            assert r[w] == pytest.approx(R_FAIL), f"terminal reward {r[w]}"
            obs2, r2, term2, trunc2, info2 = vec_env.step(crash)
            assert bool(info2["dummy"][w]), "the world must be autoreset on the next step"
            assert r2[w] == 0.0, f"dummy transition reward {r2[w]}"
            assert np.allclose(obs2[w, 38:41], 0.0), "last action must be cleared on autoreset"
            assert not bool(term2[w])
            return
    pytest.fail("no world terminated within 400 steps of falling")


def test_make_train_env_spaces_and_finite_rollout():
    env = make_train_env(2, seed=0)
    try:
        assert env.single_action_space.shape == (ACT_DIM,)
        assert env.single_observation_space.shape == (OBS_DIM,)
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, (2, ACT_DIM)).astype(np.float32))
            assert np.isfinite(obs).all() and np.isfinite(r).all()
            assert obs.shape == (2, OBS_DIM)
    finally:
        env.close()


def test_rejects_a_state_control_config():
    """The repo's level0.toml is control_mode="state"; the subclass must refuse it."""
    with pytest.raises(ValueError, match="attitude"):
        GateProgressRaceEnv(1, config=REPO_LEVEL0)
