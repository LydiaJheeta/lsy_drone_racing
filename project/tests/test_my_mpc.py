"""Tests for my_mpc.py (builds the acados solver; run in a temp cwd to keep the tree clean)."""

from __future__ import annotations

import numpy as np
import pytest

from common import PROJECT_DIR, load_controller_cls


def test_T_traj_sets_spline_duration_and_default_is_unchanged(real_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # acados writes c_generated_code/ into the cwd
    cfg, env = real_env
    cls = load_controller_cls(PROJECT_DIR / "my_mpc.py")
    obs, info = env.reset(seed=0)
    c_default = cls(obs, info, cfg)
    c_fast = cls(obs, info, cfg, T_traj=10.0)
    assert c_default._t_total == 18.75 and c_fast._t_total == 10.0
    assert len(c_default._waypoints_pos) == int(50 * 18.75)
    assert len(c_fast._waypoints_pos) == int(50 * 10.0)
    # same waypoints, just faster: the spline passes through the same end point
    assert np.allclose(c_fast._waypoints_pos[-1], c_default._waypoints_pos[-1], atol=2e-2)
    u = c_fast.compute_control(obs, info)
    assert u.shape == (4,) and np.all(np.isfinite(u))  # [roll, pitch, yaw, thrust]
    assert u[3] > 0.0
