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


def test_u_weight_scale_scales_only_the_input_weight_block(monkeypatch):
    """`u_weight_scale` multiplies R = diag([1,1,1,50]) and leaves the state weights alone."""
    import my_mpc
    from crazyflow.dynamics import Dynamics, load_params

    class FakeSolver:  # building the real solver is slow and not what we are checking here
        def __init__(self, ocp, **kwargs):
            self.ocp, self.set_W = ocp, []

        def cost_set(self, stage, field, value):
            self.set_W.append((stage, field, np.asarray(value)))

    monkeypatch.setattr(my_mpc, "AcadosOcpSolver", FakeSolver)
    params = load_params(Dynamics.so_rpy, "cf21B_500")
    solver1, ocp1 = my_mpc.create_ocp_solver(0.5, 25, params)
    solver3, ocp3 = my_mpc.create_ocp_solver(0.5, 25, params, u_weight_scale=3.0)
    nx = ocp1.model.x.rows()
    assert np.allclose(np.diag(ocp1.cost.W)[nx:], [1.0, 1.0, 1.0, 50.0])
    assert np.allclose(np.diag(ocp3.cost.W)[nx:], [3.0, 3.0, 3.0, 150.0])
    assert np.allclose(ocp3.cost.W[:nx, :nx], ocp1.cost.W[:nx, :nx])  # state weights unchanged
    assert np.allclose(ocp3.cost.W_e, ocp1.cost.W_e)  # terminal cost has no input term
    # W must also be pushed into the solver instance: the generated shared library is dlopen-cached
    # per path, so a solver built later in the same process would otherwise keep the first W.
    for solver, ocp in ((solver1, ocp1), (solver3, ocp3)):
        stages = [st for st, field, _ in solver.set_W if field == "W"]
        assert stages == list(range(26))  # stages 0..N-1 plus the terminal stage N
        for st, _, value in solver.set_W[:-1]:
            assert np.allclose(value, ocp.cost.W), st
        assert np.allclose(solver.set_W[-1][2], ocp.cost.W_e)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            my_mpc.create_ocp_solver(0.5, 25, params, u_weight_scale=bad)


def test_controller_passes_u_weight_scale_to_the_solver(real_env, tmp_path, monkeypatch):
    """The constructor argument reaches the OCP (default 1.0 = the original tuning)."""
    monkeypatch.chdir(tmp_path)
    cfg, env = real_env
    cls = load_controller_cls(PROJECT_DIR / "my_mpc.py")
    obs, info = env.reset(seed=0)
    c = cls(obs, info, cfg, T_traj=18.75, u_weight_scale=10.0)
    nx = c._ocp.model.x.rows()
    assert c.u_weight_scale == 10.0
    assert np.allclose(np.diag(c._ocp.cost.W)[nx:], [10.0, 10.0, 10.0, 500.0])
    # the weights the REAL solver is using, not just the ones we asked for
    W0 = np.asarray(c._acados_ocp_solver.cost_get(0, "W"))
    assert np.allclose(np.diag(W0)[nx:], [10.0, 10.0, 10.0, 500.0])
    u = c.compute_control(obs, info)
    assert u.shape == (4,) and np.all(np.isfinite(u))
