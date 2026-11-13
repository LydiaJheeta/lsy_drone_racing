"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.

Copy of lsy_drone_racing/control/attitude_mpc.py with two added constructor arguments; with their
defaults the controller is bit-identical to the original:

* `T_traj` (default 18.75 s = the original value): duration of the reference spline. A smaller
  `T_traj` flies the same waypoints faster.
* `u_weight_scale` (default 1.0): multiplies the WHOLE input-weight block
  `R = diag([1, 1, 1, 50])` of the LINEAR_LS cost, i.e. the only terms that penalise the commands
  (roll/pitch/yaw towards 0 and collective thrust towards hover). The formulation has no
  rate-of-change (du) term, so there is nothing else to scale. Scaling R uniformly keeps the
  original 1:1:1:50 ratio between the attitude and the thrust command penalties and moves the
  whole cost towards "cheap tracking error, expensive control effort" as the scale grows.

Also, `lsy_drone_racing` is imported first because crazyflow must be imported before scipy.
"""

from __future__ import annotations  # Python 3.10 type hints

import lsy_drone_racing  # noqa: F401  (must precede scipy, see module docstring)

import os
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


DEFAULT_T_TRAJ = 18.75  # s, duration of the reference spline in the original attitude_mpc.py


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from the symbolic drone dynamics."""
    # For more info on the models, check out https://github.com/learnsyslab/crazyflow
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False, u_weight_scale: float = 1.0
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver.

    Args:
        Tf: Prediction horizon in seconds.
        N: Number of horizon steps.
        parameters: Drone parameters from `crazyflow.dynamics.load_params`.
        verbose: Passed to `AcadosOcpSolver`.
        u_weight_scale: Factor applied to the whole input-weight block R (see module docstring).
            1.0 reproduces the original tuning; > 1 penalises the commands more.
    """
    if u_weight_scale <= 0:
        raise ValueError("u_weight_scale must be > 0")
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            50.0,  # pos
            50.0,  # pos
            400.0,  # pos
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # vel
            10.0,  # vel
            10.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust). The whole block is
    # scaled by `u_weight_scale`: the only command-penalising terms (there is no du cost).
    R = u_weight_scale * np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            50.0,  # thrust
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references. We will overwrite these later to track the trajectory
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    total_thrust_min = parameters["thrust_min"] * 4
    total_thrust_max = parameters["thrust_max"] * 4
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, total_thrust_min])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, total_thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    cmake_builder = None
    # On Windows, use CMake with MinGW to build the generated MPC solver DLL.
    if os.name == "nt":
        cmake_builder = ocp_get_default_cmake_builder()
        cmake_builder.generator = "MinGW Makefiles"
        cmake_builder.build_dir = "build-mingw"

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
        cmake_builder=cmake_builder,
    )

    # The weights have to be pushed into the solver at runtime, not only through the generated
    # code: every solver of this process is loaded from the same shared library path
    # (c_generated_code/lib..._basic_example_mpc.so), and the OS loader returns the image it
    # already has open, so a solver built later in the same process silently keeps the FIRST
    # build's W. Verified: without these calls, u_weight_scale=30 reproduced the
    # u_weight_scale=1 trajectory step for step. cost_set writes into this solver instance only.
    for stage in range(N):
        acados_ocp_solver.cost_set(stage, "W", ocp.cost.W)
    acados_ocp_solver.cost_set(N, "W", ocp.cost.W_e)

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
        T_traj: float = DEFAULT_T_TRAJ,
        u_weight_scale: float = 1.0,
    ):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
            T_traj: Duration in seconds of the reference spline through the waypoints.
            u_weight_scale: Factor on the input-weight block R of the cost (see module docstring).
                1.0 is the original tuning.
        """
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Same waypoints as in the state controller. Determined by trial and error.
        start_pos = obs["pos"]
        waypoints = np.array(
            [
                start_pos,
                [-1.0, 0.75, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [1.2, -0.15, 1.2],
                [1.05, 0.75, 1.2],
                [0.25, 1.25, 1.2],
            ]
        )
        self._t_total = T_traj  # s (original: 18.75)
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = self._des_pos_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
        self._waypoints_vel = self._des_vel_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0

        self.drone_params = load_params(Dynamics.so_rpy, config.sim.drone)
        self.u_weight_scale = float(u_weight_scale)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params, u_weight_scale=self.u_weight_scale
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._config = config
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Setting state reference
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]  # position
        # zero roll, pitch
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]  # yaw
        yref[:, 6:9] = self._waypoints_vel[i : i + self._N]  # velocity
        # zero drpy

        # Setting input reference (index > self._nx)
        # zero rpy
        # hover thrust
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]  # position
        # zero roll, pitch
        yref_e[5] = self._waypoints_yaw[i + self._N]  # yaw
        yref_e[6:9] = self._waypoints_vel[i + self._N]  # velocity
        # zero drpy
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Solving problem and getting first input
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")

        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
