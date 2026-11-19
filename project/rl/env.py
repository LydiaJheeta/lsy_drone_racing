"""Gate-progress racing env for PPO: a subclass of `VecDroneRaceEnv` (attitude control).

Nothing in the repo is modified or monkeypatched. `GateProgressRaceEnv` only overrides `reset`,
`step` and the two spaces; everything else (dynamics, contacts, autoreset, gate bookkeeping) is the
upstream `RaceCoreEnv`.

Stack used for training (innermost first), built by `make_train_env`:

    GateProgressRaceEnv        flat float32 observation + gate-progress reward, 4D attitude action
    JaxToNumpy (vector)        jax -> numpy at the env boundary
    VecThrustDegradeDelay      command delay + thrust scale k(t)  (OFF in phase 1)
    VecAttitudeAction          policy action [-1,1]^3 -> [roll, pitch, yaw=0, thrust]

### Action (3 values from the policy)

`[-1,1]^3 -> [roll, pitch, 0, thrust]`: `roll = 0.5 * a0`, `pitch = 0.5 * a1` (rad, the same
+-0.5 rad the MPC uses), `thrust = thrust_min + (a1+1)/2 * (thrust_max - thrust_min)` linear into
`[0.0854, 0.8] N`. Yaw is always commanded 0. Actions are clipped to `[-1,1]` first, because the
PPO policy samples from a Normal whose mean is tanh-squashed but whose samples are not bounded.

### Observation (41 floats, fixed normalisation constants)

| slice | content | scale |
|---|---|---|
| 0:3 | `pos` | `POS_SCALE` |
| 3:6 | `vel` | `VEL_SCALE` |
| 6:15 | rotation matrix from `quat`, row-major | 1 |
| 15:18 | `ang_vel` (body frame) | `ANG_VEL_SCALE` |
| 18:21 / 24:27 | next / second-next gate centre minus `pos` | `DPOS_SCALE` |
| 21:24 / 27:30 | that gate's pass normal (gate x-axis, signed by `gate_sequence_direction`) | 1 |
| 30:38 | the 4 obstacles, `obstacle_pos - pos` in xy | `DPOS_SCALE` |
| 38:41 | last action, in `[-1,1]^3` | 1 |

Gate/obstacle poses come from the **observation** (sensor values: nominal until an object has been
within `sensor_range`), not from the internal state. On level 0 nominal == true (checked by a test),
so this matches the brief; on level 2/3 the policy would see nominal poses until it gets close,
which is the intended partial observability.

Gates are crossed along their local x-axis (`envs/utils.py:95-97`), so the pass normal is
`R(gate_quat) @ [1,0,0]`, negated where `gate_sequence_direction < 0`. Once all gates are passed the
gate index is clamped, so the two gate slots repeat the last gate; that only ever happens on the
step where the episode ends.

### Reward

`r = ||g - p_{k-1}|| - ||g - p_k|| - 0.01*||omega||` with `g` the **true** centre of the target gate
**at the start of the step** (read from `self.data` before stepping), so the term does not jump when
the target gate advances inside the step. On a terminating step the reward is **only** the terminal
term: `+10` if the lap was completed (`n_gates_passed == len(gate_sequence)`), `-10` otherwise.
Truncation gets no terminal term (progress and the omega penalty still apply).

Two upstream details force this shape:

* disabled drones are warped to `pos = (-1,-1,-1)` before the observation is built
  (`race_core.py:557`), so `p_k` is meaningless on a terminating step;
* autoreset runs *inside* `step`, before the reward (`race_core.py:563-574`), so for a world that
  terminated in the previous step the returned observation is already the fresh episode. Those
  "dummy" transitions (gymnasium `AutoresetMode.NEXT_STEP`) get reward exactly 0 and a zeroed
  last action, and `info["dummy"]` marks them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector import JaxToNumpy as VecJaxToNumpy
from jax import Array

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.utils import load_config

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG = "att_level0.toml"

ACT_DIM = 3  # [roll, pitch, thrust], yaw is always 0
OBS_DIM = 41
N_GATE_SLOTS = 2  # how many upcoming gates are in the observation
RP_LIMIT = 0.5  # rad; roll/pitch command limit, as in the attitude MPC

# Observation normalisation (fixed constants, never estimated from data)
POS_SCALE = 3.0  # pos_limit_high = [3, 3, 2.5]
VEL_SCALE = 5.0  # m/s
ANG_VEL_SCALE = 10.0  # rad/s
DPOS_SCALE = 3.0  # relative vectors to gates/obstacles

# Reward
OMEGA_COEF = 0.01
R_SUCCESS = 10.0
R_FAIL = -10.0


def attitude_limits(config: Any) -> tuple[np.ndarray, np.ndarray]:
    """Low/high of the 4D attitude action space of the configured drone."""
    space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
    return np.asarray(space.low, dtype=np.float32), np.asarray(space.high, dtype=np.float32)


def load_rl_config(name: str = CONFIG) -> Any:
    """Load a project config by bare name (looked up in `project/config/`)."""
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_DIR / "config" / path
    return load_config(path)


def map_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """`[-1,1]^3` -> `[roll, pitch, yaw=0, thrust]`. Out-of-range actions are clipped first."""
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    t_min, t_max = float(low[3]), float(high[3])
    cmd = np.zeros((*a.shape[:-1], 4), dtype=np.float32)
    cmd[..., 0] = RP_LIMIT * a[..., 0]
    cmd[..., 1] = RP_LIMIT * a[..., 1]
    cmd[..., 3] = t_min + 0.5 * (a[..., 2] + 1.0) * (t_max - t_min)
    return cmd


def unmap_action(cmd: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Inverse of `map_action` (used to put the *applied* command into the observation)."""
    c = np.asarray(cmd, dtype=np.float32)
    t_min, t_max = float(low[3]), float(high[3])
    a = np.stack(
        [
            c[..., 0] / RP_LIMIT,
            c[..., 1] / RP_LIMIT,
            2.0 * (c[..., 3] - t_min) / (t_max - t_min) - 1.0,
        ],
        axis=-1,
    )
    return np.clip(a, -1.0, 1.0)


@jax.jit
def _quat_to_matrix(quat: Array) -> Array:
    """Rotation matrix (row-major, (..., 9)) from an xyzw quaternion."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    return jp.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        axis=-1,
    )


@jax.jit
def _gate_normals(gates_quat: Array) -> Array:
    """Each gate's local x-axis in world coordinates, shape (..., n_gates, 3)."""
    rot = _quat_to_matrix(gates_quat)  # (..., n_gates, 9)
    return rot[..., jp.array([0, 3, 6])]  # first column = R @ [1,0,0]


@jax.jit
def build_obs(obs: dict[str, Array], last_action: Array) -> Array:
    """Flat float32 observation from the env's dict observation. See the module docstring."""
    pos, quat, vel, ang_vel = obs["pos"], obs["quat"], obs["vel"], obs["ang_vel"]
    n_passed = obs["n_gates_passed"]  # (N,)
    sequence = obs["gate_sequence"]  # (N, n_passes)
    direction = obs["gate_sequence_direction"]  # (N, n_passes)
    gates_pos, gates_quat = obs["gates_pos"], obs["gates_quat"]  # (N, n_gates, 3/4)
    n_passes = sequence.shape[-1]
    worlds = jp.arange(pos.shape[0])

    normals = _gate_normals(gates_quat)  # (N, n_gates, 3)
    gate_feats = []
    for slot in range(N_GATE_SLOTS):
        step = jp.clip(n_passed + slot, 0, n_passes - 1)  # clamp past the end of the sequence
        gate_id = sequence[worlds, step]
        sign = jp.where(direction[worlds, step] < 0, -1.0, 1.0)[:, None]
        gate_feats.append((gates_pos[worlds, gate_id] - pos) / DPOS_SCALE)
        gate_feats.append(sign * normals[worlds, gate_id])

    d_obstacles = (obs["obstacles_pos"][..., :2] - pos[:, None, :2]) / DPOS_SCALE
    flat = jp.concatenate(
        [
            pos / POS_SCALE,
            vel / VEL_SCALE,
            _quat_to_matrix(quat),
            ang_vel / ANG_VEL_SCALE,
            *gate_feats,
            d_obstacles.reshape(pos.shape[0], -1),
            last_action,
        ],
        axis=-1,
    )
    return flat.astype(jp.float32)


@jax.jit
def target_gate_pos(gates_pos: Array, sequence: Array, direction: Array, n_passed: Array) -> Array:
    """Centre of the current target gate, (N, 3). `sequence`/`direction` are the 1D config arrays."""
    step = jp.clip(n_passed, 0, sequence.shape[0] - 1)
    gate_id = sequence[step]
    return gates_pos[jp.arange(gates_pos.shape[0]), gate_id]


@jax.jit
def gate_progress_reward(
    p_prev: Array,
    p_now: Array,
    g_prev: Array,
    ang_vel: Array,
    terminated: Array,
    n_passed_now: Array,
    n_passes: Array,
    dummy: Array,
) -> Array:
    """The reward of the module docstring. All arrays are (N,) or (N, 3)."""
    progress = jp.linalg.norm(g_prev - p_prev, axis=-1) - jp.linalg.norm(g_prev - p_now, axis=-1)
    shaped = progress - OMEGA_COEF * jp.linalg.norm(ang_vel, axis=-1)
    success = terminated & (n_passed_now >= n_passes)
    terminal = jp.where(success, R_SUCCESS, R_FAIL)
    reward = jp.where(terminated, terminal, shaped)
    return jp.where(dummy, 0.0, reward).astype(jp.float32)  # autoreset dummy transition


def obs_from_single(obs: dict[str, Any], last_action: np.ndarray) -> np.ndarray:
    """Build the flat observation from an UNBATCHED dict observation (the single-env eval).

    Uses exactly the same `build_obs` as training, by adding and removing a batch axis, so the
    policy cannot see a different observation in eval than in training.
    """
    batched = {k: jp.asarray(v)[None] for k, v in obs.items()}
    flat = build_obs(batched, jp.asarray(last_action, dtype=jp.float32)[None])
    return np.asarray(flat[0], dtype=np.float32)


class GateProgressRaceEnv(VecDroneRaceEnv):
    """`VecDroneRaceEnv` with a flat observation and the gate-progress reward.

    The action space stays the native 4D attitude command, so `VecThrustDegradeDelay` can sit on
    top of it with exactly the semantics of `project/wrapper.py`. `VecAttitudeAction` turns the
    policy's 3 values into that command.
    """

    def __init__(self, num_envs: int, config: Any | str = CONFIG, seed: int = 0, **kwargs: Any):
        """Create the env from a project config (default `project/config/att_level0.toml`)."""
        cfg = load_rl_config(config) if isinstance(config, str) else config
        if cfg.env.control_mode != "attitude":
            raise ValueError(f"expected control_mode='attitude', got {cfg.env.control_mode!r}")
        super().__init__(
            num_envs=num_envs,
            freq=cfg.env.freq,
            sim_config=cfg.sim,
            sensor_range=cfg.env.sensor_range,
            control_mode=cfg.env.control_mode,
            track=cfg.env.track,
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=seed,
            max_episode_steps=kwargs.pop("max_episode_steps", 1500),
            device=kwargs.pop("device", "cpu"),
        )
        self.config = cfg
        self.att_low, self.att_high = attitude_limits(cfg)
        self.n_passes = int(self.data.gate_sequence.shape[0])
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(self.att_low, self.att_high, dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self._last_action = jp.zeros((num_envs, ACT_DIM), dtype=jp.float32)

    def dict_obs(self) -> dict[str, Array]:
        """The upstream dict observation for the current state, batched as (N, ...).

        Same values `step`/`reset` build the flat observation from; used by the tests that compare
        the training observation with the single-env eval observation.
        """
        from lsy_drone_racing.envs.race_core import obs as core_obs

        return {k: v[:, 0] for k, v in core_obs(self.data).items()}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Array, dict]:
        """Reset all worlds and return the flat observation."""
        obs, info = super().reset(seed=seed, options=options)
        self._last_action = jp.zeros_like(self._last_action)
        return build_obs(obs, self._last_action), info

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Step every world with a 4D attitude command and return the gate-progress reward."""
        data = self.data  # state at the START of the step
        dummy = data.marked_for_reset  # (N,) worlds that autoreset inside this step
        p_prev = data.sim_data.states.pos[:, 0]
        g_prev = target_gate_pos(
            data.gates_pos, data.gate_sequence, data.gate_sequence_direction, data.n_gates_passed[:, 0]
        )
        obs, _, terminated, truncated, info = super().step(action)
        last_action = unmap_action(np.asarray(action), self.att_low, self.att_high)
        self._last_action = jp.where(dummy[:, None], 0.0, jp.asarray(last_action))
        reward = gate_progress_reward(
            p_prev,
            obs["pos"],
            g_prev,
            obs["ang_vel"],
            terminated,
            obs["n_gates_passed"],
            self.n_passes,
            dummy,
        )
        info = {**info, "dummy": dummy, "n_gates_passed": obs["n_gates_passed"]}
        return build_obs(obs, self._last_action), reward, terminated, truncated, info


class VecAttitudeAction(VectorWrapper):
    """Expose a `[-1,1]^3` action space and map it to `[roll, pitch, yaw=0, thrust]`."""

    def __init__(self, env: Any):
        """Wrap a vector env whose action space is the 4D attitude command."""
        super().__init__(env)
        low = np.asarray(env.single_action_space.low, dtype=np.float32)
        high = np.asarray(env.single_action_space.high, dtype=np.float32)
        if low.shape != (4,):
            raise ValueError(f"expected a 4-element attitude action space, got {low.shape}")
        self.att_low, self.att_high = low, high
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def step(self, actions: np.ndarray) -> tuple[Any, Any, Any, Any, dict]:
        """Map the policy action and step the wrapped env."""
        return self.env.step(map_action(actions, self.att_low, self.att_high))


def make_train_env(
    num_envs: int,
    config: str = CONFIG,
    seed: int = 0,
    perturb: dict[str, Any] | None = None,
    device: str = "cpu",
) -> Any:
    """Build the full training stack. `perturb=None` (phase 1) leaves the command untouched."""
    env: Any = GateProgressRaceEnv(num_envs, config=config, seed=seed, device=device)
    env = VecJaxToNumpy(env)
    if perturb is not None:
        from rl.vec_perturb import VecThrustDegradeDelay

        env = VecThrustDegradeDelay(env, **perturb)
    return VecAttitudeAction(env)
