"""Shared helpers: env creation (mirrors scripts/sim.py:60-71), episode runner, failure diagnosis."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

# lsy_drone_racing (-> crazyflow) must be imported before scipy: it sets SCIPY_ARRAY_API.
from lsy_drone_racing.utils import load_config, load_controller  # isort: skip

import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller import Controller

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "config"
RESULTS_DIR = PROJECT_DIR / "results"


def load_cfg(name_or_path: str | Path) -> ConfigDict:
    """Load a TOML config. Bare names are looked up in project/config/."""
    path = Path(name_or_path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    return load_config(path)


def make_env(config: ConfigDict, wrapper: Callable[[Any], Any] | None = None) -> Any:
    """Create the racing env exactly as scripts/sim.py:60-71 does, optionally wrapped.

    The optional `wrapper` is applied after JaxToNumpy, so it sees and returns numpy arrays.
    """
    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)
    return wrapper(env) if wrapper is not None else env


def load_controller_cls(path: str | Path) -> type[Controller]:
    """Load a controller class from any file path (not restricted to control/)."""
    return load_controller(Path(path).resolve())


@dataclass
class EpisodeResult:
    """Outcome of one episode."""

    seed: int
    success: bool
    n_gates: int
    n_gates_total: int
    steps: int
    time_s: float
    cause: str  # "success" | "gate" | "obstacle" | "floor" | "out_of_bounds" | "contact_unknown"
    # | "timeout" | "trajectory_end"
    detail: str = ""
    last_pos: list[float] = field(default_factory=list)
    # The env does not terminate on floor contact, so a crash shows up as a timeout. These let us
    # tell the difference. `min_z_airborne` is the minimum z after the drone first exceeded
    # TAKEOFF_Z (NaN if it never took off); `final_z` is the z at the end (impact z if terminated).
    min_z: float = float("nan")
    min_z_airborne: float = float("nan")
    final_z: float = float("nan")

    def as_row(self) -> dict[str, Any]:
        """Flat dict for CSV output."""
        d = asdict(self)
        d["last_pos"] = " ".join(f"{x:.3f}" for x in self.last_pos)
        return d


TAKEOFF_Z = 0.15  # m; the drone starts at z=0.01


def applied_action(env: Any, default: NDArray) -> NDArray:
    """Return the command that actually reached the base env (after any wrapper), if exposed.

    Wrappers/loggers expose it as `last_applied_action`; we walk the `.env` chain outermost-first.
    Falls back to `default` (the controller's command) when nothing exposes it.
    """
    e = env
    while e is not None:
        a = getattr(e, "last_applied_action", None)
        if a is not None:
            return a
        e = getattr(e, "env", None)
    return default


def z_stats(zs: list[float]) -> tuple[float, float, float]:
    """(min_z, min_z_airborne, final_z) from a z trace. See EpisodeResult."""
    arr = np.asarray(zs, dtype=float)
    up = np.flatnonzero(arr > TAKEOFF_Z)
    airborne = float(arr[up[0] :].min()) if up.size else float("nan")
    return float(arr.min()), airborne, float(arr[-1])


def _ring_distance(local: NDArray, half_inner: float = 0.2, half_outer: float = 0.36) -> float:
    """Distance from a point (gate frame) to the square ring of the gate frame."""
    x, y, z = local
    m = max(abs(y), abs(z))  # Chebyshev radius in the gate plane; ring = half_inner <= m <= half_outer
    in_plane = max(m - half_outer, half_inner - m, 0.0)
    return float(np.sqrt(x**2 + in_plane**2))


def diagnose_termination(env: Any, data_pre: Any, action: NDArray) -> tuple[str, str, list[float]]:
    """Replay the terminating step from the saved pre-step data and classify the cause.

    The env warps disabled drones to z=-1 after the step, so the real impact position is gone from
    the observation. We re-run the (deterministic) action+sim step on `data_pre`, which gives the
    state *before* the warp, then evaluate bounds and contacts with the env's own functions.

    Returns:
        (cause, detail, impact_position)
    """
    core = env.unwrapped
    apply_fn = core.build_apply_action_fn()
    sim_step = core.sim.build_step_fn()
    data = apply_fn(np.asarray(action), data_pre)
    n_steps = data.sim_data.core.freq // core.settings.freq
    data = data.replace(sim_data=sim_step(data.sim_data, n_steps))
    pos = np.asarray(data.sim_data.states.pos).reshape(-1, 3)[0]
    low = np.asarray(data.pos_limit_low).reshape(-1, 3)[0]
    high = np.asarray(data.pos_limit_high).reshape(-1, 3)[0]
    contact = bool(np.asarray(core.build_contact_check_fn()(data)).any())
    pos_l = [float(x) for x in pos]
    if pos[2] < low[2]:
        return "floor", f"z={pos[2]:.4f} < z_low={low[2]}", pos_l
    if np.any(pos < low) or np.any(pos > high):
        axis = [i for i in range(3) if pos[i] < low[i] or pos[i] > high[i]]
        return "out_of_bounds", f"axes={axis} pos={np.round(pos, 3).tolist()}", pos_l
    if not contact:
        return "contact_unknown", "terminated without bounds violation or detected contact", pos_l
    # Attribute the contact to the nearest gate frame or obstacle pole
    gates_pos = np.asarray(data.gates_pos).reshape(-1, 3)
    gates_quat = np.asarray(data.gates_quat).reshape(-1, 4)
    obs_pos = np.asarray(data.obstacles_pos).reshape(-1, 3)
    d_gate = [
        _ring_distance(R.from_quat(q).inv().apply(pos - p)) for p, q in zip(gates_pos, gates_quat)
    ]
    d_obs = [float(np.linalg.norm(pos[:2] - p[:2])) for p in obs_pos]
    g, o = int(np.argmin(d_gate)), int(np.argmin(d_obs))
    if d_gate[g] <= d_obs[o]:
        return "gate", f"gate_id={g} dist_to_frame={d_gate[g]:.3f} (heuristic)", pos_l
    return "obstacle", f"obstacle_id={o} xy_dist={d_obs[o]:.3f} (heuristic)", pos_l


def run_episode(
    env: Any,
    controller_cls: type[Controller],
    config: ConfigDict,
    seed: int,
    keep_trace: list | None = None,
) -> EpisodeResult:
    """Run one episode like scripts/sim.py:73-108, with a fixed seed and failure diagnosis.

    The episode ends as soon as all gates are passed (success), on termination/truncation, or when
    the controller reports it is finished. Time is `steps / env.freq`.
    """
    obs, info = env.reset(seed=seed)
    controller = controller_cls(obs, info, config)
    n_total = int(obs["gate_sequence"].shape[0])
    steps, last_pos = 0, [float(x) for x in obs["pos"]]
    terminated = truncated = finished = False
    data_pre, action = None, None
    zs = [float(obs["pos"][2])]
    while True:
        action = controller.compute_control(obs, info)
        data_pre = env.unwrapped.data
        if keep_trace is not None:
            keep_trace.append(np.concatenate([obs["pos"], [float(obs["n_gates_passed"])]]))
        last_pos = [float(x) for x in obs["pos"]]
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        if not terminated:  # a disabled drone is warped to z=-1 by the env, so skip that obs
            zs.append(float(obs["pos"][2]))
        finished = controller.step_callback(action, obs, 0.0, terminated, truncated, info)
        if int(obs["n_gates_passed"]) >= n_total or terminated or truncated or finished:
            break
    controller.episode_callback()
    controller.episode_reset()
    n_gates = int(obs["n_gates_passed"])
    res = dict(
        seed=seed,
        n_gates=n_gates,
        n_gates_total=n_total,
        steps=steps,
        time_s=steps / config.env.freq,
    )
    if n_gates >= n_total:
        z = z_stats(zs)
        return EpisodeResult(success=True, cause="success", last_pos=last_pos, **_z(z), **res)
    if terminated:
        # Replay with the command that was *applied* (differs from `action` under a wrapper).
        cause, detail, pos = diagnose_termination(env, data_pre, applied_action(env, action))
        zs.append(pos[2])
        return EpisodeResult(
            success=False, cause=cause, detail=detail, last_pos=pos, **_z(z_stats(zs)), **res
        )
    z = _z(z_stats(zs))
    if truncated:
        return EpisodeResult(
            success=False, cause="timeout", detail="max_episode_steps", last_pos=last_pos, **z, **res
        )
    return EpisodeResult(
        success=False, cause="trajectory_end", detail="controller finished", last_pos=last_pos,
        **z, **res,
    )


def _z(stats: tuple[float, float, float]) -> dict[str, float]:
    """Map z_stats output to EpisodeResult keyword arguments."""
    return dict(min_z=stats[0], min_z_airborne=stats[1], final_z=stats[2])


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run `fn` and return (result, wall seconds)."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


class Tee:
    """File-like object duplicating writes to several streams (used to log stdout to a file)."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        """Write to every stream and flush."""
        for st in self.streams:
            st.write(text)
            st.flush()
        return len(text)

    def flush(self) -> None:
        """Flush every stream."""
        for st in self.streams:
            st.flush()
