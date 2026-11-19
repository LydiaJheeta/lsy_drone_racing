"""Round 4: experiment E (noise check), F (separated cost knobs), diagnosis G (reference quality).

Run from project/ with the ROS PYTHONPATH removed (see README):

    cd project
    env -u PYTHONPATH pixi run python round4.py --exp=G          # creates results/round4_<ts>/
    env -u PYTHONPATH pixi run python round4.py --exp=EF --out_dir=results/round4_<ts>

Experiment E -- is the drop from 100% to 62% at T_traj=12 a systematic effect of the thrust drop,
    or chaotic sensitivity near the limit? d=0, original weights, level 0, seeds 0-19,
    T_traj in {14, 12} x k_end in {1.0, 0.99, 0.97, 0.95, 0.9, 0.8}. A k_end of 0.99 removes 1% of
    the thrust by the end of the lap, which is far too little to matter dynamically; if the
    conditional success rate already collapses there, the outcome is a coin flip, not a trend.
    Every episode also records, per gate pass, how close the drone came to the gate FRAME
    (`ring_pass<i>`: 0 = touching it), so a flipped seed can be checked for a grazing pass.

Experiment F -- separates the two knobs that round 3 scaled together, at the delay that round 3
    showed is recoverable: T_traj=18.75, k_end=1.0, level 0, seeds 0-19, d in {0, 1},
    rpy_weight_scale in {1, 10, 30} x thrust_weight_scale in {1, 10, 30}, i.e.
    R = diag(rpy_scale * [1, 1, 1], thrust_scale * 50).

Diagnosis G -- no simulation. For the FIXED waypoint spline of my_mpc.py (T_traj = 18.75 and 16),
    the geometry of the reference at each of the five gate passes: distance from the gate centre,
    in-plane offset, clearance to the gate frame and the approach angle to the gate normal. The
    waypoints are NOT changed; this only measures what the reference asks for.

Gate geometry (from `common._ring_distance`, which mirrors the env's own contact check): the
opening is a square of half-width 0.2 m, the frame material reaches out to half-width 0.36 m. So an
in-plane offset below 0.2 m passes through the opening, between 0.2 m and 0.36 m hits the frame.
"""

from __future__ import annotations

import lsy_drone_racing  # noqa: F401  (must be imported before scipy)

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402

import fire  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from common import PROJECT_DIR, RESULTS_DIR, Tee, _ring_distance  # noqa: E402
from common import load_cfg, load_controller_cls, make_env, run_episode  # noqa: E402
from grid import write_csv  # noqa: E402
from round3 import CMD_FIELDS, cell_field, cell_success, cell_success_on_baseline  # noqa: E402
from round3 import cmd_metrics, pivot_table, recording_factory, summarize_conditions  # noqa: E402
from run_eval import FIELDS, _build_env, _fmt, make_row  # noqa: E402
from wrapper import THRUST_IDX  # noqa: E402

E_TS = (14.0, 12.0)
E_KS = (1.0, 0.99, 0.97, 0.95, 0.9, 0.8)
E_FLIP_KS = (0.99, 0.97)  # the "too small to matter" drops the flip table is about
F_T = 18.75
F_DELAYS = (0, 1)
F_RPY_SCALES = (1.0, 10.0, 30.0)
F_THRUST_SCALES = (1.0, 10.0, 30.0)
G_TS = (18.75, 16.0)
N_SEEDS = 20
N_PASSES = 5
GATE_HALF_INNER = 0.2  # m, half-width of the opening
GATE_HALF_OUTER = 0.36  # m, outer half-width of the frame material

RING_FIELDS = [f"ring_pass{i}" for i in range(N_PASSES)]
E_FIELDS = [*FIELDS, "delay", "k_end", "u_weight_scale", *CMD_FIELDS, *RING_FIELDS,
            "ring_min_overall"]
F_FIELDS = [*FIELDS, "delay", "k_end", "rpy_weight_scale", "thrust_weight_scale", *CMD_FIELDS]


# --------------------------------------------------------------------------------------------
# gate geometry
# --------------------------------------------------------------------------------------------


def gate_poses(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """True gate positions and quaternions of the current episode (not the sensed ones)."""
    data = env.unwrapped.data
    return (np.asarray(data.gates_pos).reshape(-1, 3), np.asarray(data.gates_quat).reshape(-1, 4))


def local_points(points: np.ndarray, gate_pos: np.ndarray, gate_quat: np.ndarray) -> np.ndarray:
    """World points expressed in the gate frame (x = gate normal, y/z = in the gate plane)."""
    return Rot.from_quat(gate_quat).inv().apply(np.atleast_2d(points) - gate_pos)


def in_plane_offset(local: np.ndarray) -> np.ndarray:
    """Chebyshev radius in the gate plane: < 0.2 m goes through the opening."""
    return np.maximum(np.abs(local[:, 1]), np.abs(local[:, 2]))


def ring_clearance(local: np.ndarray) -> np.ndarray:
    """Distance to the gate frame for each point (0 = touching it)."""
    return np.array([_ring_distance(p) for p in local])


def pass_clearances(
    ep: dict[str, Any], gates_pos: np.ndarray, gates_quat: np.ndarray
) -> dict[str, float]:
    """Minimum distance to each gate's FRAME while that gate was the drone's target.

    The logger stores `next_gate_idx` (= gates passed so far) per step, so the steps with
    `next_gate_idx == p` are exactly the approach to pass `p`. 0 means the frame was touched.
    """
    pos, idx, ids = ep["state"][:, :3], ep["next_gate_idx"], ep["next_gate_id"]
    out: dict[str, float] = {f: float("nan") for f in RING_FIELDS}
    for p in range(N_PASSES):
        m = idx == p
        if not m.any():
            continue
        g = int(ids[m][0])
        out[f"ring_pass{p}"] = float(ring_clearance(local_points(pos[m], gates_pos[g],
                                                                 gates_quat[g])).min())
    finite = [v for v in out.values() if not math.isnan(v)]
    out["ring_min_overall"] = min(finite) if finite else float("nan")
    return out


# --------------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------------


def run_blocks(
    sub: Path, stage: str, blocks: list[dict[str, Any]], seeds: Sequence[int], cfg: Any,
    ctrl_cls: Any, fields: list[str], ctrl_keys: Sequence[str], with_clearances: bool,
) -> list[dict[str, Any]]:
    """Run every block on the same seeds; stream one row per episode to `episodes.csv`.

    `ctrl_keys` are the block entries handed to the controller (the rest configure the wrapper).
    """
    env, lg = _build_env(cfg, "wrapped", None,
                         dict(k_end=1.0, delay=0, T_end=blocks[0]["T_traj"]))
    thrust_max = float(np.asarray(env.action_space.high)[THRUST_IDX])
    rows: list[dict[str, Any]] = []
    with open(sub / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for b in blocks:
            T, d, k = b["T_traj"], b["delay"], b["k_end"]
            # T_end = T_traj of THIS block: otherwise a fast lap ends before k(t) reaches k_end.
            env.set_params(k_end=k, delay=d, T_end=T)
            label = "_".join(f"{key[0]}{b[key]:g}" for key in ctrl_keys if key != "T_traj")
            lg.tag = f"{stage}_T{T:g}_d{d}_k{k:g}_{label}"
            for seed in seeds:
                cmds: list[np.ndarray] = []
                kw = {key: b[key] for key in ctrl_keys}
                res = run_episode(env, recording_factory(ctrl_cls, cmds, **kw), cfg, seed)
                lg.flush()  # a successful episode does not terminate, so flush it by hand
                extra: dict[str, Any] = {}
                if with_clearances:
                    gp, gq = gate_poses(env)
                    extra = pass_clearances(lg.last_episode, gp, gq)
                row = {
                    **make_row(stage, T, f"d{d}_k{k:g}_{label}", res),
                    "delay": d, "k_end": k,
                    **{key: b[key] for key in ctrl_keys if key != "T_traj"},
                    **cmd_metrics(cmds, env.thrust_scale, thrust_max, cfg.env.freq),
                    **extra,
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()
                tilt = 100.0 * row["tilt_sat_steps"] / max(row["n_cmd_steps"], 1)
                ring = row.get("ring_min_overall", float("nan"))
                print(f"[{stage}] T={T:g} d={d} k={k:g} {label} seed={seed:2d} "
                      f"ok={res.success} gates={res.n_gates} t={res.time_s:.2f}s "
                      f"group={row['group']} tilt_sat={tilt:.1f}% "
                      f"thr_max={row['thrust_cmd_max']:.3f} ring_min={ring:.3f}", flush=True)
    env.close()
    return rows


def tables_for(cond: list[dict[str, Any]], row_key: str, col_key: str, row_vals: Sequence[float],
               col_vals: Sequence[float], title: str, specs: list) -> str:
    """Markdown tables for one experiment."""
    md = [f"## {title}"]
    for name, cell in specs:
        md.append(f"### {name}\n\n" + pivot_table(cond, row_key, col_key, row_vals, col_vals, cell))
    return "\n\n".join(md)


# --------------------------------------------------------------------------------------------
# Experiment E
# --------------------------------------------------------------------------------------------


def flip_table(rows: list[dict[str, Any]], ks: Sequence[float] = E_FLIP_KS) -> list[dict[str, Any]]:
    """Seeds that succeed at k_end=1.0 but fail at a k_end that is dynamically negligible.

    For each one: where and when it fails, and how close the SAME seed came to the gate frame in
    the k_end=1.0 run (`ring_pass<i>` of the baseline episode).
    """
    by = {(r["T_traj"], r["k_end"], int(r["seed"])): r for r in rows}
    out = []
    for T in sorted({r["T_traj"] for r in rows}, reverse=True):
        for k in ks:
            for seed in sorted({int(r["seed"]) for r in rows}):
                base, test = by.get((T, 1.0, seed)), by.get((T, k, seed))
                if base is None or test is None or not base["success"] or test["success"]:
                    continue
                failed_pass = int(test["n_gates"])  # gates passed = index of the pass it died on
                ring_key = f"ring_pass{failed_pass}"
                out.append({
                    "T_traj": T, "k_end": k, "seed": seed,
                    "failed_on_pass": failed_pass + 1,  # 1-based for reading
                    "fail_gate_id": _gate_id(test["detail"]),
                    "fail_cause": test["cause"],
                    "fail_time_s": test["time_s"],
                    "fail_dist_to_frame": _detail_dist(test["detail"]),
                    "nominal_ring_same_pass": base.get(ring_key, float("nan")),
                    "nominal_ring_min_overall": base.get("ring_min_overall", float("nan")),
                    "nominal_time_s": base["time_s"],
                })
    return out


def _gate_id(detail: str) -> str:
    """`gate_id=` of a failure detail string ('-' when the cause is not a gate contact)."""
    for part in str(detail).split():
        if part.startswith("gate_id="):
            return part.split("=", 1)[1]
    return "-"


def _detail_dist(detail: str) -> float:
    """`dist_to_frame=` of a failure detail string (NaN when absent)."""
    for part in str(detail).split():
        if part.startswith("dist_to_frame="):
            return float(part.split("=", 1)[1])
    return float("nan")


def exp_e(out: Path, seeds: Sequence[int], cfg: Any, ctrl_cls: Any,
          ts: Sequence[float] = E_TS, ks: Sequence[float] = E_KS) -> None:
    """Thrust drops down to 1% at two reference durations, with gate clearances per episode."""
    sub = out / "expE"
    sub.mkdir(parents=True, exist_ok=False)
    blocks = [dict(T_traj=float(T), delay=0, k_end=float(k), u_weight_scale=1.0)
              for T in ts for k in ks]
    rows = run_blocks(sub, "expE", blocks, seeds, cfg, ctrl_cls, E_FIELDS,
                      ("T_traj", "u_weight_scale"), with_clearances=True)
    cond = summarize_conditions(rows, ("T_traj", "k_end"), lambda key: (key[0], 1.0))
    write_csv(sub / "failures.csv", [r for r in rows if not r["success"]])
    flips = flip_table(rows)
    write_csv(sub / "flips.csv", flips)
    specs = [
        ("success on the seeds where k_end=1.0 succeeded", cell_success_on_baseline),
        ("success rate (all seeds)", cell_success),
        ("mean lap time of the successes [s]", cell_field("lap_time_mean_s")),
        ("minimum clearance to a gate frame over all episodes [m]",
         cell_field("ring_min_min")),
        ("steps with thrust command > 0.8 N * k(t)", cell_field("thrust_above_eff_pct_all",
                                                                pct=True)),
        ("failure causes", lambda c: c["failure_causes"]),
    ]
    for c in cond:  # add the clearance aggregate the table above asks for
        rs = [r for r in rows if (r["T_traj"], r["k_end"]) == (c["T_traj"], c["k_end"])]
        vals = [r["ring_min_overall"] for r in rs if not math.isnan(r["ring_min_overall"])]
        c["ring_min_min"] = min(vals) if vals else float("nan")
    write_csv(sub / "conditions.csv", cond)
    md = [tables_for(cond, "T_traj", "k_end", [float(t) for t in ts], [float(k) for k in ks],
                     f"Experiment E -- d=0, original weights, T_end=T_traj, level 0, seeds "
                     f"{min(seeds)}-{max(seeds)}. Baseline of each row: k_end=1.0 at the same "
                     f"T_traj.", specs)]
    md.append(flips_markdown(flips))
    (sub / "tables.md").write_text("\n\n".join(md) + "\n")
    print("\n" + "\n\n".join(md), flush=True)


def flips_markdown(flips: list[dict[str, Any]]) -> str:
    """The per-seed flip table of experiment E."""
    head = ("### Seeds that succeed at k_end=1.0 and fail at k_end=0.99 or 0.97\n\n"
            "`nominal_ring_same_pass` = how close the SAME seed came to that gate's frame in the "
            "k_end=1.0 run (m, 0 = touching).\n\n")
    if not flips:
        return head + "(none)"
    cols = ["T_traj", "k_end", "seed", "failed_on_pass", "fail_gate_id", "fail_cause",
            "fail_time_s", "fail_dist_to_frame", "nominal_ring_same_pass",
            "nominal_ring_min_overall", "nominal_time_s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in flips:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return head + "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Experiment F
# --------------------------------------------------------------------------------------------


def exp_f(out: Path, seeds: Sequence[int], cfg: Any, ctrl_cls: Any,
          T: float = F_T, ds: Sequence[int] = F_DELAYS,
          rpys: Sequence[float] = F_RPY_SCALES, thrusts: Sequence[float] = F_THRUST_SCALES) -> None:
    """The 3x3 weight grid at d=0 and d=1."""
    sub = out / "expF"
    sub.mkdir(parents=True, exist_ok=False)
    blocks = [dict(T_traj=float(T), delay=int(d), k_end=1.0,
                   rpy_weight_scale=float(rp), thrust_weight_scale=float(th))
              for d in ds for rp in rpys for th in thrusts]
    rows = run_blocks(sub, "expF", blocks, seeds, cfg, ctrl_cls, F_FIELDS,
                      ("T_traj", "rpy_weight_scale", "thrust_weight_scale"), with_clearances=False)
    cond = summarize_conditions(
        rows, ("delay", "rpy_weight_scale", "thrust_weight_scale"),
        lambda key: (0, key[1], key[2]),  # d=0 at the same weights is the control
    )
    write_csv(sub / "conditions.csv", cond)
    write_csv(sub / "failures.csv", [r for r in rows if not r["success"]])
    specs = [
        ("success rate", cell_success),
        ("maximum commanded thrust over all episodes [N]", cell_field("thrust_cmd_max")),
        ("steps with |roll| or |pitch| command >= 0.49 rad", cell_field("tilt_sat_pct_all",
                                                                       pct=True)),
        ("mean lap time of the successes [s]", cell_field("lap_time_mean_s")),
        ("mean gates passed, all episodes", cell_field("gates_mean_all")),
        ("failure causes", lambda c: c["failure_causes"]),
    ]
    md = []
    for d in ds:
        sel = [c for c in cond if c["delay"] == d]
        md.append(tables_for(
            sel, "rpy_weight_scale", "thrust_weight_scale", [float(r) for r in rpys],
            [float(t) for t in thrusts],
            f"Experiment F, delay d={d} -- T_traj={T:g}, k_end=1.0, level 0, seeds "
            f"{min(seeds)}-{max(seeds)}. R = diag(rpy_scale * [1,1,1], thrust_scale * 50).",
            specs))
    (sub / "tables.md").write_text("\n\n".join(md) + "\n")
    print("\n" + "\n\n".join(md), flush=True)


# --------------------------------------------------------------------------------------------
# Diagnosis G (no simulation)
# --------------------------------------------------------------------------------------------


def plane_crossings(
    local: np.ndarray, max_offset: float = 0.6
) -> list[int]:
    """Samples where the reference crosses the plane of the gate reasonably close to its centre.

    This is what "passing a gate" means: the sign of the coordinate along the gate normal flips.
    Crossings further than `max_offset` from the centre in the gate plane are ignored, so the
    unrelated plane crossings elsewhere on the track do not count as passes.

    The sign change is looked for between consecutive samples that are NOT exactly on the plane:
    a sample sitting exactly at x=0 would otherwise make both of its neighbouring products zero
    and hide the crossing completely.
    """
    x = local[:, 0]
    nonzero = np.flatnonzero(np.sign(x) != 0)
    out = []
    for a, b in zip(nonzero[:-1], nonzero[1:]):
        if np.sign(x[a]) * np.sign(x[b]) >= 0:
            continue
        span = np.arange(a, b + 1)
        j = int(span[np.argmin(np.abs(x[span]))])  # the sample nearest the gate plane
        if in_plane_offset(local[j][None, :])[0] <= max_offset:
            out.append(j)
    return out


def reference_passes(
    ref_pos: np.ndarray, ref_vel: np.ndarray, t: np.ndarray, sequence: Sequence[int],
    gates_pos: np.ndarray, gates_quat: np.ndarray, window_s: float = 1.0,
) -> list[dict[str, Any]]:
    """Geometry of the reference spline at each gate pass.

    Passes are matched in time order, so a gate that appears twice in the sequence is matched to
    two different visits: for each pass, the first plane crossing of that gate after the previous
    pass. If the reference never crosses that gate's plane again, the nearest sample is used
    instead and `found_by` says so. `window_s` is the half-width of the window for the minima.
    """
    out: list[dict[str, Any]] = []
    i0 = 0
    dt = float(t[1] - t[0])
    half = max(int(round(window_s / dt)), 1)
    for p, gate in enumerate(sequence):
        g = int(gate)
        local_all = local_points(ref_pos, gates_pos[g], gates_quat[g])
        d_centre = np.linalg.norm(local_all, axis=1)
        later = [j for j in plane_crossings(local_all) if j >= i0]
        if later:
            i, found_by = later[0], "plane_crossing"
        elif i0 < len(ref_pos):
            i, found_by = i0 + int(np.argmin(d_centre[i0:])), "nearest_sample"
        else:
            out.append({"pass": p + 1, "gate_id": g, "found_by": "not_found"})
            continue
        lo, hi = max(i - half, 0), min(i + half + 1, len(ref_pos))
        local_win = local_all[lo:hi]
        local_i = local_all[i]
        normal = Rot.from_quat(gates_quat[g]).apply([1.0, 0.0, 0.0])
        v = ref_vel[i]
        speed = float(np.linalg.norm(v))
        cos_to_normal = abs(float(np.dot(v / speed, normal))) if speed > 0 else float("nan")
        out.append({
            "pass": p + 1,
            "gate_id": g,
            "found_by": found_by,
            "t_s": float(t[i]),
            "ref_speed_mps": speed,
            "dist_to_centre_m": float(d_centre[i]),
            "in_plane_offset_m": float(in_plane_offset(local_i[None, :])[0]),
            "along_normal_m": float(local_i[0]),
            "clearance_to_frame_m": float(ring_clearance(local_i[None, :])[0]),
            "min_clearance_in_window_m": float(ring_clearance(local_win).min()),
            "min_in_plane_offset_in_window_m": float(in_plane_offset(local_win).min()),
            "approach_angle_to_normal_deg": math.degrees(math.acos(min(cos_to_normal, 1.0)))
            if not math.isnan(cos_to_normal) else float("nan"),
            "inside_opening": bool(in_plane_offset(local_i[None, :])[0] < GATE_HALF_INNER),
        })
        i0 = i + 1
    return out


def required_thrust(accel: np.ndarray, mass: float, g: float = 9.81) -> np.ndarray:
    """Collective thrust the reference demands: |m * (a + g_up)| (N), ignoring drag."""
    return mass * np.linalg.norm(np.atleast_2d(accel) + np.array([0.0, 0.0, g]), axis=1)


def reference_demand(
    ref_vel: np.ndarray, ref_acc: np.ndarray, mass: float, thrust_max: float
) -> dict[str, Any]:
    """Speed / acceleration / thrust the whole reference asks for, regardless of the gates."""
    thr = required_thrust(ref_acc, mass)
    return {
        "max_speed_mps": float(np.linalg.norm(ref_vel, axis=1).max()),
        "max_accel_mps2": float(np.linalg.norm(ref_acc, axis=1).max()),
        "max_required_thrust_N": float(thr.max()),
        "mean_required_thrust_N": float(thr.mean()),
        "thrust_max_N": thrust_max,
        "headroom_at_peak": float(thrust_max / thr.max()),
        "k_end_that_removes_the_headroom": float(thr.max() / thrust_max),
    }


def diag_g(out: Path, cfg: Any, ctrl_cls: Any, ts: Sequence[float] = G_TS, seed: int = 0) -> None:
    """Measure the reference geometry at every gate pass. Read-only: no waypoint is changed."""
    sub = out / "diagG"
    sub.mkdir(parents=True, exist_ok=False)
    env = make_env(cfg)
    obs, info = env.reset(seed=seed)
    gates_pos, gates_quat = gate_poses(env)
    thrust_max = float(np.asarray(env.action_space.high)[THRUST_IDX])
    sequence = [int(g) for g in np.asarray(obs["gate_sequence"]).reshape(-1)]
    rows: list[dict[str, Any]] = []
    demand: list[dict[str, Any]] = []
    for T in ts:
        ctrl = ctrl_cls(obs, info, cfg, T_traj=T)
        ref_pos = np.asarray(ctrl._waypoints_pos, dtype=float)
        ref_vel = np.asarray(ctrl._waypoints_vel, dtype=float)
        t = np.linspace(0.0, T, len(ref_pos))
        ref_acc = np.asarray(ctrl._des_pos_spline.derivative(2)(t), dtype=float)
        mass = float(ctrl.drone_params["mass"])
        thr = required_thrust(ref_acc, mass)
        for r in reference_passes(ref_pos, ref_vel, t, sequence, gates_pos, gates_quat):
            i = int(round(r["t_s"] / float(t[1] - t[0]))) if "t_s" in r else 0
            lo, hi = max(i - 25, 0), min(i + 26, len(thr))
            rows.append({"T_traj": T, **r,
                         "ref_accel_mps2": float(np.linalg.norm(ref_acc[i])),
                         "required_thrust_N": float(thr[i]),
                         "max_required_thrust_in_window_N": float(thr[lo:hi].max())})
        demand.append({"T_traj": T, "mass_kg": mass,
                       **reference_demand(ref_vel, ref_acc, mass, thrust_max)})
    env.close()
    write_csv(sub / "reference_demand.csv", demand)
    write_csv(sub / "reference_passes.csv", rows)
    cols = ["T_traj", "pass", "gate_id", "found_by", "t_s", "ref_speed_mps", "dist_to_centre_m",
            "in_plane_offset_m", "clearance_to_frame_m", "min_clearance_in_window_m",
            "approach_angle_to_normal_deg", "inside_opening", "ref_accel_mps2",
            "required_thrust_N", "max_required_thrust_in_window_N"]
    md = [f"## Diagnosis G -- geometry of the FIXED reference spline at each gate pass\n\n"
          f"Gate opening: half-width {GATE_HALF_INNER} m; frame material out to "
          f"{GATE_HALF_OUTER} m. `in_plane_offset_m` < {GATE_HALF_INNER} goes through the opening; "
          f"`clearance_to_frame_m` is 0 when the reference is inside the frame material. "
          f"`approach_angle_to_normal_deg` is 0 for a reference crossing straight through.\n\n"
          f"Gate sequence (0-based ids, 5 passes): {sequence}\n\n"
          + "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
          + "\n".join("| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows)]
    dcols = ["T_traj", "mass_kg", "max_speed_mps", "max_accel_mps2", "max_required_thrust_N",
             "mean_required_thrust_N", "thrust_max_N", "headroom_at_peak",
             "k_end_that_removes_the_headroom"]
    md.append("### Dynamic demand of the whole reference (drag ignored)\n\n"
              "`k_end_that_removes_the_headroom` is the thrust scale at which the peak demand of "
              "the reference equals the actuator limit.\n\n"
              + "| " + " | ".join(dcols) + " |\n|" + "---|" * len(dcols) + "\n"
              + "\n".join("| " + " | ".join(_fmt(r[c]) for c in dcols) + " |" for r in demand))
    md.append("### Gate poses used (true positions of this episode)\n\n"
              "| gate_id | position | yaw [deg] |\n|---|---|---|\n"
              + "\n".join(
                  f"| {g} | {np.round(gates_pos[g], 3).tolist()} | "
                  f"{math.degrees(Rot.from_quat(gates_quat[g]).as_euler('xyz')[2]):.1f} |"
                  for g in range(len(gates_pos))))
    (sub / "tables.md").write_text("\n\n".join(md) + "\n")
    print("\n" + "\n\n".join(md), flush=True)


# --------------------------------------------------------------------------------------------


def main(exp: str = "G", out_dir: str | None = None, n_seeds: int = N_SEEDS,
         config: str = "att_level0.toml", controller: str | None = None) -> None:
    """Run experiment E and/or F and/or diagnosis G into results/round4_<timestamp>/."""
    exp = exp.upper()
    if not exp or set(exp) - set("EFG"):
        raise ValueError("exp must be a combination of 'E', 'F' and 'G'")
    out = Path(out_dir) if out_dir else RESULTS_DIR / f"round4_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    names = {"E": "expE", "F": "expF", "G": "diagG"}
    for e in exp:
        if (out / names[e]).exists():
            raise FileExistsError(f"{out / names[e]} already exists; refusing to overwrite")
    sys.stdout = Tee(sys.stdout, open(out / f"run_{exp}.log", "x"))
    ctrl_path = controller or str(PROJECT_DIR / "my_mpc.py")
    seeds = list(range(n_seeds))
    (out / f"params_{exp}.json").write_text(json.dumps(dict(
        exp=exp, seeds=f"{seeds[0]}-{seeds[-1]}", config=config,
        controller=Path(ctrl_path).name,
        E=dict(ts=E_TS, ks=E_KS, delay=0, u_weight_scale=1.0, T_end="= T_traj",
               flip_ks=E_FLIP_KS),
        F=dict(T_traj=F_T, delays=F_DELAYS, rpy_scales=F_RPY_SCALES,
               thrust_scales=F_THRUST_SCALES, k_end=1.0),
        G=dict(ts=G_TS, seed=0, gate_half_inner=GATE_HALF_INNER,
               gate_half_outer=GATE_HALF_OUTER),
    ), indent=2))
    print(f"Writing results to {out} (exp {exp}, seeds {seeds[0]}-{seeds[-1]})", flush=True)
    cfg = load_cfg(config)
    ctrl_cls = load_controller_cls(ctrl_path)
    if "G" in exp:
        diag_g(out, cfg, ctrl_cls)
    if "E" in exp:
        exp_e(out, seeds, cfg, ctrl_cls)
    if "F" in exp:
        exp_f(out, seeds, cfg, ctrl_cls)


if __name__ == "__main__":
    fire.Fire(main)
