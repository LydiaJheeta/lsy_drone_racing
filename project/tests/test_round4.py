"""Tests for round4.py: gate geometry, pass matching, flip table (no simulation involved)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from round4 import (
    GATE_HALF_INNER,
    flip_table,
    in_plane_offset,
    local_points,
    pass_clearances,
    plane_crossings,
    reference_passes,
    required_thrust,
    ring_clearance,
)

# one gate at the origin facing +x (identity rotation), one shifted and yawed by 90 deg
GATES_POS = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
GATES_QUAT = np.array([
    Rot.identity().as_quat(),
    Rot.from_euler("z", 90, degrees=True).as_quat(),
])


def test_local_points_and_in_plane_offset_use_the_gate_frame():
    # 0.1 m in front of gate 0 and 0.05 m above its centre
    local = local_points(np.array([0.1, 0.0, 1.05]), GATES_POS[0], GATES_QUAT[0])
    assert np.allclose(local[0], [0.1, 0.0, 0.05])
    assert in_plane_offset(local)[0] == pytest.approx(0.05)
    # gate 1 is yawed 90 deg, so its normal is +y in world coordinates
    local = local_points(np.array([2.0, 0.3, 1.0]), GATES_POS[1], GATES_QUAT[1])
    assert local[0][0] == pytest.approx(0.3)  # along the gate normal
    assert in_plane_offset(local)[0] == pytest.approx(0.0)


def test_ring_clearance_is_zero_inside_the_frame_and_grows_outside():
    pts = np.array([
        [0.0, 0.0, 0.0],  # dead centre of the opening -> 0.2 from the frame
        [0.0, 0.25, 0.0],  # inside the frame material (0.2 <= 0.25 <= 0.36)
        [0.0, 0.5, 0.0],  # beyond the frame
        [0.3, 0.0, 0.0],  # 0.3 m in front of the centre
    ])
    c = ring_clearance(pts)
    assert c[0] == pytest.approx(GATE_HALF_INNER)
    assert c[1] == pytest.approx(0.0)  # touching
    assert c[2] == pytest.approx(0.5 - 0.36)
    assert c[3] == pytest.approx(math.hypot(0.3, GATE_HALF_INNER))


def test_plane_crossings_finds_fly_throughs_and_ignores_distant_ones():
    # straight line along x through the centre of gate 0, then far above it
    near = np.stack([np.linspace(-0.5, 0.5, 11), np.zeros(11), np.full(11, 1.0)], axis=1)
    local = local_points(near, GATES_POS[0], GATES_QUAT[0])
    assert plane_crossings(local) == [5]  # the sample at x = 0
    far = np.stack([np.linspace(-0.5, 0.5, 11), np.zeros(11), np.full(11, 3.0)], axis=1)
    assert plane_crossings(local_points(far, GATES_POS[0], GATES_QUAT[0])) == []


def test_reference_passes_matches_a_repeated_gate_to_two_different_visits():
    """Sequence [0, 1, 0]: the reference must be matched to the first and the second visit.

    Two gates facing +x at x=0 and x=1.2; the path sweeps x out to 2.0 and back through both
    centres, so gate 0's plane is crossed on the way out and again on the way back.
    """
    gates_pos = np.array([[0.0, 0.0, 1.0], [1.2, 0.0, 1.0]])
    gates_quat = np.array([Rot.identity().as_quat(), Rot.identity().as_quat()])
    xs = np.concatenate([np.linspace(-0.5, 2.0, 40), np.linspace(2.0, -0.5, 40)])
    pos = np.stack([xs, np.zeros(80), np.full(80, 1.0)], axis=1)
    t = np.linspace(0.0, 8.0, 80)
    out = reference_passes(pos, np.gradient(pos, t, axis=0), t, [0, 1, 0], gates_pos, gates_quat)
    assert [r["pass"] for r in out] == [1, 2, 3]
    assert [r["gate_id"] for r in out] == [0, 1, 0]
    assert all(r["found_by"] == "plane_crossing" for r in out), out
    assert out[0]["t_s"] < out[1]["t_s"] < out[2]["t_s"]  # strictly increasing in time
    assert all(r["inside_opening"] for r in out)


def test_approach_angle_is_zero_head_on_and_90_degrees_edge_on():
    """A straight run along +x meets gate 0 head-on and the yawed gate 1 edge-on."""
    pos = np.stack([np.linspace(-0.5, 0.5, 21), np.zeros(21), np.full(21, 1.0)], axis=1)
    t = np.linspace(0.0, 2.0, 21)
    vel = np.gradient(pos, t, axis=0)
    head_on = reference_passes(pos, vel, t, [0], GATES_POS, GATES_QUAT)[0]
    assert head_on["found_by"] == "plane_crossing"  # a sample sits exactly on the plane
    assert head_on["t_s"] == pytest.approx(1.0)
    assert head_on["approach_angle_to_normal_deg"] == pytest.approx(0.0, abs=1.0)
    edge_on = reference_passes(pos, vel, t, [1], GATES_POS, GATES_QUAT)[0]
    assert edge_on["found_by"] == "nearest_sample"  # gate 1's plane is never crossed
    assert edge_on["approach_angle_to_normal_deg"] == pytest.approx(90.0, abs=1.0)


def test_reference_passes_falls_back_to_the_nearest_sample():
    pos = np.stack([np.linspace(-1.0, -0.5, 20), np.zeros(20), np.full(20, 1.0)], axis=1)
    t = np.linspace(0.0, 2.0, 20)
    out = reference_passes(pos, np.gradient(pos, t, axis=0), t, [0], GATES_POS, GATES_QUAT)
    assert out[0]["found_by"] == "nearest_sample"  # the plane is never crossed
    assert out[0]["dist_to_centre_m"] == pytest.approx(0.5, abs=1e-6)


def test_pass_clearances_uses_the_target_gate_of_each_step():
    # 4 steps approaching gate 0 (target 0), then 4 steps near gate 1 (target 1)
    # gate 1 is yawed 90 deg, so approaching its centre means moving along world y
    pos = np.array([[-0.4, 0, 1], [-0.2, 0, 1], [0.0, 0, 1], [0.1, 0, 1],
                    [2.0, -0.3, 1], [2.0, -0.1, 1], [2.0, 0.0, 1], [2.0, 0.1, 1]], dtype=float)
    ep = {
        "state": np.hstack([pos, np.zeros((8, 10))]),
        "next_gate_idx": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "next_gate_id": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    }
    out = pass_clearances(ep, GATES_POS, GATES_QUAT)
    assert out["ring_pass0"] == pytest.approx(GATE_HALF_INNER)  # through the centre of gate 0
    assert out["ring_pass1"] == pytest.approx(GATE_HALF_INNER)
    assert math.isnan(out["ring_pass2"])  # that pass never happened
    assert out["ring_min_overall"] == pytest.approx(GATE_HALF_INNER)


def test_required_thrust_is_hover_at_zero_acceleration():
    m = 0.0434
    assert required_thrust(np.zeros(3), m)[0] == pytest.approx(m * 9.81)
    # 9.81 m/s^2 upwards needs twice the hover thrust
    assert required_thrust(np.array([0.0, 0.0, 9.81]), m)[0] == pytest.approx(2 * m * 9.81)


def _erow(T, k, seed, success, n_gates=5, time_s=11.0, detail="", cause="success", **extra):
    return {"T_traj": T, "k_end": k, "seed": seed, "success": success, "n_gates": n_gates,
            "time_s": time_s, "detail": detail, "cause": cause,
            "ring_min_overall": 0.09, **extra}


def test_flip_table_lists_only_baseline_successes_that_fail_at_a_tiny_drop():
    rows = [
        _erow(12.0, 1.0, 3, True, ring_pass1=0.085, ring_pass3=0.2),
        _erow(12.0, 0.99, 3, False, n_gates=1, time_s=4.2, cause="gate",
              detail="gate_id=1 dist_to_frame=0.081 (heuristic)"),
        # a seed that fails in the baseline too is not a flip
        _erow(12.0, 1.0, 5, False, n_gates=1, cause="gate", detail="gate_id=1"),
        _erow(12.0, 0.99, 5, False, n_gates=1, cause="gate", detail="gate_id=1"),
        # a drop that is not in E_FLIP_KS is ignored
        _erow(12.0, 0.8, 3, False, n_gates=1, cause="gate", detail="gate_id=1"),
    ]
    flips = flip_table(rows)
    assert len(flips) == 1
    f = flips[0]
    assert (f["seed"], f["k_end"], f["T_traj"]) == (3, 0.99, 12.0)
    assert f["failed_on_pass"] == 2  # n_gates=1 -> died on the second pass (1-based)
    assert f["fail_gate_id"] == "1"
    assert f["fail_dist_to_frame"] == pytest.approx(0.081)
    assert f["fail_time_s"] == pytest.approx(4.2)
    # the clearance of the SAME seed on the SAME pass in the k_end=1.0 run
    assert f["nominal_ring_same_pass"] == pytest.approx(0.085)
    assert f["nominal_ring_min_overall"] == pytest.approx(0.09)


def test_flip_table_is_empty_when_nothing_flips():
    rows = [_erow(12.0, 1.0, 0, True), _erow(12.0, 0.99, 0, True)]
    assert flip_table(rows) == []
