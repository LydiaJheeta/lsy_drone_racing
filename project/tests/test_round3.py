"""Tests for round3.py (metrics and aggregation; the episode loop is covered by test_common)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from round3 import (
    TILT_SAT,
    cell_success,
    cell_success_on_baseline,
    cmd_metrics,
    pivot_table,
    recording_factory,
    summarize_conditions,
)


class FakeCtrl:
    """Minimal controller: returns a fixed command, records the kwargs it was built with."""

    def __init__(self, obs, info, config, T_traj=18.75, u_weight_scale=1.0):
        self.kwargs = dict(T_traj=T_traj, u_weight_scale=u_weight_scale)
        self.n = 0

    def compute_control(self, obs, info=None):
        self.n += 1
        return np.array([0.1 * self.n, 0.0, 0.0, 0.5])


def test_recording_factory_records_commands_and_forwards_kwargs():
    store = []
    factory = recording_factory(FakeCtrl, store, T_traj=12.0, u_weight_scale=3.0)
    c = factory({}, {}, None)
    assert c.kwargs == {"T_traj": 12.0, "u_weight_scale": 3.0}
    out = [c.compute_control({}) for _ in range(3)]
    assert len(store) == 3
    assert np.allclose(np.asarray(store), np.asarray(out))  # the MPC's own (unwrapped) commands
    store[0][0] = 99.0  # stored copies, not views on the controller's output
    assert out[0][0] == pytest.approx(0.1)


def test_cmd_metrics_counts_tilt_and_thrust_saturation():
    # 4 steps at 50 Hz, k(t) = 1 - 0.5 * t (so k = 1.0, 0.99, 0.98, 0.97), thrust_max = 0.8
    cmds = [
        np.array([0.49, 0.0, 0.0, 0.80]),   # tilt at the threshold -> counted; 0.80 > 0.8*1.0 false
        np.array([0.0, -0.50, 0.0, 0.79]),  # |pitch| over the threshold -> counted; 0.79 > 0.792 f.
        np.array([0.2, 0.3, 0.0, 0.79]),    # no tilt saturation; 0.79 > 0.784 -> counted
        np.array([0.0, 0.0, 0.0, 0.10]),    # neither
    ]
    m = cmd_metrics(cmds, lambda t: 1.0 - 0.5 * t, thrust_max=0.8, freq=50.0)
    assert m["n_cmd_steps"] == 4
    assert m["tilt_sat_steps"] == 2
    assert m["tilt_cmd_max"] == pytest.approx(0.50)
    assert m["thrust_above_eff_steps"] == 1
    assert m["thrust_cmd_max"] == pytest.approx(0.80)
    assert m["thrust_cmd_mean"] == pytest.approx((0.80 + 0.79 + 0.79 + 0.10) / 4)
    assert TILT_SAT == 0.49


def test_cmd_metrics_on_an_empty_episode():
    m = cmd_metrics([], lambda t: 1.0, thrust_max=0.8, freq=50.0)
    assert m["n_cmd_steps"] == 0 and m["tilt_sat_steps"] == 0
    assert math.isnan(m["tilt_cmd_max"]) and math.isnan(m["thrust_cmd_mean"])


def _row(seed, delay, scale, success, time_s=17.0, tilt=10, n=100, over=0, gates=5):
    return {
        "seed": seed, "delay": delay, "u_weight_scale": scale, "success": success,
        "time_s": time_s, "n_gates": gates if success else 2, "group": "success" if success
        else "collision", "min_z_airborne": 0.2, "n_cmd_steps": n, "tilt_sat_steps": tilt,
        "thrust_above_eff_steps": over, "tilt_cmd_max": 0.5, "thrust_cmd_max": 0.8,
    }


def test_summarize_conditions_rates_pooled_percentages_and_baseline_pairing():
    rows = [
        # baseline d=0: seeds 0,1 succeed, seed 2 fails
        _row(0, 0, 1.0, True, tilt=50, n=100, over=5),
        _row(1, 0, 1.0, True, tilt=30, n=100, over=0),
        _row(2, 0, 1.0, False, tilt=20, n=200, over=0),
        # d=1: only seed 2 succeeds -> 1/3 overall, but 0/2 on the seeds the baseline managed
        _row(0, 1, 1.0, False, tilt=10, n=100),
        _row(1, 1, 1.0, False, tilt=10, n=100),
        _row(2, 1, 1.0, True, time_s=19.0, tilt=10, n=100),
    ]
    cond = summarize_conditions(rows, ("delay", "u_weight_scale"), lambda key: (0, key[1]))
    base, delayed = cond[0], cond[1]
    assert (base["delay"], base["u_weight_scale"]) == (0, 1.0)
    assert base["n"] == 3 and base["n_success"] == 2
    assert base["success_rate"] == pytest.approx(2 / 3)
    assert base["success_rate_on_baseline_ok"] == pytest.approx(1.0)  # compared with itself
    assert base["n_baseline_ok_seeds"] == 2
    # pooled over steps, not an average of per-episode shares: (50+30+20)/(100+100+200)
    assert base["tilt_sat_pct_all"] == pytest.approx(25.0)
    assert base["tilt_sat_pct_success"] == pytest.approx(40.0)  # (50+30)/200
    assert base["thrust_above_eff_pct_all"] == pytest.approx(5 / 400 * 100)
    assert base["lap_time_mean_s"] == pytest.approx(17.0)
    assert delayed["success_rate"] == pytest.approx(1 / 3)
    assert delayed["success_on_baseline_ok"] == 0
    assert delayed["success_rate_on_baseline_ok"] == pytest.approx(0.0)
    assert delayed["lap_time_mean_s"] == pytest.approx(19.0)  # successes only
    assert delayed["failure_causes"] == "collision:2"


def test_summarize_conditions_without_a_surviving_baseline():
    rows = [_row(0, 0, 1.0, False), _row(0, 1, 1.0, False)]
    cond = summarize_conditions(rows, ("delay", "u_weight_scale"), lambda key: (0, key[1]))
    assert all(c["n_baseline_ok_seeds"] == 0 for c in cond)
    assert all(math.isnan(c["success_rate_on_baseline_ok"]) for c in cond)
    assert all(math.isnan(c["lap_time_mean_s"]) for c in cond)
    assert cell_success_on_baseline(cond[0]) == "-"


def test_pivot_table_layout_and_missing_cells():
    cond = [
        {"delay": 0, "u_weight_scale": 1.0, "success_rate": 1.0, "n_success": 20, "n": 20},
        {"delay": 1, "u_weight_scale": 1.0, "success_rate": 0.0, "n_success": 0, "n": 20},
    ]
    md = pivot_table(cond, "delay", "u_weight_scale", [0, 1, 2], [1.0, 3.0], cell_success)
    lines = md.splitlines()
    assert lines[0] == "| delay \\ u_weight_scale | 1 | 3 |"
    assert lines[2] == "| 0 | 100% (20/20) | n/a |"
    assert lines[3] == "| 1 | 0% (0/20) | n/a |"
    assert lines[4] == "| 2 | n/a | n/a |"  # a condition that was not run
