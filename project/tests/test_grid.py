"""Tests for grid.py (aggregation, tables, plotting; no simulation)."""

from __future__ import annotations

import numpy as np
import pytest

from common import EpisodeResult
from grid import (episode_metrics, grid_table, in_band, main, plot_episode, summarize_grid,
                  window_amp)
from run_eval import make_row


def _row(T, d, k, seed, ok, t=15.0, gates=None, cause=None, min_air=1.0):
    r = EpisodeResult(seed=seed, success=ok, n_gates=5 if ok else (gates or 1), n_gates_total=5,
                      steps=int(t * 50), time_s=t, cause="success" if ok else (cause or "gate"),
                      min_z=0.01, min_z_airborne=min_air, final_z=1.0)
    return {**make_row("grid", T, f"d{d}_k{k:g}", r), "delay": d, "k_end": k}


def _rows():
    rows = []
    for seed in range(10):
        rows.append(_row(16.0, 0, 1.0, seed, seed < 8))  # nominal: seeds 0-7 succeed
        rows.append(_row(16.0, 2, 1.0, seed, seed in (0, 1, 9), cause="gate", min_air=0.05 * seed))
        rows.append(_row(16.0, 1, 0.7, seed, seed in (2, 3, 4, 5)))  # 40%
    return rows


def test_in_band():
    assert in_band(0.3) and in_band(0.7) and in_band(0.5)
    assert not in_band(0.0) and not in_band(1.0) and not in_band(0.75)


def test_summarize_grid_pairs_against_nominal_seeds():
    s = {(c["delay"], c["k_end"]): c for c in summarize_grid(_rows())}
    nom, d2, d1 = s[(0, 1.0)], s[(2, 1.0)], s[(1, 0.7)]
    assert nom["n_success"] == 8 and nom["n_nominal_ok_seeds"] == 8
    assert nom["success_rate_on_nominal_ok"] == 1.0
    assert d2["n_success"] == 3 and d2["success_on_nominal_ok"] == 2  # seed 9 not nominal-ok
    assert d2["success_rate_on_nominal_ok"] == pytest.approx(2 / 8)
    assert d2["failure_causes"] == "collision:7" and d2["min_z_airborne_min"] == 0.0
    assert d1["in_band_30_70"] and not nom["in_band_30_70"]
    assert d1["lap_time_mean_s"] == pytest.approx(15.0)


def test_grid_table_layout_and_band_marker():
    t = grid_table(summarize_grid(_rows()), 16.0, "success_rate", ks=(1.0, 0.7), ds=(0, 1, 2))
    lines = t.splitlines()
    assert lines[0].startswith("| d \\ k_end | 1 | 0.7 |") and len(lines) == 5
    assert "40% (4/10) *" in lines[3]  # d=1, k=0.7 is in the band
    assert "80% (8/10)" in lines[2] and "*" not in lines[2]
    assert "n/a" in lines[3]  # d=1, k=1.0 was not run


def test_window_amp_and_metrics():
    t = np.arange(0, 3, 0.02)
    st = np.zeros((len(t), 13))
    st[:, 6] = 1.0  # qw
    st[:, 2] = np.where(t > 1.0, 1.0, 0.01)  # takes off at t=1
    assert window_amp(t, np.sin(t), 0, 1) == pytest.approx(np.sin(1.0), abs=0.02)
    assert np.isnan(window_amp(t, t, 10, 11))
    res = EpisodeResult(seed=0, success=False, n_gates=0, n_gates_total=5, steps=len(t),
                        time_s=3.0, cause="floor", min_z_airborne=0.0)
    ep = dict(t=t, state=st, action=np.zeros((len(t), 4)))
    m = episode_metrics(ep, res)
    assert m["t_takeoff_s"] == pytest.approx(1.02) and m["z_max"] == 1.0
    assert m["tilt_peak_overall"] == 0.0 and m["cause"] == "floor"


def test_plot_episode_writes_a_png(tmp_path):
    t = np.arange(0, 2, 0.02)
    st = np.zeros((len(t), 13))
    st[:, 6] = 1.0
    ep = dict(t=t, state=st, action=np.zeros((len(t), 4)))
    path = tmp_path / "p.png"
    plot_episode(ep, np.zeros((len(t), 4)), "title", path, hover=0.4258)
    assert path.stat().st_size > 1000


def test_main_refuses_to_overwrite_an_existing_experiment_folder(tmp_path):
    (tmp_path / "expA").mkdir()
    with pytest.raises(FileExistsError):
        main(exp="A", out_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == [tmp_path / "expA"]  # nothing else created or removed
