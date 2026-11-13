"""Tests for run_eval.py (aggregation logic is pure; no simulation needed)."""

from __future__ import annotations

import math

import pytest

from common import EpisodeResult
from run_eval import choose_T, failure_group, make_row, paired_table, run_eval, summarize


def _res(seed=0, ok=True, gates=5, t=17.0, cause="success", min_air=1.0, final_z=1.0):
    return EpisodeResult(seed=seed, success=ok, n_gates=gates, n_gates_total=5, steps=int(t * 50),
                         time_s=t, cause=cause, min_z=0.01, min_z_airborne=min_air, final_z=final_z)


def test_failure_group():
    assert failure_group(_res()) == "success"
    assert failure_group(_res(ok=False, cause="gate")) == "collision"
    assert failure_group(_res(ok=False, cause="obstacle")) == "collision"
    assert failure_group(_res(ok=False, cause="floor")) == "fall"
    assert failure_group(_res(ok=False, cause="out_of_bounds")) == "out_of_bounds"
    # The env ignores floor contact: a crash to the floor ends as a timeout -> reclassified
    assert failure_group(_res(ok=False, cause="timeout", final_z=0.02)) == "fall"
    assert failure_group(_res(ok=False, cause="trajectory_end", min_air=0.03)) == "fall"
    assert failure_group(_res(ok=False, cause="timeout", final_z=1.1)) == "timeout"


def _rows():
    rows = []
    for T in (18.75, 10.0):
        for cond in ("nominal", "wrapped"):
            for seed in range(10):
                # nominal 18.75: 10/10, nominal 10: 6/10, wrapped 18.75: 8/10, wrapped 10: 0/10
                ok = {("nominal", 18.75): True, ("nominal", 10.0): seed < 6,
                      ("wrapped", 18.75): seed < 8, ("wrapped", 10.0): False}[(cond, T)]
                r = _res(seed, ok, 5 if ok else 2, 17.0 + seed * 0.1 if ok else 9.0,
                         "success" if ok else "gate")
                rows.append(make_row("sweep", T, cond, r))
    return rows


def test_summarize_lap_time_uses_successes_only():
    s = {(x["T_traj"], x["condition"]): x for x in summarize(_rows())}
    n18 = s[(18.75, "wrapped")]
    assert n18["n_success"] == 8 and n18["success_rate"] == 0.8
    assert n18["lap_time_mean_s"] == pytest.approx(17.0 + 0.1 * 3.5)  # seeds 0-7 only
    assert n18["gates_mean_failed"] == 2 and n18["failure_causes"] == "collision:2"
    assert math.isnan(s[(10.0, "wrapped")]["lap_time_mean_s"])  # no successes


def test_choose_T_picks_smallest_meeting_threshold():
    s = summarize(_rows())
    assert choose_T(s, 0.9) == 18.75  # nominal 10 s is only 60%
    assert choose_T(s, 0.6) == 10.0
    assert choose_T([x for x in s if x["condition"] == "wrapped"], 0.9) is None


def test_paired_table_counts_same_seed_outcomes():
    p = paired_table(_rows(), 18.75)
    assert p == {**p, "n_pairs": 10, "both_success": 8, "nominal_only": 2, "wrapped_only": 0,
                 "both_fail": 0}
    assert p["lap_diff_mean_s"] == pytest.approx(0.0)  # identical lap times by construction


def test_refuses_to_reuse_an_existing_output_folder(tmp_path):
    with pytest.raises(FileExistsError):
        run_eval(out_dir=str(tmp_path))
