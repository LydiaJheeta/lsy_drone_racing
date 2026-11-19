"""Tests for rl.eval_policy: the controller adapter and the CSV schema."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.env import ACT_DIM, attitude_limits, load_rl_config
from rl.eval_policy import PolicyController, evaluate
from rl.ppo import Agent, Args, _save
from run_eval import FIELDS

CFG = load_rl_config("att_level0.toml")
LOW, HIGH = attitude_limits(CFG)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An untrained checkpoint (enough to exercise the plumbing)."""
    path = tmp_path_factory.mktemp("ckpt") / "c.pt"
    torch.manual_seed(0)
    _save(Agent(), Args.create(), path, global_step=0)
    return path


def _cfg_with(ckpt: Path, deterministic: bool = True):
    cfg = load_rl_config("att_level0.toml")
    cfg.rl = {"ckpt": str(ckpt), "deterministic": deterministic}
    return cfg


def _obs() -> dict[str, np.ndarray]:
    return {
        "pos": np.array([-1.5, 0.75, 0.01], dtype=np.float32),
        "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "vel": np.zeros(3, dtype=np.float32),
        "ang_vel": np.zeros(3, dtype=np.float32),
        "n_gates_passed": np.array(0),
        "gate_sequence": np.array([0, 1, 2, 3, 1]),
        "gate_sequence_direction": np.array([1, 1, 1, 1, 1], dtype=np.int32),
        "gates_pos": np.array(
            [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]],
            dtype=np.float32,
        ),
        "gates_quat": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1)),
        "gates_visited": np.zeros(4, dtype=bool),
        "obstacles_pos": np.array(
            [[0.0, 0.75, 1.55], [1.0, 0.25, 1.55], [-1.5, -0.25, 1.55], [-0.5, -0.75, 1.55]],
            dtype=np.float32,
        ),
        "obstacles_visited": np.zeros(4, dtype=bool),
    }


def test_step_callback_never_finishes_the_episode(ckpt: Path):
    """Regression: `Controller.step_callback` returns True by default (`controller.py:95`).

    `common.run_episode` stops on a truthy return, so without the override every evaluation
    episode ended after a single step with cause `trajectory_end`.
    """
    c = PolicyController(_obs(), {}, _cfg_with(ckpt))
    assert c.step_callback(np.zeros(4), _obs(), 0.0, False, False, {}) is False


def test_command_is_inside_the_attitude_box_and_deterministic(ckpt: Path):
    c = PolicyController(_obs(), {}, _cfg_with(ckpt))
    cmd1 = c.compute_control(_obs())
    c.episode_reset()
    cmd2 = c.compute_control(_obs())
    assert cmd1.shape == (4,)
    assert np.allclose(cmd1, cmd2), "deterministic policy must repeat its command"
    assert np.all(cmd1 >= LOW) and np.all(cmd1 <= HIGH)
    assert cmd1[2] == 0.0, "yaw must be zero"


def test_last_action_memory_is_used_and_reset(ckpt: Path):
    c = PolicyController(_obs(), {}, _cfg_with(ckpt))
    assert np.allclose(c._last_action, 0.0)
    c.compute_control(_obs())
    assert c._last_action.shape == (ACT_DIM,)
    c.episode_reset()
    assert np.allclose(c._last_action, 0.0)


def test_evaluate_writes_the_run_eval_schema(ckpt: Path, tmp_path: Path):
    out = evaluate(ckpt=str(ckpt), seeds=1, out_dir=str(tmp_path / "ev"))
    rows = list(csv.DictReader((out / "episodes.csv").open()))
    assert len(rows) == 1
    assert list(rows[0]) == FIELDS
    assert rows[0]["condition"] == "nominal" and rows[0]["stage"] == "ppo"
    assert rows[0]["group"] in {"success", "collision", "fall", "timeout", "out_of_bounds"}
    assert (out / "summary.csv").exists()
