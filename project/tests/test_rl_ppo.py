"""Tests for rl.ppo: argument derivation, agent, episode bookkeeping, end-to-end loop."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.env import ACT_DIM, OBS_DIM
from rl.ppo import METRICS, Agent, Args, EpisodeTracker, _save, load_agent, machine_load, train


def test_args_defaults_and_derived_sizes():
    """train_rl.py defaults, except num_envs=256 and num_steps=32 (decision 1 of the brief)."""
    args = Args.create(total_timesteps=2_000_000)
    assert (args.num_envs, args.num_steps) == (256, 32)
    assert args.batch_size == 8192 and args.minibatch_size == 1024
    assert args.num_iterations == 2_000_000 // 8192 == 244
    # unchanged train_rl values
    assert (args.learning_rate, args.gamma, args.gae_lambda) == (1.5e-3, 0.94, 0.97)
    assert (args.clip_coef, args.ent_coef, args.vf_coef) == (0.26, 0.007, 0.7)
    assert (args.num_minibatches, args.update_epochs, args.max_grad_norm) == (8, 10, 1.5)


def test_agent_shapes_and_deterministic_action_is_the_mean():
    agent = Agent(OBS_DIM, ACT_DIM)
    x = torch.zeros((5, OBS_DIM))
    action, logprob, entropy, value = agent.get_action_and_value(x)
    assert action.shape == (5, ACT_DIM) and logprob.shape == (5,) and value.shape == (5, 1)
    assert entropy.shape == (5,)
    det, *_ = agent.get_action_and_value(x, deterministic=True)
    assert torch.allclose(det, agent.actor_mean(x))
    assert agent.actor_logstd.shape == (1, ACT_DIM)
    assert torch.allclose(agent.actor_logstd.detach(), torch.tensor([[-1.0, -1.0, 1.0]]))
    # the tanh head keeps the mean inside the action box
    big = agent.actor_mean(torch.randn((64, OBS_DIM)) * 10)
    assert big.abs().max() <= 1.0


def test_checkpoint_roundtrip(tmp_path: Path):
    agent = Agent(OBS_DIM, ACT_DIM)
    with torch.no_grad():
        agent.actor_logstd.fill_(-0.3)
    _save(agent, Args.create(), tmp_path / "c.pt", global_step=123)
    loaded = load_agent(tmp_path / "c.pt")
    x = torch.randn((4, OBS_DIM))
    a1, *_ = agent.get_action_and_value(x, deterministic=True)
    a2, *_ = loaded.get_action_and_value(x, deterministic=True)
    assert torch.allclose(a1, a2)
    assert not loaded.training


def test_episode_tracker_skips_dummy_steps_and_classifies_outcomes():
    t = EpisodeTracker(num_envs=2, n_passes=5)
    f, T = np.zeros(2, dtype=bool), np.ones(2, dtype=bool)
    # two normal steps
    t.update(np.array([1.0, 2.0]), T, f, f, np.array([0, 0]))
    t.update(np.array([1.0, 2.0]), T, f, f, np.array([1, 0]))
    # world 0 finishes the lap, world 1 crashes
    t.update(
        np.array([10.0, -10.0]), T, np.array([True, True]), f, np.array([5, 2])
    )
    eps = t.drain()
    assert len(eps) == 2
    done0 = next(e for e in eps if e["gates"] == 5)
    done1 = next(e for e in eps if e["gates"] == 2)
    assert done0["success"] and not done0["terminated_fail"]
    assert done0["ret"] == pytest.approx(12.0) and done0["len"] == 3
    assert done1["terminated_fail"] and not done1["success"]
    assert done1["ret"] == pytest.approx(-6.0)
    assert t.drain() == []
    # the dummy step after a terminal step must not be counted at all
    t.update(np.array([0.0, 0.0]), f, f, f, np.array([0, 0]))
    assert t.len.tolist() == [0, 0] and t.ret.tolist() == [0.0, 0.0]
    # truncation without termination is a timeout
    t.update(np.array([1.0, 1.0]), T, f, T, np.array([3, 3]))
    eps = t.drain()
    assert all(e["timeout"] and not e["success"] for e in eps)


def test_machine_load_reports_no_absolute_paths():
    """`meta.json` is tracked, so the load record must not leak paths or user names."""
    load = machine_load()
    assert set(load) == {"loadavg", "n_cpus", "torch_threads", "n_python_procs", "other_jobs"}
    assert all("/" not in tag for tag in load["other_jobs"])
    assert len(load["loadavg"]) == 3


def test_two_update_training_run(tmp_path: Path):
    """End-to-end: 2 tiny updates must write a complete progress.csv and a loadable checkpoint."""
    out = train(
        out_dir=str(tmp_path / "run"),
        ckpt_every=0,
        num_envs=2,
        num_steps=4,
        total_timesteps=16,
        num_minibatches=2,
        update_epochs=2,
        seed=0,
    )
    rows = list(csv.DictReader((out / "progress.csv").open()))
    assert len(rows) == 2, f"expected 2 updates, got {len(rows)}"
    assert list(rows[0]) == METRICS
    assert int(rows[-1]["global_step"]) == 16
    assert float(rows[0]["sps_loop"]) > 0 and float(rows[0]["sps_env"]) > 0
    for key in ("value_loss", "policy_loss", "entropy", "approx_kl"):
        assert np.isfinite(float(rows[0][key])), key
    agent = load_agent(out / "ckpt_final.pt")
    assert agent.obs_dim == OBS_DIM
    assert (out / "meta.json").exists()
