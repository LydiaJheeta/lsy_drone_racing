"""PPO with a gate-progress reward on `VecDroneRaceEnv` (phase 1).

* `env.py` - `GateProgressRaceEnv` (subclass of `VecDroneRaceEnv`), observation, action mapping
* `vec_perturb.py` - per-world command delay + thrust degradation k(t) (off in phase 1)
* `ppo.py` - the PPO loop (core copied from `train_rl.py`), CSV metrics + checkpoints
* `eval_policy.py` - evaluation through the same single-env pipeline as the MPC evals
"""
