# project/ — attitude-MPC robustness experiments

Everything here lives outside the upstream repo code; nothing in `lsy_drone_racing/`, `scripts/`,
`config/` or `pixi.lock` is modified.

| File | Purpose |
|---|---|
| `common.py` | env creation (mirrors `scripts/sim.py:60-71`), seeded episode runner, failure diagnosis, `Tee` |
| `diagnose.py` | Step 0: N seeded runs per level, cause of every failure, reproducibility re-runs |
| `logger.py` | `EpisodeLogger`: per-step `.npz` (time, state, **applied** command, next gate) for SINDy |
| `wrapper.py` | `ThrustDegradeDelay`: thrust scaling k(t) + command delay |
| `my_mpc.py` | copy of `attitude_mpc.py` with a `T_traj` argument (default 18.75 s) |
| `run_eval.py` | Step 4: T_traj sweep, nominal vs. wrapped, paired seeds, tables + CSV |
| `round3.py` | Experiment C (delay x cost-weight scale) and D (T_traj x k_end), tables + CSV |
| `round4.py` | Experiment E (noise check), F (rpy vs thrust weights), diagnosis G (reference) |
| `bench_rl_env.py` | steps/s benchmark of the vectorized envs (`VecDroneRaceEnv`, `train_rl`'s stack, the PPO stack) on CPU with random actions |
| `rl/env.py` | `GateProgressRaceEnv`: subclass of `VecDroneRaceEnv` with the gate-progress reward, the 41-float observation and the 3-value action |
| `rl/vec_perturb.py` | `VecThrustDegradeDelay`: per-world command delay + thrust k(t), same semantics as `wrapper.py` |
| `rl/ppo.py` | PPO (core copied from `train_rl.py:557-757`), CSV metrics + `.pt` checkpoints, no wandb |
| `rl/eval_policy.py` | evaluates a checkpoint through the **same** single-env pipeline as the MPC evals |
| `rl/plot_progress.py` | training curves PNG from a run's `progress.csv` |
| `HANDOFF.md` | the brief this round of work was given (verbatim) |
| `config/att_level{0,1}.toml` | copies of `config/level{0,1}.toml` with `control_mode="attitude"`, `render=false` |
| `tests/` | one pytest module per module above |
| `results/` | every run writes to a NEW timestamped folder (`step0_<ts>/`, `eval_<ts>/`, ...) |

## Running

Run from `project/` — acados writes `c_generated_code/` into the working directory (git-ignored).

```bash
cd project
pixi run python diagnose.py --levels="[0,1]" --n_runs=20 --seed0=0     # Step 0
pixi run python run_eval.py                                            # Step 4 (sweep + final)
pixi run python grid.py --exp=A                                        # experiments A, B
pixi run python round3.py --exp=CD                                     # experiments C, D
pixi run python round4.py --exp=GEF                                    # diagnosis G, exp. E, F
pixi run python bench_rl_env.py --pipelines="[race_state,race_attitude]"   # env throughput
```

`--exp=CD` runs both round-3 experiments into one `results/round3_<ts>/` folder; `--exp=C` and
`--exp=D --out_dir=results/round3_<ts>` run them separately into the same folder; `round4.py`
takes any combination of `E`, `F` and `G` the same way and always runs G, then E, then F.

`bench_rl_env.py --pipelines="[train_rl]"` measures the env stack of
`lsy_drone_racing/control/train_rl.py` and needs the `tests` pixi env (`pixi run -e tests --frozen
python ...`), because torch/wandb are only in feature `rl`. See
`results/rlbench_20261113_151905/FINDINGS.md` for the survey of the RL pipeline, the measured
steps/s, and the two reasons `train_rl.py` does not run against crazyflow 0.4.0 as committed.

Tests (never install pytest in the default env; use the `tests` env, locked):

```bash
cd ~/repos/lsy_drone_racing
env -u PYTHONPATH pixi run -e tests --frozen pytest project/tests -q
```

`env -u PYTHONPATH` is needed on machines where a ROS 2 Humble setup script exports `PYTHONPATH`:
pytest otherwise auto-loads ROS's `launch_testing` plugin and crashes on `import lark`.

## Conventions and decisions

* **Attitude command layout is `[roll, pitch, yaw, thrust]`** — thrust is the **last** element
  (index 3) and is the *total* collective thrust in Newtons (4 x per-motor). Evidence:
  `crazyflow/control/mellinger/control.py:677-679` (`cmd[..., :3]`, `cmd[..., 3]`),
  `crazyflow/sim/sim.py:756` (clip of `cmd[..., -1]`), `race_core.py:246-249` (action-space bounds).
  The docstring of `Controller.compute_control` (`[thrust, roll, pitch, yaw]`) is wrong.
  Limits (cf21B_500): thrust in [0.0854, 0.8] N, angles in [-pi/2, pi/2] rad.
* **Success** = `n_gates_passed == 5`. The literal "and not terminated" cannot hold: the env disables
  the drone (-> `terminated=True`) as soon as all gates are passed (`race_core.py:861`).
* **Lap time = `steps / 50`.** `scripts/sim.py` reports `(steps - 1) / 50`, so its numbers are
  **0.02 s smaller** than ours for the same episode.
* **Seeds** are passed as `env.reset(seed=...)` (JAX key `jax.random.key(seed)`). The same seeds are
  used in every condition (paired comparison). Re-running a seed reproduces the episode exactly.
* **Failure causes.** `terminated` is replayed from the pre-step state with the command that was
  *applied* (after any wrapper) to recover the impact position (the env warps disabled drones to
  z=-1): `floor` (z below limit), `out_of_bounds`, else `gate`/`obstacle` (nearest object; the
  gate-vs-obstacle attribution is a geometric heuristic, the contact itself is the env's own check).
  The env **ignores floor contact** (`race_core.py:836-837`), so a crash to the floor never
  terminates: it ends as `timeout` / `trajectory_end`. We therefore log `min_z`, `min_z_airborne`
  (min z after first exceeding 0.15 m) and `final_z`, and report `fall` when a non-terminated
  failure ends below 5 cm or dips below 5 cm after take-off.
* **`T_end` is a REQUIRED keyword-only argument of the wrapper** (round 4). There is no default,
  so a wrong value cannot be passed silently and a misplaced positional argument cannot land in it
  either: both are a `TypeError`.
* **`T_end` is the `T_traj` of the experiment, not a constant.** k(t) falls linearly from 1.0 to
  `k_end` over `T_end` seconds of absolute time, so a fixed `T_end` (the old default of 18 s) meant
  a fast lap (small `T_traj`) ended before k had reached `k_end` and the condition label was a lie.
  `wrapper.set_params(T_end=...)` now exists, `run_eval.py` sets `T_end = T_traj` of each block
  unless `--T_end` is given, and `grid.py`/`round3.py` set it per block. Consequence: the earlier
  `grid_20261113_021212/expB` numbers at T_traj=16 were run with T_end=18 s, i.e. they never
  reached the stated `k_end`; experiment D supersedes them.
* **Cost weights: `my_mpc.py` takes three scale factors** (all default 1.0 = the original
  tuning). The LINEAR_LS cost penalises the commands only through the input block
  `R = u_weight_scale * diag(rpy_weight_scale * [1,1,1], thrust_weight_scale * 50)`
  (roll/pitch/yaw towards 0, thrust towards hover); there is **no** rate-of-change (du) term, so
  there is nothing else to scale. `u_weight_scale` keeps the original 1:1:1:50 ratio (experiment
  C); `rpy_weight_scale` and `thrust_weight_scale` break it so the two can be told apart
  (experiment F). All three are multiplicative.
* **acados gotcha: the generated solver library is dlopen-cached per path.** Every solver in a
  process is loaded from `c_generated_code/lib..._basic_example_mpc.so`; once that file has been
  opened, rebuilding it does not affect a solver created later in the same process, which silently
  keeps the FIRST build's cost matrix. Measured: before the fix, `u_weight_scale=30` reproduced the
  `u_weight_scale=1` trajectory step for step. `create_ocp_solver` therefore pushes `W` into each
  solver instance with `cost_set` after creating it. This never mattered before, because nothing
  but `T_traj` (a Python-side reference, not part of the OCP) used to change between episodes.
* **Gate geometry** (`common._ring_distance`, mirroring the env's own contact check): the opening
  is a square of half-width 0.2 m and the frame material reaches out to half-width 0.36 m, so an
  in-plane offset below 0.2 m goes through the opening. The `ring`/`clearance` numbers of
  `round4.py` are the distance of a point to that frame region: 0 means touching it. Gate ids in
  failure details and in these tables are **0-based** indices into the env's gate arrays.
* **Wrapper order of operations** for step n (t = n/50 s): delay -> thrust *= k(t) -> clip to the
  env's action bounds. While the delay buffer fills (n < d) the *first* command is repeated.
  k(t) falls linearly 1.0 -> `k_end` over `T_end` seconds of **absolute** time and is constant
  afterwards, so for fast laps (short `T_traj`) the drone only sees part of the drop.
* **Logger** sits *inside* the wrapper (directly on the base env) so it stores what the env received.
  `state[1:]` + `final_state` give x_{t+1}. Episodes that end by success (no termination yet) are
  flushed on the next `reset()`/`close()`.
* `my_mpc.py` rebuilds the acados solver every episode (as the original does). Reusing a solver would
  leak warm-start state between episodes and break paired reproducibility.

## PPO (phase 1)

Reinforcement-learning side of the comparison, in `rl/`. It shares nothing with
`lsy_drone_racing/control/train_rl.py` (that file trains a *trajectory-following* env and does not
run against crazyflow 0.4.0 anyway, see `results/rlbench_20261113_151905/FINDINGS.md`); only its
PPO core was copied.

Decisions fixed by the brief:

1. **Own PPO copy.** `rl/ppo.py` is `train_rl.py:557-757` (rollout, GAE, clipped surrogate, clipped
   value loss) with CSV logging and `.pt` checkpoints instead of wandb. `Args` keeps the
   `train_rl.py` hyperparameters, except `num_envs=256` and `num_steps=32`
   (-> `batch_size=8192`, `minibatch_size=1024`).
2. **Subclass, not a patch.** `rl/env.py:GateProgressRaceEnv` subclasses `VecDroneRaceEnv` and
   overrides only `reset`/`step`/the spaces. `race_core.py` is untouched; the subclass was enough
   because everything the reward needs (true gate poses, the pre-step drone position, the autoreset
   mask) is readable from `self.data` *before* `super().step()` runs.
3. **Perturbation on the command.** `rl/vec_perturb.py` scales the commanded thrust and delays the
   command, exactly like `wrapper.py`; the plant is never changed. It is **not** in the phase-1
   training stack (`perturb=None`).

Action (3 values, tanh): `[-1,1]^3 -> [roll, pitch, yaw=0, thrust]`, `roll/pitch = 0.5 * a` rad
(the MPC's limits), thrust linear into `[0.0854, 0.8] N`. Observation: 41 floats, see the
`rl/env.py` docstring for the exact layout and the fixed normalisation constants.

Reward (Song et al. style): `||g - p_{k-1}|| - ||g - p_k|| - 0.01*||omega||`, with `g` the **true**
centre of the target gate **at the start of the step**; `+10` when the lap is completed, `-10` on
any other termination, no terminal term on truncation, and exactly `0` on the autoreset dummy
transition.

```bash
cd project
# training (torch lives only in the `tests` env)
env -u PYTHONPATH pixi run -e tests --frozen python -m rl.ppo train --total_timesteps=2000000
# curves + evaluation of a checkpoint through the MPC eval pipeline (seeds 0-19)
env -u PYTHONPATH pixi run -e tests --frozen python -m rl.plot_progress --run=results/ppo_<ts>
env -u PYTHONPATH pixi run -e tests --frozen python -m rl.eval_policy evaluate \
    --ckpt=results/ppo_<ts>/ckpt_final.pt
# cost of the perturbation wrapper at N=256
env -u PYTHONPATH pixi run -e tests --frozen python bench_rl_env.py \
    --pipelines="[rl_train,rl_train_perturb]" --num_envs="[256]"
```

A training run writes `results/ppo_<ts>/`: `progress.csv` (one row per update: episode reward,
gates, outcome fractions, steps/s of the whole loop and of `env.step` alone, PPO diagnostics),
`meta.json` (arguments + machine load at start and end), `ckpt_*.pt`, `curves.png`.

### What terminates an episode (measured, see the report in `results/ppo_smoke_*/FINDINGS.md`)

`terminated` is `data.disabled_drones` (`race_core.py:747`), which `_disabled_drones`
(`race_core.py:856-866`) sets on: contact with a gate or an obstacle (floor contacts are masked out
at `:843`), `pos > [3, 3, 2.5]`, `pos < [-3, -3, 0]` **unless** the drone is within 2 cm in xy of
its take-off position, and completing the lap (`n_gates_passed >= 5`). `truncated` is
`steps >= 1500` (30 s).

So a crash into the floor **does** terminate the episode (`clip_floor_pos` pins z at -0.001, which
is below `pos_limit_low[2] = 0`), *except* directly over the take-off pad, where the drone rests at
z = -0.001 until the 1500-step truncation. That is the one case where the eval counts a failure
(`run_eval.failure_group` reclassifies it as `fall`) while the reward gives 0 instead of -10.
