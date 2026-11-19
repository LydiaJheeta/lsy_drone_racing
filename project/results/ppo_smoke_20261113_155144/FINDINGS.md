# PPO phase 1: setup + smoke run (2026-11-13)

Gate-progress PPO on `VecDroneRaceEnv` (attitude), nominal level 0, no thrust/delay perturbation.
Line numbers refer to commit `7eb5879` of the branch `project`.

## A. What ends an episode, and how the eval defines failure

`terminated` is exactly `data.disabled_drones` (`race_core.py:747-749`), set by `_disabled_drones`
(`race_core.py:856-866`):

| condition | code | note |
|---|---|---|
| contact with a gate or an obstacle | `:864-865` | **floor contacts are masked out** (`:843`) |
| `pos > [3, 3, 2.5]` | `:862` | any axis |
| `pos < [-3, -3, 0]` **and** more than 2 cm (xy) from the take-off position | `:859-861` | the pad exemption |
| lap completed, `n_gates_passed >= 5` | `:863` | success also "disables" the drone |

`truncated` is `steps >= max_episode_steps` = 1500 = 30 s (`:752-755`). Disabled drones are warped
to `pos = (-1,-1,-1)` *before* the observation is built (`:557`), and autoreset runs *inside* `step`
before the reward (`:563-574`).

**Measured** (probe placing the drone 0.3 m above the floor with minimum thrust, `uptime` load 1.4):

* **off the pad** (placed at xy = (-2.0, 1.0), take-off pad at (-1.5, 0.75)): sinks to z = -0.001
  (`clip_floor_pos`, `crazyflow/sim/sim.py:728-735` pins z at -0.001 and zeroes the velocity),
  which is below `pos_limit_low[2] = 0` -> **terminated after 13 steps, cause `floor`**;
* **on the pad** (xy = take-off position): the same sink does **not** terminate; the drone rests at
  z = -0.001 until the 1500-step truncation.

So the old note "the env does not terminate on floor contact" (`HANDOFF.md`, and the docstring of
`run_eval.failure_group`) is only true within 2 cm of the take-off xy — which is exactly the
lift-off oscillation case the MPC experiments ran into, hence the original observation.

**Consequence for the -10 term.** `project/common.py` calls a run a failure whenever
`n_gates_passed < 5`, and `run_eval.failure_group` splits that into `collision` (causes `gate`,
`obstacle`, `contact_unknown`), `fall` (cause `floor`, or a timeout whose `min_z_airborne` or
`final_z` is below 0.05 m), `out_of_bounds` and `timeout`. The reward's `-10` fires on every
`terminated` that is not a completed lap, i.e. it covers `collision`, `out_of_bounds` and the
off-pad `fall`. The **one mismatch** is a crash that ends on the pad (or any other 30 s timeout):
the eval counts it as a failure, the reward gives 0 (per the brief: "0 on truncation"). The smoke
run's 14 timeouts are *not* of that kind - they end airborne at z = 1.0-1.3 m (`eval_final/episodes.csv`,
`final_z`), so they are genuine "survived without racing" episodes.

**Single env vs vector env.** `common.make_env` uses `gymnasium.make(config.env.id, ...)`, whose
entry point is `DroneRaceEnv` (`envs/__init__.py:22`), *not* `VecDroneRaceEnv` with `num_envs=1`
(that is the `vector_entry_point`, `:23`). Both subclass `RaceCoreEnv` with `n_envs=1`. Measured
with the same seed and the same 60 commands: **max |Δpos| = 0.0** - bit-identical dynamics and
identical observation timing. The differences are only:

* `autoreset`: `False` for `DroneRaceEnv` (`drone_race.py:73-74`, which also rebuilds `_step`),
  `True` / `AutoresetMode.NEXT_STEP` for the vector env;
* shapes: scalars/`(3,)` vs `(N,)`/`(N,3)`, and `reward`/`terminated` as python floats/bools.

A policy trained on the vector env can therefore be evaluated through the MPC pipeline unchanged.

## B-D. What was implemented

New files, all under `project/` (nothing else was touched, no monkeypatching):

| file | content |
|---|---|
| `rl/env.py` | `GateProgressRaceEnv` (subclass of `VecDroneRaceEnv`), 41-float observation, gate-progress reward, `VecAttitudeAction` (3 -> 4 action mapping), `make_train_env` |
| `rl/vec_perturb.py` | `VecThrustDegradeDelay`: per-world command delay + k(t), semantics of `wrapper.py` |
| `rl/ppo.py` | PPO, core copied from `train_rl.py:557-757`; CSV metrics, `.pt` checkpoints, no wandb |
| `rl/eval_policy.py` | `PolicyController` + `evaluate`, through `common.run_episode` |
| `rl/plot_progress.py` | `curves.png` from `progress.csv` |
| `tests/test_rl_env.py`, `tests/test_rl_perturb.py`, `tests/test_rl_ppo.py`, `tests/test_rl_eval.py` | 37 tests |

**Reward** (`rl/env.py:gate_progress_reward`), exactly as briefed:
`r = ||g - p_{k-1}|| - ||g - p_k|| - 0.01*||omega||`, `g` = **true** centre
(`self.data.gates_pos`, not the sensor values) of the target gate **read before** `super().step()`;
`+10` if the terminating step completed the lap, `-10` on any other termination, and on a
terminating step **only** that term (the warp to `(-1,-1,-1)` makes `p_k` meaningless); no terminal
term on truncation; exactly `0` on the autoreset dummy transition.

**Action**: `[-1,1]^3 -> [roll, pitch, yaw=0, thrust]`, `roll/pitch = 0.5*a` rad, thrust linear into
`[0.0854, 0.8] N`; clipped to `[-1,1]` first because PPO samples from an unbounded Normal.
Hover (0.4258 N) sits at `a_thrust = -0.047`, i.e. almost exactly in the middle of the range.

**Observation** (41): `pos/3`, `vel/5`, rotation matrix (9), `ang_vel/10`, for the next 2 gates
`(centre - pos)/3` and the signed gate normal (the gate's local x-axis, `envs/utils.py:95-97`,
negated where `gate_sequence_direction < 0`), for the 4 obstacles `(obstacle - pos)_xy / 3`, last
action (3). Gate/obstacle poses are taken from the **observation** (sensor: nominal until seen);
a test asserts that on level 0 sensor == nominal == true, so this matches the brief.

**Deviations from the brief / decisions taken without asking**

1. The action mapping is a thin wrapper (`VecAttitudeAction`) *outside* the env rather than inside
   it, so that `VecThrustDegradeDelay` can sit on the native 4D attitude command and keep exactly
   the `wrapper.py` semantics. Consequence: the "last action" in the observation is the
   **inverse-mapped applied command**. With `perturb=None` (phase 1) that is identical to the
   policy's action; with the perturbation on it would also reveal part of k(t) to the policy, which
   is a phase-2 decision to make deliberately.
2. `eval_policy` writes `T_traj = 0.0` (not NaN) in the `run_eval` schema: `T_traj` is a
   reference-trajectory duration and a policy has none, and `run_eval.summarize` groups rows by
   `T_traj`, where NaN != NaN would drop every row.
3. `PolicyController.step_callback` had to return `False`: `Controller.step_callback` returns
   `True` by default (`controller.py:95`, whose inline comment says the opposite) and
   `common.run_episode` stops on a truthy return - without the override every eval episode ended
   after one step with cause `trajectory_end`.
4. `project/.gitignore` now ignores `results/**/*.pt` (binary checkpoints).
5. `bench_rl_env.py` got two pipelines (`rl_train`, `rl_train_perturb`) and records the machine
   load in `meta.json`, instead of a separate benchmark script.

## 3. What `att_level0.toml` randomizes at reset

**Nothing.** It has no `[env.randomizations]` block and `env.track.randomize = false`, so
`build_reset_fn({})` adds no randomization to the sim reset pipeline and
`build_track_randomization_fn` only restores the nominal track. The start state is the same every
episode; only the JAX key differs with the seed.

The stochasticity is **per step**, from `[env.disturbances]` (inherited from `level0.toml:96-104`):

* `action`: `N(0, 0.001)` added to every element of the attitude command (`race_core.py:593-597`),
  after the command has been clipped to the action space;
* `dynamics`: `U(-0.1, 0.1) N` per world-frame axis, which **replaces** the CoM force every
  simulation step at 500 Hz before integration (`race_core.py:958-967`). For scale: hover thrust is
  0.4258 N, so this is up to ~24% of hover as a random sideways/vertical force.

## G. Smoke run

`results/ppo_smoke_20261113_155144/`, 244 updates x 8192 steps = 1 998 848 steps, 287.6 s wall.
Machine load (`/proc/loadavg`, 8 CPUs, torch threads 4): **3.35 at start, 7.98 at the end** (`meta.json`; `uptime` 20 s later: 6.00); the
other session's `round4.py --exp=GEF` ran throughout (recorded in `meta.json` as
`other_jobs: ["round4"]`). Throughput is therefore a lower bound.

| update | steps | reward/episode | episode length | reward/step | gates | success | crash | timeout | steps/s (loop) | entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 192 | -10.34 | 28.1 | -0.368 | 0.00 | 0.00 | 1.00 | 0.00 | 1 408 | 3.22 |
| 41 | 335 872 | -14.28 | 253.5 | -0.056 | 0.00 | 0.00 | 1.00 | 0.00 | 7 001 | 3.92 |
| 101 | 827 392 | -15.53 | 364.7 | -0.043 | 0.00 | 0.00 | 1.00 | 0.00 | 4 552 | 4.83 |
| 161 | 1 318 912 | -16.83 | 460.1 | -0.037 | 0.00 | 0.00 | 0.93 | 0.07 | 4 303 | 4.96 |
| 221 | 1 810 432 | -18.48 | 768.1 | -0.024 | 0.00 | 0.00 | 0.57 | 0.43 | 7 591 | 4.99 |
| 244 | 1 998 848 | -19.26 | 745.5 | -0.026 | 0.00 | 0.00 | 0.67 | 0.33 | 7 848 | 4.96 |

* Throughput of the **whole** loop (env + policy + update): mean 7 249 steps/s, min 1 408 (first
  update, includes JIT), max 8 911. `env.step` alone: mean 12 027 steps/s (max 14 190).
* 5 619 episodes finished in total. **Not one of them passed a single gate** (`gates_mean = 0.00`
  for all 244 updates).
* Reward per *episode* gets worse (-10.3 -> -19.3) while reward per *step* improves by a factor of
  14 (-0.368 -> -0.026) and episode length grows 27x (28 -> 746 steps). The policy is learning to
  survive, not to race: longer episodes accumulate more small negative progress.
* `explained_var` rises to 0.3-0.6, so the critic does fit the (short-horizon) returns.
* **Policy entropy grows monotonically, 3.22 -> 4.96.** For 3 independent Gaussian dimensions that
  is a per-dimension std of ~1.26 on an action space of `[-1,1]`: by the end of training the
  sampled commands are mostly saturated noise. With `ent_coef = 0.007` and a learnable `logstd`
  nothing in this reward is pulling the std back down.

### Evaluation of `ckpt_final.pt` (deterministic policy, seeds 0-19, `eval_final/`)

Load at eval start: 6.00 (round4 still running).

**0/20 successes, 0 gates passed.** 14 episodes time out at 30.00 s, 6 collide with a gate frame
after 1.5-2.0 s (`gate_id=3`, `dist_to_frame` 0.08-0.10 m; that attribution is the heuristic of
`common.diagnose_termination`). The timeouts end airborne around `(0.0, 0.2, 1.2)` m - the policy
flies off the pad towards the first gate (centre `(0.5, 0.25, 0.7)`) and then loiters ~0.5 m short
of and ~0.5 m above it.

### Reading (separating measurement from hypothesis)

Measured: the pipeline runs end to end, the reward and observation behave as specified (37 tests),
~7 250 steps/s for the full loop on a shared 8-core CPU, and 2M steps produce a policy that takes
off, heads for the first gate and then stalls, never passing one.

Hypotheses (**not tested**) for why it stalls, in the order I would test them:

1. `gamma = 0.94` at 50 Hz is a 0.33 s effective horizon (`train_rl.py` default, kept per the
   brief). The `+10` for completing the lap is discounted by `0.94^n`; at 250 steps that is `10^-7`.
   The agent is effectively optimising one-step distance gain, for which "hover near the gate" is a
   local optimum.
2. The entropy bonus (`ent_coef = 0.007`) drives the action std to ~1.26 (measured above), which
   makes threading a 0.4 m opening impossible; precision would need a decaying or fixed std.
3. The pure distance-progress term gives no reward for *aligning* with the gate normal, so passing
   through the opening is not distinguished from approaching the frame - the 6 frame collisions are
   consistent with that.

Nothing was measured about what a longer run or different hyperparameters would do.

## Tests

37 new tests (`test_rl_env.py` 17, `test_rl_perturb.py` 8, `test_rl_ppo.py` 6, `test_rl_eval.py` 6),
including: the `[-1,1]^3` -> command mapping at the box corners and with out-of-range samples, the
rotation matrix against scipy, the observation layout at the level-0 start pose, the signed gate
normals against scipy, sensor == true on level 0, train-vs-eval observation equality, the reward
sign and magnitude, the terminal-term-only rule, the dummy transition being exactly 0, a real gate
crossing showing the pre-step target is used (the post-step target would have scored the same step
-0.004 instead of -0.098), the terminal -> dummy sequence in the real env, equivalence of
`VecThrustDegradeDelay` with the scalar `ThrustDegradeDelay` for 4 (k_end, delay) combinations, the
per-world draws, the autoreset clock restart, and an end-to-end 2-update training run.

## E. Cost of the perturbation wrapper

`results/ppo_perturbcost_20261113_155105/`, N=256, 200 steps, load 1.62, round4 running:

| stack | ms per vectorized step | steps/s |
|---|---:|---:|
| `rl_train` (no perturbation) | 17.238 | 14 851 |
| `rl_train_perturb` (delay 0-2, k_end in [0.6, 1.0], T_end=18.75 s) | 17.212 | 14 873 |

**No measurable cost** (-0.15%, inside the run-to-run spread): the wrapper is a numpy ring buffer
over 256x4 floats against a 17 ms simulation step.
