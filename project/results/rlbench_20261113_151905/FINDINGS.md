# RL pipeline survey + CPU throughput benchmark (2026-11-13)

Read-only survey of `lsy_drone_racing/control/train_rl.py` and `lsy_drone_racing/envs/drone_race.py`
(`VecDroneRaceEnv`), plus a steps/s benchmark run with `project/bench_rl_env.py`. Nothing outside
`project/` was modified. Line numbers refer to commit `7eb5879` (branch `project`).

## 1. Algorithm and parallelism

* **Hand-written single-file PPO in PyTorch, cleanrl-style.** No SB3 / rllib / cleanrl package:
  `train_ppo` (`train_rl.py:557`) is an inlined copy, as its docstring states
  (`train_rl.py:562`: "An implementation of PPO from cleanrl, see https://docs.cleanrl.dev/").
  Rollout + GAE + clipped surrogate + clipped value loss: `train_rl.py:610-726`.
* Libraries: `torch` (nn/optim/distributions, `train_rl.py:14-16,36`), `gymnasium` vector API,
  `jax` for the environment, `wandb` for logging (`:17`), `fire` for the CLI (`:9`, `:830`).
  **`torch` and `wandb` are not in the `default` pixi env** — only in `tests`/`gpu-tests`
  (`pyproject.toml:47` feature `rl`), so `train_rl.py` can only be imported in those.
* **Parallel environments: `Args.num_envs` (`train_rl.py:65`, default 1024).** Path:
  `Args.num_envs` -> `make_envs(num_envs=...)` (`:579`) -> `RandTrajEnv(num_envs=...)` (`:481`) ->
  `DroneEnv.__init__` -> `Sim(n_worlds=num_envs, ...)` (`crazyflow/envs/drone_env.py:92`).
  Batch bookkeeping: `batch_size = num_envs * num_steps` (`:114`), `num_steps=8` (`:67`),
  `minibatch_size = batch_size // num_minibatches` (`:115`, `num_minibatches=8`).
  Device: `Args.jax_device` (`:53`, default `"gpu"`) for the env, `Args.cuda` (`:51`) for the policy.

## 2. Observation / action space, control mode

### `VecDroneRaceEnv` (`envs/drone_race.py:107`, id `DroneRacing-v0` vector entry point)

* Observation: `build_observation_space` (`race_core.py:253-280`), a `spaces.Dict`, batched with
  `batch_space(..., num_envs)` (`drone_race.py:161-164`): `pos(3)`, `quat(4)`, `vel(3)`,
  `ang_vel(3)`, `n_gates_passed` `Discrete(n_gate_passes+1)`, `gate_sequence`
  `MultiDiscrete(n_gate_passes)`, `gate_sequence_direction(n_gate_passes)`, `gates_pos(n_gates,3)`,
  `gates_quat(n_gates,4)`, `gates_visited(n_gates)`, `obstacles_pos(n_obstacles,3)`,
  `obstacles_visited(n_obstacles)`. Default track: 4 gates, 4 obstacles, 5 gate passes.
  `step`/`reset` drop the drone axis (`drone_race.py:178-179`, `:192-193`), so arrays are `(N, ...)`.
* Action: `build_action_space` (`race_core.py:226-250`).
  `state` -> `Box(-inf, inf, (16,))` (full state `[pos, vel, acc, quat, ang_vel]`,
  see `race_core.py:476`). `attitude` -> `Box([-pi/2, -pi/2, -pi/2, 4*thrust_min],
  [pi/2, pi/2, pi/2, 4*thrust_max])`, i.e. `[roll, pitch, yaw, total thrust]` with thrust in
  `[0.0854, 0.8] N` for `cf21B_500` — thrust **last**, as already documented in `project/README.md`.
* Control mode: constructor argument `control_mode` (`drone_race.py:119`), passed straight to
  `Sim(control=control_mode)` (`race_core.py:386`) and used to pick the action space and the
  controller function (`race_core.py:580-586`). Both modes are supported; **all shipped configs set
  `control_mode = "state"`** (`config/level0.toml:47` and level1-3 `:48`), which is why this project
  keeps attitude copies in `project/config/att_level{0,1}.toml`.

### The env `train_rl.py` actually trains on

**`train_rl.py` never touches `VecDroneRaceEnv`.** `make_envs` (`:470-500`) builds `RandTrajEnv`
(`:121`), a subclass of crazyflow's `DroneEnv` that follows a **random cubic-spline trajectory**
(`:205-235`) — no gates, no obstacles, no contacts, no track. It only borrows the racing
*disturbance* specs from `config/level0.toml` (`:478-487`).

* Control mode: **attitude only.** `RandTrajEnv` does not pass `control` to `Sim`, so
  `Control.default = attitude` applies (`crazyflow/control/core.py:63`), and `DroneEnv` raises
  `NotImplementedError` for state control (`crazyflow/envs/drone_env.py:97-98`). Measured action
  space before `NormalizeActions`: `Box([-1.5708, -1.5708, -1.5708, 0.0854], [1.5708, 1.5708,
  1.5708, 0.8])`. `NormalizeActions` then exposes `[-1, 1]^4` (`crazyflow/envs/norm_actions_wrapper.py:29-32`),
  and `AngleReward.step` zeroes the yaw command (`train_rl.py:390`).
* Observation: `DroneEnv.obs` gives `pos(3), quat(4), vel(3), ang_vel(3)`; `RandTrajEnv.obs` adds
  `local_samples(3*n_samples=30)`, the position errors to the next 10 trajectory points
  (`:256-267`); `StackObs` adds `prev_obs(13*n_obs=26)` (`:342-377`), `ActionPenalty` adds
  `last_action(4)` (`:402-440`), `FlattenJaxObservation` concatenates (`:443-456`) and `JaxToTorch`
  converts. **Measured flat observation dimension: 73** (3+4+3+3+30+26+4), action dimension 4.
* Policy: 2x64 tanh MLP actor/critic, `actor_logstd` initialised `[-1,-1,-1,1]` ("smaller std for
  roll, pitch, yaw", `train_rl.py:532-534`) — further confirmation that index 3 is thrust.

### `train_rl.py` does not run as committed (crazyflow 0.4.0)

Two independent incompatibilities, both reproduced:

1. `train_rl.py:196` — `insert_fn_before(self.sim.step_pipeline, "integrate", disturbance_fn)`.
   The stage is named **`"integration"`** in crazyflow 0.4.0; `race_core.py:674` already uses the new
   name. Any config with `env.disturbances.dynamics` (all of level0-3) raises
   `KeyError: No stage named 'integrate'. Available stages: ('attitude_controller',
   'force_torque_controller', 'clip_rotor_vel_cmd', 'integration', 'clip_floor_pos',
   'increment_steps')`.
2. `train_rl.py:318`, `:325` — the reset-randomization callbacks take `(data, mask)`, but crazyflow
   0.4.0 calls every reset-pipeline function as `fn(data, default_data, mask)`
   (`crazyflow/sim/sim.py:441`). The first `reset()` raises `TypeError: ... takes 2 positional
   arguments but 3 were given`. `race_core.py:905` uses the 3-argument signature.

`project/bench_rl_env.py:_patch_train_rl_for_crazyflow_040` monkeypatches both **in-process only**
(signature adapters, no change to the dynamics) so the pipeline can be benchmarked. Fixing the two
lines upstream is a 2-line change, but that is outside `project/`.

## 3. Reward, and where a gate-progress reward goes

Two unrelated reward functions:

* **Racing env**: `reward(data)` at `race_core.py:733-744` returns
  `-1.0 * (n_gates_passed >= gate_sequence.shape[0])` — i.e. **-1 when the lap is finished, 0
  otherwise**; its own docstring says it "will most likely not work directly for training an agent".
  Sign aside, it carries no shaping at all.
* **`train_rl` env**: `RandTrajEnv.reward` (`:269-282`) = `exp(-2 * ||pos - goal||)` with
  `goal` = the trajectory point for the current step, and `-1.0` where `terminated()`
  (out of a `[-4,-4,0] .. [4,4,4]` box, `:304-310`). Wrappers then add:
  `AngleReward` `-rpy_coef * ||rpy||` (`:394-399`), `ActionPenalty`
  `-act_coef * thrust^2 - d_act_th_coef * dthrust^2 - d_act_xy_coef * ||d(rpy)||^2` (`:424-435`).
  Coefficients: `Args` `:102-108`.

**Gate-progress reward (Song et al. style).** Change `reward(data)` at `race_core.py:733`; it is
called at `race_core.py:574` inside the jitted `step`, after `_update_target_gates` (`:559`), so the
gate counter it sees is already post-pass. Fields needed, all present in `EnvData`:

| quantity | source |
|---|---|
| current drone position | `data.sim_data.states.pos` (obs `pos`) |
| previous drone position | `data.last_drone_pos` (`race_core.py:814`) — **in `EnvData`, not in obs** |
| index of the target gate | `data.gate_sequence[data.n_gates_passed]` (cf. `:807`) |
| target gate pose | `data.gates_pos[...]`, `data.gates_quat[...]` (`:809-810`) |
| pass direction | `data.gate_sequence_direction[data.n_gates_passed]` (`:808`) |
| gate passed this step | recomputable via `gate_passed(...)` (`:811`), or from the increment of `n_gates_passed` |
| crash / out of bounds | `data.disabled_drones` (`terminated`, `:747`) |

Progress reward = `||p_{t-1} - g|| - ||p_t - g||` (optionally projected on the gate normal from
`gates_quat`) + a bonus on each pass + a crash penalty. Two traps:

* **Autoreset happens before the reward.** `step` runs the masked reset at `:563-572` and *then*
  computes `reward(data)` at `:574`, so for an env that was marked for reset the reward is computed
  on the already-reset state (`n_gates_passed = 0`, `last_drone_pos = ` new start pos). A progress
  reward must mask those envs (e.g. with `marked_for_reset`, saved at `:548`) or the last transition
  of every episode produces garbage.
* **`obs` reports sensor poses, not true poses**: `gates_pos`/`gates_quat` in the observation are
  the *nominal* poses until the gate has been within `sensor_range` (`race_core.py:710-716`).
  Computing the reward *inside* the env (from `data.gates_pos`) uses the true poses; computing it in
  an outer wrapper from the observation would use nominal poses for gates not yet seen.
* `n_gates_passed` is a `Discrete` in the obs space but an int array in the data; the policy
  observation has no "distance/direction to the next gate" field — it would have to be added to
  `build_observation_space` (`race_core.py:253`) plus `obs()` (`:705`), like `local_samples` does in
  `train_rl.py:199-203`, `:256-267`.

## 4. Randomization, and a random thrust gain `k` per run

Three mechanisms, all driven from TOML through `rng_spec2fn` (`race_core.py:872-884`,
`{fn, scale, offset, kwargs}` -> `jax.random.<fn>() * scale + offset`):

| what | where it is applied | config key |
|---|---|---|
| `drone_pos`, `drone_rpy`, `drone_mass`, `drone_inertia` | `build_reset_fn` (`race_core.py:887-910`) -> appended to the **sim reset pipeline** (`:670-671`), functions in `envs/randomize.py:24-81` | `[env.randomizations.*]` (level1-3 `:119-128`) |
| `gate_pos`, `gate_rpy`, `obstacle_pos`, plus full-layout `track.randomize` | `build_track_randomization_fn` (`:913-955`) -> called inside the jitted reset (`:529`), functions in `envs/randomize.py:84-120` | `[env.randomizations.*]`, `[env.track].randomize` (level3 `:54`) |
| `disturbances.action` (per step, added to the command) / `disturbances.dynamics` (per step, world-frame force) | `apply_action` (`:593-597`) / `build_dynamics_disturbance_fn` (`:958-967`) inserted before `integration` (`:672-675`) | `[env.disturbances.action]`, `[env.disturbances.dynamics]` (level0 `:96-104`) |

Reset randomization is per-world and masked (`leaf_replace(..., mask, ...)`), so with autoreset each
world redraws when *it* resets — that is exactly the "per run" granularity wanted for `k`.

**Adding a random thrust gain `k` per run.** Two places, with different physical meaning:

1. **Plant-side (true motor/battery degradation, recommended for domain randomization).** The
   first-principles plant converts rotor speed to thrust with `params.rpm2thrust`
   (`crazyflow/dynamics/first_principles/dynamics.py:332`, per-world, `CORE_NDIM_KEY: 2`,
   shape `(N, M, 1, 3)`), while the onboard attitude controller inverts with its **own** static
   parameter set (`force_torque2rotor_vel(..., **ft_ctrl.params)`,
   `crazyflow/control/mellinger/control.py:748-750`, loaded by `load_fn_params` at
   `:632`). The two are independent, so scaling `params.rpm2thrust` by `k` is **not** cancelled by
   the controller — exactly the mismatch we want. The pattern is `randomize_drone_mass_fn`
   (`envs/randomize.py:53-65`): a function `(data, default_data, mask) -> data` that draws
   `k ~ U(k_min, 1)` with shape `(n_worlds, n_drones, 1, 1)` and does
   `leaf_replace(data.params, mask, rpm2thrust=default_data.params.rpm2thrust * k)` (use
   `default_data`, not `data`, or `k` compounds over episodes). Registering it upstream means a new
   `case "thrust_k"` in `build_reset_fn` (`race_core.py:890-903`) plus a factory in
   `envs/randomize.py` and a `[env.randomizations.thrust_k]` block — all outside `project/`. From
   `project/` only, the same effect is reachable without editing the repo:
   `append_fn(env.unwrapped.sim.reset_pipeline, fn)`, then rebuild
   `env.unwrapped.sim.build_reset_fn()` **and** `env.unwrapped._reset = env.unwrapped.build_reset_fn()`
   and `env.unwrapped._step = env.unwrapped.build_step_fn()`, because `RaceCoreEnv.build_reset_fn`
   captures the sim reset function at build time (`race_core.py:509`). In `train_rl`'s env the hook
   is cleaner: extend `RandTrajEnv.build_reset_randomization_fn` (`train_rl.py:312-338`), which is
   already the function appended to the sim reset pipeline.
2. **Command-side (what `project/wrapper.py` already does).** `applied_thrust = k * commanded`.
   To make it random per run, draw `k` in `reset()` from the episode seed instead of using a fixed
   `k_end`. Note the ordering: the env clips the action to the action space *before* adding the
   action disturbance (`race_core.py:592`), and `project/wrapper.py` applies "delay, then `k(t)`,
   then clip", so for `k < 0.532` hover is impossible (`project/HANDOFF.md`). This is a mismatch
   between the controller's assumed and the realised actuator gain, but it is *not* a plant
   parameter change — the MPC's own model and the sim dynamics both stay nominal.

Observability is a separate decision: neither mechanism puts `k` in the observation, so a policy
must infer it from `prev_obs`/`last_action` history (`StackObs`, `ActionPenalty`). For an
MPC-with-learning baseline (E1/E2) `k` is exactly the kind of parameter RLS should estimate.

## 5. Throughput on CPU (random actions, no training)

`project/bench_rl_env.py`, 200 timed steps per configuration after 20 warmup steps; compile time
excluded (reported separately); actions pre-sampled with a fixed seed and replayed from a ring
buffer; `jax.block_until_ready` after **every** step (what a policy in the loop sees, since it must
read the observation before acting). Host: Linux 6.8.0, 8 CPUs, jax 0.11.0 (`cpu:0`), gymnasium
1.4.0, Python 3.13.15. Env step rate is 50 Hz with 10 sim substeps at 500 Hz, so
`real-time factor = steps/s / 50`.

`steps/s` = `num_envs * n_steps / elapsed` (environment transitions per second).

| pipeline | N=1 | N=16 | N=64 | N=256 |
|---|---:|---:|---:|---:|
| `race_state` (`VecDroneRaceEnv`, state) | 890 | 8 572 | 12 149 | 13 737 |
| `race_attitude` (`VecDroneRaceEnv`, attitude) | 617 | 5 732 | 12 765 | 18 857 |
| `train_rl` (`RandTrajEnv` stack, patched) | 110 | 537 | 1 898 | 6 797 |

Per vectorized step (ms), same runs:

| pipeline | N=1 | N=16 | N=64 | N=256 |
|---|---:|---:|---:|---:|
| `race_state` | 1.124 | 1.866 | 5.268 | 18.636 |
| `race_attitude` | 1.620 | 2.791 | 5.014 | 13.576 |
| `train_rl` | 9.068 | 29.796 | 33.723 | 37.663 |

Raw numbers, percentiles, build/compile times: `race/throughput.csv`, `train_rl/throughput.csv`,
`*/results.json`, logs `race.log`, `train_rl.log`.

Observations (measurements, not hypotheses):

* Vectorization pays off but sublinearly on 8 cores: `race_attitude` gains 31x from N=1 to N=256
  (617 -> 18 857 steps/s), i.e. 377x real time; the marginal gain from 64 to 256 is only 1.5x.
  `race_state` saturates earlier (12 149 -> 13 737, 1.13x) — at N=256 the extra 12-dim state
  controller work per world shows up (18.6 ms/step vs 13.6 ms for attitude).
* The `train_rl` stack is **3-8x slower per environment step** than the racing env at the same N
  (6 797 vs 18 857 steps/s at N=256) and has a large fixed cost per step: 9.1 ms at N=1 vs 1.6 ms.
  At N=1 -> 16 the time per step triples although the work per step is 16x, so the cost is
  dominated by host-side work, not by the sim. Candidates (not verified): `RandTrajEnv.obs` indexes
  the host-side numpy array `self.trajectories` of shape `(N, 7500, 3)` on every call
  (`train_rl.py:256-267`), `reward()` calls `obs()` a second time (`:271`), and the four python
  wrappers each add a dict copy and a device transfer per step.
* Autoreset load was comparable across pipelines (`done_fraction` 0.019-0.023 for the racing env,
  0.075-0.080 for `train_rl`, whose episodes end sooner under random actions), so it does not
  explain the difference.
* Pipelined timings (block only after the last step) differ from the blocked ones by less than
  ~10% everywhere, i.e. nothing here is hidden behind async dispatch; on CPU the step is
  essentially synchronous.
* `total_timesteps = 1_500_000` (`train_rl.py:61`) at the measured N=256 rate is ~3.7 min of pure
  environment stepping (6 797 steps/s) — the policy update cost is **not** included in any of these
  numbers.

### Caveats

* All numbers are CPU-only (no GPU on this machine); `Args.jax_device` defaults to `"gpu"`
  (`train_rl.py:53`), so training as shipped is meant to run elsewhere.
* The two `race_*` rows were measured in the `default` pixi env, `train_rl` in `tests` (it needs
  torch/wandb). Same solve group, but torch's thread pool is loaded in the second case.
* Random actions, so the state distribution is not the one a trained policy would visit; the
  `race_state` actions come from `Box(-inf, inf)`, which gymnasium samples as standard normal.
* The `train_rl` row required the two in-process shims of section 2; they only adapt call
  signatures, but the pipeline is not byte-for-byte what a fixed `train_rl.py` would run.
* A `RuntimeWarning: overflow encountered in cast` appears while constructing the racing env; it
  comes from casting the infinite action/observation-space bounds and is unrelated to the timings.
