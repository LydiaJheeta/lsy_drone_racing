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
```

`--exp=CD` runs both round-3 experiments into one `results/round3_<ts>/` folder; `--exp=C` and
`--exp=D --out_dir=results/round3_<ts>` run them separately into the same folder.

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
* **`T_end` is the `T_traj` of the experiment, not a constant.** k(t) falls linearly from 1.0 to
  `k_end` over `T_end` seconds of absolute time, so a fixed `T_end` (the old default of 18 s) meant
  a fast lap (small `T_traj`) ended before k had reached `k_end` and the condition label was a lie.
  `wrapper.set_params(T_end=...)` now exists, `run_eval.py` sets `T_end = T_traj` of each block
  unless `--T_end` is given, and `grid.py`/`round3.py` set it per block. Consequence: the earlier
  `grid_20261113_021212/expB` numbers at T_traj=16 were run with T_end=18 s, i.e. they never
  reached the stated `k_end`; experiment D supersedes them.
* **Cost weights: `my_mpc.py` takes `u_weight_scale`** (default 1.0 = the original tuning). The
  LINEAR_LS cost penalises the commands only through the input block `R = diag([1, 1, 1, 50])`
  (roll/pitch/yaw towards 0, thrust towards hover); there is **no** rate-of-change (du) term, so
  there is nothing else to scale. `u_weight_scale` multiplies all four entries, which keeps the
  original 1:1:1:50 ratio between the attitude and thrust penalties. Scaling the attitude and the
  thrust penalty separately is the obvious follow-up but is not implemented.
* **acados gotcha: the generated solver library is dlopen-cached per path.** Every solver in a
  process is loaded from `c_generated_code/lib..._basic_example_mpc.so`; once that file has been
  opened, rebuilding it does not affect a solver created later in the same process, which silently
  keeps the FIRST build's cost matrix. Measured: before the fix, `u_weight_scale=30` reproduced the
  `u_weight_scale=1` trajectory step for step. `create_ocp_solver` therefore pushes `W` into each
  solver instance with `cost_set` after creating it. This never mattered before, because nothing
  but `T_traj` (a Python-side reference, not part of the OCP) used to change between episodes.
* **Wrapper order of operations** for step n (t = n/50 s): delay -> thrust *= k(t) -> clip to the
  env's action bounds. While the delay buffer fills (n < d) the *first* command is repeated.
  k(t) falls linearly 1.0 -> `k_end` over `T_end` seconds of **absolute** time and is constant
  afterwards, so for fast laps (short `T_traj`) the drone only sees part of the drop.
* **Logger** sits *inside* the wrapper (directly on the base env) so it stores what the env received.
  `state[1:]` + `final_state` give x_{t+1}. Episodes that end by success (no termination yet) are
  flushed on the next `reset()`/`close()`.
* `my_mpc.py` rebuilds the acados solver every episode (as the original does). Reusing a solver would
  leak warm-start state between episodes and break paired reproducibility.
