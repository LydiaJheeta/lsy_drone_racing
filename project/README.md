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
| `config/att_level{0,1}.toml` | copies of `config/level{0,1}.toml` with `control_mode="attitude"`, `render=false` |
| `tests/` | one pytest module per module above |
| `results/` | every run writes to a NEW timestamped folder (`step0_<ts>/`, `eval_<ts>/`) |

## Running

Run from `project/` — acados writes `c_generated_code/` into the working directory (git-ignored).

```bash
cd project
pixi run python diagnose.py --levels="[0,1]" --n_runs=20 --seed0=0     # Step 0
pixi run python run_eval.py                                            # Step 4 (sweep + final)
```

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
* **Wrapper order of operations** for step n (t = n/50 s): delay -> thrust *= k(t) -> clip to the
  env's action bounds. While the delay buffer fills (n < d) the *first* command is repeated.
  k(t) falls linearly 1.0 -> `k_end` over `T_end` seconds of **absolute** time and is constant
  afterwards, so for fast laps (short `T_traj`) the drone only sees part of the drop.
* **Logger** sits *inside* the wrapper (directly on the base env) so it stores what the env received.
  `state[1:]` + `final_state` give x_{t+1}. Episodes that end by success (no termination yet) are
  flushed on the next `reset()`/`close()`.
* `my_mpc.py` rebuilds the acados solver every episode (as the original does). Reusing a solver would
  leak warm-start state between episodes and break paired reproducibility.
