# Round 3 — decisions taken without asking

Context: experiments C (delay x cost-weight scale) and D (T_traj x k_end) plus the `T_end` fix.
`!` marks a decision that can change the conclusions.

## A. Experiment design

1. **! One knob for the whole `R`.** `u_weight_scale` multiplies all three attitude weights *and*
   the thrust weight (`R = diag([1, 1, 1, 50])`, ratio unchanged). Rejected alternative: scaling
   only `R[0:3]`. Chosen as the literal reading of the brief ("the weights that penalise the
   attitude/thrust commands"). **Cost:** at scale 30 the maximum commanded thrust drops from
   0.80 N to 0.62 N, so the "soft" controller is also less aggressive in thrust. `d=0` within each
   column is the control, which keeps the comparison *inside* a column valid but not *across*
   columns.
2. **! No du term was added.** The brief said "and the rate of change if there is one" — there is
   none in the LINEAR_LS cost. Adding one would have been a new controller, not a scale parameter,
   so it was left out.
3. **! Per-condition baseline.** C: `d=0` at the *same* scale. D: `k_end=1.0` at the *same*
   `T_traj`. Rejected alternative for C: one global baseline (`d=0, w=1`). The per-column choice
   prevents a column from looking bad merely because its own baseline already failed.
4. **C and D in one process** (`--exp=CD`) to share the JIT setup. A crash in D would have cost
   time but no data: `episodes.csv` is flushed after every episode.
5. **No per-step `.npz` logging** for C/D (`log_dir=None`). Keeps the folder small, but the
   trajectories of this round **cannot be re-analysed** without re-running them.

## B. Code

6. **`cost_set` after the solver is created**, to push `W` into the solver instance. Rejected
   alternative: a distinct `model.name`/`json_file` per scale, i.e. a distinct `.so`. `cost_set`
   needs no recompile and writes to that one instance only.
7. **Kept the per-episode solver rebuild** (not optimised away) to preserve the documented paired
   reproducibility property.
8. **Fixed `T_end` in `grid.py` and `run_eval.py` too**, not only in the new script: same bug, two
   lines. This changes the default behaviour of `run_eval.py`.
9. **The wrapper's `T_end=18.0` default was left in place** in round 3, to avoid breaking existing
   calls and tests; callers pass the right value explicitly. (Superseded in round 4: `T_end` is now
   a required argument.)
10. **A new `round3.py` instead of extending `grid.py`.** `grid.py` is keyed on `(T, d, k)` and has
    already produced committed results; changing its semantics was not worth it.
11. **One generic `summarize_conditions(rows, keys, baseline_of)`** instead of two bespoke
    aggregators for C and D.
12. **Testing `W` without a real build:** `AcadosOcpSolver` is monkeypatched with a stub that
    records the `cost_set` calls, plus *one* real-solver test using `cost_get(0, "W")` that catches
    exactly the dlopen bug. Otherwise the suite would pay for a recompile per test.

## C. Metrics

13. **`0.8 * k(t)` taken literally**, read as "what the degraded actuator can still deliver". The
    0.8 N comes from `action_space.high[3]`, it is not hardcoded.
14. **`k` is evaluated when the command is issued.** For `d=0` (experiment D) that is also when it
    is applied; for `d>0` the two differ by `d` steps, but in experiment C the metric is 0%
    everywhere (k == 1), so no reported number depends on this.
15. **Tilt threshold 0.49 rad, inclusive (`>=`).**
16. **Pooled over all steps** of all episodes of a condition — literally "share of steps", not the
    mean of per-episode shares. The raw per-episode counts are in `episodes.csv`, so the other
    definition can be recomputed.
17. **Commands recorded by monkeypatching the instance's `compute_control`** (same trick as
    `grid.exp_a`): this records the **MPC's own** command, before the wrapper touches it.
18. **Extra columns beyond the brief:** conditional success for C as well, `thrust_cmd_max/mean`,
    `lap_time_std`, `gates_mean`, `min_z_airborne_min`.

## D. Process

19. **The smoke test wrote to the scratchpad**, not to `project/results/`, to keep throwaway runs
    out of the results tree.
20. **The raw stdout was copied to `run_CD_stdout.log`** inside this folder: acados prints from C
    and **bypasses the `Tee`**, which is why the older `run.log` files contain none of it.
21. **The 1391 `QP status 3` messages were investigated** with four single episodes rather than
    ignored. They occur only in episodes that are already diverging (0 in nominal `d=0 w=1`, 0 in
    `d=0 w=30`, 0 in `T=12 k=0.6`, 17 in `d=1 w=1 seed 0`): a symptom, not a cause.
22. **`README.md` was updated** (dlopen bug, `T_end`, `u_weight_scale`, and the fact that the old
    T=16 numbers are void). Not explicitly requested, but the brief makes the README the place for
    conventions.
23. **No ruff/formatter run.** Line lengths were checked by hand against the 100-char limit: every
    over-long line in the touched files is pre-existing, none is new.
24. **`expB` was not re-run** with the corrected `T_end`; recorded as outstanding instead of
    widening the scope.

## E. Deliberately untouched

No commit or push during the work, no `rm`, nothing outside `project/`, no existing results file,
no change to levels or configs, nothing for E1/E2/E3.
