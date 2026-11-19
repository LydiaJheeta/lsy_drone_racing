# Round 4 — decisions taken without asking

Context: `T_end` made required, experiment E (noise check), F (separated cost knobs), diagnosis G
(reference quality). `!` marks a decision that can change the conclusions.

## A. The `T_end` change

1. **Keyword-only, not just "first positional without a default".** `T_end` is now required *and*
   keyword-only, so `ThrustDegradeDelay(env, 0.85)` is a `TypeError` instead of silently putting a
   k_end value into `T_end`. That is the whole point of the change, and a positional-but-required
   argument would have reintroduced the same class of silent error.
2. **`grid.exp_b` was the only caller that did not pass it**; it now passes `T_end=ts[0]` at
   construction and `set_params(T_end=T)` per block, as it already did.
3. The example in `logger.py`'s docstring was updated too, so copying it still works.

## B. Experiment E

4. **! `k_end=0.99` and `0.97` are the test of the hypothesis.** A 1% thrust loss by the end of the
   lap is far below anything that can matter dynamically (diagnosis G measures the reference peak
   demand at 0.45 N against a 0.80 N limit), so a collapse there cannot be a dose-response. That is
   why the flip table is restricted to those two.
5. **Flips are reported in both directions.** The brief asked only for seeds that *lose* success,
   but seeds that *gain* success are the same phenomenon and are the stronger evidence, so the
   analysis reports `lost` and `gained` per condition.
6. **! Gate clearance is measured against the frame, per pass, from the drone's own trajectory.**
   `ring_pass<i>` is the minimum distance from the drone to that gate's frame region over the steps
   where that gate was the target (`next_gate_idx == i` from the logger). 0 means touching. This is
   the same geometric measure the failure details already use (`dist_to_frame`), so the two are
   directly comparable, but note it is the heuristic ring distance, **not** the env's own collision
   geometry: a success at 0.085 m and a contact at 0.085 m can both occur.
7. **The logger runs in memory** (`log_dir=None`) and is flushed by hand after each episode, since
   a successful episode never terminates and would otherwise stay buffered. No `.npz` is written.

## C. Experiment F

8. **Three multiplicative factors rather than replacing `u_weight_scale`.**
   `R = u_weight_scale * diag(rpy_weight_scale * [1,1,1], thrust_weight_scale * 50)`, all defaulting
   to 1.0. This keeps round 3 exactly reproducible (`u_weight_scale=s`) while allowing F to separate
   the two, and gives three cross-checks: `r1_t1`, `r10_t10` and `r30_t30` must reproduce round 3's
   C columns 1, 10 and 30. They do, exactly.
9. **! The baseline of each F condition is `d=0` at the same weights**, so the d=1 row is read
   against its own undelayed control rather than against the original tuning.
10. **d=0 was run as well, not assumed.** The brief asked for both; it is what shows that a high
    attitude penalty costs nothing without delay.

## D. Diagnosis G

11. **! Passes are found by plane crossing, not by nearest approach.** The first attempt matched
    each pass to the nearest sample of the reference to the gate centre; that breaks when the track
    revisits a gate (the sequence is `[0, 1, 2, 3, 1]`) and crashed outright. A pass is now the
    first crossing of the gate's plane, after the previous pass, within 0.6 m of the centre in the
    plane, which is what the env itself means by passing a gate. A `found_by` column records when
    the fallback (nearest sample) had to be used; in the real reference it never was.
12. **A sample sitting exactly on the gate plane is handled explicitly.** The sign-change test is
    taken between consecutive samples that are not exactly zero, otherwise an exact zero hides the
    crossing. This was found by a test, after the main run; re-running G with the fix reproduces
    `reference_passes.csv` and `reference_demand.csv` byte for byte, so the published numbers are
    unaffected (real float data never lands exactly on the plane).
13. **Added beyond the brief: the dynamic demand of the reference.** The brief asked for distance,
    clearance and approach angle. Required thrust `m * |a + g|` per pass and over the whole lap was
    added because it answers "is the reference even feasible", which is what experiments D and E
    are about. Drag is ignored, so these are lower bounds.
14. **Gate poses are read from `env.unwrapped.data` after a seed-0 reset**, i.e. the true positions,
    not the sensed ones. Level 0 has no randomisation, so one reset is representative.
15. **No waypoint was touched**, as instructed. G only measures.

## E. Process

16. **G runs before E and F** regardless of the order of the letters in `--exp`, because it is
    instant and its output frames the reading of the other two.
17. **Nothing was committed.** Only step 1 of the brief authorised a commit, and that was round 3.
18. **Another session was found working in this repo** (`project/bench_rl_env.py`,
    `project/tests/test_bench_rl_env.py`, `project/results/rlbench_20261113_151905/`, plus README
    edits, between 15:19 and 15:25). None of it was touched; the round-4 README edits were applied
    as exact-match replacements so both sets of changes coexist. Their files are not part of this
    work and were left untracked.
19. **Nothing that builds an acados solver was run while the experiments were running**, because
    `project/c_generated_code/` is shared and a concurrent build would corrupt it.
