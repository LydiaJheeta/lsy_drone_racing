## Sweep (level 0, seeds 0-9, same seeds in all rows)

| T_traj | condition | n | n_success | success_rate | lap_time_mean_s | lap_time_std_s | gates_mean_all | gates_mean_failed | failure_causes |
|---|---|---|---|---|---|---|---|---|---|
| 18.75 | nominal | 10 | 10 | 1.00 | 17.12 | 0.031 | 5.00 | - | - |
| 18.75 | wrapped | 10 | 9 | 0.900 | 17.11 | 0.031 | 4.80 | 3.00 | collision:1 |

### Failures (sweep)

| T_traj | condition | seed | n_gates | time_s | group | cause | min_z_airborne | final_z | detail |
|---|---|---|---|---|---|---|---|---|---|
| 18.75 | wrapped | 3 | 3 | 13.64 | collision | gate | 0.152 | 1.16 | gate_id=3 dist_to_frame=0.096 (heuristic) |

## Final (T_traj=18.75, seeds 0--1, paired)

| T_traj | condition | n | n_success | success_rate | lap_time_mean_s | lap_time_std_s | gates_mean_all | gates_mean_failed | failure_causes |
|---|---|---|---|---|---|---|---|---|---|


### Paired outcomes (same seeds)

- n_pairs: 0
- both_success: 0
- nominal_only: 0
- wrapped_only: 0
- both_fail: 0
- lap_diff_mean_s: -

### Failures (final)

(no failures)
