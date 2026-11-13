## Sweep (level 0, seeds 0-9, same seeds in all rows)

| T_traj | condition | n | n_success | success_rate | lap_time_mean_s | lap_time_std_s | gates_mean_all | gates_mean_failed | failure_causes |
|---|---|---|---|---|---|---|---|---|---|
| 18.75 | nominal | 10 | 10 | 1.00 | 17.12 | 0.031 | 5.00 | - | - |
| 18.75 | wrapped | 10 | 0 | 0.000 | - | - | 0.000 | 0.000 | collision:3, fall:6, out_of_bounds:1 |

### Failures (sweep)

| T_traj | condition | seed | n_gates | time_s | group | cause | min_z_airborne | final_z | detail |
|---|---|---|---|---|---|---|---|---|---|
| 18.75 | wrapped | 0 | 0 | 3.28 | collision | obstacle | 0.058 | 0.499 | obstacle_id=0 xy_dist=0.070 (heuristic) |
| 18.75 | wrapped | 1 | 0 | 3.00 | out_of_bounds | out_of_bounds | 0.039 | 0.740 | axes=[1] pos=[0.8930000066757202, 3.01200008392334, 0.7400000095367432] |
| 18.75 | wrapped | 2 | 0 | 1.02 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 18.75 | wrapped | 3 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 18.75 | wrapped | 4 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 18.75 | wrapped | 5 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 18.75 | wrapped | 6 | 0 | 2.98 | collision | gate | 0.004 | 0.381 | gate_id=0 dist_to_frame=0.078 (heuristic) |
| 18.75 | wrapped | 7 | 0 | 2.64 | collision | gate | 0.021 | 0.471 | gate_id=3 dist_to_frame=0.384 (heuristic) |
| 18.75 | wrapped | 8 | 0 | 1.00 | fall | floor | -0.000 | -0.000 | z=-0.0001 < z_low=0.0 |
| 18.75 | wrapped | 9 | 0 | 0.960 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |

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
