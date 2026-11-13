## Sweep (level 0, seeds 0-19, same seeds in all rows)

| T_traj | condition | n | n_success | success_rate | lap_time_mean_s | lap_time_std_s | gates_mean_all | gates_mean_failed | failure_causes |
|---|---|---|---|---|---|---|---|---|---|
| 18.75 | nominal | 20 | 20 | 1.00 | 17.12 | 0.030 | 5.00 | - | - |
| 18.75 | wrapped | 20 | 0 | 0.000 | - | - | 0.000 | 0.000 | collision:3, fall:17 |
| 16.00 | nominal | 20 | 18 | 0.900 | 14.63 | 0.022 | 4.80 | 3.00 | collision:2 |
| 16.00 | wrapped | 20 | 0 | 0.000 | - | - | 0.000 | 0.000 | collision:2, fall:18 |
| 14.00 | nominal | 20 | 13 | 0.650 | 12.80 | 0.031 | 4.00 | 2.14 | collision:7 |
| 14.00 | wrapped | 20 | 0 | 0.000 | - | - | 0.000 | 0.000 | collision:1, fall:19 |
| 12.00 | nominal | 20 | 13 | 0.650 | 10.98 | 0.030 | 3.65 | 1.14 | collision:7 |
| 12.00 | wrapped | 20 | 0 | 0.000 | - | - | 0.050 | 0.050 | collision:17, fall:3 |
| 10.00 | nominal | 20 | 2 | 0.100 | 9.14 | 0.020 | 1.65 | 1.28 | collision:18 |
| 10.00 | wrapped | 20 | 0 | 0.000 | - | - | 0.050 | 0.050 | collision:10, fall:10 |

### Failures (sweep)

| T_traj | condition | seed | n_gates | time_s | group | cause | min_z_airborne | final_z | detail |
|---|---|---|---|---|---|---|---|---|---|
| 18.75 | wrapped | 0 | 0 | 6.58 | collision | gate | 0.052 | 0.623 | gate_id=0 dist_to_frame=0.066 (heuristic) |
| 18.75 | wrapped | 1 | 0 | 3.94 | collision | obstacle | 0.036 | 0.103 | obstacle_id=3 xy_dist=0.467 (heuristic) |
| 18.75 | wrapped | 2 | 0 | 3.62 | collision | gate | 0.013 | 0.579 | gate_id=0 dist_to_frame=0.033 (heuristic) |
| 18.75 | wrapped | 3 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 18.75 | wrapped | 4 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 18.75 | wrapped | 5 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 18.75 | wrapped | 6 | 0 | 1.40 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 18.75 | wrapped | 7 | 0 | 2.92 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 18.75 | wrapped | 8 | 0 | 1.00 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 18.75 | wrapped | 9 | 0 | 0.960 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 18.75 | wrapped | 10 | 0 | 1.44 | fall | floor | -0.001 | -0.001 | z=-0.0008 < z_low=0.0 |
| 18.75 | wrapped | 11 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 18.75 | wrapped | 12 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 18.75 | wrapped | 13 | 0 | 2.16 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 18.75 | wrapped | 14 | 0 | 3.96 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 18.75 | wrapped | 15 | 0 | 1.18 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 18.75 | wrapped | 16 | 0 | 0.940 | fall | floor | - | -0.001 | z=-0.0007 < z_low=0.0 |
| 18.75 | wrapped | 17 | 0 | 0.980 | fall | floor | -0.000 | -0.000 | z=-0.0003 < z_low=0.0 |
| 18.75 | wrapped | 18 | 0 | 1.68 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 18.75 | wrapped | 19 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | nominal | 10 | 3 | 11.68 | collision | gate | 0.152 | 1.19 | gate_id=3 dist_to_frame=0.082 (heuristic) |
| 16.00 | nominal | 18 | 3 | 11.68 | collision | gate | 0.152 | 1.19 | gate_id=3 dist_to_frame=0.073 (heuristic) |
| 16.00 | wrapped | 0 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 1 | 0 | 1.80 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 2 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 3 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 4 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 5 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 6 | 0 | 2.42 | collision | obstacle | 0.039 | 0.566 | obstacle_id=0 xy_dist=0.093 (heuristic) |
| 16.00 | wrapped | 7 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 8 | 0 | 3.42 | fall | floor | -0.001 | -0.001 | z=-0.0007 < z_low=0.0 |
| 16.00 | wrapped | 9 | 0 | 0.940 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 10 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 11 | 0 | 0.860 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 12 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 13 | 0 | 2.20 | collision | gate | 0.072 | 0.505 | gate_id=2 dist_to_frame=0.085 (heuristic) |
| 16.00 | wrapped | 14 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 15 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 16 | 0 | 0.860 | fall | floor | -0.000 | -0.000 | z=-0.0005 < z_low=0.0 |
| 16.00 | wrapped | 17 | 0 | 1.10 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 18 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0007 < z_low=0.0 |
| 16.00 | wrapped | 19 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0008 < z_low=0.0 |
| 14.00 | nominal | 2 | 1 | 4.88 | collision | gate | 0.151 | 1.20 | gate_id=1 dist_to_frame=0.090 (heuristic) |
| 14.00 | nominal | 4 | 3 | 10.22 | collision | gate | 0.155 | 1.19 | gate_id=3 dist_to_frame=0.101 (heuristic) |
| 14.00 | nominal | 5 | 1 | 4.92 | collision | gate | 0.157 | 1.20 | gate_id=1 dist_to_frame=0.081 (heuristic) |
| 14.00 | nominal | 7 | 3 | 10.24 | collision | gate | 0.152 | 1.19 | gate_id=3 dist_to_frame=0.074 (heuristic) |
| 14.00 | nominal | 11 | 3 | 10.20 | collision | gate | 0.153 | 1.20 | gate_id=3 dist_to_frame=0.088 (heuristic) |
| 14.00 | nominal | 12 | 3 | 10.20 | collision | gate | 0.153 | 1.20 | gate_id=3 dist_to_frame=0.080 (heuristic) |
| 14.00 | nominal | 18 | 1 | 4.92 | collision | gate | 0.155 | 1.21 | gate_id=1 dist_to_frame=0.076 (heuristic) |
| 14.00 | wrapped | 0 | 0 | 1.06 | fall | floor | -0.000 | -0.000 | z=-0.0001 < z_low=0.0 |
| 14.00 | wrapped | 1 | 0 | 1.16 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 2 | 0 | 0.960 | fall | floor | -0.001 | -0.001 | z=-0.0008 < z_low=0.0 |
| 14.00 | wrapped | 3 | 0 | 1.28 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 4 | 0 | 0.960 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 5 | 0 | 0.940 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 6 | 0 | 1.00 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 7 | 0 | 1.12 | fall | floor | -0.000 | -0.000 | z=-0.0001 < z_low=0.0 |
| 14.00 | wrapped | 8 | 0 | 2.90 | collision | gate | 0.013 | 0.476 | gate_id=0 dist_to_frame=0.254 (heuristic) |
| 14.00 | wrapped | 9 | 0 | 2.86 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 10 | 0 | 0.980 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 11 | 0 | 0.940 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 12 | 0 | 0.940 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 14.00 | wrapped | 13 | 0 | 0.960 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 14 | 0 | 4.24 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 15 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 16 | 0 | 0.940 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 14.00 | wrapped | 17 | 0 | 1.02 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 14.00 | wrapped | 18 | 0 | 0.960 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 14.00 | wrapped | 19 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 12.00 | nominal | 1 | 1 | 4.22 | collision | gate | 0.152 | 1.22 | gate_id=1 dist_to_frame=0.081 (heuristic) |
| 12.00 | nominal | 6 | 1 | 4.18 | collision | gate | 0.152 | 1.21 | gate_id=1 dist_to_frame=0.090 (heuristic) |
| 12.00 | nominal | 12 | 1 | 4.24 | collision | gate | 0.154 | 1.21 | gate_id=1 dist_to_frame=0.091 (heuristic) |
| 12.00 | nominal | 13 | 3 | 8.76 | collision | gate | 0.159 | 1.20 | gate_id=3 dist_to_frame=0.089 (heuristic) |
| 12.00 | nominal | 15 | 1 | 4.24 | collision | gate | 0.153 | 1.21 | gate_id=1 dist_to_frame=0.098 (heuristic) |
| 12.00 | nominal | 18 | 0 | 2.42 | collision | gate | 0.156 | 0.733 | gate_id=0 dist_to_frame=0.088 (heuristic) |
| 12.00 | nominal | 19 | 1 | 4.18 | collision | gate | 0.153 | 1.21 | gate_id=1 dist_to_frame=0.076 (heuristic) |
| 12.00 | wrapped | 0 | 0 | 2.58 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 12.00 | wrapped | 1 | 0 | 1.42 | collision | gate | 0.087 | 0.285 | gate_id=2 dist_to_frame=0.073 (heuristic) |
| 12.00 | wrapped | 2 | 0 | 2.20 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 12.00 | wrapped | 3 | 0 | 1.50 | collision | gate | 0.085 | 0.343 | gate_id=2 dist_to_frame=0.086 (heuristic) |
| 12.00 | wrapped | 4 | 0 | 1.50 | collision | gate | 0.086 | 0.312 | gate_id=2 dist_to_frame=0.065 (heuristic) |
| 12.00 | wrapped | 5 | 0 | 2.82 | collision | gate | 0.155 | 0.769 | gate_id=0 dist_to_frame=0.076 (heuristic) |
| 12.00 | wrapped | 6 | 0 | 3.60 | collision | gate | 0.105 | 1.03 | gate_id=0 dist_to_frame=0.087 (heuristic) |
| 12.00 | wrapped | 7 | 0 | 1.56 | collision | gate | 0.127 | 0.317 | gate_id=2 dist_to_frame=0.066 (heuristic) |
| 12.00 | wrapped | 8 | 0 | 1.72 | collision | obstacle | 0.123 | 0.196 | obstacle_id=0 xy_dist=0.063 (heuristic) |
| 12.00 | wrapped | 9 | 0 | 0.980 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 12.00 | wrapped | 10 | 0 | 2.72 | collision | gate | 0.080 | 0.470 | gate_id=0 dist_to_frame=0.083 (heuristic) |
| 12.00 | wrapped | 11 | 0 | 2.18 | collision | obstacle | 0.141 | 0.786 | obstacle_id=0 xy_dist=0.087 (heuristic) |
| 12.00 | wrapped | 12 | 0 | 1.54 | collision | gate | 0.073 | 0.345 | gate_id=2 dist_to_frame=0.080 (heuristic) |
| 12.00 | wrapped | 13 | 0 | 1.48 | collision | gate | 0.037 | 0.353 | gate_id=2 dist_to_frame=0.065 (heuristic) |
| 12.00 | wrapped | 14 | 0 | 3.46 | collision | gate | 0.016 | 1.00 | gate_id=0 dist_to_frame=0.083 (heuristic) |
| 12.00 | wrapped | 15 | 0 | 1.50 | collision | gate | 0.082 | 0.315 | gate_id=2 dist_to_frame=0.032 (heuristic) |
| 12.00 | wrapped | 16 | 0 | 1.42 | collision | gate | 0.080 | 0.298 | gate_id=2 dist_to_frame=0.055 (heuristic) |
| 12.00 | wrapped | 17 | 0 | 1.86 | collision | obstacle | 0.094 | 0.316 | obstacle_id=2 xy_dist=0.070 (heuristic) |
| 12.00 | wrapped | 18 | 0 | 2.02 | collision | gate | 0.153 | 0.645 | gate_id=0 dist_to_frame=0.062 (heuristic) |
| 12.00 | wrapped | 19 | 1 | 2.24 | collision | gate | 0.089 | 0.549 | gate_id=0 dist_to_frame=0.070 (heuristic) |
| 10.00 | nominal | 0 | 3 | 7.30 | collision | gate | 0.157 | 1.20 | gate_id=3 dist_to_frame=0.091 (heuristic) |
| 10.00 | nominal | 2 | 2 | 3.58 | collision | gate | 0.161 | 1.21 | gate_id=1 dist_to_frame=0.089 (heuristic) |
| 10.00 | nominal | 3 | 1 | 3.46 | collision | gate | 0.157 | 1.20 | gate_id=1 dist_to_frame=0.075 (heuristic) |
| 10.00 | nominal | 4 | 1 | 3.50 | collision | gate | 0.158 | 1.21 | gate_id=1 dist_to_frame=0.082 (heuristic) |
| 10.00 | nominal | 5 | 3 | 7.30 | collision | gate | 0.157 | 1.20 | gate_id=3 dist_to_frame=0.090 (heuristic) |
| 10.00 | nominal | 6 | 1 | 3.50 | collision | gate | 0.152 | 1.21 | gate_id=1 dist_to_frame=0.072 (heuristic) |
| 10.00 | nominal | 7 | 1 | 3.54 | collision | gate | 0.155 | 1.21 | gate_id=1 dist_to_frame=0.092 (heuristic) |
| 10.00 | nominal | 8 | 0 | 2.02 | collision | gate | 0.152 | 0.740 | gate_id=0 dist_to_frame=0.086 (heuristic) |
| 10.00 | nominal | 9 | 1 | 3.52 | collision | gate | 0.152 | 1.22 | gate_id=1 dist_to_frame=0.071 (heuristic) |
| 10.00 | nominal | 11 | 1 | 3.52 | collision | gate | 0.159 | 1.22 | gate_id=1 dist_to_frame=0.077 (heuristic) |
| 10.00 | nominal | 12 | 3 | 7.30 | collision | gate | 0.155 | 1.20 | gate_id=3 dist_to_frame=0.086 (heuristic) |
| 10.00 | nominal | 13 | 1 | 3.50 | collision | gate | 0.156 | 1.21 | gate_id=1 dist_to_frame=0.067 (heuristic) |
| 10.00 | nominal | 14 | 0 | 2.02 | collision | gate | 0.155 | 0.741 | gate_id=0 dist_to_frame=0.080 (heuristic) |
| 10.00 | nominal | 15 | 0 | 2.04 | collision | gate | 0.153 | 0.745 | gate_id=0 dist_to_frame=0.095 (heuristic) |
| 10.00 | nominal | 16 | 1 | 3.52 | collision | gate | 0.155 | 1.21 | gate_id=1 dist_to_frame=0.076 (heuristic) |
| 10.00 | nominal | 17 | 0 | 2.04 | collision | gate | 0.152 | 0.747 | gate_id=0 dist_to_frame=0.091 (heuristic) |
| 10.00 | nominal | 18 | 1 | 3.52 | collision | gate | 0.156 | 1.21 | gate_id=1 dist_to_frame=0.077 (heuristic) |
| 10.00 | nominal | 19 | 3 | 7.30 | collision | gate | 0.153 | 1.19 | gate_id=3 dist_to_frame=0.090 (heuristic) |
| 10.00 | wrapped | 0 | 0 | 1.08 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 10.00 | wrapped | 1 | 0 | 1.86 | collision | gate | 0.108 | 0.433 | gate_id=3 dist_to_frame=0.410 (heuristic) |
| 10.00 | wrapped | 2 | 0 | 1.02 | fall | floor | -0.001 | -0.001 | z=-0.0006 < z_low=0.0 |
| 10.00 | wrapped | 3 | 0 | 2.54 | collision | gate | 0.087 | 0.711 | gate_id=0 dist_to_frame=0.068 (heuristic) |
| 10.00 | wrapped | 4 | 0 | 1.96 | collision | gate | 0.089 | 0.817 | gate_id=0 dist_to_frame=0.069 (heuristic) |
| 10.00 | wrapped | 5 | 0 | 1.02 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 10.00 | wrapped | 6 | 0 | 1.74 | collision | obstacle | 0.009 | 0.534 | obstacle_id=0 xy_dist=0.093 (heuristic) |
| 10.00 | wrapped | 7 | 0 | 5.46 | collision | obstacle | 0.089 | 0.409 | obstacle_id=2 xy_dist=0.095 (heuristic) |
| 10.00 | wrapped | 8 | 0 | 2.10 | collision | gate | 0.128 | 0.545 | gate_id=0 dist_to_frame=0.084 (heuristic) |
| 10.00 | wrapped | 9 | 0 | 2.18 | collision | gate | 0.162 | 0.795 | gate_id=0 dist_to_frame=0.093 (heuristic) |
| 10.00 | wrapped | 10 | 0 | 2.34 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 10.00 | wrapped | 11 | 0 | 1.08 | fall | floor | -0.000 | -0.000 | z=-0.0000 < z_low=0.0 |
| 10.00 | wrapped | 12 | 0 | 0.960 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 10.00 | wrapped | 13 | 0 | 2.42 | collision | gate | 0.167 | 0.848 | gate_id=0 dist_to_frame=0.090 (heuristic) |
| 10.00 | wrapped | 14 | 0 | 1.98 | collision | obstacle | 0.165 | 0.562 | obstacle_id=0 xy_dist=0.074 (heuristic) |
| 10.00 | wrapped | 15 | 0 | 2.74 | collision | obstacle | 0.163 | 0.683 | obstacle_id=2 xy_dist=0.058 (heuristic) |
| 10.00 | wrapped | 16 | 0 | 1.10 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 10.00 | wrapped | 17 | 0 | 0.960 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 10.00 | wrapped | 18 | 0 | 1.00 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 10.00 | wrapped | 19 | 1 | 4.28 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |

## Final (T_traj=16, seeds 0-49, paired)

| T_traj | condition | n | n_success | success_rate | lap_time_mean_s | lap_time_std_s | gates_mean_all | gates_mean_failed | failure_causes |
|---|---|---|---|---|---|---|---|---|---|
| 16.00 | nominal | 50 | 44 | 0.880 | 14.63 | 0.029 | 4.68 | 2.33 | collision:6 |
| 16.00 | wrapped | 50 | 0 | 0.000 | - | - | 0.000 | 0.000 | collision:3, fall:46, out_of_bounds:1 |

### Paired outcomes (same seeds)

- n_pairs: 50
- both_success: 0
- nominal_only: 44
- wrapped_only: 0
- both_fail: 6
- lap_diff_mean_s: -

### Failures (final)

| T_traj | condition | seed | n_gates | time_s | group | cause | min_z_airborne | final_z | detail |
|---|---|---|---|---|---|---|---|---|---|
| 16.00 | nominal | 10 | 3 | 11.68 | collision | gate | 0.152 | 1.19 | gate_id=3 dist_to_frame=0.082 (heuristic) |
| 16.00 | nominal | 18 | 3 | 11.68 | collision | gate | 0.152 | 1.19 | gate_id=3 dist_to_frame=0.073 (heuristic) |
| 16.00 | nominal | 24 | 3 | 11.68 | collision | gate | 0.154 | 1.19 | gate_id=3 dist_to_frame=0.092 (heuristic) |
| 16.00 | nominal | 27 | 3 | 11.66 | collision | gate | 0.153 | 1.20 | gate_id=3 dist_to_frame=0.098 (heuristic) |
| 16.00 | nominal | 35 | 1 | 5.64 | collision | gate | 0.150 | 1.20 | gate_id=1 dist_to_frame=0.091 (heuristic) |
| 16.00 | nominal | 45 | 1 | 5.66 | collision | gate | 0.151 | 1.21 | gate_id=1 dist_to_frame=0.093 (heuristic) |
| 16.00 | wrapped | 0 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 1 | 0 | 1.80 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 2 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 3 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 4 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 5 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 6 | 0 | 2.42 | collision | obstacle | 0.039 | 0.566 | obstacle_id=0 xy_dist=0.093 (heuristic) |
| 16.00 | wrapped | 7 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 8 | 0 | 3.42 | fall | floor | -0.001 | -0.001 | z=-0.0007 < z_low=0.0 |
| 16.00 | wrapped | 9 | 0 | 0.940 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 10 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 11 | 0 | 0.860 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 12 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 13 | 0 | 2.20 | collision | gate | 0.072 | 0.505 | gate_id=2 dist_to_frame=0.085 (heuristic) |
| 16.00 | wrapped | 14 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 15 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 16 | 0 | 0.860 | fall | floor | -0.000 | -0.000 | z=-0.0005 < z_low=0.0 |
| 16.00 | wrapped | 17 | 0 | 1.10 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 18 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0007 < z_low=0.0 |
| 16.00 | wrapped | 19 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0008 < z_low=0.0 |
| 16.00 | wrapped | 20 | 0 | 2.96 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 21 | 0 | 0.860 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 22 | 0 | 0.840 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 23 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 16.00 | wrapped | 24 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 25 | 0 | 0.940 | fall | floor | -0.000 | -0.000 | z=-0.0005 < z_low=0.0 |
| 16.00 | wrapped | 26 | 0 | 0.860 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 27 | 0 | 0.980 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 16.00 | wrapped | 28 | 0 | 0.920 | fall | floor | -0.000 | -0.000 | z=-0.0000 < z_low=0.0 |
| 16.00 | wrapped | 29 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 16.00 | wrapped | 30 | 0 | 0.860 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 31 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 32 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 33 | 0 | 0.960 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
| 16.00 | wrapped | 34 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 35 | 0 | 0.860 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 36 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 37 | 0 | 3.16 | collision | gate | 0.023 | 0.770 | gate_id=0 dist_to_frame=0.081 (heuristic) |
| 16.00 | wrapped | 38 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 39 | 0 | 1.58 | fall | floor | -0.001 | -0.001 | z=-0.0008 < z_low=0.0 |
| 16.00 | wrapped | 40 | 0 | 0.920 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 41 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 42 | 0 | 4.28 | out_of_bounds | out_of_bounds | 0.003 | 0.589 | axes=[0] pos=[3.0940001010894775, -1.190000057220459, 0.5889999866485596] |
| 16.00 | wrapped | 43 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 44 | 0 | 0.860 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 45 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0008 < z_low=0.0 |
| 16.00 | wrapped | 46 | 0 | 0.940 | fall | floor | -0.000 | -0.000 | z=-0.0005 < z_low=0.0 |
| 16.00 | wrapped | 47 | 0 | 0.900 | fall | floor | -0.001 | -0.001 | z=-0.0010 < z_low=0.0 |
| 16.00 | wrapped | 48 | 0 | 0.860 | fall | floor | -0.001 | -0.001 | z=-0.0008 < z_low=0.0 |
| 16.00 | wrapped | 49 | 0 | 0.880 | fall | floor | -0.001 | -0.001 | z=-0.0009 < z_low=0.0 |
