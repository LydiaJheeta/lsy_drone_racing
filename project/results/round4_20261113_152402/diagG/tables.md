## Diagnosis G -- geometry of the FIXED reference spline at each gate pass

Gate opening: half-width 0.2 m; frame material out to 0.36 m. `in_plane_offset_m` < 0.2 goes through the opening; `clearance_to_frame_m` is 0 when the reference is inside the frame material. `approach_angle_to_normal_deg` is 0 for a reference crossing straight through.

Gate sequence (0-based ids, 5 passes): [0, 1, 2, 3, 1]

| T_traj | pass | gate_id | found_by | t_s | ref_speed_mps | dist_to_centre_m | in_plane_offset_m | clearance_to_frame_m | min_clearance_in_window_m | approach_angle_to_normal_deg | inside_opening | ref_accel_mps2 | required_thrust_N | max_required_thrust_in_window_N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 18.75 | 1 | 0 | plane_crossing | 3.65 | 0.934 | 0.031 | 0.024 | 0.176 | 0.175 | 12.84 | True | 0.189 | 0.423 | 0.428 |
| 18.75 | 2 | 1 | plane_crossing | 6.49 | 0.732 | 0.002 | 0.002 | 0.198 | 0.198 | 6.45 | True | 1.14 | 0.410 | 0.418 |
| 18.75 | 3 | 2 | plane_crossing | 9.52 | 0.374 | 0.036 | 0.034 | 0.166 | 0.161 | 18.35 | True | 0.460 | 0.437 | 0.441 |
| 18.75 | 4 | 3 | plane_crossing | 13.64 | 0.881 | 0.050 | 0.050 | 0.150 | 0.150 | 3.83 | True | 0.633 | 0.430 | 0.430 |
| 18.75 | 5 | 1 | plane_crossing | 17.05 | 0.613 | 0.001 | 0.000 | 0.200 | 0.199 | 4.11 | True | 0.262 | 0.426 | 0.426 |
| 16.00 | 1 | 0 | plane_crossing | 3.10 | 1.10 | 0.032 | 0.023 | 0.177 | 0.175 | 12.94 | True | 0.263 | 0.423 | 0.430 |
| 16.00 | 2 | 1 | plane_crossing | 5.55 | 0.854 | 0.008 | 0.002 | 0.198 | 0.198 | 6.54 | True | 1.58 | 0.405 | 0.418 |
| 16.00 | 3 | 2 | plane_crossing | 8.13 | 0.436 | 0.034 | 0.032 | 0.168 | 0.161 | 19.00 | True | 0.624 | 0.441 | 0.448 |
| 16.00 | 4 | 3 | plane_crossing | 11.63 | 1.03 | 0.050 | 0.050 | 0.150 | 0.150 | 4.11 | True | 0.870 | 0.431 | 0.431 |
| 16.00 | 5 | 1 | plane_crossing | 14.54 | 0.718 | 0.005 | 0.000 | 0.200 | 0.199 | 4.36 | True | 0.363 | 0.426 | 0.426 |

### Dynamic demand of the whole reference (drag ignored)

`k_end_that_removes_the_headroom` is the thrust scale at which the peak demand of the reference equals the actuator limit.

| T_traj | mass_kg | max_speed_mps | max_accel_mps2 | max_required_thrust_N | mean_required_thrust_N | thrust_max_N | headroom_at_peak | k_end_that_removes_the_headroom |
|---|---|---|---|---|---|---|---|---|
| 18.75 | 0.043 | 1.17 | 1.63 | 0.446 | 0.426 | 0.800 | 1.79 | 0.557 |
| 16.00 | 0.043 | 1.38 | 2.23 | 0.454 | 0.426 | 0.800 | 1.76 | 0.567 |

### Gate poses used (true positions of this episode)

| gate_id | position | yaw [deg] |
|---|---|---|
| 0 | [0.5, 0.25, 0.699999988079071] | -44.7 |
| 1 | [1.0499999523162842, 0.75, 1.2000000476837158] | 134.6 |
| 2 | [-1.0, -0.25, 0.699999988079071] | 179.9 |
| 3 | [0.0, -0.75, 1.2000000476837158] | 0.0 |
