## Experiment F, delay d=0 -- T_traj=18.75, k_end=1.0, level 0, seeds 0-19. R = diag(rpy_scale * [1,1,1], thrust_scale * 50).

### success rate

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | 100% (20/20) | 90% (18/20) | 95% (19/20) |
| 10 | 100% (20/20) | 100% (20/20) | 95% (19/20) |
| 30 | 100% (20/20) | 100% (20/20) | 100% (20/20) |

### maximum commanded thrust over all episodes [N]

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | 0.800 | 0.671 | 0.620 |
| 10 | 0.800 | 0.671 | 0.620 |
| 30 | 0.800 | 0.671 | 0.620 |

### steps with |roll| or |pitch| command >= 0.49 rad

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | 50.3% | 50.8% | 53.2% |
| 10 | 8.1% | 7.3% | 7.7% |
| 30 | 0.0% | 0.0% | 0.0% |

### mean lap time of the successes [s]

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | 17.12 | 17.12 | 17.12 |
| 10 | 17.14 | 17.15 | 17.15 |
| 30 | 17.12 | 17.12 | 17.12 |

### mean gates passed, all episodes

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | 5.00 | 4.70 | 4.90 |
| 10 | 5.00 | 5.00 | 4.90 |
| 30 | 5.00 | 5.00 | 5.00 |

### failure causes

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | - | collision:2 | collision:1 |
| 10 | - | - | collision:1 |
| 30 | - | - | - |

## Experiment F, delay d=1 -- T_traj=18.75, k_end=1.0, level 0, seeds 0-19. R = diag(rpy_scale * [1,1,1], thrust_scale * 50).

### success rate

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | 0% (0/20) | 5% (1/20) | 10% (2/20) |
| 10 | 10% (2/20) | 65% (13/20) | 20% (4/20) |
| 30 | 90% (18/20) | 85% (17/20) | 70% (14/20) |

### maximum commanded thrust over all episodes [N]

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | 0.800 | 0.671 | 0.621 |
| 10 | 0.800 | 0.671 | 0.621 |
| 30 | 0.800 | 0.671 | 0.621 |

### steps with |roll| or |pitch| command >= 0.49 rad

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | 86.7% | 93.5% | 93.8% |
| 10 | 88.8% | 96.1% | 94.9% |
| 30 | 94.0% | 95.3% | 95.4% |

### mean lap time of the successes [s]

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | - | 17.12 | 17.21 |
| 10 | 17.15 | 17.29 | 17.28 |
| 30 | 17.13 | 17.18 | 17.17 |

### mean gates passed, all episodes

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | 0.150 | 1.75 | 2.35 |
| 10 | 0.900 | 3.80 | 2.40 |
| 30 | 4.50 | 4.50 | 4.10 |

### failure causes

| rpy_weight_scale \ thrust_weight_scale | 1 | 10 | 30 |
|---|---|---|---|
| 1 | collision:15, fall:5 | collision:19 | collision:18 |
| 10 | collision:11, fall:6, out_of_bounds:1 | collision:7 | collision:16 |
| 30 | collision:2 | collision:3 | collision:6 |
