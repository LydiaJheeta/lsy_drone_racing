## Experiment C -- T_traj=18.75, k_end=1, level 0, seeds 0-19. Baseline of each column: d=0 at the same scale.

### success rate

| delay \ u_weight_scale | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| 0 | 100% (20/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) |
| 1 | 0% (0/20) | 50% (10/20) | 65% (13/20) | 70% (14/20) |
| 2 | 0% (0/20) | 0% (0/20) | 0% (0/20) | 0% (0/20) |

### success on the seeds where the baseline succeeded

| delay \ u_weight_scale | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| 0 | 100% (20/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) |
| 1 | 0% (0/20) | 50% (10/20) | 65% (13/20) | 70% (14/20) |
| 2 | 0% (0/20) | 0% (0/20) | 0% (0/20) | 0% (0/20) |

### mean lap time of the successes [s]

| delay \ u_weight_scale | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| 0 | 17.12 | 17.13 | 17.15 | 17.12 |
| 1 | - | 17.26 | 17.29 | 17.17 |
| 2 | - | - | - | - |

### steps with |roll| or |pitch| command >= 0.49 rad, all episodes

| delay \ u_weight_scale | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| 0 | 50.3% | 32.3% | 7.3% | 0.0% |
| 1 | 86.7% | 94.4% | 96.1% | 95.4% |
| 2 | 79.8% | 89.6% | 92.6% | 84.2% |

### steps with thrust command > 0.8 N * k(t), all episodes

| delay \ u_weight_scale | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| 0 | 0.0% | 0.0% | 0.0% | 0.0% |
| 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 | 0.0% | 0.0% | 0.0% | 0.0% |

### mean gates passed, all episodes

| delay \ u_weight_scale | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| 0 | 5.00 | 5.00 | 5.00 | 5.00 |
| 1 | 0.150 | 3.15 | 3.80 | 4.10 |
| 2 | 0.000 | 0.250 | 0.550 | 0.000 |

### failure causes

| delay \ u_weight_scale | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| 0 | - | - | - | - |
| 1 | collision:15, fall:5 | collision:9, fall:1 | collision:7 | collision:6 |
| 2 | collision:5, fall:14, out_of_bounds:1 | collision:16, fall:4 | collision:20 | collision:20 |
