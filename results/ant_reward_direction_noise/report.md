| Metric | σ | mean cos | Uniform ± SE | D-LEVER ± SE | D − Uniform ± SE | Δ vs σ=0: D − Uniform ± SE |
|---|---:|---:|---:|---:|---:|---:|
| Mean task return | 0 | 1.000 | 451.0 ± 17.5 | 438.7 ± 15.5 | -12.3 ± 16.5 | +0.0 ± 0.0 |
| Mean task return | 0.25 | 0.982 | 443.4 ± 18.8 | 433.6 ± 16.1 | -9.8 ± 14.7 | +2.5 ± 2.4 |
| Mean task return | 0.5 | 0.934 | 427.8 ± 18.6 | 423.0 ± 13.8 | -4.9 ± 16.3 | +7.5 ± 1.0 |
| Mean task return | 1 | 0.869 | 408.1 ± 19.5 | 393.5 ± 16.2 | -14.5 ± 16.9 | -2.2 ± 5.1 |
| Mean task return | 2 | 0.849 | 401.2 ± 21.0 | 390.6 ± 16.3 | -10.6 ± 16.5 | +1.7 ± 3.3 |
| Mean task return | 4 | 0.839 | 394.0 ± 20.1 | 386.3 ± 16.1 | -7.7 ± 16.6 | +4.6 ± 1.5 |
| Bottom-25% tasks | 0 | 1.000 | 435.9 ± 19.0 | 420.0 ± 19.2 | -15.9 ± 19.3 | +0.0 ± 0.0 |
| Bottom-25% tasks | 0.25 | 0.982 | 423.1 ± 20.1 | 418.1 ± 16.6 | -4.9 ± 12.6 | +11.0 ± 8.7 |
| Bottom-25% tasks | 0.5 | 0.934 | 404.0 ± 22.4 | 405.2 ± 13.4 | +1.2 ± 17.5 | +17.1 ± 10.0 |
| Bottom-25% tasks | 1 | 0.869 | 371.1 ± 19.1 | 362.4 ± 14.4 | -8.8 ± 16.9 | +7.1 ± 5.8 |
| Bottom-25% tasks | 2 | 0.849 | 355.9 ± 21.9 | 343.4 ± 18.1 | -12.5 ± 18.2 | +3.4 ± 4.5 |
| Bottom-25% tasks | 4 | 0.839 | 336.4 ± 20.7 | 344.0 ± 12.4 | +7.6 ± 14.4 | +23.5 ± 8.7 |

Checkpoints: latest.pt of runs_2m_sphere/mo_ant (d seed 0: 1650k, d seed 1: 1700k, d seed 2: 1600k, d seed 3: 2000k, uniform seed 0: 1700k, uniform seed 1: 1700k, uniform seed 2: 1700k, uniform seed 3: 2000k). Each task's utility is the IQM (middle 6) of ten rollouts; one noise draw per rollout. Bottom-25% is the mean of the lowest 3 of 10 task IQMs. Mean ± SE across the four training seeds; differences are paired by seed. σ=0 vs the saved evaluation: 0–100% of rollouts identical per checkpoint (rollouts diverge chaotically under tiny float differences, e.g. CPU vs another GPU); checkpoint mean return within 1.07%. All σ levels of a checkpoint run on the same device here, so the σ comparison is internally consistent. The noise has relative scale σ like the Walker experiment, but is folded into the positive quadrant, so cosines are not comparable to Walker's high-dimensional cosines.
