| Metric | σ | mean cos | Uniform ± SE | D-LEVER ± SE | D − Uniform ± SE | Δ vs σ=0: D − Uniform ± SE |
|---|---:|---:|---:|---:|---:|---:|
| Mean task return | 0 | 1.000 | 217.3 ± 29.6 | 244.1 ± 31.1 | +26.8 ± 37.9 | +0.0 ± 0.0 |
| Mean task return | 0.25 | 0.982 | 215.2 ± 27.6 | 248.5 ± 28.5 | +33.3 ± 35.3 | +6.5 ± 4.2 |
| Mean task return | 0.5 | 0.934 | 219.6 ± 26.0 | 254.6 ± 20.4 | +35.0 ± 30.4 | +8.3 ± 12.2 |
| Mean task return | 1 | 0.869 | 211.7 ± 29.4 | 252.7 ± 15.2 | +41.0 ± 34.7 | +14.2 ± 14.7 |
| Mean task return | 2 | 0.849 | 212.8 ± 27.7 | 250.0 ± 14.4 | +37.2 ± 31.3 | +10.4 ± 16.2 |
| Mean task return | 4 | 0.839 | 210.0 ± 27.4 | 249.2 ± 14.0 | +39.2 ± 30.7 | +12.4 ± 16.4 |
| Bottom-25% tasks | 0 | 1.000 | 157.5 ± 48.8 | 193.4 ± 55.7 | +35.9 ± 62.6 | +0.0 ± 0.0 |
| Bottom-25% tasks | 0.25 | 0.982 | 162.4 ± 49.1 | 207.5 ± 51.9 | +45.1 ± 66.4 | +9.2 ± 13.3 |
| Bottom-25% tasks | 0.5 | 0.934 | 183.6 ± 37.1 | 227.0 ± 36.9 | +43.4 ± 51.0 | +7.5 ± 18.8 |
| Bottom-25% tasks | 1 | 0.869 | 190.2 ± 34.4 | 221.0 ± 34.0 | +30.8 ± 52.7 | -5.1 ± 25.7 |
| Bottom-25% tasks | 2 | 0.849 | 188.8 ± 31.6 | 217.6 ± 28.6 | +28.8 ± 43.4 | -7.1 ± 27.8 |
| Bottom-25% tasks | 4 | 0.839 | 182.4 ± 33.8 | 223.0 ± 18.9 | +40.7 ± 38.4 | +4.8 ± 33.3 |

Checkpoints: latest.pt of runs_2m_sphere/mo_hopper (d seed 0: 2000k, d seed 1: 2000k, d seed 2: 2000k, d seed 4: 1750k, uniform seed 0: 2000k, uniform seed 1: 2000k, uniform seed 2: 2000k, uniform seed 4: 1700k). Each task's utility is the IQM (middle 6) of ten rollouts; one noise draw per rollout. Bottom-25% is the mean of the lowest 3 of 10 task IQMs. Mean ± SE across the four training seeds; differences are paired by seed. σ=0 vs the saved evaluation: 41–78% of rollouts identical per checkpoint (rollouts diverge chaotically under tiny float differences, e.g. CPU vs another GPU); checkpoint mean return within 0.00%. All σ levels of a checkpoint run on the same device here, so the σ comparison is internally consistent. The noise has relative scale σ like the Walker experiment, but is folded into the positive quadrant, so cosines are not comparable to Walker's high-dimensional cosines.
