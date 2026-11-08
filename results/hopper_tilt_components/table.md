| Model | Rank corr score–norm² | Rank corr score–direction | Remove norm: TV | Remove direction: TV |
|---|---:|---:|---:|---:|
| Uniform (n=4) | -0.278 ± 0.253 | 0.941 ± 0.023 | 7.8% ± 5.3%p | 11.8% ± 1.9%p |
| D-LEVER (n=4) | -0.256 ± 0.192 | 0.964 ± 0.029 | 1.6% ± 0.1%p | 11.6% ± 1.3%p |

Saved training design at the same step (actual candidates, μ and Gram; no resampling):

| Model | Rank corr score–norm² | Rank corr score–direction | Remove norm: TV | Remove direction: TV |
|---|---:|---:|---:|---:|
| Uniform (n=4) | -0.291 ± 0.255 | 0.942 ± 0.022 | 7.9% ± 5.3%p | 11.8% ± 2.0%p |
| D-LEVER (n=4) | -0.314 ± 0.178 | 0.972 ± 0.021 | 1.6% ± 0.1%p | 11.4% ± 1.3%p |

Score ell = ||μ||² · uᵀG⁻¹u with u = μ/||μ|| (exact). Gram: the training EMA Gram saved at the checkpoint step, frozen (D-LEVER seed 0: 2000k, D-LEVER seed 1: 2000k, D-LEVER seed 2: 2000k, D-LEVER seed 4: 1750k, Uniform seed 0: 2000k, Uniform seed 1: 2000k, Uniform seed 2: 2000k, Uniform seed 4: 1700k). Each checkpoint was evaluated at five candidate seeds (42–46, 2560 prior draws each) × five fresh 8-state reset banks (42–46); results are averaged within the checkpoint, then mean ± SE across training seeds 0, 1, 2, 4. Both methods use the D-LEVER mixture with η = 0.9 (uniform share 0.1); for Uniform this is counterfactual. TV is on the candidate draw probabilities before without-replacement depletion. μ includes the fixed common-term SF coordinate.
