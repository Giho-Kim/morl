# `runs_2m_sphere` 결과 스냅샷

2026-09-26 12:18 KST 기준. 입력은 `runs_2m_sphere`의 `metrics.csv`와 평가 `.npz`만 사용했다. 아래 수치는 같은 환경, seed, environment step에서 Uniform과 D를 짝지어 비교한 후 seed 평균을 낸 것이다. Return은 고정 평가 task별 rollout IQM을 집계한 값이다.

| 환경 | Steps | 짝지은 seeds | Mean return: Uniform / D | CVaR25: Uniform / D | CVaR10: Uniform / D | Rollout logdet: Uniform / D |
|---|---:|---|---:|---:|---:|---:|
| Ant | 1.5M | 0, 1, 2, 3 | 434.55 / 424.50 | 424.25 / 407.93 | 417.92 / 399.99 | 23.99 / 23.60 |
| Ant | 2M | 3 | 399.90 / 425.11 | 380.57 / 404.07 | 368.70 / 396.07 | 24.69 / 23.97 |
| Hopper | 0.8M | 0, 1, 2, 4 | 228.00 / 247.89 | 211.93 / 215.58 | 199.12 / 187.03 | 18.94 / 19.98 |
| Hopper | 1.5M | 0, 1, 2, 4 | 191.70 / 265.07 | 155.14 / 248.75 | 143.42 / 237.33 | 18.59 / 19.27 |
| Hopper | 2M | 0, 1, 2 | 213.45 / 229.58 | 160.52 / 165.23 | 151.47 / 149.07 | 18.45 / 21.21 |

2M까지 양쪽 방법이 완료된 seed는 Ant 3번 하나, Hopper 0번, 1번, 2번이다. 나머지 run은 2M 평가에 도달하지 않아 2M 평균에 포함하지 않았다. Ant D seed 4는 step 0 평가만 있으므로 학습 추이 그림에서 제외했다. 2M 결과의 표본 수가 작고 Hopper의 seed별 return 변동이 커서 방법의 우위를 결론내리기에는 이르다.

## 평가 곡선

| 환경 | Seed별: 체크포인트 4개 평균 | 전체 평균 ± SE: 체크포인트 4개 평균 |
|---|---|---|
| Ant | [Return·CVaR·train logdet](plots/ant_4ckpt_seed.png), [방향별 reward](plots/ant_dir_4ckpt_seed.png) | [Return·CVaR·train logdet](plots/ant_4ckpt_mean_se.png), [방향별 reward](plots/ant_dir_4ckpt_mean_se.png) |
| Hopper | [Return·CVaR·train logdet](plots/hopper_4ckpt_seed.png), [방향별 reward](plots/hopper_dir_4ckpt_seed.png) | [Return·CVaR·train logdet](plots/hopper_4ckpt_mean_se.png), [방향별 reward](plots/hopper_dir_4ckpt_mean_se.png) |

Step 0 평가는 제외하고 50k 간격의 연속된 체크포인트를 네 개씩 묶는다. 먼저 각 seed와 방법에서 네 평가값의 평균을 계산하고, 그다음 같은 묶음을 완료한 짝지은 seed들로 방법별 평균과 seed 간 standard error를 계산한다. 가로축은 묶음의 마지막 step이다. 하단 패널은 각 묶음에 포함된 짝지은 seed 수다. Seed가 하나인 구간은 SE 음영을 표시하지 않았다.

마지막 2M 묶음(1.85M, 1.90M, 1.95M, 2.00M)의 mean return은 Ant Uniform 388.62 / D 415.58(짝지은 seed 1개), Hopper Uniform 222.96 / D 249.32(짝지은 seed 3개)이다.

원래 체크포인트별 곡선도 보관했다: [Ant seed별](plots/ant_seed.png), [Ant 평균±SE](plots/ant_mean_se.png), [Hopper seed별](plots/hopper_seed.png), [Hopper 평균±SE](plots/hopper_mean_se.png). 위쪽 표의 수치는 4개 평균을 적용하지 않은 해당 step의 평가값이다.
