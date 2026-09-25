# D-LEVER with conditional SF-SAC — Ant and Hopper

기존 SF Double-Q 구현을 conditional SF-SAC로 변경한 모델 프리 실험 코드입니다.
두 환경 모두 actor `pi(a|s,z)`와 twin SF critics `psi_i(s,a,z)`를 학습합니다.
환경마다 별개의 conditional model을 학습하며, 한 환경 내 모든 z가 모델을 공유합니다.

| CLI 환경 | 공식 환경 ID | 행동 / actor | 보상 차원 | 기본 steps |
|---|---|---|---:|---:|
| `mo_hopper` | `mo-hopper-2obj-v5` | 연속 3 / tanh Gaussian | 공식 2D → 학습 feature 3D | 1,000,000 |
| `mo_ant` | `mo-ant-2obj-v5` | 연속 8 / tanh Gaussian | 공식 2D → 학습 feature 3D | 1,000,000 |

Hopper와 Ant는 공식 2-objective v5 환경입니다.

## 설치와 실행

**Python 3.12**에서 검증했습니다. 예:

```bash
conda create -n dlever-sfsac python=3.12 -y
conda activate dlever-sfsac
cd dlever_bench
python -m pip install --upgrade pip
# CPU 버전. GPU 사용 시 이 줄 대신 해당 시스템에 맞는 PyTorch 2.6.0 설치.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python run_suite.py --smoke --seeds 0 --output smoke_sfsac
```

MuJoCo Python 패키지는 requirements에 포함됩니다. 별도 MuJoCo license나 rendering은
필요하지 않습니다. Linux에서 `libGL.so.1` 관련 import 오류가 있으면 시스템 `libgl1`이 필요합니다.

**두 환경 × Uniform/D-LEVER × 5 seeds:**

```bash
python run_suite.py --seeds 0 1 2 3 4 --output runs_sfsac
```

개별 실행 / 추가 ablation:

```bash
python train.py --env mo_ant --method d --seed 0 --output ant_runs
python run_suite.py --methods uniform d a e td --seeds 0 1 2 3 4 --output all_methods
python run_suite.py --methods uniform d --seeds 0 --steps 20000 --learning-starts 1000 --eval-every 10000 --output pilot
```

각 run을 별도 프로세스로 순차 실행합니다. `--device cuda` 지정도 가능합니다.
기존 run 폴더를 덮어쓰지 않습니다. smoke는 160 steps, 짧은 horizon, 작은 모델이므로
수렴·성능 실험이 아닙니다. 기본 학습 예산과 hyperparameter도 성능을 튜닝한 값은 아닙니다.
기본 학습 시작 시점과 평가 간격은 각각 50,000 environment steps입니다
(`--learning-starts 50000`, `--eval-every 50000`).

## SF-SAC 정의: 보상 SF와 엔트로피를 분리

공식 환경이 반환하는 2D 보상을 매 transition에서 `[방향 1, 방향 2, 공통항]`의
3D feature `phi_t`로 분해하고 `r_z = z^T phi_t`를 사용합니다.
모델, transition table, 환경 내부 reward table, DP, scripted policy 없이 `reset/step` 경험과
replay TD로만 학습합니다. 초기 상태 bank는 별도 reset 표본입니다.

각 critic은 d차원 **순수 reward SF**와 1차원 **미래 entropy return**을 출력합니다:

```text
psi(s,a,z) = E[sum_{t>=0} gamma^t phi_t | s0=s, a0=a, pi_z]
h(s,a,z)   = E[sum_{t>=1} gamma^t (-log pi_z(a_t|s_t)) | s0=s, a0=a]
Q_soft_i(s,a,z) = z^T psi_i(s,a,z) + temperature * h_i(s,a,z)
```

현재 action의 entropy는 h에 포함하지 않습니다. `--temperature` 기본값 `0.1`은 자동
temperature tuning의 초기값입니다. `log(temperature)`를 매 update 학습하며 target entropy는
이산 환경에서 `0.98*log(|A|)`, 연속 환경에서 `-|A|`입니다.
SAC의 entropy regularization은 양쪽 비교군에 동일하게 적용됩니다. 따라서 학습 목적은
순수 return 최대화에 entropy 항이 추가된 목적이고, 평가 return에는 entropy를 더하지 않습니다.

다음 action에서 `Q_soft`가 작은 target critic의 **전체 (psi,h) 쌍**을 선택합니다.
SF 좌표별 minimum은 사용하지 않습니다.

```text
y_psi = phi + gamma * (1-terminated) * E_{a'~pi_z}[psi_target_selected(s',a',z)]
y_h   =       gamma * (1-terminated) * E_{a'~pi_z}[h_target_selected(s',a',z) - log pi_z(a'|s')]
critic_loss = sum_i [MSE(psi_i,y_psi) + temperature^2 * MSE(h_i,y_h)]
actor_loss  = E_{a~pi_z}[temperature * log pi_z(a|s) - min_i Q_soft_i(s,a,z)]
```

`z^T y_psi + temperature*y_h`는 통상적인 clipped-twin SAC scalar target과 정확히 일치합니다.
벡터 회귀 loss는 scalar Q loss 하나와 동일하지 않습니다. 각 보상 성분을 개별 학습하는
SF parameterization입니다. twin 선택으로 생기는 finite-sample/approximation bias는 존재합니다.

- **이산**: 모든 action에 대해 위 기댓값과 actor loss를 확률 가중 합으로 정확히 계산합니다.
- **연속**: reparameterized tanh Gaussian 표본으로 계산합니다. tanh와 action scaling의
  log-density Jacobian을 모두 반영합니다. actor gradient는 critic의 action 입력을 통과합니다.
- target은 critic만 Polyak update합니다. target action은 현재 actor가 같은 z로 생성합니다.
- warmup은 random action, 이후 behavior는 SAC stochastic actor입니다. epsilon-greedy는 제거했습니다.

## D-LEVER 구현과 비교 protocol

기본적으로 episode 시작마다 behavior z는 원래 prior에서 추출합니다. `--tilted-behavior`를
켜면 현재 cached 후보와 확률에서 behavior z를 episode마다 독립 추출합니다. 따라서 episode 간에는
같은 후보가 반복될 수 있습니다. cache가 생기기 전 warmup에는
prior를 사용하며 behavior sampling 때문에 score나 cache를 새로 계산하지 않습니다.
D-LEVER는 replay 학습 시 **critic와 actor 양쪽에 쓰는 z minibatch 분포**를 바꿉니다.
두 업데이트는 같은 z를 씁니다.
벡터 보상과 dynamics가 z에 독립이므로 replay transition을 다른 z로 relabel할 수 있습니다.
중요도 보정으로 prior 분포로 되돌리지 않으며, leverage를 reward에 더하지 않습니다.

`core.py:Curriculum.sample`의 순서:

1. K updates마다 현재 actor와 twin critics의 detached snapshot을 함께 갱신합니다.
2. prior에서 `n=N*B`개 후보 z를 독립 추출합니다.
3. 공통 초기 상태 bank에서
   `mu_z = mean_s0 E_{a~pi_z}[ (psi_1(s0,a,z)+psi_2(s0,a,z))/2 ]`를 추정합니다.
   이산 action 기대값은 정확한 합, 연속 action 기대값은 기본 2개 표본입니다.
   **h와 entropy는 mu에 포함하지 않습니다.** 초기 상태 평균을 먼저 취합니다.
4. `G_hat = mean_z mu_z mu_z^T`, `lambda_ada=max(ridge*trace(G_hat)/d,1e-8)`.
5. 첫 refresh는 `G=G_hat+lambda_ada*I`; 이후 regularized Gram에 EMA를 적용합니다.
6. `ell_z=mu_z^T solve(G,mu_z)`와
   `q_i=(1-eta)/n + eta*ell_i/sum(ell)`를 계산합니다.
7. 매 update마다 전체 cached pool에서 B개 z를 **batch 내 비복원 추출**합니다.
   다음 update에는 전체 pool을 다시 사용하므로 같은 index가 반복될 수 있습니다.
   refresh 사이에는 후보와 확률을 cache합니다.

Snapshot scoring, Gram, scores, sampling은 모두 no-grad입니다. Gram/solve는 float64입니다.
Scoring용 torch RNG와 NumPy RNG를 분리해 candidate 평가가 behavior/learner RNG를 바꾸지 않습니다.
Uniform도 동일한 후보 수와 scoring 진단을 사용하되 `eta=0`입니다.

| method | score |
|---|---|
| `uniform` | 후보에서 균등 재샘플링 |
| `d` | `mu^T G^-1 mu` |
| `a` | `mu^T G^-2 mu / (1 + mu^T G^-1 mu)` |
| `e` | `lambda_min(G + mu mu^T) - lambda_min(G)` |
| `td` | 공통 replay probe에서 twin soft-Q의 평균 absolute TD error |

모두 0인 score는 uniform으로 fallback합니다. 기본 `eta=.9`는 prior 10%, tilted 90%입니다.
기본 `ridge=.001`, Gram EMA `alpha=.005`, `refresh=5`, `multiplier=10`, `batch_size=256`입니다.
여기서 `--alpha`는 SAC temperature가 아니라 **Gram EMA 계수**입니다.
연속 환경의 scoring 비용이 크면 `--refresh 20`을 두 방법에 동일하게 적용할 수 있습니다.
이는 refresh 설정 변경이므로 실험에 기록해야 합니다.

`td`는 PLR-inspired task-level baseline입니다. 매 refresh에서 모든 후보를 동일한 replay
transition probe(`--td-probes`, 기본 8개)로 평가하고, 두 critic의 scalar soft-Q absolute TD error를
평균합니다. transition 자체를 우선순위화하는 PER나 discrete level을 저장하는 원래 PLR과는
구별됩니다. target critic snapshot과 scoring 전용 RNG를 사용하므로 learner RNG를 바꾸지 않습니다.

동일 environment-step/update budget, 구조, optimizer, entropy 계수를 사용합니다.
온라인 정책이 달라지므로 replay trajectory 자체가 동일한 실험은 아닙니다.

## Task prior와 환경 의미

두 환경 모두 공식 2-objective v5 환경을 그대로 실행합니다. 단, 공식 2D 벡터
보상은 공통항이 각 성분에 이미 더해져 있으므로, 학습 시 `info`에 기록된 값으로
다음 3D transition feature를 만듭니다:

```text
Ant:    phi = [x_velocity, y_velocity, reward_ctrl + reward_survive + reward_contact]
Hopper: phi = [x_velocity, 10*z_distance_from_origin, reward_ctrl + reward_survive]
```

task는 `z=[u1,u2,1]`이며 `u1,u2 >= 0`, `u1²+u2²=1`인 양의 단위 구면에서
방향 가중치 `u`를 뽑습니다. 따라서 `r_z=u1*phi1+u2*phi2+phi3`이고 공통항 계수는
모든 task에서 1입니다. 축 방향 `[1,0,1]`, `[0,1,1]`은 각각 공식 2D 보상의
첫째, 둘째 성분을 재현합니다. `--radius` 기본값은 **1**이며 방향 부분의 L2 반지름입니다.

```bash
# 방향 가중치만 simplex로 바꾸는 ablation (공통항 계수는 계속 1):
python run_suite.py --prior simplex --eta .9 --seeds 0 1 2 3 4 --output simplex_runs
# 방향 가중치에 음수도 허용하는 전체 구면:
python run_suite.py --prior sphere --output sphere_runs
```

prior가 바뀌면 평가 task 분포도 바뀝니다. 공식 보상 항의 크기를 바꾸지 않고
공통항만 별도 feature로 분리합니다. 양의 구면은 x/y 양의 방향을 선호하므로 Ant의
전방향 이동 suite는 아닙니다. `--prior sphere`는 방향 보상에 음수 가중치도 허용하지만
공통항 계수는 여전히 1입니다.

두 환경 모두 공식 raw observation을 사용하며 기본 horizon은 1,000입니다. 환경 termination에서는
bootstrap을 끄고, time-limit truncation에서는 episode를 reset하되 bootstrap은 유지합니다.
observation에 별도의 시간 좌표를 추가하지 않습니다.

## 평가 / 체크포인트

```bash
python plot.py runs_sfsac --output plots
python evaluate.py runs_sfsac/mo_ant/d/seed_0/latest.pt --output ant_heldout.npz
```

평가에는 학습된 SAC policy의 **deterministic action**을 사용합니다. 이산 환경은 categorical
argmax, 연속 환경은 Gaussian mean을 tanh 변환한 action입니다. 학습과 data collection은
stochastic policy를 그대로 사용합니다.
평가 중 업데이트는 없습니다. task/episode seed를 고정하고 Python/NumPy/torch RNG를 복구합니다.

기본 평가는 Ant와 Hopper 모두 고정 seed로 무작위 추출한 10개 weight를
사용하며 weight당 10 rollouts입니다. 각 weight의 scalar utility는
하위/상위 25%를 제외한 IQM(10개 중 중앙 6개 평균)으로 집계하고, weight별 IQM은 평가
`.npz`의 `utility_iqm`에 모두 저장합니다. `mean_return` 로그는 weight별 IQM의 평균입니다.
Ant의 평가 weight는 학습 prior에서 `eval_seed`로 한 번 뽑아 모든 방법과 seed에 고정합니다.
Hopper의 평가 weight도 같은 양의 구면 prior에서 고정 seed로 뽑습니다. 학습 task는 기존 prior와
curriculum에 따라 별도로 샘플됩니다. 위의 고정 공통항 보상으로, entropy bonus 없이:

- weight별 rollout IQM 전체 저장과 그 IQM들의 mean / worst-decile / worst-quartile / minimum 로그
- rollout으로 측정한 mu의 logdet / minimum eigenvalue
- predicted mu와 rollout mu 사이 RMSE
- 길이, timeout fraction, evaluation transition 수

Ant의 weight가 10개일 때 worst-decile은 minimum과 같고, worst-quartile은 낮은 3개
weight의 IQM 평균입니다.

`rollout_logdet`는 일정한 absolute ridge를 사용합니다. 훈련 EMA logdet와 구별합니다.
RMSE에는 서로 다른 reset 표본 bank와 action/rollout Monte Carlo 오차도 포함됩니다.
Reward 크기가 다른 task 사이 raw lower-tail return은 lower-tail regret이 아닙니다.

```text
runs_sfsac/<env>/<method>/seed_<n>/
  config.json, versions.json
  eval_tasks.npy, initial_states.npy
  metrics.csv
  eval_000050000.npz
  design_000050000.npz
  latest.pt
```

`latest.pt`는 actor와 twin critics를 포함한 **inference checkpoint**입니다.
optimizer/replay/environment state는 없으므로 정확한 학습 재개용은 아닙니다.
새 3D feature 모델은 기존 2D 모델 checkpoint를 이어 학습할 수 없습니다. `evaluate.py`는
이전 2D checkpoint를 공식 2D 보상 방식으로 계속 평가합니다.
`plot.py` 음영은 seed 간 standard error입니다.

현재 배포 검증은 수학적 일관성과 실제 학습 경로 실행 검사이며, 장기 수렴·D-LEVER 성능 우위는
실험으로 확인해야 합니다. 상세 결과는 `VALIDATION.md`에 기록했습니다.

## 공식 참고

- https://mo-gymnasium.farama.org/environments/mo-hopper/
- https://mo-gymnasium.farama.org/environments/mo-ant/
- https://spinningup.openai.com/en/latest/algorithms/sac.html
- https://proceedings.mlr.press/v139/jiang21b.html (PLR; `td` baseline의 task-priority 근거)
