# Conditional SF-SAC validation — 2026-09-21

Runtime: Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 1.26.4,
Gymnasium 1.3.0, MO-Gymnasium 1.3.1, MuJoCo 3.13.0.

## Automated checks

`python -m pytest -q`: **19 passed**.

Checks cover:

- D/A/E scores against matrix definitions, mixture floor and uniform fallback.
- Initial-state average before outer product, stochastic categorical expectation,
  and exclusion of the entropy head from policy embeddings.
- Both action types: scalarizing the SF and entropy Bellman targets gives the
  standard clipped-twin SAC target; terminal targets contain no bootstrap.
- Selection of an entire twin SF vector, rather than coordinatewise minima.
- Actual updates to actor and critic parameters for both action types.
- Continuous action bounds and finite transformed Gaussian log probabilities.
- Detached scorer snapshot, refresh cache, and Gram EMA.
- Prior sampling domains; all four official environment adapters and timeouts.
- Repeated evaluation and preservation of Python/NumPy/torch training RNG.
- Continuous embedding Monte Carlo reproducibility and RNG isolation.
- Inference state-dict round trip.

## Real environment smoke runs

Command:

```bash
python run_suite.py --smoke --seeds 0 --output sfsac_smoke_runs
```

All eight runs completed. Each used **160 training transitions, 145 learner
updates**, hidden width 32, batch 16, warmup 16, 3 evaluation tasks, 1 episode,
maximum horizon 12. Fruit Tree terminates at depth 6.

| Environment | Uniform | D-LEVER |
|---|---|---|
| Fruit Tree | Passed | Passed |
| Minecart | Passed | Passed |
| MO-Hopper-v5 | Passed | Passed |
| MO-Ant-v5 | Passed | Passed |

Actor/critic losses and recorded evaluation metrics were finite.
All four D-LEVER inference checkpoints were reloaded by `evaluate.py` and
re-evaluated on a separate 2-task, 2-episode bank. Fruit Tree evaluation-only
reference generation also completed.

Minecart emits Gymnasium float64-to-float32 Box-bound warnings. They did not
prevent environment creation, stepping, learning, or evaluation.

These checks establish execution and numerical consistency, **not convergence
or an advantage over Uniform**. Long-budget training, performance tuning,
GPU execution, and the A/E end-to-end suite were not run. A/E score formulas
were checked by the unit tests.
