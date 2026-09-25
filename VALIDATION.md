# Conditional SF-SAC validation — Ant and Hopper

The supported benchmarks are MO-Ant-2obj-v5 and MO-Hopper-2obj-v5.
Their official 2D rewards are decomposed into two directional features and one
shared feature. The directional task weight has unit L2 norm in the positive
orthant; the shared feature always has weight one.

`python -m pytest -q`: **31 passed**. These checks cover the D/A/E and TD
sampling scores, SF-SAC targets and updates, task priors, evaluation RNG
isolation, checkpoint state-dict round trip, and both official environment
adapters and vector rewards.

`python run_suite.py --smoke --seeds 0 --output /tmp/dlever-fixed-common-smoke-20260925`:
Ant and Hopper each completed Uniform and D runs (160 training steps per run).
Both environments also completed 160-step TD smoke runs.
The new Hopper smoke checkpoint and an existing legacy 2D Ant checkpoint were
also reloaded and evaluated successfully with `evaluate.py`.

These checks establish execution and numerical consistency, not convergence
or an advantage over Uniform. Long-budget results must be assessed separately.
