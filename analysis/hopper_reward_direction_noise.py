"""MO-Hopper / MO-Ant analogue of note/analysis/walker_reward_direction_noise.py.

Usage: python analysis/hopper_reward_direction_noise.py all --env {hopper,ant}

The policy receives a noisy task vector, while the return is scored with the
true task.  Only the direction part u (z=[u1,u2,1]) is perturbed; the fixed
common coefficient stays 1.  The noise has relative scale sigma, matching the
Walker experiment, where N(0, sigma^2 I) noise was added to a radius-sqrt(d) z:

    u_noisy = |u + sigma * eps / sqrt(2)|, renormalized to |u| = 1.

The absolute value folds the vector back into the positive quadrant, so the
noisy task stays on the training prior's support (fixed_common_positive_sphere).

Rollouts reuse the saved evaluator's protocol: deterministic actions, the ten
fixed evaluation tasks, reset seed eval_seed+10000+k for rollout k, discounted
utility, then the IQM over the ten rollouts of each task.  Noise draw k for task
j is shared across methods, checkpoints and sigma levels (paired design).
Sigma=0 reproduces the saved evaluation up to CUDA-vs-CPU float error; this is checked.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import SFSAC, isolated_torch_rng  # noqa: E402
from envs import Benchmark, isolated_global_rng  # noqa: E402
from train import Config, tensor  # noqa: E402

SIGMAS = [0., .25, .5, 1., 2., 4.]
METHODS = ["uniform", "d"]
# Training seeds with both methods trained past step 0 (Ant D seed 4 has only step 0).
ENVS = {"hopper": [0, 1, 2, 4], "ant": [0, 1, 2, 3]}


def configure(env):
    global ENV, RUNS, OUT, SEEDS
    ENV, SEEDS = env, ENVS[env]
    RUNS = ROOT / "runs_2m_sphere" / f"mo_{env}"
    OUT = ROOT / "results" / f"{env}_reward_direction_noise"


configure("hopper")


def load(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    cfg = Config(**{k: v for k, v in saved["config"].items() if k in Config.__dataclass_fields__})
    cfg.device = device
    net = SFSAC(saved["obs_dim"], saved["actions"], saved["dim"], cfg.hidden,
                saved["discrete"], saved["low"], saved["high"], saved["temperature"]).to(device)
    net.load_state_dict(saved["model"])
    return net.eval(), cfg, int(saved["steps"])


def noisy_task(z, sigma, task_index, noise_seed):
    rng = np.random.default_rng(np.random.SeedSequence([9109, task_index, noise_seed]))
    eps = rng.normal(size=2)
    u = np.asarray(z[:2], np.float64)
    radius = np.linalg.norm(u)
    v = np.abs(u + sigma * radius * eps / np.sqrt(2))
    v = radius * v / max(np.linalg.norm(v), 1e-12)
    return np.array([v[0], v[1], z[2]], np.float32), float(u @ v / radius ** 2)


@torch.no_grad()
def run(method, seed, only_task=None, env_name=None, device="cpu"):
    # Ant contact dynamics amplify CPU/CUDA float differences; the saved evaluations
    # ran on CUDA, so Ant must also run on CUDA to reproduce sigma=0.
    if env_name is not None:
        configure(env_name)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / (f"{method}_seed{seed}.json" if only_task is None
                  else f"{method}_seed{seed}_task{only_task}.json")
    if dest.exists():
        return
    torch.set_num_threads(1)
    directory = RUNS / method / f"seed_{seed}"
    net, cfg, step = load(directory / "latest.pt", device)
    tasks = np.load(directory / "eval_tasks.npy")
    env = Benchmark(cfg.env, cfg.horizon, cfg.reward_layout)
    rows = []
    for j, z in enumerate(tasks):
        if only_task is not None and j != only_task:
            continue
        for sigma in SIGMAS:
            for k in range(cfg.eval_episodes):
                znoisy, cosine = noisy_task(z, sigma, j, k)
                reset = cfg.eval_seed + 10000 + k
                ret = np.zeros(net.dim)
                with isolated_global_rng(reset), isolated_torch_rng(reset, device):
                    obs, _ = env.reset(seed=reset)
                    for t in range(env.horizon):
                        action = net.act(tensor(obs[None], device), tensor(znoisy[None], device),
                                         deterministic=True)[0].cpu().numpy()
                        obs, reward, terminated, truncated, _ = env.step(action)
                        ret += cfg.gamma ** t * reward
                        if terminated or truncated:
                            break
                rows.append(dict(method=method, seed=seed, step=step, task=j, sigma=sigma,
                                 noise_seed=k, reset_seed=reset, cosine=cosine,
                                 utility=float(z @ ret), length=t + 1,
                                 noisy_task=znoisy.tolist()))
        print(method, seed, "task", j, "done", flush=True)
    env.close()
    dest.write_text(json.dumps(rows))


def summarize():
    import pandas as pd
    from scipy.stats import trim_mean
    df = pd.DataFrame([r for p in sorted(OUT.glob("*_seed*.json")) for r in json.loads(p.read_text())])
    assert len(df) == len(METHODS) * len(SEEDS) * 10 * len(SIGMAS) * 10, len(df)
    df.drop(columns="noisy_task").to_csv(OUT / "rollouts.csv", index=False)
    # Sigma=0 must reproduce the saved evaluation at the checkpoint step.
    check = []
    for (method, seed), g in df[df.sigma == 0].groupby(["method", "seed"]):
        step = int(g.step.iloc[0])
        saved = np.load(RUNS / method / f"seed_{seed}" / f"eval_{step:09d}.npz")["utilities"]
        mine = g.sort_values(["task", "noise_seed"]).utility.to_numpy().reshape(10, 10)
        trim = lambda x: np.sort(x, 1)[:, 2:8].mean()
        check.append((float((np.abs(mine - saved) < 1e-3).mean()),
                      float(abs(trim(mine) - trim(saved)) / abs(trim(saved)))))
    task = df.groupby(["method", "seed", "step", "sigma", "task"], as_index=False).agg(
        iqm=("utility", lambda x: trim_mean(x.to_numpy(), .25)), cosine=("cosine", "mean"))
    ckpt = task.groupby(["method", "seed", "step", "sigma"], as_index=False).agg(
        mean=("iqm", "mean"), cosine=("cosine", "mean"),
        cvar25=("iqm", lambda x: np.sort(x.to_numpy())[:3].mean()))
    ckpt.to_csv(OUT / "checkpoint_summary.csv", index=False)

    def values(metric, method, sigma):
        g = ckpt[(ckpt["method"] == method) & (ckpt.sigma == sigma)].sort_values("seed")
        assert g.seed.tolist() == SEEDS
        return g[metric].to_numpy()

    se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
    lines = ["| Metric | σ | mean cos | Uniform ± SE | D-LEVER ± SE | D − Uniform ± SE | Δ vs σ=0: D − Uniform ± SE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for metric, label in [("mean", "Mean task return"), ("cvar25", "Bottom-25% tasks")]:
        for sigma in SIGMAS:
            u, d = values(metric, "uniform", sigma), values(metric, "d", sigma)
            u0, d0 = values(metric, "uniform", 0.), values(metric, "d", 0.)
            diff, adv = d - u, (d - d0) - (u - u0)
            cos = ckpt[ckpt.sigma == sigma].cosine.mean()
            rows.append(dict(metric=metric, sigma=sigma, cosine=cos, uniform=u.mean(), uniform_se=se(u),
                             d=d.mean(), d_se=se(d), diff=diff.mean(), diff_se=se(diff),
                             change_advantage=adv.mean(), change_advantage_se=se(adv)))
            lines.append(f"| {label} | {sigma:g} | {cos:.3f} | {u.mean():.1f} ± {se(u):.1f} | "
                         f"{d.mean():.1f} ± {se(d):.1f} | {diff.mean():+.1f} ± {se(diff):.1f} | "
                         f"{adv.mean():+.1f} ± {se(adv):.1f} |")
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    steps = ckpt[ckpt.sigma == 0].sort_values(["method", "seed"])
    step_text = ", ".join(f"{m} seed {s}: {st // 1000}k" for m, s, st in
                          zip(steps["method"], steps.seed, steps.step))
    lines += ["", f"Checkpoints: latest.pt of runs_2m_sphere/mo_{ENV} ({step_text}). "
              "Each task's utility is the IQM (middle 6) of ten rollouts; one noise draw per rollout. "
              "Bottom-25% is the mean of the lowest 3 of 10 task IQMs. Mean ± SE across the four "
              "training seeds; differences are paired by seed. σ=0 vs the saved evaluation: "
              f"{100 * min(c[0] for c in check):.0f}–{100 * max(c[0] for c in check):.0f}% of rollouts identical per checkpoint "
              f"(rollouts diverge chaotically under tiny float differences, e.g. CPU vs another GPU); "
              f"checkpoint mean return within {100 * max(c[1] for c in check):.2f}%. All σ levels of a checkpoint run "
              "on the same device here, so the σ comparison is internally consistent. The noise has relative scale σ like the Walker "
              "experiment, but is folded into the positive quadrant, so cosines are not comparable "
              "to Walker's high-dimensional cosines."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "all", "summarize"])
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--seed", type=int)
    p.add_argument("--env", choices=list(ENVS), default="hopper")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    configure(a.env)
    if a.command == "run":
        run(a.method, a.seed)
    elif a.command == "all":
        from multiprocessing import get_context
        jobs = [(m, s, j, a.env, a.device) for m in METHODS for s in SEEDS for j in range(10)]
        with get_context("spawn").Pool(a.workers) as pool:
            pool.starmap(run, jobs, chunksize=1)
        summarize()
    else:
        summarize()
