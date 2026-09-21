"""Optional evaluation-only Fruit Tree oracle, obtained via simulator trajectories.

Never imported by training. No transition table, DP or leaf-reward array access.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from core import sample_latents
from envs import Benchmark, isolated_global_rng


def enumerate_returns(depth, gamma):
    env = Benchmark("fruit_tree", depth=depth)
    outcomes = []
    with isolated_global_rng(0):
        for leaf in range(2 ** depth):
            env.reset(seed=0)
            total = np.zeros(env.dim)
            for t in range(depth):
                action = (leaf >> (depth - t - 1)) & 1
                _, reward, terminated, truncated, _ = env.step(action)
                total += gamma ** t * reward
            assert terminated or truncated
            outcomes.append(total)
    env.close()
    return np.stack(outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="fruit_tree/method/seed_N run directory")
    parser.add_argument("--directions", type=int, default=20000)
    args = parser.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    if cfg["env"] != "fruit_tree":
        raise ValueError("This oracle is only valid for deterministic Fruit Tree")
    leaf_returns = enumerate_returns(cfg["depth"], cfg["gamma"])
    prior_z = sample_latents(np.random.default_rng(333), args.directions, 6,
                             cfg["prior"], cfg["radius"])
    best_leaf = (prior_z @ leaf_returns.T).argmax(1)
    masses = np.bincount(best_leaf, minlength=len(leaf_returns)) / len(prior_z)
    np.savez_compressed(args.run / "fruit_reference.npz", leaf_returns=leaf_returns,
                        estimated_prior_optimal_leaf_mass=masses)
    report = []
    for path in sorted(args.run.glob("eval_*.npz")):
        data = np.load(path)
        optimal = (data["tasks"] @ leaf_returns.T).max(1)
        obtained = data["utility_iqm"] if "utility_iqm" in data else data["utilities"].mean(1)
        regret = optimal - obtained
        report.append(dict(file=path.name, mean_regret=float(regret.mean()),
                           worst_decile_regret=float(np.sort(regret)[-max(1, int(np.ceil(.1 * len(regret)))):].mean()),
                           max_regret=float(regret.max())))
    payload = dict(reference_environment_steps=int(cfg["depth"] * 2 ** cfg["depth"]),
                   prior=cfg["prior"], directions=args.directions, checkpoints=report)
    (args.run / "fruit_regret.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
