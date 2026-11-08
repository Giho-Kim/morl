"""Plot per-seed curves and paired-seed mean ± SE at each evaluation checkpoint."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {"uniform": ("Uniform", "C0", "-", "o"),
           "d": ("D-tilted", "C1", "--", "s")}
METRICS = [("mean_return", "Mean"),
           ("worst_quartile_return", "CVaR25"),
           ("worst_decile_return", "CVaR10"),
           ("train_logdet", "Train logdet")]


def load_metrics(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def average_checkpoint_blocks(data, block_size=4, stride=4):
    """Average complete blocks of positive-step checkpoints per run."""
    steps = sorted({step for method_runs in data.values()
                    for run in method_runs.values() for step in run if step > 0})
    blocks = [steps[start:start + block_size]
              for start in range(0, len(steps), stride)]
    blocks = [block for block in blocks if len(block) == block_size
              and np.all(np.diff(block) == 50_000)]
    return {method: {seed: {block[-1]: np.mean([run[step] for step in block], axis=0)
                             for block in blocks if all(step in run for step in block)}
                     for seed, run in method_runs.items()}
            for method, method_runs in data.items()}


def plot_seed_blocks(data, titles, seeds, name, output, ylabels, moving=False,
                     block_size=4):
    fig, axes = plt.subplots(len(seeds), len(titles),
                             figsize=(4.2 * len(titles), 3.1 * len(seeds)),
                             squeeze=False, sharex="col")
    for row, seed in enumerate(seeds):
        for method, (label, color, style, marker) in METHODS.items():
            run = data[method].get(seed, {})
            steps = sorted(run)
            if not steps:
                continue
            values = np.asarray([run[step] for step in steps])
            for col, title in enumerate(titles):
                axes[row, col].plot(steps, values[:, col], label=label,
                                    color=color, ls=style, marker=marker,
                                    markersize=3, linewidth=1.6)
                axes[row, col].set_title(title if row == 0 else "")
        for col, ax in enumerate(axes[row]):
            ax.grid(alpha=0.25)
            ax.set_ylabel(f"seed {seed}\n{ylabels[col]}")
    for ax in axes[-1]:
        ax.set_xlabel(f"End step of {block_size}-checkpoint moving window" if moving
                      else f"End step of {block_size}-checkpoint block")
    axes[0, 0].legend()
    fig.suptitle(f"{name.title()} by seed: {block_size}-checkpoint moving average" if moving
                 else f"{name.title()} by seed: {block_size}-checkpoint averages")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(output)


def plot_mean_se(data, titles, name, output, ylabels, checkpoint_block=1,
                 moving=False):
    """Use only seeds with both methods at each checkpoint."""
    common = {seed: set(data["uniform"].get(seed, {})) & set(data["d"].get(seed, {}))
              for seed in set(data["uniform"]) | set(data["d"])}
    steps = sorted(set().union(*common.values()))
    if not steps:
        return
    counts = np.asarray([sum(step in seed_steps for seed_steps in common.values())
                         for step in steps])
    fig = plt.figure(figsize=(4.2 * len(titles), 5.3))
    grid = fig.add_gridspec(2, len(titles), height_ratios=[4, 1])
    axes = [fig.add_subplot(grid[0, col]) for col in range(len(titles))]
    count_ax = fig.add_subplot(grid[1, :], sharex=axes[0])
    for method, (label, color, style, marker) in METHODS.items():
        for col, title in enumerate(titles):
            values = [np.asarray([data[method][seed][step][col]
                                  for seed in sorted(common) if step in common[seed]],
                                 dtype=float)
                      for step in steps]
            means = np.asarray([value.mean() for value in values])
            se = np.asarray([value.std(ddof=1) / np.sqrt(len(value))
                             if len(value) > 1 else np.nan for value in values])
            axes[col].plot(steps, means, label=label, color=color, ls=style,
                           marker=marker, markersize=2.5, linewidth=1.6)
            axes[col].fill_between(steps, means - se, means + se,
                                   color=color, alpha=0.18)
            axes[col].set_title(title)
            axes[col].set_ylabel(ylabels[col])
            axes[col].grid(alpha=0.25)
    axes[0].legend()
    count_ax.step(steps, counts, where="mid", color="0.3")
    count_ax.set_yticks(range(1, int(counts.max()) + 1))
    count_ax.set_ylabel("Paired seeds")
    count_ax.set_xlabel("Environment steps" if checkpoint_block == 1
                        else (f"End step of {checkpoint_block}-checkpoint moving window"
                              if moving else f"End step of {checkpoint_block}-checkpoint block"))
    count_ax.grid(alpha=0.25)
    suffix = ("paired-seed mean ± SE" if checkpoint_block == 1 else
              f"{checkpoint_block}-checkpoint {'moving average' if moving else 'average'}, then paired-seed mean ± SE")
    fig.suptitle(f"{name.title()}: {suffix}")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="run directory containing environment folders")
    parser.add_argument("--env", default="mo_hopper")
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument("--moving-window", type=int, default=4,
                        help="number of evaluation checkpoints in each moving average")
    args = parser.parse_args()
    if args.moving_window < 2:
        parser.error("--moving-window must be at least 2")

    envdir = args.root / args.env
    seeds = sorted({int(path.parent.name.removeprefix("seed_"))
                    for method in METHODS
                    for path in (envdir / method).glob("seed_*/metrics.csv")
                    if any(int(row["env_steps"]) > 0 for row in load_metrics(path))})
    if not seeds:
        parser.error(f"no seed metrics found under {envdir}")
    args.output.mkdir(parents=True, exist_ok=True)
    name = args.env.removeprefix("mo_")

    fig, axes = plt.subplots(len(seeds), len(METRICS),
                             figsize=(16, 3.1 * len(seeds)), squeeze=False,
                             sharex="col")
    for row, seed in enumerate(seeds):
        for method, (label, color, style, marker) in METHODS.items():
            path = envdir / method / f"seed_{seed}" / "metrics.csv"
            if not path.exists():
                continue
            records = load_metrics(path)
            steps = [int(record["env_steps"]) for record in records]
            for col, (key, title) in enumerate(METRICS):
                axes[row, col].plot(steps, [float(record[key]) for record in records],
                                    label=label, color=color, ls=style,
                                    marker=marker, markersize=2.5, linewidth=1.6)
                axes[row, col].set_title(title if row == 0 else "")
        for ax in axes[row]:
            ax.grid(alpha=0.25)
        axes[row, 0].set_ylabel(f"seed {seed}\nReturn")
        axes[row, -1].set_ylabel("Logdet")
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps")
    axes[0, 0].legend()
    fig.suptitle(f"{name.title()} by seed: every eval checkpoint")
    fig.tight_layout()
    metrics_output = args.output / f"{name}_seed.png"
    fig.savefig(metrics_output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    metric_data = {method: {} for method in METHODS}
    for method in METHODS:
        for seed in seeds:
            path = envdir / method / f"seed_{seed}" / "metrics.csv"
            if path.exists():
                metric_data[method][seed] = {
                    int(record["env_steps"]):
                    np.asarray([float(record[key]) for key, _ in METRICS])
                    for record in load_metrics(path)}
    plot_mean_se(metric_data, [title for _, title in METRICS], name,
                 args.output / f"{name}_mean_se.png", ["Return"] * 3 + ["Logdet"])
    metric_blocks = average_checkpoint_blocks(metric_data)
    plot_seed_blocks(metric_blocks, [title for _, title in METRICS], seeds, name,
                     args.output / f"{name}_4ckpt_seed.png",
                     ["Return"] * 3 + ["Logdet"])
    plot_mean_se(metric_blocks,
                 [title for _, title in METRICS], name,
                 args.output / f"{name}_4ckpt_mean_se.png",
                 ["Return"] * 3 + ["Logdet"], checkpoint_block=4)
    window = args.moving_window
    metric_moving = average_checkpoint_blocks(metric_data, block_size=window, stride=1)
    plot_seed_blocks(metric_moving, [title for _, title in METRICS], seeds, name,
                     args.output / f"{name}_{window}ckpt_moving_seed.png",
                     ["Return"] * 3 + ["Logdet"], moving=True, block_size=window)
    plot_mean_se(metric_moving, [title for _, title in METRICS], name,
                 args.output / f"{name}_{window}ckpt_moving_mean_se.png",
                 ["Return"] * 3 + ["Logdet"], checkpoint_block=window, moving=True)

    fig, axes = plt.subplots(len(seeds), 2, figsize=(12, 3.1 * len(seeds)),
                             squeeze=False, sharex="col")
    directions = ("Forward x reward", "Vertical z reward") if name == "hopper" else (
                  "Forward x reward", "Lateral y reward")
    direction_data = {method: {} for method in METHODS}
    for row, seed in enumerate(seeds):
        for method, (label, color, style, marker) in METHODS.items():
            rundir = envdir / method / f"seed_{seed}"
            checkpoints = sorted(rundir.glob("eval_*.npz"))
            steps, values = [], []
            for path in checkpoints:
                with np.load(path) as evaluation:
                    values.append(evaluation["measured_mu"].mean(axis=0)[:2])
                steps.append(int(path.stem.removeprefix("eval_")))
            if not steps:
                continue
            values = np.asarray(values)
            direction_data[method][seed] = dict(zip(steps, values))
            for col, title in enumerate(directions):
                axes[row, col].plot(steps, values[:, col], label=label,
                                    color=color, ls=style, marker=marker,
                                    markersize=2.5, linewidth=1.6)
                axes[row, col].set_title(title if row == 0 else "")
        for ax in axes[row]:
            ax.grid(alpha=0.25)
        axes[row, 0].set_ylabel(f"seed {seed}\nDiscounted return")
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps")
    axes[0, 0].legend()
    fig.suptitle(f"{name.title()} directional rewards by seed: every eval checkpoint")
    fig.tight_layout()
    direction_output = args.output / f"{name}_dir_seed.png"
    fig.savefig(direction_output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plot_mean_se(direction_data, directions, name,
                 args.output / f"{name}_dir_mean_se.png", ["Discounted return"] * 2)
    direction_blocks = average_checkpoint_blocks(direction_data)
    plot_seed_blocks(direction_blocks, directions, seeds, name,
                     args.output / f"{name}_dir_4ckpt_seed.png",
                     ["Discounted return"] * 2)
    plot_mean_se(direction_blocks, directions, name,
                 args.output / f"{name}_dir_4ckpt_mean_se.png",
                 ["Discounted return"] * 2, checkpoint_block=4)
    direction_moving = average_checkpoint_blocks(direction_data,
                                                  block_size=window, stride=1)
    plot_seed_blocks(direction_moving, directions, seeds, name,
                     args.output / f"{name}_dir_{window}ckpt_moving_seed.png",
                     ["Discounted return"] * 2, moving=True, block_size=window)
    plot_mean_se(direction_moving, directions, name,
                 args.output / f"{name}_dir_{window}ckpt_moving_mean_se.png",
                 ["Discounted return"] * 2, checkpoint_block=window, moving=True)
    print(metrics_output)
    print(direction_output)


if __name__ == "__main__":
    main()
