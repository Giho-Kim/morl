"""Compare the current D-optimal prior and tilted-behavior runs."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "prior": Path("runs_ant_d_prior/mo_ant/d"),
    "tilted": Path("runs_ant_d_tilted/mo_ant/d"),
}
METRICS = [
    ("mean_return", "Mean return"),
    ("worst_decile_return", "Worst-decile return"),
    ("rollout_logdet", "Rollout log-det"),
    ("embedding_rmse", "Embedding RMSE"),
]


def load_runs(root):
    runs = []
    for path in sorted(root.glob("seed_*/metrics.csv")):
        with path.open() as file:
            rows = list(csv.DictReader(file))
        runs.append({int(row["env_steps"]): row for row in rows})
    return runs


def main():
    output = Path("plots/prior_vs_tilted_current.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    for label, root in RUNS.items():
        runs = load_runs(root)
        if not runs:
            continue
        steps = sorted(set.intersection(*(set(run) for run in runs)))
        for ax, (key, title) in zip(axes.flat, METRICS):
            values = np.array(
                [[float(run[step][key]) for step in steps] for run in runs]
            )
            mean = values.mean(axis=0)
            line, = ax.plot(steps, mean, marker="o", ms=3, lw=2, label=label)
            if len(runs) > 1:
                se = values.std(axis=0, ddof=1) / np.sqrt(len(runs))
                ax.fill_between(
                    steps, mean - se, mean + se, color=line.get_color(), alpha=0.15
                )
            ax.set_title(title)
            ax.grid(alpha=0.25)

    for ax in axes[-1]:
        ax.set_xlabel("Environment steps")
    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Value")
    axes[0, 0].legend(title="D-optimal behavior")
    fig.suptitle("MO-Ant: prior vs tilted (current results, seed 0)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
