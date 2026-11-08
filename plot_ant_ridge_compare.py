"""Compare Ant seed means with the available low-ridge D runs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("runs_2m_sphere/mo_ant")
RIDGE = Path("runs_ridge_0001/mo_ant/d")
OUTPUT = Path("plots/ant_ridge_0001_compare.png")
COMPLETED_STEPS = 2_000_000
METRICS = (("mean_return", "Mean return"),
           ("worst_quartile_return", "CVaR25"),
           ("worst_decile_return", "CVaR10"),
           ("train_logdet", "Train logdet"))


def read(path):
    with path.open(newline="") as file:
        return {int(row["env_steps"]): row for row in csv.DictReader(file)}


def main():
    uniform_all = {int(path.parent.name.removeprefix("seed_")): read(path)
                   for path in (BASE / "uniform").glob("seed_*/metrics.csv")}
    tilted_all = {int(path.parent.name.removeprefix("seed_")): read(path)
                  for path in (BASE / "d").glob("seed_*/metrics.csv")}
    uniform_complete = {seed: rows for seed, rows in uniform_all.items()
                        if rows and max(rows) >= COMPLETED_STEPS}
    tilted_complete = {seed: rows for seed, rows in tilted_all.items()
                       if rows and max(rows) >= COMPLETED_STEPS}
    seeds = sorted(set(uniform_complete) & set(tilted_complete))
    uniform = {seed: uniform_complete[seed] for seed in seeds}
    tilted = {seed: tilted_complete[seed] for seed in seeds}
    steps = sorted(set().union(*(set(uniform[s]) & set(tilted[s]) for s in seeds)))
    ridge = {int(path.parent.name.removeprefix("seed_")): read(path)
             for path in RIDGE.glob("seed_*/metrics.csv")}
    ridge = {seed: rows for seed, rows in ridge.items()
             if rows and max(rows) >= COMPLETED_STEPS}
    if not steps or not ridge:
        raise FileNotFoundError("Ant baseline or ridge metrics are missing")
    ridge_steps = sorted(set().union(*(set(rows) for rows in ridge.values())))

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.1))
    for ax, (key, title) in zip(axes, METRICS):
        for runs, label, color, style in ((uniform, "UNIFORM (ridge 0.01)", "#0072B2", "--"),
                                           (tilted, "D-LEVER (ridge 0.01)", "#D62728", "-")):
            means, errors = [], []
            for step in steps:
                values = np.asarray([float(runs[seed][step][key]) for seed in seeds
                                     if step in uniform[seed] and step in tilted[seed]])
                means.append(values.mean())
                errors.append(values.std(ddof=1) / np.sqrt(len(values))
                              if len(values) > 1 else np.nan)
            means, errors = np.asarray(means), np.asarray(errors)
            ax.plot(steps, means, color=color, ls=style, lw=2, label=label)
            ax.fill_between(steps, means - errors, means + errors,
                            color=color, alpha=0.16)
        ridge_means, ridge_errors = [], []
        for step in ridge_steps:
            values = np.asarray([float(rows[step][key]) for rows in ridge.values()
                                 if step in rows])
            ridge_means.append(values.mean())
            ridge_errors.append(values.std(ddof=1) / np.sqrt(len(values))
                                if len(values) > 1 else np.nan)
        ridge_means, ridge_errors = np.asarray(ridge_means), np.asarray(ridge_errors)
        ax.plot(ridge_steps, ridge_means, color="#C00000", ls=":", lw=2.3,
                marker="o", markevery=4, markersize=3.2,
                label="D-LEVER (ridge 0.0001)")
        ax.fill_between(ridge_steps, ridge_means - ridge_errors,
                        ridge_means + ridge_errors, color="#C00000", alpha=0.08)
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Return")
    axes[-1].set_ylabel("Logdet")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Ant: completed {COMPLETED_STEPS:,}-step seeds only (baseline: {seeds}; ridge 0.0001: {sorted(ridge)})")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
