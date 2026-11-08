"""Plot Hopper mean, CVaR10, and directional returns from saved D-LEVER runs."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


METHODS = ("uniform", "d")
STYLES = {"uniform": ("UNIFORM", "#0072B2", "--"),
          "d": ("D-LEVER", "#D62728", "-")}
WINDOW = 5
INTERVAL = 50_000


def load_runs(root):
    runs = {method: {} for method in METHODS}
    for method in METHODS:
        for csv_path in sorted((root / method).glob("seed_*/metrics.csv")):
            seed = int(csv_path.parent.name.removeprefix("seed_"))
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            run = {}
            for row in rows:
                step = int(row["env_steps"])
                run[step] = [float(row["mean_return"]),
                             float(row["worst_decile_return"]), None, None]
            for path in sorted(csv_path.parent.glob("eval_*.npz")):
                step = int(path.stem.removeprefix("eval_"))
                with np.load(path) as evaluation:
                    directions = evaluation["measured_mu"].mean(axis=0)
                run.setdefault(step, [None] * 4)
                run[step][2:] = [float(directions[0]), float(directions[1])]
            runs[method][seed] = run
    if not runs["uniform"] or not runs["d"]:
        raise FileNotFoundError(f"Expected uniform/ and d/ seed metrics under {root}")
    return runs


def smooth(run):
    result = {}
    for end in sorted(run):
        window = [end - i * INTERVAL for i in range(WINDOW)]
        if all(step in run for step in window):
            values = np.asarray([run[step] for step in window], dtype=float)
            if np.isfinite(values).all():
                result[end] = values.mean(axis=0)
    return result


def paired_summary(runs):
    smoothed = {method: {seed: smooth(run) for seed, run in runs[method].items()}
                for method in METHODS}
    seeds = sorted(set(smoothed["uniform"]) & set(smoothed["d"]))
    paired = {seed: set(smoothed["uniform"][seed]) & set(smoothed["d"][seed])
              for seed in seeds}
    steps = sorted(set().union(*paired.values())) if paired else []
    if not steps:
        raise ValueError("No paired five-evaluation Hopper windows found")
    summary = {}
    for method in METHODS:
        means, errors = [], []
        for step in steps:
            values = np.stack([smoothed[method][seed][step] for seed in seeds
                               if step in paired[seed]])
            means.append(values.mean(axis=0))
            errors.append(values.std(axis=0, ddof=1) / np.sqrt(len(values))
                          if len(values) > 1 else np.zeros(values.shape[1]))
        summary[method] = (np.stack(means), np.stack(errors))
    return np.asarray(steps) / 1e6, summary


def plot(root, output):
    runs = load_runs(root)
    x, summary = paired_summary(runs)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25), sharex=True)
    panels = ((0, "Mean"), (1, "CVaR 10%"),
              (2, r"Forward $x$"), (3, r"Vertical $z$"))
    for ax, (index, title) in zip(axes, panels):
        for method in METHODS:
            _, color, linestyle = STYLES[method]
            mean, se = summary[method]
            ax.plot(x, mean[:, index], color=color, ls=linestyle, lw=1.65)
            ax.fill_between(x, mean[:, index] - se[:, index],
                            mean[:, index] + se[:, index], color=color,
                            alpha=.15, linewidth=0)
        ax.set_title(title, loc="left", pad=6)
        ax.grid(True, color="0.85", alpha=.45, linewidth=.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(.22, 2.03)
        ax.set_xticks((.5, 1., 1.5, 2.))
        ax.set_xlabel(r"Steps ($\times 10^6$)")
    axes[0].set_ylabel("Discounted Return")
    handles = [Line2D([], [], color=STYLES[m][1], ls=STYLES[m][2],
                      lw=2, label=STYLES[m][0]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.04))
    fig.subplots_adjust(left=.085, right=.99, bottom=.28, top=.82, wspace=.38)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "hopper_four_panel"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="run root, e.g. runs or runs/mo_hopper")
    parser.add_argument("--output", type=Path, default=Path("plots"))
    args = parser.parse_args()
    # Accept either the Hopper run directory itself or a suite directory containing it.
    root = args.root if (args.root / "uniform").is_dir() else args.root / "mo_hopper"
    plot(root, args.output)
    print(f"Saved Hopper four-panel figure to {args.output / 'hopper_four_panel.pdf'} and PNG")


if __name__ == "__main__":
    main()
