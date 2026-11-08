"""Generate publication-ready Hopper figures from five-evaluation moving averages."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "runs_2m_sphere" / "mo_hopper"
FIGURES = Path(__file__).resolve().parent / "figures"
WINDOW = 5
INTERVAL = 50_000
METHODS = ("uniform", "d")
STYLE = {"uniform": ("UNIFORM", "#0072B2", "--"),
         "d": ("D-LEVER", "#D62728", "-")}
METRICS = (("mean_return", "Mean task return"),
           ("worst_quartile_return", "Bottom 25% task return"),
           ("worst_decile_return", "Bottom 10% task return"),
           ("train_logdet", "Training Gram log det"))


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def load_runs():
    metrics, directions = ({method: {} for method in METHODS} for _ in range(2))
    for method in METHODS:
        for csv_path in sorted((RUNS / method).glob("seed_*/metrics.csv")):
            seed = int(csv_path.parent.name.removeprefix("seed_"))
            with csv_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            metrics[method][seed] = {
                int(row["env_steps"]): np.array([float(row[key]) for key, _ in METRICS])
                for row in rows
            }
            directions[method][seed] = {}
            for path in sorted(csv_path.parent.glob("eval_*.npz")):
                step = int(path.stem.removeprefix("eval_"))
                with np.load(path) as evaluation:
                    directions[method][seed][step] = evaluation["measured_mu"].mean(0)[:2]
    return metrics, directions


def moving_average(runs):
    output = {method: {} for method in METHODS}
    for method in METHODS:
        for seed, run in runs[method].items():
            smooth = {}
            for end in sorted(step for step in run if step >= WINDOW * INTERVAL):
                window = [end - offset * INTERVAL for offset in range(WINDOW)]
                if all(step in run for step in window):
                    smooth[end] = np.mean([run[step] for step in window], axis=0)
            output[method][seed] = smooth
    return output


def paired_series(runs):
    seeds = sorted(set(runs["uniform"]) & set(runs["d"]))
    paired = {seed: set(runs["uniform"][seed]) & set(runs["d"][seed])
              for seed in seeds}
    steps = sorted(set().union(*paired.values()))
    if not steps:
        raise ValueError("No paired Hopper evaluation windows found")
    summary = {}
    for method in METHODS:
        means, errors = [], []
        for step in steps:
            values = np.stack([runs[method][seed][step] for seed in seeds
                               if step in paired[seed]])
            means.append(values.mean(0))
            errors.append(values.std(0, ddof=1) / np.sqrt(len(values))
                          if len(values) > 1 else np.full(values.shape[1], np.nan))
        summary[method] = (np.stack(means), np.stack(errors))
    counts = [sum(step in paired[seed] for seed in seeds) for step in steps]
    return np.array(steps) / 1e6, summary, counts


def format_axis(ax, title, ylabel=None):
    ax.set_title(title, loc="left", pad=6)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="0.85", alpha=0.45, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.22, 2.03)
    ax.set_xticks(np.arange(0.25, 2.01, 0.25))


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(runs, stem, panels, figsize):
    x, summary, counts = paired_series(runs)
    fig, axes = plt.subplots(1 if len(panels) == 2 else 2, 2,
                             figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)
    for ax, (index, title, ylabel) in zip(axes.flat, panels):
        for method in METHODS:
            label, color, line = STYLE[method]
            mean, se = summary[method]
            ax.plot(x, mean[:, index], color=color, ls=line, lw=1.7,
                    label=label, zorder=3)
            ax.fill_between(x, mean[:, index] - se[:, index],
                            mean[:, index] + se[:, index], color=color,
                            alpha=0.14, linewidth=0)
        format_axis(ax, title, ylabel)
    for ax in axes[-1]:
        ax.set_xlabel(r"Steps ($\times 10^6$)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.77 if len(panels) == 2 else 0.87,
                        bottom=0.16 if len(panels) == 2 else 0.11,
                        left=0.1, right=0.98,
                        wspace=0.3, hspace=0.38)
    save(fig, stem)
    return x, counts


def plot_by_seed(runs, stem, index, ylabel):
    seeds = sorted(set(runs["uniform"]) & set(runs["d"]))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for ax, seed in zip(axes.flat, seeds):
        for method in METHODS:
            label, color, line = STYLE[method]
            run = runs[method][seed]
            steps = sorted(run)
            if steps:
                values = np.stack([run[step] for step in steps])
                ax.plot(np.array(steps) / 1e6, values[:, index],
                        color=color, ls=line, lw=1.6, label=label)
        format_axis(ax, f"Seed {seed}", ylabel if ax in axes[:, 0] else None)
    for ax in axes[-1]:
        ax.set_xlabel(r"Steps ($\times 10^6$)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.1, right=0.98,
                        wspace=0.25, hspace=0.3)
    save(fig, stem)


def plot_four_panel(metrics, directions):
    x, metric_summary, metric_counts = paired_series(metrics)
    direction_x, direction_summary, direction_counts = paired_series(directions)
    if not np.array_equal(x, direction_x) or metric_counts != direction_counts:
        raise ValueError("Performance and direction checkpoints do not match")
    panels = ((metric_summary, 0, "Mean"),
              (metric_summary, 2, "CVaR 10%"),
              (direction_summary, 0, r"Forward $x$"),
              (direction_summary, 1, r"Vertical $z$"))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25))
    for ax, (summary, index, title) in zip(axes, panels):
        for method in METHODS:
            _, color, line = STYLE[method]
            mean, se = summary[method]
            ax.plot(x, mean[:, index], color=color, ls=line, lw=1.65)
            ax.fill_between(x, mean[:, index] - se[:, index],
                            mean[:, index] + se[:, index], color=color,
                            alpha=0.15, linewidth=0)
        format_axis(ax, title)
        ax.set_xticks((0.5, 1.0, 1.5, 2.0))
        ax.set_xlabel(r"Steps ($\times 10^6$)")
    axes[0].set_ylabel("Discounted Return")
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.28, top=0.84,
                        wspace=0.38)
    save(fig, "hopper_5eval_four_panel")

    fig = plt.figure(figsize=(3.1, 0.38))
    handles = [Line2D([], [], color=STYLE[method][1],
                      linestyle=STYLE[method][2], linewidth=2,
                      label=STYLE[method][0]) for method in METHODS]
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False,
               handlelength=2.5, columnspacing=1.8)
    save(fig, "hopper_5eval_legend")


def plot_mean_cvar10(runs):
    x, summary, _ = paired_series(runs)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6), sharex=True)
    for ax, index, title in zip(axes, (0, 2), ("Mean", "CVaR 10%")):
        for method in METHODS:
            _, color, line = STYLE[method]
            mean, se = summary[method]
            ax.plot(x, mean[:, index], color=color, ls=line, lw=1.65, zorder=3)
            ax.fill_between(x, mean[:, index] - se[:, index],
                            mean[:, index] + se[:, index], color=color,
                            alpha=0.15, linewidth=0)
        format_axis(ax, title)
        ax.set_xlim(.22, 2.0)
        ax.set_xticks((.5, 1., 1.5, 2.))
        ax.set_xlabel("Steps (× 10⁶)")
        ax.set_ylabel("Discounted Return")
    axes[-1].set_xlabel(r"Steps ($\times 10^6$)")
    fig.subplots_adjust(left=.12, right=.98, bottom=.24, top=.88, wspace=.32)
    save(fig, "hopper_5eval_mean_cvar10")


def main():
    metrics, directions = load_runs()
    metrics, directions = moving_average(metrics), moving_average(directions)
    plot_four_panel(metrics, directions)
    plot_mean_cvar10(metrics)
    x, counts = plot_aggregate(
        metrics, "hopper_5eval_performance",
        ((0, "Mean", "Discounted Return"),
         (1, "Lowest 3 of 10", "Discounted Return"),
         (2, "CVaR 10%", "Discounted Return"),
         (3, "Training log det", "Log det")),
        (7.2, 5.1))
    plot_aggregate(
        directions, "hopper_5eval_directions",
        ((0, r"Forward $x$", "Discounted Return"),
         (1, r"Vertical $z$", "Discounted Return")),
        (7.2, 2.85))
    plot_by_seed(metrics, "hopper_5eval_seeds", 0, "Mean return")
    plot_by_seed(directions, "hopper_5eval_forward_seeds", 0,
                 "Forward velocity return")
    plot_by_seed(directions, "hopper_5eval_vertical_seeds", 1,
                 "Vertical distance return")
    print(f"Saved seven PDF and PNG figures in {FIGURES}")
    changes = [(x[0], counts[0])]
    changes.extend((step, count) for step, count, previous in
                   zip(x[1:], counts[1:], counts[:-1]) if count != previous)
    print("Paired-seed counts:", ", ".join(
        f"{step:.2f}M: n={count}" for step, count in changes))


if __name__ == "__main__":
    main()
