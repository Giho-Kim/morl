"""Ant four-panel comparisons with five-evaluation smoothing."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOTS = (("UNIFORM", Path("runs_2m_sphere/mo_ant/uniform"), None),
         ("D-LEVER", Path("runs_ridge_0001/mo_ant/d"), 1e-4))
ROOTS_001 = (("UNIFORM", Path("runs_2m_sphere/mo_ant/uniform"), None),
             ("D-LEVER", Path("runs_2m_sphere/mo_ant/d"), 1e-2))
OUTPUT_DIR = Path("plots")
OUTPUT = OUTPUT_DIR / "ant_ridge_compare_four_panel.png"
RIDGE_OUTPUT = OUTPUT_DIR / "ant_ridge_compare_with_0001_2M_four_panel.png"
NO_LEGEND_OUTPUT = OUTPUT_DIR / "ant_ridge_compare_four_panel_no_legend.png"
OUTPUT_001 = OUTPUT_DIR / "ant_ridge_compare_ridge001_four_panel.png"
NO_LEGEND_OUTPUT_001 = OUTPUT_DIR / "ant_ridge_compare_ridge001_four_panel_no_legend.png"
MEAN_CVAR_OUTPUT = OUTPUT_DIR / "ant_ridge_compare_mean_cvar10.png"
WINDOW = 5
INTERVAL = 50_000
BASELINE_LOGGED_END = 1_700_000
BASELINE_ACTUAL_END = 2_000_000
INCLUDED_SEEDS = {1, 3}
METRICS = (("mean_return", "Mean"),
           ("worst_decile_return", "CVaR 10%"),
           ("forward_x", r"Forward $x$"),
           ("lateral_y", r"Lateral $y$"))
METHOD_STYLES = {"UNIFORM": ("#0072B2", "--"),
                 "D-LEVER": ("#D62728", "-")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def load_runs(roots=ROOTS):
    runs = {}
    for method, root, expected_ridge in roots:
        runs[method] = {}
        for path in sorted(root.glob("seed_*/metrics.csv")):
            if expected_ridge is not None:
                config = json.loads(path.with_name("config.json").read_text())
                if not np.isclose(config["ridge"], expected_ridge):
                    continue
            seed = int(path.parent.name.removeprefix("seed_"))
            if seed not in INCLUDED_SEEDS:
                continue
            with path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            run = {}
            for row in rows:
                step = int(row["env_steps"])
                run[step] = {"mean_return": float(row["mean_return"]),
                             "worst_decile_return": float(row["worst_decile_return"]),
                             "worst_quartile_return": float(row["worst_quartile_return"])}
            for eval_path in sorted(path.parent.glob("eval_*.npz")):
                step = int(eval_path.stem.removeprefix("eval_"))
                with np.load(eval_path) as evaluation:
                    measured = evaluation["measured_mu"].mean(0)
                # In Ant, the directional reward coordinates are x and y
                # velocity. The vertical z component is specific to Hopper.
                run.setdefault(step, {})["forward_x"] = float(measured[0])
                run[step]["lateral_y"] = float(measured[1])
            if run:
                runs[method][seed] = run
    if any(not method_runs for method_runs in runs.values()):
        raise FileNotFoundError("A requested Ant method has no runs")
    return runs


def smooth(run, metric):
    result = {}
    for end in sorted(step for step in run if step >= WINDOW * INTERVAL):
        window_steps = [end - i * INTERVAL for i in range(WINDOW)]
        if all(step in run and metric in run[step] for step in window_steps):
            result[end] = float(np.mean([run[step][metric] for step in window_steps]))
    return result


def draw(runs, methods, output, legend_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25), sharex=True)
    max_x = BASELINE_ACTUAL_END / 1e6
    for ax, (metric, title) in zip(axes, METRICS):
        for method in methods:
            color, linestyle = METHOD_STYLES[method]
            seed_curves = [smooth(run, metric) for run in runs[method].values()]
            steps = sorted(set().union(*(curve.keys() for curve in seed_curves)))
            means, ses = [], []
            for step in steps:
                values = np.asarray([curve[step] for curve in seed_curves if step in curve])
                means.append(values.mean())
                ses.append(values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.)
            # Map 1.7M logged steps to 2M; steps beyond that are extra training.
            step_array = np.asarray(steps)
            x = np.where(
                step_array <= BASELINE_LOGGED_END,
                step_array * (BASELINE_ACTUAL_END / BASELINE_LOGGED_END),
                BASELINE_ACTUAL_END + (step_array - BASELINE_LOGGED_END),
            ) / 1e6
            max_x = max(max_x, float(x.max(initial=0.)))
            means, ses = np.asarray(means), np.asarray(ses)
            ax.plot(x, means, color=color, ls=linestyle, lw=1.65, zorder=3)
            ax.fill_between(x, means - ses, means + ses, color=color,
                            alpha=.15, linewidth=0)
        ax.set_title(title, loc="left", pad=6)
        ax.grid(True, color="0.85", alpha=.45, linewidth=.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(.22, BASELINE_ACTUAL_END / 1e6)
        ax.set_xticks((.5, 1., 1.5, 2.))
        ax.set_xlabel("Steps (× 10⁶)")
        ax.set_xlabel(r"Steps ($\times 10^6$)")
    axes[0].set_ylabel("Discounted Return")
    axes[2].set_ylabel("Directional Return")
    legend_handles = [Line2D([], [], color=METHOD_STYLES[method][0],
                             ls=METHOD_STYLES[method][1], lw=2, label=method)
                      for method in methods]
    if legend_path is not None:
        fig.legend(handles=legend_handles, loc="upper center", ncol=len(methods),
                   frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.subplots_adjust(left=.085, right=.99, bottom=.28,
                        top=.76 if legend_path is not None else .91, wspace=.38)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    if legend_path is not None:
        legend = plt.figure(figsize=(3.1, .38))
        legend.legend(handles=legend_handles, loc="center", ncol=len(methods), frameon=False,
                      handlelength=2.5, columnspacing=1.8)
        legend.savefig(legend_path, dpi=300, bbox_inches="tight")
        legend.savefig(legend_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(legend)
    print(f"Saved {output}")


def draw_mean_cvar10(runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6), sharex=True)
    for ax, (metric, title) in zip(axes, METRICS[:2]):
        for method in ("UNIFORM", "D-LEVER"):
            color, linestyle = METHOD_STYLES[method]
            seed_curves = [smooth(run, metric) for run in runs[method].values()]
            steps = sorted(set().union(*(curve.keys() for curve in seed_curves)))
            means, ses = [], []
            for step in steps:
                values = np.asarray([curve[step] for curve in seed_curves if step in curve])
                means.append(values.mean())
                ses.append(values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.)
            step_array = np.asarray(steps)
            x = np.where(
                step_array <= BASELINE_LOGGED_END,
                step_array * (BASELINE_ACTUAL_END / BASELINE_LOGGED_END),
                BASELINE_ACTUAL_END + (step_array - BASELINE_LOGGED_END),
            ) / 1e6
            means, ses = np.asarray(means), np.asarray(ses)
            ax.plot(x, means, color=color, ls=linestyle, lw=1.65, zorder=3)
            ax.fill_between(x, means - ses, means + ses, color=color,
                            alpha=.15, linewidth=0)
        ax.set_title(title, loc="left", pad=6)
        ax.grid(True, color="0.85", alpha=.45, linewidth=.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(.22, BASELINE_ACTUAL_END / 1e6)
        ax.set_xticks((.5, 1., 1.5, 2.))
        ax.set_ylabel("Discounted Return")
    axes[-1].set_xlabel(r"Steps ($\times 10^6$)")
    fig.subplots_adjust(left=.12, right=.98, bottom=.24, top=.88, wspace=.32)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    runs = load_runs()
    runs_001 = load_runs(ROOTS_001)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw(runs, ("UNIFORM", "D-LEVER"), OUTPUT,
         OUTPUT_DIR / "ant_ridge_compare_legend.png")
    draw(runs, ("UNIFORM", "D-LEVER"), RIDGE_OUTPUT,
         OUTPUT_DIR / "ant_ridge_compare_with_0001_2M_legend.png")
    draw(runs, ("UNIFORM", "D-LEVER"), NO_LEGEND_OUTPUT)
    draw(runs_001, ("UNIFORM", "D-LEVER"), OUTPUT_001,
         OUTPUT_DIR / "ant_ridge_compare_ridge001_legend.png")
    draw(runs_001, ("UNIFORM", "D-LEVER"), NO_LEGEND_OUTPUT_001)
    draw_mean_cvar10(runs, MEAN_CVAR_OUTPUT)
    print("Seed endpoints:", "; ".join(
        f"{method} seed {seed}: {max(run):,}"
        for method, method_runs in runs.items()
        for seed, run in sorted(method_runs.items())))


if __name__ == "__main__":
    main()
