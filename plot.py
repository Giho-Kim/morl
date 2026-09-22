"""Per-environment learning curves; shaded SE across independent training seeds."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("plots"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    keys = ["mean_return", "train_logdet", "rollout_logdet", "embedding_rmse"]
    for envdir in sorted(args.root.iterdir()):
        if not envdir.is_dir():
            continue
        fig, axes = plt.subplots(1, len(keys), figsize=(17, 3.5))
        found = False
        for method in ("uniform", "d", "a", "e"):
            data = []
            for file in sorted((envdir / method).glob("seed_*/metrics.csv")):
                with file.open() as f:
                    rows = list(csv.DictReader(f))
                data.append({int(row["env_steps"]): row for row in rows})
            if not data:
                continue
            found = True
            steps = sorted(set.intersection(*(set(d) for d in data)))
            for ax, key in zip(axes, keys):
                y = np.array([[float(d[s][key]) for s in steps] for d in data])
                mean = y.mean(0)
                se = y.std(0, ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.zeros_like(mean)
                line, = ax.plot(steps, mean, label=f"{method} (n={len(y)})")
                ax.fill_between(steps, mean - se, mean + se, color=line.get_color(), alpha=.15)
                ax.set_title(key.replace("_", " "))
                ax.set_xlabel("training environment steps")
        if found:
            axes[0].legend()
            fig.suptitle(envdir.name + " — mean ± SE across seeds")
            fig.tight_layout()
            fig.savefig(args.output / f"{envdir.name}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
