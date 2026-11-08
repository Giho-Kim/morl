"""MO-Hopper analogue of note/analyze_tilt_components.py.

D-LEVER score ell(z) = mu_z^T G^-1 mu_z, decomposed as
    norm2(z)     = ||mu_z||^2
    direction(z) = u_z^T G^-1 u_z,  u_z = mu_z / ||mu_z||
so ell = norm2 * direction exactly.  Sampling probabilities follow training:
q = (1-eta)/n + eta * s / sum(s), with eta = 0.9 (uniform share 0.1).

Per checkpoint (latest.pt), the Gram is FROZEN: the training EMA Gram saved in
design_<step>.npz at the checkpoint step.  Candidates (2560 = 256*10 prior draws)
and the initial-state bank (8 resets) are drawn fresh and crossed over five
candidate seeds x five bank seeds.  Results are averaged within a checkpoint
first, then mean +- SE across training seeds.  Uniform checkpoints use the same
counterfactual eta = 0.9.  A sanity pass decomposes the saved training design
(the actual candidates, mu and Gram) and checks it against the saved scores.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import embeddings, sample_latents  # noqa: E402
from envs import Benchmark, isolated_global_rng  # noqa: E402
from hopper_reward_direction_noise import RUNS, METHODS, SEEDS, load  # noqa: E402

OUT = ROOT / "results" / "hopper_tilt_components"
CANDIDATE_SEEDS = BANK_SEEDS = [42, 43, 44, 45, 46]


def components(mu, gram):
    mu, gram = np.asarray(mu, np.float64), np.asarray(gram, np.float64)
    norm2 = (mu ** 2).sum(1)
    u = mu / np.sqrt(np.maximum(norm2, 1e-30))[:, None]
    direction = (u * np.linalg.solve(gram, u.T).T).sum(1)
    return norm2, direction, norm2 * direction


def metrics(norm2, direction, score, eta):
    def prob(s):
        s = np.clip(s, 0, None)
        return (1 - eta) / len(s) + eta * s / s.sum()
    full = prob(score)
    logn, logd = np.log(norm2), np.log(direction)
    n = max(1, len(score) // 10)
    top = lambda x: set(np.argsort(x)[-n:].tolist())
    return dict(rho_score_norm2=float(spearmanr(score, norm2).statistic),
                rho_score_direction=float(spearmanr(score, direction).statistic),
                rho_norm_direction=float(spearmanr(norm2, direction).statistic),
                log_norm2_sd=float(logn.std()), log_direction_sd=float(logd.std()),
                top10_overlap_norm=len(top(score) & top(norm2)) / n,
                top10_overlap_direction=len(top(score) & top(direction)) / n,
                # Removing norm leaves direction-only weights, and vice versa.
                tv_remove_norm=float(abs(full - prob(direction)).sum() / 2),
                tv_remove_direction=float(abs(full - prob(norm2)).sum() / 2),
                prob_max_over_uniform=float(full.max() * len(full)))


def bank(cfg, bank_seed):
    with isolated_global_rng(bank_seed):
        env = Benchmark(cfg.env, cfg.horizon, cfg.reward_layout)
        states = [env.reset(seed=1_000_000 * bank_seed + j)[0] for j in range(cfg.start_samples)]
        env.close()
    return torch.as_tensor(np.stack(states), dtype=torch.float32)


@torch.no_grad()
def main():
    import pandas as pd
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sanity = [], []
    for method in METHODS:
        for seed in SEEDS:
            directory = RUNS / method / f"seed_{seed}"
            net, cfg, step = load(directory / "latest.pt")
            design = np.load(directory / f"design_{step:09d}.npz")
            gram, eta = design["gram"], .9
            # Sanity: the saved training design decomposes into its saved scores.
            n2, dr, sc = components(design["mu"], gram)
            assert np.allclose(sc, design["scores"], rtol=1e-6, atol=1e-9), (method, seed)
            sanity.append(dict(method=method, seed=seed, step=step,
                               **metrics(n2, dr, sc, eta)))
            count = cfg.batch_size * cfg.multiplier
            for bank_seed in BANK_SEEDS:
                starts = bank(cfg, bank_seed)
                for cand_seed in CANDIDATE_SEEDS:
                    z = torch.as_tensor(sample_latents(np.random.default_rng(cand_seed), count,
                                                       net.dim, cfg.prior, cfg.radius))
                    mu = embeddings(net, starts, z, action_samples=cfg.embedding_action_samples,
                                    seed=cand_seed * 100 + bank_seed).numpy()
                    n2, dr, sc = components(mu, gram)
                    rows.append(dict(method=method, seed=seed, step=step, bank_seed=bank_seed,
                                     candidate_seed=cand_seed, **metrics(n2, dr, sc, eta)))
            print(method, seed, step, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_results.csv", index=False)
    pd.DataFrame(sanity).to_csv(OUT / "saved_design_sanity.csv", index=False)
    numeric = [c for c in df.columns if c not in ("method", "seed", "step", "bank_seed", "candidate_seed")]
    per = df.groupby(["method", "seed", "step"], as_index=False)[numeric].mean()
    per.to_csv(OUT / "per_checkpoint_mean.csv", index=False)
    se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
    summary = []
    for method in METHODS:
        g = per[per["method"] == method]
        summary.append(dict(method=method, n_checkpoints=len(g),
                            **{f"{c}_mean": g[c].mean() for c in numeric},
                            **{f"{c}_se": se(g[c]) for c in numeric}))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "mean_se.csv", index=False)
    sane = pd.DataFrame(sanity)
    names = {"uniform": "Uniform", "d": "D-LEVER"}
    lines = ["| Model | Rank corr score–norm² | Rank corr score–direction | Remove norm: TV | Remove direction: TV |",
             "|---|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        lines.append(f"| {names[r['method']]} (n={r.n_checkpoints}) | "
                     f"{r.rho_score_norm2_mean:.3f} ± {r.rho_score_norm2_se:.3f} | "
                     f"{r.rho_score_direction_mean:.3f} ± {r.rho_score_direction_se:.3f} | "
                     f"{100 * r.tv_remove_norm_mean:.1f}% ± {100 * r.tv_remove_norm_se:.1f}%p | "
                     f"{100 * r.tv_remove_direction_mean:.1f}% ± {100 * r.tv_remove_direction_se:.1f}%p |")
    lines += ["", "Saved training design at the same step (actual candidates, μ and Gram; no resampling):", "",
              "| Model | Rank corr score–norm² | Rank corr score–direction | Remove norm: TV | Remove direction: TV |",
              "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        g = sane[sane["method"] == method]
        lines.append(f"| {names[method]} (n={len(g)}) | {g.rho_score_norm2.mean():.3f} ± {se(g.rho_score_norm2):.3f} | "
                     f"{g.rho_score_direction.mean():.3f} ± {se(g.rho_score_direction):.3f} | "
                     f"{100 * g.tv_remove_norm.mean():.1f}% ± {100 * se(g.tv_remove_norm):.1f}%p | "
                     f"{100 * g.tv_remove_direction.mean():.1f}% ± {100 * se(g.tv_remove_direction):.1f}%p |")
    steps = ", ".join(f"{names[m]} seed {s}: {st // 1000}k" for m, s, st in
                      zip(per["method"], per.seed, per.step))
    lines += ["", "Score ell = ||μ||² · uᵀG⁻¹u with u = μ/||μ|| (exact). Gram: the training EMA Gram saved at the "
              f"checkpoint step, frozen ({steps}). Each checkpoint was evaluated at five candidate seeds (42–46, "
              "2560 prior draws each) × five fresh 8-state reset banks (42–46); results are averaged within the "
              "checkpoint, then mean ± SE across training seeds 0, 1, 2, 4. Both methods use the D-LEVER mixture "
              "with η = 0.9 (uniform share 0.1); for Uniform this is counterfactual. TV is on the candidate draw "
              "probabilities before without-replacement depletion. μ includes the fixed common-term SF coordinate."]
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
