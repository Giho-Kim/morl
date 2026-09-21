"""Train/evaluate D-LEVER with model-free conditional SF-SAC."""
import argparse
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from pymoo.util.ref_dirs import get_reference_directions
from tqdm.auto import tqdm

from core import (Curriculum, Replay, SFSAC, isolated_torch_rng, embeddings, learner_step,
                  sample_latents)
from envs import Benchmark, SPECS, isolated_global_rng


@dataclass
class Config:
    env: str = "fruit_tree"
    method: str = "d"
    seed: int = 0
    depth: int = 6
    horizon: int | None = None
    steps: int | None = None
    prior: str | None = None
    radius: float = 1.
    gamma: float = .99
    batch_size: int = 128
    multiplier: int = 10
    eta: float = .9
    ridge: float = 1e-2
    alpha: float = .005
    refresh: int = 5
    hidden: int = 256
    lr: float = 3e-4
    tau: float = .005
    replay_size: int = 200_000
    learning_starts: int = 2000
    train_every: int = 1
    temperature: float = .1
    embedding_action_samples: int = 8
    eval_every: int = 20_000
    eval_tasks: int = 10
    eval_episodes: int = 10
    eval_seed: int = 2027
    eval_ridge: float = 1e-3
    start_samples: int = 8
    device: str = "auto"
    threads: int = 1
    output: str = "runs"

    def resolve(self):
        if self.steps is None:
            self.steps = SPECS[self.env][3]
        if self.prior is None:
            self.prior = SPECS[self.env][2]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        positive = (self.steps, self.batch_size, self.multiplier, self.refresh,
                    self.replay_size, self.eval_every, self.eval_tasks,
                    self.eval_episodes, self.start_samples, self.train_every,
                    self.hidden, self.lr, self.ridge, self.eval_ridge, self.radius,
                    self.threads, self.temperature, self.embedding_action_samples)
        if min(positive) <= 0:
            raise ValueError("Counts, scales, rates and ridge must be positive")
        if not (0 <= self.eta <= 1 and 0 < self.alpha <= 1 and 0 < self.tau <= 1
                and 0 < self.gamma < 1):
            raise ValueError("Invalid probability, discount, EMA or exploration settings")
        if self.learning_starts < 0 or (self.horizon is not None and self.horizon < 1):
            raise ValueError("Invalid warmup or horizon")
        if self.env == "fruit_tree" and self.horizon is not None and self.horizon < self.depth:
            raise ValueError("Fruit Tree horizon must allow reaching a leaf")
        return self


def tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def initial_bank(cfg):
    with isolated_global_rng(cfg.eval_seed + 777):
        env = Benchmark(cfg.env, cfg.depth, cfg.horizon)
        states = [env.reset(seed=cfg.eval_seed + 777 + j)[0] for j in range(cfg.start_samples)]
        env.close()
    return np.stack(states)


def evaluation_tasks(cfg, dim):
    """Fixed benchmark preferences; simplex tasks use Riesz-energy directions."""
    if cfg.prior == "simplex":
        tasks = get_reference_directions("energy", dim, cfg.eval_tasks,
                                         seed=cfg.eval_seed)
        return (cfg.radius * tasks).astype(np.float32)
    return sample_latents(np.random.default_rng(cfg.eval_seed), cfg.eval_tasks,
                          dim, cfg.prior, cfg.radius)


def iqm(values, axis=-1):
    """25%-trimmed mean (IQM), matching scipy.stats.trim_mean semantics."""
    values = np.sort(np.asarray(values), axis=axis)
    cut = int(values.shape[axis] * .25)
    if cut == 0:
        return values.mean(axis=axis)
    keep = [slice(None)] * values.ndim
    keep[axis] = slice(cut, -cut)
    return values[tuple(keep)].mean(axis=axis)


@torch.no_grad()
def evaluate(net, cfg, tasks, starts):
    """Independent fixed tasks + common episode seeds. No test-task updates."""
    device = next(net.parameters()).device
    returns = np.zeros((len(tasks), cfg.eval_episodes, net.dim), np.float64)
    terminal_ids = np.full((len(tasks), cfg.eval_episodes), -1, np.int64)
    lengths = np.zeros((len(tasks), cfg.eval_episodes), np.int64)
    endings = np.zeros((len(tasks), cfg.eval_episodes), bool)
    with isolated_global_rng(cfg.eval_seed), isolated_torch_rng(cfg.eval_seed, device):
        env = Benchmark(cfg.env, cfg.depth, cfg.horizon)
        for j, z in enumerate(tasks):
            zt = tensor(z[None], device)
            for k in range(cfg.eval_episodes):
                seed = cfg.eval_seed + 10000 + k  # common randomness across tasks/methods
                with isolated_global_rng(seed), isolated_torch_rng(seed, device):
                    obs, _ = env.reset(seed=seed)
                    for t in range(env.horizon):
                        action_tensor = net.act(tensor(obs[None], device), zt)
                        action = action_tensor.item() if net.discrete else action_tensor[0].cpu().numpy()
                        obs, reward, terminated, truncated, info = env.step(action)
                        returns[j, k] += cfg.gamma ** t * reward
                        if terminated or truncated:
                            lengths[j, k] = t + 1
                            endings[j, k] = terminated
                            if cfg.env == "fruit_tree" and terminated:
                                terminal_ids[j, k] = int(info["raw_state"][1])
                            break
        env.close()
    utility = np.einsum("nd,nkd->nk", tasks, returns)
    # One robust scalar score per weight. With the default 10 rollouts this
    # discards the lowest/highest two and averages the central six.
    utility_iqm = iqm(utility, axis=1)
    measured = returns.mean(1)
    predicted = embeddings(net, tensor(starts, device), tensor(tasks, device),
                           action_samples=cfg.embedding_action_samples, seed=cfg.eval_seed).cpu().numpy()
    gram = measured.T @ measured / len(tasks) + cfg.eval_ridge * np.eye(net.dim)
    tail_n = max(1, int(np.ceil(.1 * len(tasks))))
    metrics = dict(mean_return=float(utility_iqm.mean()),
                   worst_decile_return=float(np.sort(utility_iqm)[:tail_n].mean()),
                   min_return=float(utility_iqm.min()),
                   rollout_logdet=float(np.linalg.slogdet(gram)[1]),
                   rollout_min_eigenvalue=float(np.linalg.eigvalsh(gram)[0]),
                   embedding_rmse=float(np.sqrt(np.mean((predicted - measured) ** 2))),
                   mean_episode_length=float(lengths.mean()),
                   timeout_fraction=float(1 - endings.mean()),
                   eval_transitions=int(lengths.sum()))
    if cfg.env == "fruit_tree":
        metrics["unique_leaves"] = int(len(np.unique(terminal_ids[terminal_ids >= 0])))
    arrays = dict(tasks=tasks, returns=returns, utilities=utility,
                  utility_iqm=utility_iqm, measured_mu=measured,
                  predicted_mu=predicted, lengths=lengths, terminal_ids=terminal_ids)
    return metrics, arrays


def make_model(cfg, env):
    return SFSAC(env.obs_dim, env.actions, env.dim, cfg.hidden, env.discrete,
                 env.low, env.high, cfg.temperature).to(cfg.device)


def train(cfg):
    cfg.resolve()
    torch.set_num_threads(cfg.threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    # Independent streams: evaluation/candidate counts do not shift replay/exploration RNG.
    streams = np.random.SeedSequence(cfg.seed).spawn(3)
    behavior_rng, replay_rng, design_rng = [np.random.default_rng(s) for s in streams]
    directory = Path(cfg.output) / cfg.env / cfg.method / f"seed_{cfg.seed}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "torch", "mo-gymnasium", "gymnasium", "mujoco")}
    (directory / "versions.json").write_text(json.dumps(versions, indent=2))
    env = Benchmark(cfg.env, cfg.depth, cfg.horizon)
    online = make_model(cfg, env)
    target = deepcopy(online.critics).eval().requires_grad_(False)
    critic_optimizer = torch.optim.Adam(online.critics.parameters(), lr=cfg.lr)
    actor_optimizer = torch.optim.Adam(online.actor.parameters(), lr=cfg.lr)
    replay = Replay(cfg.replay_size, env.obs_dim, env.dim, env.actions, env.discrete)
    starts = initial_bank(cfg)
    curriculum = Curriculum(online, tensor(starts, cfg.device), design_rng, cfg.method,
                            cfg.prior, cfg.batch_size, cfg.multiplier, cfg.eta,
                            cfg.ridge, cfg.alpha, cfg.refresh, cfg.radius,
                            cfg.embedding_action_samples, cfg.seed + 17000)
    tasks = evaluation_tasks(cfg, env.dim)
    np.save(directory / "eval_tasks.npy", tasks)
    np.save(directory / "initial_states.npy", starts)
    obs, _ = env.reset(seed=cfg.seed)
    z_behavior = sample_latents(behavior_rng, 1, env.dim, cfg.prior, cfg.radius)[0]
    updates, episode = 0, 0
    learner_metrics = dict(sf_td_loss=0., entropy_td_loss=0., actor_loss=0.,
                           policy_entropy=0., grad_norm=0.)
    training_seconds = 0.
    total_eval_steps = 0
    columns = None
    progress = tqdm(total=cfg.steps, desc=f"{cfg.env}/{cfg.method}/seed_{cfg.seed}",
                    unit="step", dynamic_ncols=True)

    def record(step):
        nonlocal columns, total_eval_steps
        progress.set_postfix(stage=f"evaluating@{step}", refresh=True)
        metrics, arrays = evaluate(online, cfg, tasks, starts)
        total_eval_steps += metrics["eval_transitions"]
        metrics.update(env_steps=step, updates=updates, episodes=episode,
                       training_seconds=training_seconds,
                       cumulative_eval_transitions=total_eval_steps,
                       **learner_metrics)
        metrics.update({k: curriculum.stats.get(k, 0.) for k in
                        ("train_logdet", "mean_leverage", "adaptive_ridge", "sampling_ess")})
        if columns is None:
            columns = list(metrics)
        with (directory / "metrics.csv").open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if step == 0:
                writer.writeheader()
            writer.writerow(metrics)
        np.savez_compressed(directory / f"eval_{step:09d}.npz", **arrays)
        if curriculum.z is not None:
            np.savez_compressed(directory / f"design_{step:09d}.npz",
                z=curriculum.z.cpu().numpy(), mu=curriculum.mu.cpu().numpy(),
                scores=curriculum.scores.cpu().numpy(), probs=curriculum.probs.cpu().numpy(),
                gram=curriculum.gram.cpu().numpy())
        # Inference checkpoint, not an exact training-resume snapshot.
        torch.save(dict(config=asdict(cfg), model=online.state_dict(), steps=step,
                        obs_dim=env.obs_dim, actions=env.actions, dim=env.dim,
                        discrete=env.discrete,
                        low=None if env.discrete else env.low.tolist(),
                        high=None if env.discrete else env.high.tolist(),
                        algorithm="conditional_sf_sac_v1"),
                   directory / "latest.pt")
        progress.write(json.dumps(dict(env=cfg.env, method=cfg.method, seed=cfg.seed,
                                       **metrics)))
        progress.set_postfix(stage="training", updates=updates, episodes=episode,
                             refresh=True)

    record(0)
    try:
        for step in range(1, cfg.steps + 1):
            tic = time.perf_counter()
            if step <= cfg.learning_starts:
                action = (int(behavior_rng.integers(env.actions)) if env.discrete else
                          behavior_rng.uniform(env.low, env.high).astype(np.float32))
            else:
                with torch.no_grad():
                    action_tensor = online.act(tensor(obs[None], cfg.device),
                                               tensor(z_behavior[None], cfg.device))
                    action = action_tensor.item() if env.discrete else action_tensor[0].cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            if step >= cfg.learning_starts and step % cfg.train_every == 0:
                z = curriculum.sample(online, updates)
                batch = replay.sample(replay_rng, cfg.batch_size, cfg.device)
                learner_metrics = learner_step(online, target, critic_optimizer, actor_optimizer,
                                                batch, z, cfg.gamma, cfg.tau)
                updates += 1
            if terminated or truncated:
                episode += 1
                obs, _ = env.reset()
                z_behavior = sample_latents(behavior_rng, 1, env.dim, cfg.prior, cfg.radius)[0]
            if cfg.device.startswith("cuda"):
                torch.cuda.synchronize()
            training_seconds += time.perf_counter() - tic
            progress.update()
            if step % cfg.eval_every == 0 or step == cfg.steps:
                record(step)
    finally:
        progress.close()
        env.close()
    return directory


def parse_config():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=SPECS, default="fruit_tree")
    parser.add_argument("--method", choices=("uniform", "d", "a", "e"), default="d")
    parser.add_argument("--prior", choices=("sphere", "positive_sphere", "simplex"))
    defaults = Config()
    for name in Config.__dataclass_fields__:
        if name in ("env", "method", "prior"):
            continue
        default = getattr(defaults, name)
        kind = int if name in ("steps", "horizon") else type(default)
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=default)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_config())
