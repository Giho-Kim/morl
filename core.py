"""Conditional SF-SAC, pure reward SFs, and detached design curricula.

Each twin critic outputs (psi, h); Q_soft = z^T psi + temperature*h.
h is discounted future entropy, excluding the current action's entropy.
Temperature is tuned automatically; psi and h remain unweighted components.
"""
from copy import deepcopy
from contextlib import contextmanager
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def isolated_torch_rng(seed, device):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        yield


def sample_latents(rng, count, dim, prior, radius=1.0):
    if prior == "simplex":
        return rng.dirichlet(np.ones(dim), count).astype(np.float32) * radius
    z = rng.normal(size=(count, dim))
    if prior == "positive_sphere":
        z = np.abs(z)
    elif prior != "sphere":
        raise ValueError(prior)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return (radius * z).astype(np.float32)


def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


class Actor(nn.Module):
    def __init__(self, obs_dim, actions, dim, hidden, discrete, low=None, high=None):
        super().__init__()
        self.discrete, self.actions = discrete, actions
        self.net = mlp(obs_dim + dim, actions if discrete else 2 * actions, hidden)
        if not discrete:
            low, high = torch.as_tensor(low).float(), torch.as_tensor(high).float()
            self.register_buffer("scale", (high - low) / 2)
            self.register_buffer("bias", (high + low) / 2)

    def distribution(self, obs, z):
        output = self.net(torch.cat((obs, z), -1))
        if self.discrete:
            logp = output.log_softmax(-1)
            return logp.exp(), logp
        mean, logstd = output.chunk(2, -1)
        return torch.distributions.Normal(mean, logstd.clamp(-5, 2).exp())

    def sample(self, obs, z, deterministic=False):
        if self.discrete:
            probs, logp = self.distribution(obs, z)
            action = probs.argmax(-1) if deterministic else torch.multinomial(probs, 1).squeeze(-1)
            return action, logp.gather(1, action[:, None]).squeeze(-1)
        dist = self.distribution(obs, z)
        raw = dist.mean if deterministic else dist.rsample()
        action = raw.tanh() * self.scale + self.bias
        # Stable tanh Jacobian, plus affine action-rescaling Jacobian.
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        logp = (dist.log_prob(raw) - correction - self.scale.log()).sum(-1)
        return action, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, actions, dim, hidden, discrete):
        super().__init__()
        self.discrete, self.actions, self.dim = discrete, actions, dim
        self.net = mlp(obs_dim + dim + (0 if discrete else actions),
                       (dim + 1) * (actions if discrete else 1), hidden)

    def forward(self, obs, z, action=None):
        inputs = (obs, z) if self.discrete else (obs, z, action)
        out = self.net(torch.cat(inputs, -1))
        if self.discrete:
            out = out.reshape(-1, self.actions, self.dim + 1)
        return out[..., :self.dim], out[..., self.dim]


class SFSAC(nn.Module):
    def __init__(self, obs_dim, actions, dim, hidden=256, discrete=True,
                 low=None, high=None, temperature=.1):
        super().__init__()
        self.dim, self.discrete, self.temperature = dim, discrete, temperature
        self.actor = Actor(obs_dim, actions, dim, hidden, discrete, low, high)
        self.critics = nn.ModuleList([Critic(obs_dim, actions, dim, hidden, discrete)
                                     for _ in range(2)])

    def q(self, psi, h, z):
        weights = z[:, None, :] if psi.ndim == 3 else z
        return (psi * weights).sum(-1) + self.temperature * h

    @torch.no_grad()
    def act(self, obs, z, deterministic=False):
        action, _ = self.actor.sample(obs, z, deterministic)
        return action


def select_twin(outputs, z, temperature):
    """Select a whole (psi,h) pair by scalar soft Q, NEVER coordinatewise minima."""
    (p1, h1), (p2, h2) = outputs
    weights = z[:, None, :] if p1.ndim == 3 else z
    first = (p1 * weights).sum(-1) + temperature * h1 <= (p2 * weights).sum(-1) + temperature * h2
    return torch.where(first[..., None], p1, p2), torch.where(first, h1, h2)


@torch.no_grad()
def embeddings(net, starts, z, chunk=2048, action_samples=2, seed=1729):
    """E_s0 E_a~pi psi(s0,a,z), excluding entropy; twin mean reduces noise.

    Exact action expectation for discrete SAC; Monte Carlo for continuous SAC.
    Both critic and actor come from the same snapshot. Mean before outer product.
    Scoring has an isolated RNG, so it cannot perturb training action samples.
    """
    answer = []
    with isolated_torch_rng(seed, z.device):
        for zs in z.split(max(1, chunk // len(starts))):
            n, m = len(zs), len(starts)
            states, latents = starts.repeat(n, 1), zs.repeat_interleave(m, dim=0)
            if net.discrete:
                probs, _ = net.actor.distribution(states, latents)
                sf = sum(c(states, latents)[0] for c in net.critics) / 2
                mu = (probs[..., None] * sf).sum(1)
            else:
                mu = torch.zeros(n * m, net.dim, device=z.device)
                for _ in range(action_samples):
                    action, _ = net.actor.sample(states, latents)
                    mu += sum(c(states, latents, action)[0] for c in net.critics) / (2 * action_samples)
            answer.append(mu.reshape(n, m, -1).mean(1))
    return torch.cat(answer)


def design_scores(mu, gram, method):
    """Exact draft scores, computed in float64. No explicit matrix inverse."""
    mu, gram = mu.double(), gram.double()
    solved = torch.linalg.solve(gram, mu.T).T
    lev = (mu * solved).sum(-1).clamp_min(0)
    if method in ("d", "uniform"):
        score = lev
    elif method == "a":
        score = solved.square().sum(-1) / (1 + lev)
    elif method == "e":
        base = torch.linalg.eigvalsh(gram)[0]
        score = torch.linalg.eigvalsh(gram[None] + mu[:, :, None] * mu[:, None, :])[:, 0] - base
    else:
        raise ValueError(method)
    return score.clamp_min(0), lev


def mixture_probs(scores, eta):
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0,1]")
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise FloatingPointError("Invalid curriculum scores")
    uniform = torch.full_like(scores, 1 / len(scores))
    if scores.sum() <= 0:
        return uniform
    return (1 - eta) * uniform + eta * scores / scores.sum()


class Curriculum:
    def __init__(self, net, starts, rng, method, prior, batch_size=256,
                 multiplier=10, eta=.9, ridge=1e-3, alpha=.005, refresh=5,
                 radius=1., action_samples=2, score_seed=1729):
        self.scorer = deepcopy(net).eval().requires_grad_(False)
        self.starts, self.rng = starts, rng
        self.method, self.prior = method, prior
        self.batch_size, self.multiplier = batch_size, multiplier
        self.eta, self.ridge, self.alpha, self.refresh = eta, ridge, alpha, refresh
        self.radius, self.dim = radius, net.dim
        self.action_samples, self.score_seed = action_samples, score_seed
        self.gram = self.z = self.probs = None
        self.stats = {}

    @torch.no_grad()
    def sample(self, net, update):
        if self.z is None or update % self.refresh == 0:
            # Detached ONLINE actor + critic snapshot: current stochastic policy.
            self.scorer.load_state_dict(net.state_dict())
            count = self.batch_size * self.multiplier
            self.z = torch.as_tensor(sample_latents(self.rng, count, self.dim,
                                      self.prior, self.radius), device=self.starts.device)
            self.mu = embeddings(self.scorer, self.starts, self.z,
                                 action_samples=self.action_samples,
                                 seed=self.score_seed + update).double()
            empirical = self.mu.T @ self.mu / count
            ada = max(self.ridge * empirical.trace().item() / self.dim, 1e-8)
            regularized = empirical + ada * torch.eye(self.dim, device=empirical.device,
                                                      dtype=torch.float64)
            self.gram = regularized if self.gram is None else (
                (1 - self.alpha) * self.gram + self.alpha * regularized)
            self.scores, lev = design_scores(self.mu, self.gram, self.method)
            self.probs = mixture_probs(self.scores, 0 if self.method == "uniform" else self.eta)
            self.stats = dict(train_logdet=torch.linalg.slogdet(self.gram).logabsdet.item(),
                              mean_leverage=lev.mean().item(), adaptive_ridge=ada,
                              sampling_ess=(1 / self.probs.square().sum()).item())
        # Replacement is intentional; do not differentiate through q_eta.
        index = self.rng.choice(len(self.z), self.batch_size, replace=True,
                                p=self.probs.cpu().numpy() / self.probs.sum().item())
        return self.z[torch.as_tensor(index, device=self.z.device)].detach()


class Replay:
    def __init__(self, capacity, obs_dim, dim, actions=1, discrete=True):
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.obs = np.empty((capacity, obs_dim), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.reward = np.empty((capacity, dim), np.float32)
        self.action = np.empty(capacity if discrete else (capacity, actions),
                               np.int64 if discrete else np.float32)
        self.terminal = np.empty(capacity, np.float32)

    def add(self, obs, action, reward, next_obs, terminal):
        i = self.pos
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.terminal[i] = next_obs, terminal
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, batch, device):
        i = rng.integers(self.size, size=batch)
        return tuple(torch.as_tensor(x[i], device=device) for x in
                     (self.obs, self.action, self.reward, self.next_obs, self.terminal))


@torch.no_grad()
def bellman_target(online, target_critics, next_obs, z, reward, terminal, gamma):
    if online.discrete:
        probs, logp = online.actor.distribution(next_obs, z)
        psi, h = select_twin([c(next_obs, z) for c in target_critics], z, online.temperature)
        future_sf = (probs[..., None] * psi).sum(1)
        future_h = (probs * (h - logp)).sum(1)
    else:
        action, logp = online.actor.sample(next_obs, z)
        future_sf, h = select_twin([c(next_obs, z, action) for c in target_critics],
                                   z, online.temperature)
        future_h = h - logp
    continuation = gamma * (1 - terminal)
    return reward + continuation[:, None] * future_sf, continuation * future_h


def learner_step(online, target_critics, critic_optimizer, actor_optimizer,
                 batch, z, gamma, tau):
    obs, actions, reward, next_obs, terminal = batch
    wanted_sf, wanted_h = bellman_target(online, target_critics, next_obs, z, reward, terminal, gamma)
    sf_loss, h_loss = 0., 0.
    for critic in online.critics:
        psi, h = critic(obs, z, None if online.discrete else actions)
        if online.discrete:
            idx = torch.arange(len(z), device=z.device)
            psi, h = psi[idx, actions], h[idx, actions]
        sf_loss = sf_loss + F.mse_loss(psi, wanted_sf)
        h_loss = h_loss + F.mse_loss(h, wanted_h)
    loss = sf_loss + online.temperature ** 2 * h_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite critic loss")
    critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = nn.utils.clip_grad_norm_(online.critics.parameters(), 10.)
    critic_optimizer.step()
    online.critics.requires_grad_(False)
    try:
        if online.discrete:
            probs, logp = online.actor.distribution(obs, z)
            qs = [online.q(*c(obs, z), z) for c in online.critics]
            actor_loss = (probs * (online.temperature * logp - torch.minimum(*qs))).sum(-1).mean()
            entropy = -(probs * logp).sum(-1).mean()
        else:
            action, logp = online.actor.sample(obs, z)
            qs = [online.q(*c(obs, z, action), z) for c in online.critics]
            actor_loss = (online.temperature * logp - torch.minimum(*qs)).mean()
            entropy = -logp.mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("Non-finite actor loss")
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(online.actor.parameters(), 10.)
        actor_optimizer.step()
    finally:
        online.critics.requires_grad_(True)
    with torch.no_grad():
        for p, q in zip(online.critics.parameters(), target_critics.parameters()):
            q.lerp_(p, tau)
    return dict(sf_td_loss=sf_loss.item(), entropy_td_loss=h_loss.item(),
                actor_loss=actor_loss.item(), policy_entropy=entropy.item(), grad_norm=float(grad))


def temperature_step(online, optimizer, log_temperature, obs, z, target_entropy):
    """Tune log alpha using exact discrete entropy or a continuous policy sample."""
    with torch.no_grad():
        if online.discrete:
            probs, logp = online.actor.distribution(obs, z)
            entropy = -(probs * logp).sum(-1)
        else:
            _, logp = online.actor.sample(obs, z)
            entropy = -logp
        entropy_error = entropy - target_entropy
    loss = (log_temperature * entropy_error).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite temperature loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    temperature = log_temperature.detach().exp()
    if not torch.isfinite(temperature):
        raise FloatingPointError("Non-finite temperature")
    online.temperature = temperature.item()
    return dict(temperature=online.temperature, temperature_loss=loss.item(),
                target_entropy=float(target_entropy))
