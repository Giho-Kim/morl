from copy import deepcopy
import random

import numpy as np
import mo_gymnasium as mo
import pytest
import torch

from core import (Curriculum, SFSAC, bellman_target, learner_step, isolated_torch_rng,
                  select_twin, design_scores, embeddings, mixture_probs, sample_latents,
                  temperature_step)
from envs import Benchmark, isolated_global_rng


def test_design_scores_against_definitions():
    mu = torch.tensor([[2., 1., 0.], [0., 1., 3.]], dtype=torch.float64)
    gram = torch.tensor([[3., .2, 0.], [.2, 2., .1], [0., .1, 1.]], dtype=torch.float64)
    inverse = torch.linalg.inv(gram)
    d, _ = design_scores(mu, gram, "d")
    a, _ = design_scores(mu, gram, "a")
    e, _ = design_scores(mu, gram, "e")
    for i, v in enumerate(mu):
        updated = gram + torch.outer(v, v)
        assert d[i] == pytest.approx((v @ inverse @ v).item())
        assert a[i] == pytest.approx((inverse.trace() - torch.linalg.inv(updated).trace()).item())
        assert e[i] == pytest.approx((torch.linalg.eigvalsh(updated)[0] - torch.linalg.eigvalsh(gram)[0]).item())
        h = 1e-6
        gain = (torch.logdet(gram + h * torch.outer(v, v)) - torch.logdet(gram)) / h
        assert gain.item() == pytest.approx(d[i].item(), rel=1e-5)


def test_mixture_floor_zero_and_covariance_identity():
    scores = torch.tensor([0., 1., 5.], dtype=torch.float64)
    f = torch.tensor([4., -2., 3.], dtype=torch.float64)
    for eta in (0., .3, 1.):
        q = mixture_probs(scores, eta)
        assert q.sum().item() == pytest.approx(1.)
        assert torch.all(q >= (1 - eta) / 3)
        cov = (f * scores).mean() - f.mean() * scores.mean()
        assert (q @ f).item() == pytest.approx((f.mean() + eta * cov / scores.mean()).item())
    assert torch.allclose(mixture_probs(torch.zeros(3), 1), torch.ones(3) / 3)


def test_mean_before_outer_product_and_no_entropy_contamination():
    class Actor:
        def distribution(self, obs, z):
            probs = torch.tensor([.25, .75]).expand(len(z), -1)
            return probs, probs.log()
    class Critic:
        def __call__(self, obs, z):
            return torch.stack((obs, 3 * obs), dim=1), torch.full((len(z), 2), 1e6)
    class Fake:
        discrete = True
        actor = Actor()
        critics = [Critic(), Critic()]
    starts = torch.tensor([[1., 0.], [0., 2.]])
    z = torch.tensor([[1., 1.], [1., 1.]])
    mu = embeddings(Fake(), starts, z)
    assert torch.allclose(mu, torch.tensor([[1.25, 2.5], [1.25, 2.5]]))
    assert not torch.allclose(mu.T @ mu / 2, (starts * 2.5).T @ (starts * 2.5) / 2)


@pytest.mark.parametrize("discrete", [True, False])
def test_soft_bellman_identity_and_terminal_mask(discrete):
    # Scalarizing the two targets must give the standard SAC target exactly.
    torch.manual_seed(42)
    net = SFSAC(4, 2, 3, 8, discrete, [-2., -1.], [2., 3.], temperature=.2)
    target = deepcopy(net.critics)
    obs, z, reward = torch.randn(5, 4), torch.randn(5, 3), torch.randn(5, 3)
    done = torch.tensor([0., 0., 1., 0., 1.])
    with isolated_torch_rng(8, "cpu"):
        psi_y, h_y = bellman_target(net, target, obs, z, reward, done, .9)
    with isolated_torch_rng(8, "cpu"), torch.no_grad():
        if discrete:
            probs, logp = net.actor.distribution(obs, z)
            qmin = torch.minimum(*[net.q(*c(obs, z), z) for c in target])
            value = (probs * (qmin - .2 * logp)).sum(-1)
        else:
            action, logp = net.actor.sample(obs, z)
            qmin = torch.minimum(*[net.q(*c(obs, z, action), z) for c in target])
            value = qmin - .2 * logp
        expected = (reward * z).sum(-1) + .9 * (1 - done) * value
    assert torch.allclose((psi_y * z).sum(-1) + .2 * h_y, expected, atol=1e-6)
    assert torch.equal(psi_y[done.bool()], reward[done.bool()])
    assert torch.equal(h_y[done.bool()], torch.zeros(2))
    assert not psi_y.requires_grad and not h_y.requires_grad


def test_twin_selection_is_whole_vector():
    z = torch.tensor([[1., 1.]])
    p1, p2 = torch.tensor([[10., 0.]]), torch.tensor([[0., 20.]])
    selected, h = select_twin([(p1, torch.tensor([1.])), (p2, torch.tensor([2.]))], z, .1)
    assert torch.equal(selected, p1) and h.item() == 1.


@pytest.mark.parametrize("discrete", [True, False])
def test_actor_and_critic_update_and_action_bounds(discrete):
    torch.manual_seed(3)
    net = SFSAC(4, 2, 3, 16, discrete, [-2., -1.], [2., 3.])
    target = deepcopy(net.critics).requires_grad_(False)
    obs, z = torch.randn(8, 4), torch.randn(8, 3)
    action, logp = net.actor.sample(obs, z)
    assert torch.isfinite(logp).all()
    if not discrete:
        assert (action >= torch.tensor([-2., -1.])).all()
        assert (action <= torch.tensor([2., 3.])).all()
    actor_before = [p.detach().clone() for p in net.actor.parameters()]
    critic_before = [p.detach().clone() for p in net.critics.parameters()]
    result = learner_step(net, target, torch.optim.Adam(net.critics.parameters()),
                          torch.optim.Adam(net.actor.parameters()),
                          (obs, action.detach(), torch.randn(8, 3), obs, torch.zeros(8)), z, .99, .005)
    assert all(np.isfinite(v) for v in result.values())
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, net.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, net.critics.parameters()))


@pytest.mark.parametrize("discrete", [True, False])
def test_automatic_temperature_tuning(discrete):
    torch.manual_seed(11)
    net = SFSAC(4, 2, 3, 8, discrete, [-1., -1.], [1., 1.], temperature=.1)
    obs, z = torch.randn(16, 4), torch.randn(16, 3)
    log_temperature = torch.tensor(np.log(.1), requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=3e-4)
    target = .98 * np.log(2) if discrete else -2.
    before = log_temperature.detach().clone()
    metrics = temperature_step(net, optimizer, log_temperature, obs, z, target)
    assert not torch.equal(before, log_temperature)
    assert metrics["temperature"] == pytest.approx(log_temperature.exp().item())
    assert metrics["target_entropy"] == pytest.approx(target)
    assert all(np.isfinite(v) for v in metrics.values())


def test_curriculum_snapshot_cache_ema_and_without_replacement():
    torch.manual_seed(5)
    net = SFSAC(4, 2, 3, 8)
    curriculum = Curriculum(net, torch.zeros(2, 4), np.random.default_rng(7),
                             "d", "sphere", batch_size=16, multiplier=2,
                             alpha=.2, refresh=2)
    selected = curriculum.sample(net, 0)
    assert len(torch.unique(selected, dim=0)) == len(selected)
    old_z, old_g = curriculum.z.clone(), curriculum.gram.clone()
    expected = curriculum.mu.T @ curriculum.mu / 32
    expected += curriculum.stats["adaptive_ridge"] * torch.eye(3)
    assert torch.allclose(old_g, expected)
    assert not selected.requires_grad
    assert all(not p.requires_grad for p in curriculum.scorer.parameters())
    with torch.no_grad():
        for p in net.parameters():
            p.add_(.1)
    second = curriculum.sample(net, 1)
    assert len(torch.unique(torch.cat((selected, second)), dim=0)) == 2 * len(selected)
    assert torch.equal(old_z, curriculum.z)
    assert torch.equal(old_g, curriculum.gram)
    curriculum.sample(net, 2)
    new_g = curriculum.mu.T @ curriculum.mu / 32
    new_g += curriculum.stats["adaptive_ridge"] * torch.eye(3)
    assert torch.allclose(curriculum.gram, .8 * old_g + .2 * new_g)
    assert not torch.equal(old_z, curriculum.z)


@pytest.mark.parametrize("prior", ["sphere", "positive_sphere", "simplex"])
def test_latent_domains(prior):
    z = sample_latents(np.random.default_rng(1), 100, 6, prior)
    if prior == "simplex":
        assert np.allclose(z.sum(1), 1)
    else:
        assert np.allclose(np.linalg.norm(z, axis=1), 1)
    if prior != "sphere":
        assert (z >= 0).all()


@pytest.mark.parametrize("name", ["fruit_tree", "minecart", "mo_hopper", "mo_ant"])
def test_real_environment_contract_and_timeout(name):
    with isolated_global_rng(5):
        env = Benchmark(name, horizon=8)
        obs, _ = env.reset(seed=5)
        assert obs.shape == (env.obs_dim,)
        assert env.obs_dim == env.state_dim
        total = np.zeros(env.dim)
        for i in range(8):
            obs, r, terminated, truncated, info = env.step(
                0 if env.discrete else np.zeros(env.actions))
            assert r.shape == (env.dim,) and np.isfinite(obs).all()
            total += r
            if terminated or truncated:
                break
        assert terminated or truncated
        if name == "fruit_tree":
            assert (total > 0).sum() > 1  # Original nutrients, NOT one-hot leaves.
        env.close()


def test_time_limit_truncates_without_termination():
    env = Benchmark("mo_ant", horizon=1)
    obs, _ = env.reset(seed=5)
    assert obs.shape == (105,)
    _, _, terminated, truncated, _ = env.step(np.zeros(env.actions))
    assert truncated and not terminated
    env.close()


def test_mo_ant_matches_official_vector_reward():
    benchmark = Benchmark("mo_ant")
    official = mo.make("mo-ant-v5")
    obs, _ = benchmark.reset(seed=17)
    expected_obs, _ = official.reset(seed=17)
    assert np.allclose(obs, expected_obs)
    action = np.linspace(-.8, .8, 8, dtype=np.float32)
    obs, reward, terminated, truncated, _ = benchmark.step(action)
    expected_obs, expected_reward, expected_terminated, expected_truncated, _ = official.step(action)
    assert np.allclose(obs, expected_obs)
    assert np.array_equal(reward, expected_reward)
    assert terminated == expected_terminated
    assert truncated == expected_truncated
    benchmark.close()
    official.close()


def test_mo_ant_uses_simplex_preferences_by_default():
    from train import Config, evaluation_tasks
    cfg = Config(env="mo_ant", device="cpu").resolve()
    assert cfg.prior == "simplex"
    assert cfg.radius == pytest.approx(np.sqrt(3))
    z = sample_latents(np.random.default_rng(3), 100, 3, cfg.prior)
    assert (z >= 0).all()
    assert np.allclose(z.sum(1), 1.)
    first = evaluation_tasks(cfg, 3)
    second = evaluation_tasks(cfg, 3)
    assert first.shape == (20, 3)
    assert np.array_equal(first, second)
    assert (first >= 0).all()
    assert np.allclose(first.sum(1), cfg.radius)


def test_tilted_behavior_uses_existing_cache_only():
    from train import Config, behavior_task
    cfg = Config(env="mo_ant", tilted_behavior=True, device="cpu").resolve()
    net = SFSAC(4, 2, 3, 8)
    curriculum = Curriculum(net, torch.zeros(2, 4), np.random.default_rng(7),
                            "d", "simplex", batch_size=2, multiplier=2)
    cached_z = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    curriculum.z, curriculum.probs = cached_z, torch.tensor([.1, .9])
    curriculum.behavior_remaining = np.arange(2)
    before = cached_z.clone()
    rng = np.random.default_rng(9)
    first = behavior_task(cfg, rng, 3, curriculum)
    second = behavior_task(cfg, rng, 3, curriculum)
    assert np.array_equal(first, np.array([4., 5., 6.], np.float32))
    assert np.array_equal(second, np.array([1., 2., 3.], np.float32))
    assert torch.equal(cached_z, before)
    curriculum.z = None
    fallback = behavior_task(cfg, np.random.default_rng(9), 3, curriculum)
    assert (fallback >= 0).all()
    assert fallback.sum() == pytest.approx(cfg.radius)


def test_iqm_of_ten_rollouts_averages_central_six():
    from train import iqm
    values = np.array([[100., 1., 9., 2., 8., 3., 7., 4., 6., 5.]])
    assert np.array_equal(iqm(values, axis=1), np.array([5.5]))


def test_evaluation_preserves_rng_and_repeats():
    from train import Config, evaluate, initial_bank, make_model
    cfg = Config(env="minecart", horizon=10, eval_tasks=2, eval_episodes=1,
                 hidden=8, steps=10, device="cpu").resolve()
    with isolated_global_rng(20):
        env = Benchmark(cfg.env, horizon=cfg.horizon)
        net = make_model(cfg, env)
        env.close()
    z = sample_latents(np.random.default_rng(5), 2, 3, cfg.prior)
    start = initial_bank(cfg)
    np.random.seed(123)
    random.seed(123)
    expected_np, expected_py = np.random.random(), random.random()
    np.random.seed(123)
    random.seed(123)
    torch_state = torch.get_rng_state().clone()
    first, arrays = evaluate(net, cfg, z, start)
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.random.random() == expected_np
    assert random.random() == expected_py
    second, repeated = evaluate(net, cfg, z, start)
    assert first == second
    assert np.array_equal(arrays["returns"], repeated["returns"])
    assert arrays["utility_iqm"].shape == (2,)


def test_safe_checkpoint_roundtrip(tmp_path):
    from dataclasses import asdict
    from train import Config, make_model
    cfg = Config(steps=10, hidden=8, device="cpu").resolve()
    env = Benchmark("fruit_tree")
    model = make_model(cfg, env)
    path = tmp_path / "model.pt"
    torch.save(dict(model=model.state_dict(), config=asdict(cfg), obs_dim=env.obs_dim,
                    actions=env.actions, dim=env.dim), path)
    restored = torch.load(path, weights_only=True)
    clone = make_model(Config(**restored["config"]), env)
    clone.load_state_dict(restored["model"])
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters()))
    env.close()


def test_continuous_embedding_rng_isolation_and_reproducibility():
    net = SFSAC(4, 2, 3, 8, False, [-1., -1.], [1., 1.])
    starts, z = torch.randn(2, 4), torch.randn(5, 3)
    state = torch.get_rng_state().clone()
    a = embeddings(net, starts, z)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(a, embeddings(net, starts, z))
