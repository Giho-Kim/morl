"""Official benchmark environments; no custom maps, rewards, or dynamics."""
import os
import random
from contextlib import contextmanager

import numpy as np
import mo_gymnasium as mo
from gymnasium.spaces import Discrete

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

SPECS = {
    "fruit_tree": ("fruit-tree-v0", 7, "positive_sphere", 200_000),
    "minecart": ("minecart-v0", 1000, "positive_sphere", 2_000_000),
    "mo_hopper": ("mo-hopper-v5", 1000, "positive_sphere", 1_000_000),
    "mo_ant": ("mo-ant-v5", 1000, "simplex", 1_000_000),
}

REWARD_DIMS = {"fruit_tree": 6, "minecart": 3, "mo_hopper": 3, "mo_ant": 3}


@contextmanager
def isolated_global_rng(seed):
    """MO-Gymnasium 1.3.1 Minecart uses scipy/global NumPy RNG for ores.

    Isolate evaluation/reference calls so they cannot alter training randomness.
    """
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class Benchmark:
    def __init__(self, name, depth=6, horizon=None):
        self.name, self.depth = name, depth
        self.horizon = horizon or (depth if name == "fruit_tree" else SPECS[name][1])
        kwargs = {"depth": depth} if name == "fruit_tree" else {}
        self.env = mo.make(SPECS[name][0], max_episode_steps=self.horizon, **kwargs)
        self.dim = int(self.env.unwrapped.reward_space.shape[0])
        self.discrete = isinstance(self.env.action_space, Discrete)
        self.actions = int(self.env.action_space.n if self.discrete else self.env.action_space.shape[0])
        self.low = None if self.discrete else self.env.action_space.low.copy()
        self.high = None if self.discrete else self.env.action_space.high.copy()
        self.raw_shape = int(self.env.observation_space.shape[0])
        # One-hot STATE encoding does not replace the original vector reward.
        if name == "fruit_tree":
            self.state_dim = 2 ** (depth + 1) - 1
        else:
            self.state_dim = self.raw_shape
        self.obs_dim = self.state_dim

    def encode(self, raw):
        x = np.zeros(self.obs_dim, np.float32)
        if self.name == "fruit_tree":
            level, offset = map(int, raw)
            x[2 ** level - 1 + offset] = 1.
        else:
            x[:] = raw
        return x

    def reset(self, seed=None):
        raw, info = self.env.reset(seed=seed)
        return self.encode(raw), info

    def step(self, action):
        raw, reward, terminated, truncated, info = self.env.step(int(action) if self.discrete else np.asarray(action, np.float32))
        info = dict(info, raw_state=np.asarray(raw).copy())
        return (self.encode(raw), np.asarray(reward, np.float32), bool(terminated),
                bool(truncated), info)

    def close(self):
        self.env.close()
