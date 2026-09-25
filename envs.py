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
    "mo_hopper": ("mo-hopper-2obj-v5", 1000, "fixed_common_positive_sphere", 1_000_000),
    "mo_ant": ("mo-ant-2obj-v5", 1000, "fixed_common_positive_sphere", 1_000_000),
}

REWARD_DIMS = {"mo_hopper": 3, "mo_ant": 3}


@contextmanager
def isolated_global_rng(seed):
    """Keep evaluation calls from altering training randomness."""
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class Benchmark:
    def __init__(self, name, horizon=None, reward_layout="directional_common_3d"):
        self.name = name
        self.reward_layout = reward_layout
        self.horizon = horizon or SPECS[name][1]
        if reward_layout not in ("directional_common_3d", "official_2d"):
            raise ValueError(f"Unknown reward layout: {reward_layout}")
        self.env = mo.make(SPECS[name][0], max_episode_steps=self.horizon)
        self.dim = 3 if reward_layout == "directional_common_3d" else 2
        self.discrete = isinstance(self.env.action_space, Discrete)
        self.actions = int(self.env.action_space.n if self.discrete else self.env.action_space.shape[0])
        self.low = None if self.discrete else self.env.action_space.low.copy()
        self.high = None if self.discrete else self.env.action_space.high.copy()
        self.raw_shape = int(self.env.observation_space.shape[0])
        self.state_dim = self.raw_shape
        self.obs_dim = self.state_dim

    def encode(self, raw):
        return np.asarray(raw, dtype=np.float32)

    def reset(self, seed=None):
        raw, info = self.env.reset(seed=seed)
        return self.encode(raw), info

    def step(self, action):
        raw, reward, terminated, truncated, info = self.env.step(int(action) if self.discrete else np.asarray(action, np.float32))
        info = dict(info, raw_state=np.asarray(raw).copy())
        if self.reward_layout == "directional_common_3d":
            common = info["reward_ctrl"] + info["reward_survive"]
            if self.name == "mo_ant":
                common += info["reward_contact"]
                directions = (info["x_velocity"], info["y_velocity"])
            else:
                directions = (info["x_velocity"], 10 * info["z_distance_from_origin"])
            reward = np.asarray((*directions, common), dtype=np.float32)
        return (self.encode(raw), np.asarray(reward, np.float32), bool(terminated),
                bool(truncated), info)

    def close(self):
        self.env.close()
