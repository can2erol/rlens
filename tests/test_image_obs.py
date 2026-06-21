"""Image-observation support: CNN encoder, uint8 replay, env handling, end-to-end.

Uses a tiny self-contained image env (no Atari/ROM dependency) so the whole pipeline —
image env -> channel-first -> CNN -> uint8 replay -> DQN update -> learning — is exercised
fast and deterministically.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rlens.core.buffers import ReplayBuffer
from rlens.core.env import EnvManager
from rlens.core.networks import CNNCategoricalActor, CNNCritic, CNNQNetwork, NatureCNN
from rlens.experiment.eval import evaluate, load_trained_algo
from rlens.experiment.run import train_single

CPU = torch.device("cpu")
SHAPE_HWC = (42, 42, 1)   # channels-last; EnvManager transposes to (1, 42, 42)
SHAPE_CHW = (1, 42, 42)


class _TinyImageEnv(gym.Env):
    """One-step image bandit: a bright patch on the left (action 0) or right (action 1)."""

    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, SHAPE_HWC, np.uint8)
        self.action_space = spaces.Discrete(2)
        self._side = 0

    def _obs(self):
        img = np.zeros(SHAPE_HWC, np.uint8)
        if self._side == 0:
            img[:, :21] = 255
        else:
            img[:, 21:] = 255
        return img

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._side = int(self.np_random.integers(2))
        return self._obs(), {}

    def step(self, action):
        reward = 1.0 if int(action) == self._side else 0.0
        return self._obs(), reward, True, False, {}


if "rlensTinyImage-v0" not in gym.registry:
    gym.register(id="rlensTinyImage-v0", entry_point=_TinyImageEnv)


# ---- unit: networks --------------------------------------------------------
def test_nature_cnn_and_heads_shapes():
    x = torch.randint(0, 256, (5, *SHAPE_CHW), dtype=torch.uint8)
    assert NatureCNN(SHAPE_CHW)(x).shape == (5, 512)
    assert CNNQNetwork(SHAPE_CHW, 2)(x).shape == (5, 2)
    assert CNNCritic(SHAPE_CHW)(x).shape == (5,)
    dist = CNNCategoricalActor(SHAPE_CHW, 2).dist(x)
    assert dist.sample().shape == (5,)


# ---- unit: uint8 replay ----------------------------------------------------
def test_replay_buffer_stores_uint8_images():
    buf = ReplayBuffer(50, SHAPE_CHW, (), CPU, action_dtype=np.int64, obs_dtype=np.uint8)
    obs = np.random.randint(0, 256, (4, *SHAPE_CHW), dtype=np.uint8)
    buf.add_batch(obs, np.zeros(4, np.int64), np.ones(4, np.float32), obs, np.zeros(4, np.float32))
    assert buf.obs.dtype == np.uint8
    b = buf.sample(8)
    assert b["obs"].dtype == torch.uint8 and b["obs"].shape == (8, *SHAPE_CHW)


# ---- unit: env detection ---------------------------------------------------
def test_envmanager_detects_and_transposes_image():
    env = EnvManager("rlensTinyImage-v0", num_envs=2, seed=0)
    try:
        assert env.is_image
        assert env.obs_shape == SHAPE_CHW          # HWC -> CHW
        assert env.obs_dtype == np.uint8
        obs, _ = env.reset()
        assert obs.shape == (2, *SHAPE_CHW) and obs.dtype == np.uint8
    finally:
        env.close()


# ---- end-to-end: DQN learns the image bandit -------------------------------
def test_dqn_learns_on_images(tmp_path):
    run = train_single(
        algo="dqn", env_id="rlensTinyImage-v0", total_steps=2500, seed=0,
        device="cpu", runs_dir=tmp_path, name="img-dqn", progress=False,
        algo_overrides={"batch_size": 32, "eps_decay_steps": 800, "buffer_size": 5000},
    )
    assert (run / "policy.pt").exists()
    algo, env_id, _ = load_trained_algo(run, device="cpu")
    # the bandit is trivially separable; a working CNN+DQN should clear chance (0.5)
    res = evaluate(algo, env_id, algo.device, episodes=30, seed=0)
    assert res["return_mean"] > 0.7, f"did not learn image bandit: {res['return_mean']}"


# ---- end-to-end: PPO runs on images ----------------------------------------
def test_ppo_runs_on_images(tmp_path):
    run = train_single(
        algo="ppo", env_id="rlensTinyImage-v0", total_steps=2048, seed=0,
        device="cpu", runs_dir=tmp_path, name="img-ppo", progress=False,
        num_envs=4, algo_overrides={"num_minibatches": 2},
    )
    algo, env_id, _ = load_trained_algo(run, device="cpu")
    res = evaluate(algo, env_id, algo.device, episodes=10, seed=0)
    assert np.isfinite(res["return_mean"])
