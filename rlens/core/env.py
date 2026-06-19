"""Environment management on top of Gymnasium vector envs.

We deliberately wrap Gymnasium rather than invent an env API — it is the de-facto standard
that every benchmark targets. We pin the vector autoreset mode to ``SAME_STEP`` (classic
behavior): on episode end the env resets immediately and the true terminal observation is
returned in ``info["final_obs"]``. That keeps the rollout loop and GAE bootstrapping simple
and correct (no spurious post-reset transitions).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, SyncVectorEnv


def _make_thunk(env_id: str, seed: int, idx: int, render_mode: str | None):
    def thunk() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk


class EnvManager:
    """A vectorized set of environments with episode-statistics tracking.

    Attributes:
        envs: the underlying gymnasium vector env.
        single_observation_space / single_action_space: per-env spaces.
        obs_dim: flattened observation dimension (Box only).
        is_discrete: whether the action space is Discrete.
        act_dim: number of actions (discrete) or action dimension (continuous).
    """

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        seed: int = 0,
        render_mode: str | None = None,
    ):
        self.env_id = env_id
        self.num_envs = num_envs
        self.seed = seed
        thunks = [_make_thunk(env_id, seed, i, render_mode) for i in range(num_envs)]
        base = SyncVectorEnv(thunks, autoreset_mode=AutoresetMode.SAME_STEP)
        self.envs = gym.wrappers.vector.RecordEpisodeStatistics(base)

        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space

        if not isinstance(self.single_observation_space, gym.spaces.Box):
            raise NotImplementedError(
                f"Only Box observation spaces supported, got {self.single_observation_space}"
            )
        self.obs_dim = int(np.prod(self.single_observation_space.shape))

        if isinstance(self.single_action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.act_dim = int(self.single_action_space.n)
        elif isinstance(self.single_action_space, gym.spaces.Box):
            self.is_discrete = False
            self.act_dim = int(np.prod(self.single_action_space.shape))
            self.action_low = self.single_action_space.low
            self.action_high = self.single_action_space.high
        else:
            raise NotImplementedError(f"Unsupported action space {self.single_action_space}")

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        return self.envs.reset(seed=self.seed)

    def step(self, actions: np.ndarray):
        return self.envs.step(actions)

    def close(self) -> None:
        self.envs.close()


def extract_final_obs(infos: dict[str, Any], num_envs: int, obs_dim: int) -> np.ndarray | None:
    """Pull terminal observations out of a vector ``info`` dict (SAME_STEP autoreset).

    Returns an array of shape (num_envs, obs_dim) where rows for envs that did not finish
    are zeros, or ``None`` if no env finished this step. Used to bootstrap value targets
    for *truncated* (time-limit) episodes.
    """
    if "final_obs" not in infos:
        return None
    mask = np.asarray(infos["_final_obs"])
    if not mask.any():
        return None
    out = np.zeros((num_envs, obs_dim), dtype=np.float32)
    raw = infos["final_obs"]
    for i in range(num_envs):
        if mask[i] and raw[i] is not None:
            out[i] = np.asarray(raw[i], dtype=np.float32).ravel()
    return out
