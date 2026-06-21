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


class _TransposeImage(gym.ObservationWrapper):
    """Channels-last ``(H, W, C)`` uint8 image -> channels-first ``(C, H, W)`` for conv nets."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.transpose(obs, (2, 0, 1))


def default_frame_stack(env_id: str) -> int:
    """Atari defaults to a 4-frame stack; everything else to none."""
    return 4 if env_id.startswith("ALE/") else 1


def make_env(
    env_id: str,
    render_mode: str | None = None,
    frame_stack: int | None = None,
    seed: int | None = None,
) -> gym.Env:
    """Build a single (non-vectorized) env with rlens's standard wrappers.

    Atari ids (``ALE/...``) get the canonical 84x84 grayscale + frameskip pipeline; other
    image envs are transposed to channel-first. Used by both the vector EnvManager and the
    evaluator so preprocessing always matches.
    """
    if frame_stack is None:
        frame_stack = default_frame_stack(env_id)
    if env_id.startswith("ALE/"):
        env = gym.make(env_id, render_mode=render_mode, frameskip=1)
        env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, screen_size=84)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, frame_stack)  # -> (k, 84, 84)
    else:
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        sp = env.observation_space
        if isinstance(sp, gym.spaces.Box) and len(sp.shape) == 3 and sp.shape[-1] in (1, 3, 4):
            env = _TransposeImage(env)  # HWC -> CHW
        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, frame_stack)
    if seed is not None:
        env.action_space.seed(seed)
    return env


def _make_thunk(env_id: str, seed: int, idx: int, render_mode: str | None, frame_stack: int):
    return lambda: make_env(env_id, render_mode, frame_stack, seed + idx)


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
        frame_stack: int | None = None,
    ):
        self.env_id = env_id
        self.num_envs = num_envs
        self.seed = seed
        if frame_stack is None:
            frame_stack = default_frame_stack(env_id)
        thunks = [_make_thunk(env_id, seed, i, render_mode, frame_stack) for i in range(num_envs)]
        base = SyncVectorEnv(thunks, autoreset_mode=AutoresetMode.SAME_STEP)
        self.envs = gym.wrappers.vector.RecordEpisodeStatistics(base)

        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space

        if not isinstance(self.single_observation_space, gym.spaces.Box):
            raise NotImplementedError(
                f"Only Box observation spaces supported, got {self.single_observation_space}"
            )
        shape = tuple(self.single_observation_space.shape)
        # 3D Box -> image (channel-first after our wrappers); kept as uint8 for the CNN
        self.is_image = len(shape) == 3
        self.obs_shape = shape if self.is_image else (int(np.prod(shape)),)
        self.obs_dtype = np.uint8 if self.is_image else np.float32
        self.obs_dim = int(np.prod(shape))

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


def extract_final_obs(
    infos: dict[str, Any],
    num_envs: int,
    obs_shape: tuple[int, ...],
    dtype: Any = np.float32,
) -> np.ndarray | None:
    """Pull terminal observations out of a vector ``info`` dict (SAME_STEP autoreset).

    Returns an array of shape ``(num_envs, *obs_shape)`` where rows for envs that did not
    finish are zeros, or ``None`` if no env finished this step. Used to bootstrap value
    targets for *truncated* (time-limit) episodes and to store true terminal transitions.
    """
    if "final_obs" not in infos:
        return None
    mask = np.asarray(infos["_final_obs"])
    if not mask.any():
        return None
    out = np.zeros((num_envs, *obs_shape), dtype=dtype)
    raw = infos["final_obs"]
    for i in range(num_envs):
        if mask[i] and raw[i] is not None:
            out[i] = np.asarray(raw[i], dtype=dtype).reshape(obs_shape)
    return out
