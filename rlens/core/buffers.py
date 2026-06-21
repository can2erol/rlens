"""Experience buffers.

- ``RolloutBuffer``: fixed-length on-policy storage with GAE(λ) advantage estimation (PPO).
- ``ReplayBuffer``: uniform off-policy storage (DQN/SAC) — added in phase 5.

The GAE math is factored into a free function so it can be unit-tested against hand
computed values independently of the buffer plumbing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    last_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    Args:
        rewards, values, dones: shape (T, N). ``dones[t]`` marks that the episode ended
            *after* the transition at step t (so it cuts the bootstrap from t to t+1).
        last_value: shape (N,) — V(s_T) for bootstrapping the final step.
        last_done: shape (N,) — done flag for s_T.

    Returns:
        (advantages, returns), each shape (T, N). ``returns = advantages + values``.
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - last_done
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    """On-policy rollout storage shaped (rollout_len, num_envs, ...)."""

    def __init__(
        self,
        rollout_len: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: torch.device,
        obs_dtype: torch.dtype = torch.float32,
    ):
        self.rollout_len = rollout_len
        self.num_envs = num_envs
        self.device = device
        T, N = rollout_len, num_envs
        self.obs = torch.zeros((T, N, *obs_shape), dtype=obs_dtype, device=device)
        self.actions = torch.zeros((T, N, *action_shape), device=device)
        self.logprobs = torch.zeros((T, N), device=device)
        self.rewards = torch.zeros((T, N), device=device)
        self.dones = torch.zeros((T, N), device=device)
        self.values = torch.zeros((T, N), device=device)
        self.advantages = torch.zeros((T, N), device=device)
        self.returns = torch.zeros((T, N), device=device)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value) -> None:
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self.ptr += 1

    def reset(self) -> None:
        self.ptr = 0

    def compute_advantages(self, last_value, last_done, gamma: float, gae_lambda: float) -> None:
        adv, ret = compute_gae(
            self.rewards, self.values, self.dones, last_value, last_done, gamma, gae_lambda
        )
        self.advantages = adv
        self.returns = ret

    def iter_minibatches(self, num_minibatches: int):
        """Yield flattened minibatches over the whole rollout (shuffled each epoch)."""
        batch_size = self.rollout_len * self.num_envs
        mb_size = batch_size // num_minibatches
        flat = {
            "obs": self.obs.reshape(batch_size, *self.obs.shape[2:]),
            "actions": self.actions.reshape(batch_size, *self.actions.shape[2:]),
            "logprobs": self.logprobs.reshape(batch_size),
            "advantages": self.advantages.reshape(batch_size),
            "returns": self.returns.reshape(batch_size),
            "values": self.values.reshape(batch_size),
        }
        idx = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, mb_size):
            sel = idx[start : start + mb_size]
            yield {k: v[sel] for k, v in flat.items()}


class ReplayBuffer:
    """Uniform off-policy replay (DQN/SAC).

    Transitions are appended per-env (the trainer passes vectorized batches). Stored as
    numpy ring buffers; ``sample`` returns torch tensors on the target device.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: torch.device,
        action_dtype=np.float32,
        obs_dtype=np.float32,
    ):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((capacity, *action_shape), dtype=action_dtype)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, action, reward, next_obs, done) -> None:
        n = len(reward)
        for i in range(n):
            j = self.ptr
            self.obs[j] = obs[i]
            self.actions[j] = action[i]
            self.rewards[j] = reward[i]
            self.next_obs[j] = next_obs[i]
            self.dones[j] = done[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, Any]:
        idx = np.random.randint(0, self.size, size=batch_size)
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.device)  # noqa: E731
        # obs keep their stored dtype (uint8 for images — the encoder normalizes)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=self.device),
            "actions": t(self.actions[idx]),
            "rewards": t(self.rewards[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=self.device),
            "dones": t(self.dones[idx]),
        }

    def __len__(self) -> int:
        return self.size
