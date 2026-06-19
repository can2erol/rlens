"""Neural network building blocks.

Kept small and readable: an MLP factory plus actor/critic heads for discrete and
continuous action spaces. Orthogonal initialization with the conventional gains is used
throughout (small final-layer gain on policy heads stabilizes early PPO training).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(sizes: list[int], activation: type[nn.Module] = nn.Tanh, std: float = np.sqrt(2)) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        is_last = i == len(sizes) - 2
        layers.append(layer_init(nn.Linear(sizes[i], sizes[i + 1]), std=std if not is_last else 1.0))
        if not is_last:
            layers.append(activation())
    return nn.Sequential(*layers)


class CategoricalActor(nn.Module):
    """Discrete policy: logits -> Categorical."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim])
        # shrink final layer for a near-uniform initial policy
        layer_init(self.net[-1], std=0.01)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))


class GaussianActor(nn.Module):
    """Continuous policy: state-dependent mean, state-independent log-std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.mean = mlp([obs_dim, *hidden, act_dim])
        layer_init(self.mean[-1], std=0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.mean(obs)
        std = torch.exp(self.log_std.expand_as(mean))
        return Normal(mean, std)


class Critic(nn.Module):
    """State-value function V(s)."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1])
        layer_init(self.net[-1], std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---- off-policy heads (DQN / SAC) ----------------------------------------
class QNetwork(nn.Module):
    """Discrete action-value network: Q(s, ·) over all actions (DQN)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (120, 84)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], activation=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ContinuousQ(nn.Module):
    """Continuous action-value network: Q(s, a) -> scalar (SAC critic)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1], activation=nn.ReLU)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy with the log-prob correction term (SAC)."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        hidden: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], activation=nn.ReLU)
        self.mean = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def _mean_logstd(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.net(obs))
        mean = self.mean(h)
        log_std = torch.tanh(self.log_std(h))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action, log_prob). Action is scaled to the env's bounds."""
        mean, log_std = self._mean_logstd(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        return action, log_prob.sum(-1)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._mean_logstd(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias
