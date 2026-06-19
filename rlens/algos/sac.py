"""Soft Actor-Critic (continuous control) with twin critics and automatic temperature.

Continuous (Box) action spaces only. Uses a tanh-squashed Gaussian policy, clipped double-Q
targets, Polyak-averaged target critics, and (optionally) automatic entropy-temperature tuning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlens.algos.base import Algorithm
from rlens.core.buffers import ReplayBuffer
from rlens.core.env import EnvManager
from rlens.core.networks import ContinuousQ, SquashedGaussianActor
from rlens.telemetry.grads import grad_norm_metrics
from rlens.telemetry.recorder import Recorder


@dataclass
class SACConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1_000_000
    batch_size: int = 256
    autotune_alpha: bool = True
    alpha: float = 0.2               # used when autotune is off
    policy_freq: int = 2             # delayed policy updates (per critic update)
    hidden: tuple[int, ...] = (256, 256)


class SAC(Algorithm):
    mode = "off_policy"

    def __init__(self, env: EnvManager, device: torch.device, cfg: SACConfig | None = None):
        super().__init__(env, device)
        if env.is_discrete:
            raise ValueError("SAC requires a continuous (Box) action space")
        self.cfg = cfg or SACConfig()

        scale = torch.tensor((env.action_high - env.action_low) / 2.0, dtype=torch.float32)
        bias = torch.tensor((env.action_high + env.action_low) / 2.0, dtype=torch.float32)
        self.actor = SquashedGaussianActor(env.obs_dim, env.act_dim, scale, bias, self.cfg.hidden).to(device)
        self.q1 = ContinuousQ(env.obs_dim, env.act_dim, self.cfg.hidden).to(device)
        self.q2 = ContinuousQ(env.obs_dim, env.act_dim, self.cfg.hidden).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.cfg.lr
        )

        if self.cfg.autotune_alpha:
            self.target_entropy = -float(env.act_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.cfg.alpha

        self.buffer = ReplayBuffer(self.cfg.buffer_size, env.obs_dim, (env.act_dim,), device)
        self.updates = 0

    def modules(self) -> dict[str, nn.Module]:
        return {"actor": self.actor, "q1": self.q1, "q2": self.q2}

    # ---- interaction ------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        if deterministic:
            a = self.actor.act_deterministic(obs)
        else:
            a, _ = self.actor.sample(obs)
        return a.cpu().numpy(), {}

    def observe(self, tr: dict[str, Any]) -> None:
        self.buffer.add_batch(tr["obs"], tr["action"], tr["reward"], tr["next_obs"], tr["done"])

    # ---- learning ---------------------------------------------------------
    def update_off_policy(self, rec: Recorder, step: int) -> None:
        c = self.cfg
        if len(self.buffer) < c.batch_size:
            return
        b = self.buffer.sample(c.batch_size)

        # --- critic update ---
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(b["next_obs"])
            q1_t = self.q1_target(b["next_obs"], next_a)
            q2_t = self.q2_target(b["next_obs"], next_a)
            min_q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target = b["rewards"] + c.gamma * (1 - b["dones"]) * min_q_t

        q1 = self.q1(b["obs"], b["actions"])
        q2 = self.q2(b["obs"], b["actions"])
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.q_opt.zero_grad()
        q_loss.backward()
        grad_metrics = grad_norm_metrics(self.q1, prefix="grad_norm/q1")
        self.q_opt.step()
        self.updates += 1

        actor_loss_val = float("nan")
        ent_val = float("nan")
        # --- delayed actor + temperature update ---
        if self.updates % c.policy_freq == 0:
            a, logp = self.actor.sample(b["obs"])
            q1_pi = self.q1(b["obs"], a)
            q2_pi = self.q2(b["obs"], a)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * logp - min_q_pi).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            grad_metrics.update(grad_norm_metrics(self.actor, prefix="grad_norm/actor"))
            self.actor_opt.step()
            actor_loss_val = float(actor_loss.item())
            ent_val = float(-logp.mean().item())

            if c.autotune_alpha:
                with torch.no_grad():
                    _, logp_detached = self.actor.sample(b["obs"])
                alpha_loss = -(self.log_alpha.exp() * (logp_detached + self.target_entropy)).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                self.alpha = self.log_alpha.exp().item()

        # --- target critic Polyak update ---
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, pt in zip(net.parameters(), tgt.parameters(), strict=True):
                    pt.mul_(1 - c.tau).add_(c.tau * p)

        if self.updates % 100 == 0:
            rec.scalars(
                {
                    "loss/q": float(q_loss.item()),
                    "loss/actor": actor_loss_val,
                    "sac/alpha": float(self.alpha),
                    "sac/entropy": ent_val,
                    "sac/q1_mean": float(q1.mean().item()),
                    **grad_metrics,
                },
                step=step,
            )
