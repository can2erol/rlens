"""Proximal Policy Optimization (clip objective) with GAE.

Single-file and readable. Supports both discrete (Categorical) and continuous (Gaussian)
action spaces, selected from the env's action space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlens.algos.base import Algorithm
from rlens.core.env import EnvManager
from rlens.core.networks import CategoricalActor, Critic, GaussianActor
from rlens.telemetry.grads import grad_norm_metrics
from rlens.telemetry.recorder import Recorder


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: float | None = None
    anneal_lr: bool = False           # opt-in: linearly decay LR to 0 over training
    hidden: tuple[int, ...] = (64, 64)


class PPO(Algorithm):
    mode = "on_policy"

    def __init__(self, env: EnvManager, device: torch.device, cfg: PPOConfig | None = None):
        super().__init__(env, device)
        self.cfg = cfg or PPOConfig()
        if env.is_discrete:
            self.actor: nn.Module = CategoricalActor(env.obs_dim, env.act_dim, self.cfg.hidden)
        else:
            self.actor = GaussianActor(env.obs_dim, env.act_dim, self.cfg.hidden)
        self.critic = Critic(env.obs_dim, self.cfg.hidden)
        self.actor.to(device)
        self.critic.to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.lr,
            eps=1e-5,
        )

    def modules(self) -> dict[str, nn.Module]:
        return {"actor": self.actor, "critic": self.critic}

    def checkpoint_state(self) -> dict[str, Any]:
        s = super().checkpoint_state()
        s["opt"] = self.opt.state_dict()
        return s

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        super().load_checkpoint_state(state)
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])

    # ---- interaction ------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        dist = self.actor.dist(obs)
        if deterministic:
            action = dist.mode if self.env.is_discrete else dist.mean
        else:
            action = dist.sample()
        logprob = self._logprob(dist, action)
        value = self.critic(obs)
        return self._to_env_action(action), {"action": action, "logprob": logprob, "value": value}

    def _logprob(self, dist, action: torch.Tensor) -> torch.Tensor:
        if self.env.is_discrete:
            return dist.log_prob(action)
        return dist.log_prob(action).sum(-1)

    def _entropy(self, dist) -> torch.Tensor:
        return dist.entropy() if self.env.is_discrete else dist.entropy().sum(-1)

    def _to_env_action(self, action: torch.Tensor) -> np.ndarray:
        a = action.detach().cpu().numpy()
        if not self.env.is_discrete:
            a = np.clip(a, self.env.action_low, self.env.action_high)
        return a

    @torch.no_grad()
    def _value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    @property
    def gamma(self) -> float:
        return self.cfg.gamma

    def value_of(self, obs: torch.Tensor) -> torch.Tensor:
        return self._value(obs)

    # ---- learning ---------------------------------------------------------
    def update_on_policy(
        self,
        buffer,
        bootstrap_obs: torch.Tensor,
        bootstrap_done: torch.Tensor,
        rec: Recorder,
        step: int,
        progress: float = 1.0,
    ) -> None:
        cfg = self.cfg
        # linear LR annealing — stabilizes late training (e.g. PPO drifting after solving)
        if cfg.anneal_lr:
            lr_now = cfg.lr * max(0.0, 1.0 - progress)
            for g in self.opt.param_groups:
                g["lr"] = lr_now
        else:
            lr_now = cfg.lr

        last_value = self._value(bootstrap_obs)
        buffer.compute_advantages(last_value, bootstrap_done, cfg.gamma, cfg.gae_lambda)

        clipfracs: list[float] = []
        approx_kl = 0.0
        pg_loss = v_loss = ent = torch.tensor(0.0)
        grad_metrics: dict[str, float] = {}

        for _epoch in range(cfg.update_epochs):
            for mb in buffer.iter_minibatches(cfg.num_minibatches):
                dist = self.actor.dist(mb["obs"])
                action = mb["actions"]
                if self.env.is_discrete:
                    action = action.long()
                new_logprob = self._logprob(dist, action)
                entropy = self._entropy(dist).mean()
                new_value = self.critic(mb["obs"])

                logratio = new_logprob - mb["logprobs"]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                adv = mb["advantages"]
                if cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_vloss:
                    v_unclipped = (new_value - mb["returns"]) ** 2
                    v_clipped = mb["values"] + torch.clamp(
                        new_value - mb["values"], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_clipped = (v_clipped - mb["returns"]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - mb["returns"]) ** 2).mean()

                ent = entropy
                loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                grad_metrics = grad_norm_metrics(self.actor, prefix="grad_norm/actor")
                grad_metrics.update(grad_norm_metrics(self.critic, prefix="grad_norm/critic"))
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.opt.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        # explained variance: how much of the return variance the value fn captures
        y_pred = buffer.values.reshape(-1)
        y_true = buffer.returns.reshape(-1)
        var_y = y_true.var()
        explained_var = float("nan") if var_y == 0 else float(1 - (y_true - y_pred).var() / var_y)

        rec.scalars(
            {
                "loss/policy": float(pg_loss.item()),
                "loss/value": float(v_loss.item()),
                "loss/entropy": float(ent.item()),
                "ppo/approx_kl": float(approx_kl),
                "ppo/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "ppo/explained_variance": explained_var,
                "ppo/lr": lr_now,
                **grad_metrics,
            },
            step=step,
        )
