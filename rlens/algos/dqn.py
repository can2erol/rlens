"""Deep Q-Network with a target network and Polyak averaging.

Discrete action spaces only. Exploration is epsilon-greedy with a linear schedule driven by
the algorithm's own step counter (so the shared Trainer needs no DQN-specific knowledge).
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
from rlens.core.networks import QNetwork
from rlens.telemetry.grads import grad_norm_metrics
from rlens.telemetry.recorder import Recorder


@dataclass
class DQNConfig:
    lr: float = 2.5e-4
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 128
    tau: float = 1.0                 # 1.0 = hard target update
    target_update_freq: int = 500    # updates between target syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    hidden: tuple[int, ...] = (120, 84)


class DQN(Algorithm):
    mode = "off_policy"

    def __init__(self, env: EnvManager, device: torch.device, cfg: DQNConfig | None = None):
        super().__init__(env, device)
        if not env.is_discrete:
            raise ValueError("DQN requires a discrete action space")
        self.cfg = cfg or DQNConfig()
        self.q = QNetwork(env.obs_dim, env.act_dim, self.cfg.hidden).to(device)
        self.q_target = copy.deepcopy(self.q).to(device)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(
            self.cfg.buffer_size, env.obs_dim, (), device, action_dtype=np.int64
        )
        self.t = 0          # env steps observed (drives epsilon)
        self.updates = 0

    def modules(self) -> dict[str, nn.Module]:
        return {"q": self.q}

    # ---- exploration ------------------------------------------------------
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.t / c.eps_decay_steps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        eps = 0.0 if deterministic else self.epsilon()
        q = self.q(obs)
        greedy = q.argmax(dim=-1).cpu().numpy()
        if eps > 0:
            n = greedy.shape[0]
            rand = np.random.randint(0, self.env.act_dim, size=n)
            mask = np.random.random(n) < eps
            greedy = np.where(mask, rand, greedy)
        return greedy.astype(np.int64), {}

    # ---- learning ---------------------------------------------------------
    def observe(self, tr: dict[str, Any]) -> None:
        self.t += len(tr["reward"])
        self.buffer.add_batch(tr["obs"], tr["action"], tr["reward"], tr["next_obs"], tr["done"])

    def update_off_policy(self, rec: Recorder, step: int) -> None:
        c = self.cfg
        if len(self.buffer) < c.batch_size:
            return
        b = self.buffer.sample(c.batch_size)
        actions = b["actions"].long()

        with torch.no_grad():
            next_q = self.q_target(b["next_obs"]).max(dim=-1).values
            target = b["rewards"] + c.gamma * (1 - b["dones"]) * next_q

        q_pred = self.q(b["obs"]).gather(1, actions.view(-1, 1)).squeeze(1)
        loss = F.mse_loss(q_pred, target)

        self.opt.zero_grad()
        loss.backward()
        grad_metrics = grad_norm_metrics(self.q, prefix="grad_norm/q")
        self.opt.step()
        self.updates += 1

        # target network update (Polyak, or hard sync on a cadence)
        if c.tau >= 1.0:
            if self.updates % c.target_update_freq == 0:
                self.q_target.load_state_dict(self.q.state_dict())
        else:
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.q_target.parameters(), strict=True):
                    pt.mul_(1 - c.tau).add_(c.tau * p)

        if self.updates % 50 == 0:
            rec.scalars(
                {
                    "loss/q": float(loss.item()),
                    "dqn/q_mean": float(q_pred.mean().item()),
                    "dqn/target_mean": float(target.mean().item()),
                    "dqn/epsilon": self.epsilon(),
                    **grad_metrics,
                },
                step=step,
            )
