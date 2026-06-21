"""Algorithm base class.

An algorithm owns its networks/optimizers and exposes a small, uniform surface that the
shared Trainer drives. Two execution modes are supported:

- ``"on_policy"``  (PPO): the trainer fills a RolloutBuffer, then calls ``update_on_policy``.
- ``"off_policy"`` (DQN/SAC): the trainer pushes each transition via ``observe`` and calls
  ``update_off_policy`` on a cadence.

Observability is *not* the algorithm's job beyond emitting its own loss/metric scalars via
the Recorder it is handed — episode stats, action histograms, gradient norms and FPS are
captured by the Trainer for every algorithm uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlens.core.env import EnvManager
from rlens.telemetry.recorder import Recorder


class Algorithm(ABC):
    mode: str = "on_policy"  # or "off_policy"

    def __init__(self, env: EnvManager, device: torch.device):
        self.env = env
        self.device = device

    # ---- interaction ------------------------------------------------------
    @abstractmethod
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        """Return (action_numpy, extras). ``extras`` may hold logprob/value tensors."""

    # ---- learning ---------------------------------------------------------
    def update_on_policy(
        self,
        buffer,
        bootstrap_obs: torch.Tensor,
        bootstrap_done: torch.Tensor,
        rec: Recorder,
        step: int,
        progress: float = 1.0,
    ) -> None:  # pragma: no cover
        """``progress`` is the fraction of total training elapsed (0→1), for LR schedules."""
        raise NotImplementedError

    def value_of(self, obs: torch.Tensor) -> torch.Tensor | None:
        """On-policy value estimate for truncation bootstrapping. None if N/A."""
        return None

    def observe(self, transition: dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def update_off_policy(self, rec: Recorder, step: int) -> None:  # pragma: no cover
        raise NotImplementedError

    # ---- introspection / checkpointing -----------------------------------
    @abstractmethod
    def modules(self) -> dict[str, nn.Module]:
        """Named modules, used for gradient capture and checkpointing."""

    def state_dict(self) -> dict[str, Any]:
        return {name: m.state_dict() for name, m in self.modules().items()}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        for name, m in self.modules().items():
            if name in sd:
                m.load_state_dict(sd[name])

    # ---- full-state checkpointing (for resume) ---------------------------
    def checkpoint_state(self) -> dict[str, Any]:
        """Everything needed to *resume* training, not just deploy: weights plus
        optimizer state, target networks and counters. Subclasses extend this; the base
        captures the trainable modules (enough for an optimizer-free algo)."""
        return {"modules": self.state_dict()}

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if "modules" in state:
            self.load_state_dict(state["modules"])
