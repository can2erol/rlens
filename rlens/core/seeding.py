"""Seed-everything for reproducible runs."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    """Snapshot the python / numpy / torch (CPU) RNG states for checkpointing."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by :func:`rng_state`."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch RNGs.

    Args:
        seed: the base seed.
        deterministic: if True, force deterministic cuDNN/algorithms. This can slow
            training and is unnecessary on MPS/CPU, so it defaults off.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
