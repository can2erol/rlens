"""Seed-everything for reproducible runs."""

from __future__ import annotations

import random

import numpy as np
import torch


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
