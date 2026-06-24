"""Gradient & parameter introspection helpers.

These produce most of the "visibility" value for free: call ``grad_norm_metrics`` right
after ``loss.backward()`` (before the optimizer step) to get the global gradient norm plus
a per-module breakdown, ready to hand straight to the Recorder.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def grad_norm_metrics(module: nn.Module, prefix: str = "grad_norm") -> dict[str, float]:
    """Global L2 grad norm + a per-top-level-child breakdown."""
    metrics: dict[str, float] = {}
    total_sq = 0.0
    for name, child in module.named_children():
        child_sq = 0.0
        for p in child.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                child_sq += float(g.pow(2).sum().item())
        if child_sq > 0:
            metrics[f"{prefix}/{name}"] = child_sq**0.5
        total_sq += child_sq
    # include params that are direct attributes (e.g. log_std)
    for _name, p in module.named_parameters(recurse=False):
        if p.grad is not None:
            total_sq += float(p.grad.detach().pow(2).sum().item())
    metrics[f"{prefix}/global"] = total_sq**0.5
    return metrics


def _sample(t: torch.Tensor, cap: int) -> np.ndarray:
    """Flatten to 1-D and (deterministically) downsample to at most ``cap`` values.

    Evenly-strided sampling keeps the distribution's shape while bounding the cost of
    capturing big layers (e.g. a CNN's conv stack) every inspection interval.
    """
    flat = t.detach().reshape(-1)
    if flat.numel() > cap:
        idx = torch.linspace(0, flat.numel() - 1, cap, device=flat.device).long()
        flat = flat[idx]
    return flat.float().cpu().numpy()


@torch.no_grad()
def param_distributions(
    module: nn.Module, prefix: str = "weights", sample_cap: int = 4096
) -> dict[str, np.ndarray]:
    """Per-parameter weight values, sampled — ready to hand to ``Recorder.histogram``.

    One entry per parameter tensor (``net.0.weight`` etc.), so the dashboard can show how
    each layer's weight distribution evolves over training, not just its L2 norm.
    """
    return {
        f"{prefix}/{name}": _sample(p, sample_cap) for name, p in module.named_parameters()
    }


@torch.no_grad()
def grad_distributions(
    module: nn.Module, prefix: str = "grads", sample_cap: int = 4096
) -> dict[str, np.ndarray]:
    """Per-parameter gradient values, sampled. Skips params with no gradient yet."""
    return {
        f"{prefix}/{name}": _sample(p.grad, sample_cap)
        for name, p in module.named_parameters()
        if p.grad is not None
    }
