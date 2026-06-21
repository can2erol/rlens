"""Gradient & parameter introspection helpers.

These produce most of the "visibility" value for free: call ``grad_norm_metrics`` right
after ``loss.backward()`` (before the optimizer step) to get the global gradient norm plus
a per-module breakdown, ready to hand straight to the Recorder.
"""

from __future__ import annotations

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
