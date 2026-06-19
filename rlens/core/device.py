"""Device selection with Apple-MPS awareness.

The reference machine is an Apple M4 Pro (no CUDA). MPS is the accelerator, but it
has sharp edges: no float64, and a handful of ops are unimplemented. We default to
float32 everywhere and surface the `PYTORCH_ENABLE_MPS_FALLBACK` knob so unsupported
ops transparently fall back to CPU instead of crashing mid-run.
"""

from __future__ import annotations

import os

import torch


def pick_device(prefer: str = "auto") -> torch.device:
    """Pick a torch device.

    Args:
        prefer: "auto" (mps > cuda > cpu), or an explicit "mps"/"cuda"/"cpu".
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enable_mps_fallback() -> None:
    """Let unsupported MPS ops fall back to CPU rather than raising.

    Must be set before the offending op runs; safe to call unconditionally.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def default_dtype(device: torch.device) -> torch.dtype:
    """float32 everywhere — MPS does not support float64."""
    return torch.float32
