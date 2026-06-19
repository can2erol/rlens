"""Full-state checkpointing for crash-safe, resumable training.

A checkpoint is everything needed to continue training *exactly* where it stopped: the
algorithm's full state (weights + optimizers + target nets + counters), the global step,
and the RNG state. These live under ``runs/<id>/checkpoints/step_<N>.pt`` — separate from
``policy.pt`` (which holds just the weights for eval/deploy). Old checkpoints are pruned so
disk use stays bounded.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import torch

from rlens.algos.base import Algorithm
from rlens.core.seeding import rng_state

_CKPT_RE = re.compile(r"step_(\d+)\.pt$")


def _step_of(path: Path) -> int:
    m = _CKPT_RE.search(path.name)
    return int(m.group(1)) if m else -1


def save_checkpoint(
    run_dir: Path,
    algo: Algorithm,
    global_step: int,
    config: dict[str, Any],
    keep_last: int = 3,
) -> Path:
    """Write a full-state checkpoint and prune all but the ``keep_last`` newest."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{int(global_step):09d}.pt"
    torch.save(
        {
            "global_step": int(global_step),
            "algo": algo.checkpoint_state(),
            "rng": rng_state(),
            "config": config,
            "saved_at": time.time(),
        },
        path,
    )
    if keep_last > 0:
        existing = sorted(ckpt_dir.glob("step_*.pt"), key=_step_of)
        for old in existing[:-keep_last]:
            old.unlink(missing_ok=True)
    return path


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """The highest-step checkpoint in a run dir, or ``None`` if there are none."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=_step_of)
    return ckpts[-1] if ckpts else None


def load_checkpoint(path: Path, map_location: Any = None) -> dict[str, Any]:
    """Load a checkpoint file (trusted, so ``weights_only=False`` to allow RNG state)."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)
