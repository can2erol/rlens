"""Run configuration + reproducibility snapshot."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    algo: str = "ppo"
    env_id: str = "CartPole-v1"
    total_steps: int = 100_000
    seed: int = 0
    device: str = "auto"
    num_envs: int = 8
    rollout_len: int = 128            # on-policy only
    update_every: int = 1             # off-policy: env steps between updates
    learning_starts: int = 1000       # off-policy: random steps before learning
    log_interval_updates: int = 1
    eval_interval_steps: int = 0
    eval_episodes: int = 10
    record_video: bool = False
    video_interval_steps: int = 20_000
    algo_overrides: dict[str, Any] = field(default_factory=dict)

    def to_yaml(self, path: Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def version_snapshot() -> dict[str, Any]:
    """Capture library versions + git SHA for reproducibility."""
    import gymnasium
    import numpy
    import torch

    snap: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "gymnasium": gymnasium.__version__,
    }
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        snap["git_sha"] = sha
    except Exception:
        snap["git_sha"] = None
    return snap
