"""Parse and apply ``--set key=value`` hyperparameter overrides.

A single ``--set`` namespace covers both *algorithm* hyperparameters (``lr``, ``gamma``,
``batch_size``, ``hidden`` ...) and *run-level* knobs (``num_envs``, ``rollout_len``,
``learning_starts`` ...). Each value is coerced to the target field's declared type — read
straight off the dataclass annotations — so ``--set lr=3e-4 --set hidden=[256,256]`` Just
Works, and an unknown key fails loudly with the list of valid keys for the chosen algo.
"""

from __future__ import annotations

import json
import typing
from typing import Any, get_args, get_origin, get_type_hints

from rlens.experiment.config import TrainConfig


def algo_config_class(name: str) -> type:
    """The dataclass holding hyperparameters for an algorithm."""
    name = name.lower()
    if name == "ppo":
        from rlens.algos.ppo import PPOConfig

        return PPOConfig
    if name == "dqn":
        from rlens.algos.dqn import DQNConfig

        return DQNConfig
    if name == "sac":
        from rlens.algos.sac import SACConfig

        return SACConfig
    raise ValueError(f"Unknown algo '{name}' (expected ppo | dqn | sac)")


# run-level fields that don't make sense to set via --set (use the dedicated flag instead)
_RUN_EXCLUDE = {"algo", "algo_overrides"}


def _coerce(raw: str, ann: Any) -> Any:
    """Coerce a CLI string to the declared field type ``ann``."""
    origin = get_origin(ann)

    # Optional[X] / X | None  ->  coerce against the first non-None arg
    if origin is typing.Union:
        non_none = [a for a in get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return _coerce(raw, non_none[0])

    if ann is bool:
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if ann is int:
        return int(raw)
    if ann is float:
        return float(raw)
    if ann is str or ann is None or ann is Any:
        return raw
    if origin in (tuple, list):
        text = raw.strip()
        if text[:1] in "[(" and text[-1:] in "])":
            text = text[1:-1]
        args = get_args(ann)
        elem = args[0] if args and args[0] is not Ellipsis else int
        items = [_coerce(p.strip(), elem) for p in text.split(",") if p.strip() != ""]
        return tuple(items) if origin is tuple else items
    # dict / anything else: parse as JSON
    return json.loads(raw)


def parse_set(items: list[str] | None) -> dict[str, str]:
    """Turn ``["lr=3e-4", "hidden=[256,256]"]`` into ``{"lr": "3e-4", ...}``."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def apply_overrides(cfg: TrainConfig, raw: dict[str, str]) -> TrainConfig:
    """Route each override to the algo config or a run-level field, coercing its type.

    Mutates and returns ``cfg``. Algorithm hyperparameters land in ``cfg.algo_overrides``;
    run-level fields are set directly. Raises ``ValueError`` on an unknown key.
    """
    algo_types = get_type_hints(algo_config_class(cfg.algo))
    train_types = {
        k: v for k, v in get_type_hints(TrainConfig).items() if k not in _RUN_EXCLUDE
    }

    for key, value in raw.items():
        if key in algo_types:
            cfg.algo_overrides[key] = _coerce(value, algo_types[key])
        elif key in train_types:
            setattr(cfg, key, _coerce(value, train_types[key]))
        else:
            valid = sorted(set(algo_types) | set(train_types))
            raise ValueError(
                f"unknown override key '{key}' for algo '{cfg.algo}'.\n"
                f"  valid keys: {', '.join(valid)}"
            )
    return cfg
