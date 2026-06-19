"""Policy evaluation — run episodes with a trained (or in-training) policy.

Two callers share this code:

- ``rlens eval``: load a finished run's ``policy.pt`` and score / re-watch the policy.
- periodic eval during training: the Trainer calls :func:`evaluate` on a cadence and logs
  an ``eval/*`` curve — a clean, exploration-free signal distinct from the noisy
  ``rollout/episodic_return`` (which mixes in epsilon-greedy / stochastic exploration).

Evaluation always runs on a single, freshly-built env so it never perturbs the training
envs' RNG or autoreset state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rlens.algos.base import Algorithm
from rlens.core.device import enable_mps_fallback, pick_device
from rlens.core.env import EnvManager
from rlens.telemetry.store import read_meta


@torch.no_grad()
def evaluate(
    algo: Algorithm,
    env_id: str,
    device: torch.device,
    episodes: int = 10,
    seed: int = 0,
    deterministic: bool = True,
    max_steps: int = 10_000,
) -> dict[str, Any]:
    """Run ``episodes`` episodes and return aggregate return/length statistics.

    Each episode gets a distinct seed (``seed + i``) for a stable-but-varied sample.
    """
    env = gym.make(env_id)
    returns: list[float] = []
    lengths: list[int] = []
    try:
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + i)
            total = 0.0
            steps = 0
            done = False
            while not done and steps < max_steps:
                obs_t = torch.as_tensor(
                    np.asarray(obs, dtype=np.float32).ravel(), device=device
                ).unsqueeze(0)
                action_np, _ = algo.act(obs_t, deterministic=deterministic)
                obs, reward, term, trunc, _ = env.step(action_np[0])
                total += float(reward)
                steps += 1
                done = bool(term) or bool(trunc)
            returns.append(total)
            lengths.append(steps)
    finally:
        env.close()

    rets = np.asarray(returns, dtype=np.float64)
    lens = np.asarray(lengths, dtype=np.float64)
    return {
        "episodes": episodes,
        "deterministic": deterministic,
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std()),
        "return_min": float(rets.min()),
        "return_max": float(rets.max()),
        "length_mean": float(lens.mean()),
        "returns": rets.tolist(),
    }


def load_trained_algo(
    run_dir: Path, device: str | torch.device = "auto"
) -> tuple[Algorithm, str, dict[str, Any]]:
    """Rebuild the algorithm from a run dir's ``run.json`` and load its ``policy.pt``.

    Returns ``(algo, env_id, meta)``. The temporary EnvManager built to recover the
    observation/action dimensions is closed before returning; callers evaluate against a
    fresh env.
    """
    from rlens.experiment.run import build_algo

    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    config = meta.get("config")
    if not config:
        raise ValueError(f"No 'config' in {run_dir / 'run.json'} — is this a rlens run dir?")

    algo_name = config["algo"]
    env_id = config["env_id"]
    overrides = config.get("algo_overrides", {})

    policy_path = run_dir / "policy.pt"
    if not policy_path.exists():
        raise FileNotFoundError(f"No policy.pt in {run_dir}")

    enable_mps_fallback()
    dev = device if isinstance(device, torch.device) else pick_device(device)

    env = EnvManager(env_id, num_envs=1, seed=0)
    try:
        algo = build_algo(algo_name, env, dev, overrides)
    finally:
        env.close()

    state = torch.load(policy_path, map_location=dev)
    algo.load_state_dict(state)
    return algo, env_id, meta
