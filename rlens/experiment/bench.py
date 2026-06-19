"""Benchmark grid runner.

Expands an (algo x env x seed) grid from a YAML spec and runs each cell as a single run into
a shared ``runs/`` directory — where the dashboard overlays them all for comparison.
Incompatible cells (e.g. DQN on a continuous env) are skipped with a note rather than
aborting the grid. Sequential by design for v1; the loop is structured so a process pool is
a drop-in later.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import yaml

from rlens.experiment.run import train_single

# action-space compatibility: which algos need discrete vs continuous
_DISCRETE_ONLY = {"dqn"}
_CONTINUOUS_ONLY = {"sac"}


def _is_discrete_env(env_id: str) -> bool:
    import gymnasium as gym

    env = gym.make(env_id)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    env.close()
    return discrete


def _compatible(algo: str, env_id: str) -> bool:
    discrete = _is_discrete_env(env_id)
    if algo in _DISCRETE_ONLY and not discrete:
        return False
    if algo in _CONTINUOUS_ONLY and discrete:
        return False
    return True


def run_benchmark(config_path: Path, runs_dir: Path = Path("runs")) -> list[Path]:
    spec: dict[str, Any] = yaml.safe_load(Path(config_path).read_text())
    grid = spec.get("grid", {})
    algos = grid.get("algo", ["ppo"])
    envs = grid.get("env", ["CartPole-v1"])
    seeds = grid.get("seed", [0])
    total_steps = spec.get("total_steps", 100_000)
    device = spec.get("device", "auto")
    overrides_by_algo: dict[str, dict] = spec.get("algo_overrides", {})

    cells = list(itertools.product(algos, envs, seeds))
    print(f"benchmark: {len(cells)} cells ({len(algos)} algos x {len(envs)} envs x {len(seeds)} seeds)")

    results: list[Path] = []
    t0 = time.time()
    for i, (algo, env_id, seed) in enumerate(cells, 1):
        if not _compatible(algo, env_id):
            print(f"[{i}/{len(cells)}] skip {algo} on {env_id} (action-space mismatch)")
            continue
        name = f"{algo}-{env_id}-s{seed}"
        print(f"[{i}/{len(cells)}] {name} ...", flush=True)
        run = train_single(
            algo=algo,
            env_id=env_id,
            total_steps=total_steps,
            seed=seed,
            device=device,
            runs_dir=runs_dir,
            name=name,
            algo_overrides=overrides_by_algo.get(algo, {}),
            progress=False,
        )
        results.append(run)

    print(f"\nbenchmark done: {len(results)} runs in {time.time() - t0:.1f}s -> {runs_dir}")
    print("view with:  rlens dashboard")
    return results
