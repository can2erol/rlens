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


def _expand_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the list of (algo, env, seed, steps, overrides) cells from a spec.

    Two spec styles are supported:
    - ``grid``: a cartesian product of ``algo`` x ``env`` x ``seed`` (one global step budget).
    - ``runs``: an explicit list of ``{algo, env, steps?, overrides?}`` entries, each fanned
      out over the top-level ``seeds`` — better for benchmarks where envs need different
      budgets and you want to avoid incompatible (algo, env) cells.
    """
    default_steps = spec.get("total_steps", 100_000)
    overrides_by_algo: dict[str, dict] = spec.get("algo_overrides", {})

    cells: list[dict[str, Any]] = []
    if "runs" in spec:
        seeds = spec.get("seeds", [0])
        for entry in spec["runs"]:
            for seed in seeds:
                cells.append(
                    {
                        "algo": entry["algo"],
                        "env": entry["env"],
                        "seed": seed,
                        "steps": entry.get("steps", default_steps),
                        "overrides": {**overrides_by_algo.get(entry["algo"], {}),
                                      **entry.get("overrides", {})},
                    }
                )
    else:
        grid = spec.get("grid", {})
        for algo, env_id, seed in itertools.product(
            grid.get("algo", ["ppo"]), grid.get("env", ["CartPole-v1"]), grid.get("seed", [0])
        ):
            cells.append(
                {
                    "algo": algo,
                    "env": env_id,
                    "seed": seed,
                    "steps": default_steps,
                    "overrides": overrides_by_algo.get(algo, {}),
                }
            )
    return cells


def run_benchmark(config_path: Path, runs_dir: Path = Path("runs")) -> list[Path]:
    spec: dict[str, Any] = yaml.safe_load(Path(config_path).read_text())
    device = spec.get("device", "auto")
    eval_interval = spec.get("eval_interval", 0)
    eval_episodes = spec.get("eval_episodes", 10)

    cells = _expand_cells(spec)
    print(f"benchmark: {len(cells)} runs -> {runs_dir}")

    results: list[Path] = []
    t0 = time.time()
    for i, cell in enumerate(cells, 1):
        algo, env_id, seed = cell["algo"], cell["env"], cell["seed"]
        if not _compatible(algo, env_id):
            print(f"[{i}/{len(cells)}] skip {algo} on {env_id} (action-space mismatch)")
            continue
        name = f"{algo}-{env_id}-s{seed}"
        print(f"[{i}/{len(cells)}] {name} ({cell['steps']:,} steps) ...", flush=True)
        run = train_single(
            algo=algo,
            env_id=env_id,
            total_steps=cell["steps"],
            seed=seed,
            device=device,
            runs_dir=runs_dir,
            name=name,
            algo_overrides=cell["overrides"],
            progress=False,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
        )
        results.append(run)

    print(f"\nbenchmark done: {len(results)} runs in {time.time() - t0:.1f}s -> {runs_dir}")
    print(f"report with:  rlens report {runs_dir} --targets {config_path}")
    return results
