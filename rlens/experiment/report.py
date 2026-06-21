"""Aggregate a benchmark runs directory into a reproducibility report.

For every run dir it loads ``policy.pt`` and scores it with a fresh deterministic
evaluation (so the headline number is exploration-free and comparable across algorithms),
then groups by ``(algo, env)`` and aggregates across seeds. If reference ``targets`` are
supplied, each group is marked pass/fail — that table is the credibility anchor: "our PPO
actually reaches the numbers everyone else reports."
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rlens.experiment.eval import evaluate, load_trained_algo
from rlens.telemetry.store import read_meta


def summarize_runs(
    runs_dir: Path,
    episodes: int = 20,
    eval_seed: int = 0,
    targets: dict[str, dict[str, float]] | None = None,
    device: str = "cpu",
    prefer_best: bool = False,
) -> dict[str, Any]:
    """Evaluate every run under ``runs_dir`` and aggregate by (algo, env).

    With ``prefer_best`` the highest-eval checkpoint is scored instead of the final policy.
    """
    runs_dir = Path(runs_dir)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for run_dir in sorted(p for p in runs_dir.iterdir() if (p / "run.json").exists()):
        if not (run_dir / "policy.pt").exists():
            continue
        cfg = read_meta(run_dir).get("config", {})
        algo, env = cfg.get("algo"), cfg.get("env_id")
        algo_obj, env_id, _ = load_trained_algo(run_dir, device=device, prefer_best=prefer_best)
        res = evaluate(algo_obj, env_id, algo_obj.device, episodes=episodes, seed=eval_seed)
        groups[(algo, env)].append(
            {"run": run_dir.name, "seed": cfg.get("seed"), "return_mean": res["return_mean"]}
        )

    rows: list[dict[str, Any]] = []
    for (algo, env), runs in sorted(groups.items()):
        means = [r["return_mean"] for r in runs]
        target = (targets or {}).get(algo, {}).get(env)
        rows.append(
            {
                "algo": algo,
                "env": env,
                "seeds": len(runs),
                "mean": float(np.mean(means)),
                "std": float(np.std(means)),
                "per_seed": sorted(means),
                "target": target,
                "passed": None if target is None else bool(np.mean(means) >= target),
            }
        )
    return {"rows": rows, "episodes": episodes}


def format_markdown(summary: dict[str, Any]) -> str:
    """Render the summary as a Markdown table."""
    lines = [
        "| algo | env | seeds | eval return (mean ± std) | reference | status |",
        "|------|-----|-------|--------------------------|-----------|--------|",
    ]
    for r in summary["rows"]:
        if r["target"] is None:
            ref, status = "—", "—"
        else:
            ref = f"≥ {r['target']:g}"
            status = "✅ pass" if r["passed"] else "❌ miss"
        lines.append(
            f"| {r['algo']} | {r['env']} | {r['seeds']} | "
            f"{r['mean']:.1f} ± {r['std']:.1f} | {ref} | {status} |"
        )
    return "\n".join(lines)
