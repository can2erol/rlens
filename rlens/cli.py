"""rlens command-line interface.

    rlens train --algo ppo --env CartPole-v1
    rlens bench configs/bench.yaml
    rlens dashboard
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Observability-first RL training & benchmarking.")

DEFAULT_RUNS_DIR = Path("runs")


@app.command()
def train(
    algo: str | None = typer.Option(None, help="Algorithm: ppo | dqn | sac."),
    env: str | None = typer.Option(None, help="Gymnasium env id."),
    steps: int | None = typer.Option(None, help="Total environment steps."),
    seed: int | None = typer.Option(None, help="Random seed."),
    device: str | None = typer.Option(None, help="auto | mps | cuda | cpu."),
    num_envs: int | None = typer.Option(None, help="Parallel envs (0/unset = auto)."),
    config: Path | None = typer.Option(
        None, help="YAML run config used as the base (CLI flags and --set override it)."
    ),
    set_: list[str] = typer.Option(
        None, "--set", help="Override a hyperparameter, e.g. --set lr=3e-4 (repeatable)."
    ),
    resume: Path | None = typer.Option(
        None, help="Resume an existing run dir from its latest checkpoint."
    ),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, help="Where run dirs are written."),
    name: str | None = typer.Option(None, help="Run name (default: auto-generated)."),
    record_video: bool = typer.Option(False, help="Capture rollout frames for video."),
    eval_interval: int | None = typer.Option(
        None, help="Env steps between deterministic eval episodes (0 = off)."
    ),
    eval_episodes: int | None = typer.Option(None, help="Episodes per evaluation."),
    checkpoint_interval: int | None = typer.Option(
        None, help="Env steps between checkpoints (0 = only a final checkpoint)."
    ),
) -> None:
    """Train a single policy and stream telemetry to a run dir.

    Config precedence (low to high): defaults < --config YAML < explicit flags < --set.
    """
    import yaml

    from rlens.experiment.config import TrainConfig
    from rlens.experiment.overrides import apply_overrides, parse_set
    from rlens.experiment.run import resume_training, run_config

    # resume short-circuits config building: everything comes from the saved run
    if resume is not None:
        run_dir = resume_training(resume, total_steps=steps, device=device, progress=True)
        typer.echo(f"Resumed run written to {run_dir}")
        return

    # base: a config file if given, else dataclass defaults
    if config is not None:
        cfg = TrainConfig.from_dict(yaml.safe_load(Path(config).read_text()) or {})
    else:
        cfg = TrainConfig()

    if config is None and algo is None:
        raise typer.BadParameter("specify --algo (or provide --config)")

    # explicit CLI flags override the base
    flag_overrides = {
        "algo": algo,
        "env_id": env,
        "total_steps": steps,
        "seed": seed,
        "device": device,
        "num_envs": num_envs,
        "eval_interval_steps": eval_interval,
        "eval_episodes": eval_episodes,
        "checkpoint_interval_steps": checkpoint_interval,
    }
    for key, value in flag_overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if record_video:
        cfg.record_video = True

    # --set has the final say
    try:
        apply_overrides(cfg, parse_set(set_))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    run_dir = run_config(cfg, runs_dir=runs_dir, name=name)
    typer.echo(f"Run written to {run_dir}")


@app.command(name="eval")
def eval_run(
    run: Path = typer.Argument(..., help="Run directory (with policy.pt + run.json)."),
    episodes: int = typer.Option(10, help="Number of evaluation episodes."),
    seed: int = typer.Option(0, help="Base seed (episode i uses seed+i)."),
    device: str = typer.Option("auto", help="auto | mps | cuda | cpu."),
    stochastic: bool = typer.Option(
        False, "--stochastic", help="Sample actions instead of acting greedily."
    ),
    video: bool = typer.Option(False, help="Also record one episode to <run>/videos/."),
) -> None:
    """Load a trained policy and score it over several episodes (optionally record a video)."""
    from rlens.experiment.eval import evaluate, load_trained_algo

    algo_obj, env_id, _ = load_trained_algo(run, device=device)
    res = evaluate(
        algo_obj,
        env_id,
        algo_obj.device,
        episodes=episodes,
        seed=seed,
        deterministic=not stochastic,
    )
    mode = "stochastic" if stochastic else "deterministic"
    typer.echo(
        f"{env_id} | {episodes} episodes ({mode})\n"
        f"  return: {res['return_mean']:.2f} ± {res['return_std']:.2f} "
        f"(min {res['return_min']:.2f}, max {res['return_max']:.2f})\n"
        f"  length: {res['length_mean']:.1f}"
    )

    if video:
        from rlens.telemetry.frames import record_episode_video

        out = Path(run) / "videos" / "eval.mp4"
        path = record_episode_video(env_id, algo_obj, algo_obj.device, out, seed=seed)
        if path is not None:
            typer.echo(f"  video:  {path}")
        else:
            typer.echo("  video:  (render unavailable for this env)")


@app.command()
def bench(
    config: Path = typer.Argument(..., help="YAML benchmark spec."),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, help="Where run dirs are written."),
) -> None:
    """Run an (algo x env x seed) grid into a shared runs dir."""
    from rlens.experiment.bench import run_benchmark

    run_benchmark(config, runs_dir=runs_dir)


@app.command()
def report(
    runs_dir: Path = typer.Argument(..., help="Benchmark runs dir to summarize."),
    targets: Path | None = typer.Option(
        None, help="YAML spec with a 'targets' map for pass/fail checking."
    ),
    episodes: int = typer.Option(20, help="Eval episodes per run."),
    device: str = typer.Option("cpu", help="auto | mps | cuda | cpu."),
    out: Path | None = typer.Option(None, help="Also write the Markdown table here."),
) -> None:
    """Evaluate every run in a dir and print a benchmark table (vs reference targets)."""
    import yaml

    from rlens.experiment.report import format_markdown, summarize_runs

    target_map = None
    if targets is not None:
        target_map = (yaml.safe_load(Path(targets).read_text()) or {}).get("targets")

    summary = summarize_runs(runs_dir, episodes=episodes, targets=target_map, device=device)
    md = format_markdown(summary)
    typer.echo(md)
    if out is not None:
        Path(out).write_text(md + "\n")
        typer.echo(f"\nwrote {out}")


@app.command()
def dashboard(
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, help="Runs dir to serve."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Launch the observability dashboard."""
    from rlens.dashboard.server import serve

    serve(runs_dir=runs_dir, host=host, port=port)


if __name__ == "__main__":
    app()
