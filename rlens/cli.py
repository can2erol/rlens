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
    algo: str = typer.Option(..., help="Algorithm: ppo | dqn | sac."),
    env: str = typer.Option("CartPole-v1", help="Gymnasium env id."),
    steps: int = typer.Option(100_000, help="Total environment steps."),
    seed: int = typer.Option(0, help="Random seed."),
    device: str = typer.Option("auto", help="auto | mps | cuda | cpu."),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, help="Where run dirs are written."),
    name: str | None = typer.Option(None, help="Run name (default: auto-generated)."),
    record_video: bool = typer.Option(False, help="Capture rollout frames for video."),
    eval_interval: int = typer.Option(
        0, help="Env steps between deterministic eval episodes (0 = off)."
    ),
    eval_episodes: int = typer.Option(10, help="Episodes per evaluation."),
) -> None:
    """Train a single policy and stream telemetry to a run dir."""
    from rlens.experiment.run import train_single

    run_dir = train_single(
        algo=algo,
        env_id=env,
        total_steps=steps,
        seed=seed,
        device=device,
        runs_dir=runs_dir,
        name=name,
        record_video=record_video,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
    )
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
