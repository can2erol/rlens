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
    )
    typer.echo(f"Run written to {run_dir}")


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
