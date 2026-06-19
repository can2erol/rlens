"""Single-run orchestration: build env + algo + recorder, train, checkpoint."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from rlens.algos.base import Algorithm
from rlens.core.device import enable_mps_fallback, pick_device
from rlens.core.env import EnvManager
from rlens.core.seeding import seed_everything
from rlens.experiment.config import TrainConfig, version_snapshot
from rlens.telemetry.recorder import Recorder
from rlens.trainer import Trainer


def build_algo(name: str, env: EnvManager, device: torch.device, overrides: dict[str, Any]) -> Algorithm:
    name = name.lower()
    if name == "ppo":
        from rlens.algos.ppo import PPO, PPOConfig

        return PPO(env, device, PPOConfig(**overrides))
    if name == "dqn":
        from rlens.algos.dqn import DQN, DQNConfig

        return DQN(env, device, DQNConfig(**overrides))
    if name == "sac":
        from rlens.algos.sac import SAC, SACConfig

        return SAC(env, device, SACConfig(**overrides))
    raise ValueError(f"Unknown algo '{name}' (expected ppo | dqn | sac)")


def default_run_name(cfg: TrainConfig) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{cfg.algo}-{cfg.env_id}-s{cfg.seed}-{stamp}"


def train_single(
    algo: str,
    env_id: str,
    total_steps: int = 100_000,
    seed: int = 0,
    device: str = "auto",
    runs_dir: Path = Path("runs"),
    name: str | None = None,
    record_video: bool = False,
    num_envs: int | None = None,
    algo_overrides: dict[str, Any] | None = None,
    progress: bool = True,
    eval_interval: int = 0,
    eval_episodes: int = 10,
) -> Path:
    """Convenience wrapper: build a TrainConfig from kwargs and run it."""
    cfg = TrainConfig(
        algo=algo,
        env_id=env_id,
        total_steps=total_steps,
        seed=seed,
        device=device,
        num_envs=num_envs if num_envs is not None else 0,
        record_video=record_video,
        algo_overrides=algo_overrides or {},
        eval_interval_steps=eval_interval,
        eval_episodes=eval_episodes,
    )
    return run_config(cfg, runs_dir=runs_dir, name=name, progress=progress)


def run_config(
    cfg: TrainConfig,
    runs_dir: Path = Path("runs"),
    name: str | None = None,
    progress: bool = True,
    resume_from: Path | None = None,
) -> Path:
    """Train from a fully-formed :class:`TrainConfig` and write a run dir.

    Resolves the device and the ``num_envs=0`` auto default in place, so the config
    persisted to ``run.json`` reflects exactly what ran. If ``resume_from`` points at a
    checkpoint, the algorithm/optimizer/RNG state is restored and training continues from
    the saved step (telemetry is appended to the existing run dir).
    """
    from rlens.experiment.checkpoint import load_checkpoint, save_checkpoint

    enable_mps_fallback()
    dev = pick_device(cfg.device)
    cfg.device = str(dev)
    seed_everything(cfg.seed)

    # off-policy algos default to a single env; on-policy benefits from several
    if cfg.num_envs <= 0:
        cfg.num_envs = 1 if cfg.algo.lower() in ("dqn", "sac") else 8

    algo, env_id, seed = cfg.algo, cfg.env_id, cfg.seed

    run_name = name or default_run_name(cfg)
    run_dir = Path(runs_dir) / run_name
    rec = Recorder(run_dir)
    rec.meta(
        {
            "name": run_name,
            "status": "running",
            "config": cfg.__dict__,
            "versions": version_snapshot(),
            "started_at": time.time(),
        }
    )

    env = EnvManager(env_id, num_envs=cfg.num_envs, seed=seed)
    algo_obj = build_algo(algo, env, dev, cfg.algo_overrides)

    start_step = 0
    if resume_from is not None:
        from rlens.core.seeding import set_rng_state

        ckpt = load_checkpoint(resume_from, map_location=dev)
        algo_obj.load_checkpoint_state(ckpt["algo"])
        set_rng_state(ckpt["rng"])
        start_step = int(ckpt["global_step"])
        if progress:
            print(f"resumed {run_name} from step {start_step:,} -> target {cfg.total_steps:,}")

    video_cb = None
    if cfg.record_video:
        from rlens.telemetry.frames import record_episode_video

        def video_cb(step: int) -> None:
            out = run_dir / "videos" / f"step_{step:08d}.mp4"
            path = record_episode_video(env_id, algo_obj, dev, out, seed=seed)
            if path is not None:
                rec.frame(step, episode=0, path=str(path.relative_to(run_dir)))
                rec.flush()

    eval_cb = None
    if cfg.eval_interval_steps > 0:
        from rlens.experiment.eval import evaluate

        def eval_cb(step: int) -> None:
            res = evaluate(
                algo_obj, env_id, dev, episodes=cfg.eval_episodes, seed=seed + 10_000
            )
            rec.scalars(
                {
                    "eval/return_mean": res["return_mean"],
                    "eval/return_std": res["return_std"],
                    "eval/length_mean": res["length_mean"],
                },
                step=step,
            )

    def checkpoint_cb(step: int) -> None:
        save_checkpoint(run_dir, algo_obj, step, cfg.__dict__, keep_last=cfg.checkpoint_keep)

    trainer = Trainer(
        algo_obj,
        env,
        rec,
        dev,
        total_steps=cfg.total_steps,
        rollout_len=cfg.rollout_len,
        update_every=cfg.update_every,
        learning_starts=cfg.learning_starts,
        progress=progress,
        video_cb=video_cb,
        video_interval=cfg.video_interval_steps if cfg.record_video else 0,
        eval_cb=eval_cb,
        eval_interval=cfg.eval_interval_steps,
        checkpoint_cb=checkpoint_cb,
        checkpoint_interval=cfg.checkpoint_interval_steps,
        start_step=start_step,
    )

    status = "completed"
    try:
        trainer.train()
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        torch.save(algo_obj.state_dict(), run_dir / "policy.pt")
        save_checkpoint(run_dir, algo_obj, trainer.global_step, cfg.__dict__, keep_last=cfg.checkpoint_keep)
        rec.meta(
            {
                "name": run_name,
                "status": status,
                "config": cfg.__dict__,
                "versions": version_snapshot(),
                "final_step": trainer.global_step,
                "ended_at": time.time(),
            }
        )
        rec.close()
        env.close()

    return run_dir


def resume_training(
    run_dir: Path,
    total_steps: int | None = None,
    device: str | None = None,
    progress: bool = True,
) -> Path:
    """Continue an existing run from its latest checkpoint.

    Reconstructs the original :class:`TrainConfig` from ``run.json``, optionally raises the
    step target (``total_steps``) or changes the device, and appends to the same run dir.
    """
    from rlens.experiment.checkpoint import find_latest_checkpoint
    from rlens.telemetry.store import read_meta

    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    if not meta.get("config"):
        raise ValueError(f"{run_dir / 'run.json'} has no config — cannot resume")
    cfg = TrainConfig.from_dict(meta["config"])

    ckpt = find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoints found under {run_dir / 'checkpoints'}")

    if device is not None:
        cfg.device = device
    if total_steps is not None:
        cfg.total_steps = total_steps

    return run_config(
        cfg, runs_dir=run_dir.parent, name=run_dir.name, progress=progress, resume_from=ckpt
    )
