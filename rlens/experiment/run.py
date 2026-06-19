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
) -> Path:
    enable_mps_fallback()
    dev = pick_device(device)
    seed_everything(seed)

    # off-policy algos default to a single env; on-policy benefits from several
    if num_envs is None:
        num_envs = 1 if algo.lower() in ("dqn", "sac") else 8

    cfg = TrainConfig(
        algo=algo,
        env_id=env_id,
        total_steps=total_steps,
        seed=seed,
        device=str(dev),
        num_envs=num_envs,
        record_video=record_video,
        algo_overrides=algo_overrides or {},
    )

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

    env = EnvManager(env_id, num_envs=num_envs, seed=seed)
    algo_obj = build_algo(algo, env, dev, cfg.algo_overrides)

    video_cb = None
    if record_video:
        from rlens.telemetry.frames import record_episode_video

        def video_cb(step: int) -> None:
            out = run_dir / "videos" / f"step_{step:08d}.mp4"
            path = record_episode_video(env_id, algo_obj, dev, out, seed=seed)
            if path is not None:
                rec.frame(step, episode=0, path=str(path.relative_to(run_dir)))
                rec.flush()

    trainer = Trainer(
        algo_obj,
        env,
        rec,
        dev,
        total_steps=total_steps,
        rollout_len=cfg.rollout_len,
        update_every=cfg.update_every,
        learning_starts=cfg.learning_starts,
        progress=progress,
        video_cb=video_cb,
        video_interval=cfg.video_interval_steps if record_video else 0,
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
