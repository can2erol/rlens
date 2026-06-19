"""Rollout video capture.

Rendering during vectorized training is wasteful, so instead we periodically spin up a
single ``rgb_array`` env, run one *deterministic* episode with the current policy, and write
it to ``runs/<id>/videos/step_XXXXXXXX.mp4``. The dashboard discovers and plays these — so
you can literally watch the policy get better over the course of training.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch


@torch.no_grad()
def record_episode_video(
    env_id: str,
    algo,
    device: torch.device,
    out_path: Path,
    seed: int = 0,
    max_steps: int = 1000,
    fps: int = 30,
) -> Path | None:
    """Render one greedy episode to an MP4. Returns the path, or None on failure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except Exception:
        return None

    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_np, _ = algo.act(obs_t, deterministic=True)
        action = action_np[0]
        obs, _, term, trunc, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))
        if term or trunc:
            break
    env.close()

    if not frames:
        return None
    imageio.mimwrite(
        out_path, frames, fps=fps, macro_block_size=1, codec="libx264", pixelformat="yuv420p"
    )
    return out_path
