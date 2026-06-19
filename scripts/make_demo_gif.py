"""Generate a demo GIF of a trained policy for the README.

By default trains a quick PPO on CartPole-v1 and records the greedy policy balancing the
pole. Point ``--run`` at an existing run dir to reuse its ``policy.pt`` instead.

    python scripts/make_demo_gif.py                       # train + record
    python scripts/make_demo_gif.py --run runs/<name>     # reuse a trained policy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from rlens.core.device import enable_mps_fallback, pick_device


def _downscale(frame: np.ndarray, width: int) -> np.ndarray:
    img = Image.fromarray(frame)
    if img.width > width:
        height = round(img.height * width / img.width)
        img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None, help="Existing run dir to load policy from.")
    ap.add_argument("--out", type=Path, default=Path("docs/demo.gif"))
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--steps", type=int, default=60_000, help="Training steps if no --run.")
    ap.add_argument("--max-frames", type=int, default=260)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=380)
    args = ap.parse_args()

    enable_mps_fallback()
    device = pick_device("auto")

    from rlens.experiment.eval import load_trained_algo

    if args.run is None:
        from rlens.experiment.run import train_single

        print(f"training PPO on {args.env} for {args.steps:,} steps ...")
        run = train_single(
            algo="ppo", env_id=args.env, total_steps=args.steps,
            seed=0, device=str(device), runs_dir=Path("runs"), name="demo-gif",
        )
    else:
        run = args.run

    algo, env_id, _ = load_trained_algo(run, device=device)

    env = gym.make(env_id, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=123)
    for _ in range(args.max_frames):
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).ravel(), device=device).unsqueeze(0)
        action, _ = algo.act(obs_t, deterministic=True)
        obs, _, term, trunc, _ = env.step(action[0])
        frame = env.render()
        if frame is not None:
            frames.append(_downscale(frame, args.width))
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=args.fps, loop=0)
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} — {len(frames)} frames, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
