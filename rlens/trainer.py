"""The shared training loop.

One Trainer drives every algorithm. It owns the env interaction, episode bookkeeping, and
— crucially — the *uniform* observability: episode return/length, steps-per-second, action
distributions and (via the algos) gradient norms all land in the Recorder the same way for
PPO, DQN and SAC. Adding an algorithm never means re-plumbing telemetry.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from rlens.algos.base import Algorithm
from rlens.core.buffers import RolloutBuffer
from rlens.core.env import EnvManager, extract_final_obs
from rlens.telemetry.recorder import Recorder


class Trainer:
    def __init__(
        self,
        algo: Algorithm,
        env: EnvManager,
        rec: Recorder,
        device: torch.device,
        total_steps: int,
        rollout_len: int = 128,
        update_every: int = 1,
        gradient_steps: int = 1,
        learning_starts: int = 1000,
        log_interval_updates: int = 1,
        progress: bool = True,
        video_cb=None,
        video_interval: int = 0,
        eval_cb=None,
        eval_interval: int = 0,
        checkpoint_cb=None,
        checkpoint_interval: int = 0,
        start_step: int = 0,
    ):
        self.algo = algo
        self.env = env
        self.rec = rec
        self.device = device
        self.total_steps = total_steps
        self.rollout_len = rollout_len
        self.update_every = max(1, update_every)
        self.gradient_steps = max(1, gradient_steps)
        self.learning_starts = learning_starts
        self.log_interval_updates = log_interval_updates
        self.progress = progress
        self.video_cb = video_cb
        self.video_interval = video_interval
        self.eval_cb = eval_cb
        self.eval_interval = eval_interval
        self.checkpoint_cb = checkpoint_cb
        self.checkpoint_interval = checkpoint_interval

        self.global_step = start_step
        self._session_start_step = start_step
        self._t_start = time.time()
        self._recent_returns: list[float] = []
        self._last_video_step = start_step
        self._last_eval_step = start_step
        self._last_checkpoint_step = start_step

    # ---- shared helpers ---------------------------------------------------
    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _obs_t(self, x: np.ndarray) -> torch.Tensor:
        # images travel as uint8 (the CNN encoder normalizes); vectors as float32
        dt = torch.uint8 if self.env.is_image else torch.float32
        return torch.as_tensor(x, dtype=dt, device=self.device)

    def _log_episodes(self, infos: dict) -> None:
        if "episode" not in infos:
            return
        mask = np.asarray(infos["_episode"])
        rets = np.asarray(infos["episode"]["r"])
        lens = np.asarray(infos["episode"]["l"])
        for i in range(len(mask)):
            if mask[i]:
                self.rec.episode(float(rets[i]), int(lens[i]), step=self.global_step)
                self.rec.scalar("rollout/episodic_return", float(rets[i]), step=self.global_step)
                self.rec.scalar("rollout/episodic_length", int(lens[i]), step=self.global_step)
                self._recent_returns.append(float(rets[i]))
                self._recent_returns = self._recent_returns[-100:]

    def _log_throughput(self) -> None:
        elapsed = max(time.time() - self._t_start, 1e-9)
        done = self.global_step - self._session_start_step
        self.rec.scalar("perf/steps_per_sec", done / elapsed, step=self.global_step)

    def _maybe_video(self) -> None:
        if self.video_cb is None or self.video_interval <= 0:
            return
        if self.global_step - self._last_video_step >= self.video_interval:
            self._last_video_step = self.global_step
            self.video_cb(self.global_step)

    def _maybe_eval(self) -> None:
        if self.eval_cb is None or self.eval_interval <= 0:
            return
        if self.global_step - self._last_eval_step >= self.eval_interval:
            self._last_eval_step = self.global_step
            self.eval_cb(self.global_step)

    def _maybe_checkpoint(self) -> None:
        if self.checkpoint_cb is None or self.checkpoint_interval <= 0:
            return
        if self.global_step - self._last_checkpoint_step >= self.checkpoint_interval:
            self._last_checkpoint_step = self.global_step
            self.checkpoint_cb(self.global_step)

    def _print_progress(self) -> None:
        if not self.progress:
            return
        mean_ret = np.mean(self._recent_returns) if self._recent_returns else float("nan")
        pct = 100 * self.global_step / self.total_steps
        print(
            f"\r[{pct:5.1f}%] step {self.global_step:>9} | "
            f"mean_return(100) {mean_ret:8.2f}",
            end="",
            flush=True,
        )

    # ---- entry point ------------------------------------------------------
    def train(self) -> None:
        if self.algo.mode == "on_policy":
            self._train_on_policy()
        elif self.algo.mode == "off_policy":
            self._train_off_policy()
        else:
            raise ValueError(f"Unknown algo mode {self.algo.mode}")
        self.rec.flush()
        if self.progress:
            print()

    # ---- on-policy (PPO) --------------------------------------------------
    def _train_on_policy(self) -> None:
        env, algo, N = self.env, self.algo, self.env.num_envs
        action_shape = () if env.is_discrete else (env.act_dim,)
        obs_dtype = torch.uint8 if env.is_image else torch.float32
        buf = RolloutBuffer(self.rollout_len, N, env.obs_shape, action_shape, self.device, obs_dtype)
        gamma = getattr(algo, "gamma", 0.99)

        next_obs_np, _ = env.reset()
        next_obs = self._obs_t(next_obs_np)
        next_done = torch.zeros(N, device=self.device)

        update = 0
        while self.global_step < self.total_steps:
            buf.reset()
            action_log: list[np.ndarray] = []
            for _ in range(self.rollout_len):
                self.global_step += N
                action_np, extras = algo.act(next_obs)
                step_obs, reward, term, trunc, infos = env.step(action_np)
                reward = np.asarray(reward, dtype=np.float32)

                # bootstrap value for *truncated* (time-limit) episodes
                trunc_only = np.asarray(trunc) & ~np.asarray(term)
                if trunc_only.any():
                    final = extract_final_obs(infos, N, env.obs_shape, env.obs_dtype)
                    vf = algo.value_of(self._obs_t(final)) if final is not None else None
                    if vf is not None:
                        reward = reward + gamma * vf.detach().cpu().numpy() * trunc_only

                buf.add(
                    next_obs,
                    extras["action"],
                    extras["logprob"],
                    self._t(reward),
                    next_done,
                    extras["value"],
                )
                next_obs = self._obs_t(step_obs)
                next_done = self._t((np.asarray(term) | np.asarray(trunc)).astype(np.float32))
                action_log.append(np.asarray(action_np))
                self._log_episodes(infos)

            progress = min(1.0, self.global_step / self.total_steps)
            algo.update_on_policy(buf, next_obs, next_done, self.rec, self.global_step, progress)

            update += 1
            if update % self.log_interval_updates == 0:
                self.rec.histogram("actions", np.concatenate([a.ravel() for a in action_log]),
                                    step=self.global_step)
                self._log_throughput()
                self._print_progress()
            self._maybe_video()
            self._maybe_eval()
            self._maybe_checkpoint()

    # ---- off-policy (DQN/SAC) --------------------------------------------
    def _train_off_policy(self) -> None:
        env, algo, N = self.env, self.algo, self.env.num_envs
        next_obs_np, _ = env.reset()
        next_obs = self._obs_t(next_obs_np)
        action_log: list[np.ndarray] = []
        updates = 0
        steps_since_train = 0  # env steps collected since the last training trigger
        last_log = 0

        while self.global_step < self.total_steps:
            self.global_step += N
            if self.global_step < self.learning_starts:
                action_np = np.array([env.single_action_space.sample() for _ in range(N)])
            else:
                action_np, _ = algo.act(next_obs)

            step_obs, reward, term, trunc, infos = env.step(action_np)
            reward = np.asarray(reward, dtype=np.float32)

            # store true terminal obs (SAME_STEP autoreset replaces it in step_obs)
            real_next = np.asarray(step_obs, dtype=env.obs_dtype).copy()
            final = extract_final_obs(infos, N, env.obs_shape, env.obs_dtype)
            if final is not None:
                fin_mask = np.asarray(infos["_final_obs"])
                real_next[fin_mask] = final[fin_mask]

            algo.observe(
                {
                    "obs": next_obs_np,
                    "action": action_np,
                    "reward": reward,
                    "next_obs": real_next,
                    # bootstrap mask: only true terminations cut the value target
                    "done": np.asarray(term, dtype=np.float32),
                }
            )

            next_obs_np = np.asarray(step_obs, dtype=env.obs_dtype)
            next_obs = self._obs_t(next_obs_np)
            action_log.append(np.asarray(action_np))
            self._log_episodes(infos)

            # Training cadence is decoupled from collection: we collect N transitions per
            # env step (N = num_envs) and, every `update_every` collected steps, run
            # `gradient_steps` gradient updates. This keeps the replay ratio well-defined
            # for any num_envs, and lets you trade speed vs sample-efficiency.
            if self.global_step >= self.learning_starts:
                steps_since_train += N
                trained = False
                while steps_since_train >= self.update_every:
                    steps_since_train -= self.update_every
                    for _ in range(self.gradient_steps):
                        algo.update_off_policy(self.rec, self.global_step)
                        updates += 1
                    trained = True
                if trained and updates - last_log >= 200:
                    last_log = updates
                    self.rec.histogram(
                        "actions", np.concatenate([a.ravel() for a in action_log[-200:]]),
                        step=self.global_step,
                    )
                    self._log_throughput()
                    self._print_progress()
            self._maybe_video()
            self._maybe_eval()
            self._maybe_checkpoint()
