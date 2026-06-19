# rlens

An **observability-first** reinforcement-learning training & benchmarking library, built on
PyTorch and Gymnasium. The headline feature is *visibility*: a local web dashboard that
streams reward curves, per-layer gradient norms, action/value distributions, and rollout
video — live, while you train — and overlays multiple runs for benchmarking.

> Designed and tested on Apple Silicon (M-series, MPS). No CUDA required.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

```bash
# train a policy (writes telemetry to ./runs/<run-name>)
rlens train --algo ppo --env CartPole-v1

# in another terminal, watch it learn
rlens dashboard         # open http://127.0.0.1:8000

# score a trained policy and record a rollout video
rlens eval runs/<run-name> --episodes 10 --video

# benchmark a grid of algo x env x seed
rlens bench configs/bench.yaml
```

## Evaluation

Training returns mix in exploration (epsilon-greedy, stochastic sampling), so they
undersell a policy. `rlens eval` loads a run's `policy.pt` and scores it greedily:

```bash
rlens eval runs/ppo-CartPole-v1-s0-20260619  # mean ± std return over 10 episodes
rlens eval runs/<name> --episodes 20 --video # also writes videos/eval.mp4
rlens eval runs/<name> --stochastic          # sample actions instead of greedy
```

To track a clean eval curve *during* training (logged as `eval/return_mean`, distinct
from the noisy `rollout/episodic_return`), pass `--eval-interval`:

```bash
rlens train --algo dqn --env CartPole-v1 --eval-interval 5000 --eval-episodes 10
```

## Configuration

Set any hyperparameter from the command line with `--set key=value` — algorithm knobs
(`lr`, `gamma`, `batch_size`, `hidden`, ...) and run-level knobs (`num_envs`, `rollout_len`,
`learning_starts`, ...) share one namespace and are type-checked against the config schema:

```bash
rlens train --algo sac --env Pendulum-v1 --set lr=3e-4 --set hidden=[256,256] --set tau=0.01
```

An unknown key fails immediately and lists the valid ones. For repeatable runs, put the
config in YAML and override pieces on the command line:

```bash
rlens train --config configs/ppo_cartpole.yaml --steps 200000 --set lr=1e-3
```

Precedence is **defaults < `--config` < explicit flags < `--set`**. The fully-resolved
config (including library versions and git SHA) is saved to each run's `run.json`.

## Algorithms

| Algo | Type        | Action space |
|------|-------------|--------------|
| PPO  | on-policy   | discrete + continuous |
| DQN  | off-policy  | discrete |
| SAC  | off-policy  | continuous |

All three share one trainer and one telemetry layer, so adding an algorithm means writing
`act()` and `update()` — observability comes for free.

## Layout

```
rlens/
  core/         device, seeding, envs, buffers, networks
  algos/        ppo, dqn, sac (+ base Algorithm)
  trainer.py    shared on-policy / off-policy loop
  telemetry/    recorder, sqlite store, frame/video writers
  experiment/   config, single-run, benchmark grid
  dashboard/    FastAPI server + no-build static SPA
```
