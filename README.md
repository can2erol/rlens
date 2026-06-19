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

# benchmark a grid of algo x env x seed
rlens bench configs/bench.yaml
```

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
