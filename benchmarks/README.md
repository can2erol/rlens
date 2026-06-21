# Benchmarks

These benchmarks check that rlens's algorithms reach the returns commonly reported for each
task. Every entry trains from scratch over several seeds; the reported number is a
deterministic (exploration-free) evaluation of the saved policy, averaged across seeds.

"Reference" is the widely cited solved / good-policy threshold for the task, and ✅ means the
across-seed mean clears it. Numbers come from `rlens report` (20 evaluation episodes per run);
training budgets and seeds live in each spec file. "Best policy" evaluates the highest-scoring
checkpoint (`--best`).

## Classic control

CPU-only, no extra dependencies. Spec: [`classic_control.yaml`](classic_control.yaml).

```bash
rlens bench benchmarks/classic_control.yaml --runs-dir runs_bench
rlens report runs_bench --targets benchmarks/classic_control.yaml --best
```

References: CartPole-v1 is solved at ≥ 475 (max 500); a good Acrobot-v1 policy reaches about
−100; SAC on Pendulum-v1 reaches roughly −150 to −200 (a random policy scores about −1200).

| algorithm | env | seeds | eval return (best policy) | reference | status |
|-----------|-----|-------|---------------------------|-----------|--------|
| PPO | CartPole-v1 | 3 | 500.0 ± 0.0 | ≥ 475 | ✅ |
| DQN | CartPole-v1 | 3 | 500.0 ± 0.0 | ≥ 475 | ✅ |
| PPO | Acrobot-v1 | 3 | −86.4 ± 4.9 | ≥ −100 | ✅ |
| DQN | Acrobot-v1 | 3 | −81.1 ± 0.6 | ≥ −100 | ✅ |
| SAC | Pendulum-v1 | 3 | −131.4 ± 0.3 | ≥ −250 | ✅ |

Evaluating the *final* checkpoint instead of the best gives the same numbers except
PPO/CartPole-v1 = 463.1 ± 41.7: every seed reaches 500 during training, but a PPO policy can
oscillate after solving CartPole, so the last checkpoint is not always the best one. rlens
saves the best-eval checkpoint for exactly this reason. Full table: [`results.md`](results.md).

## LunarLander-v3

A Box2D control task — land a craft on a pad, with a large reward or penalty delivered at the
end of the episode; ≥ 200 average return is considered solved. Needs the Box2D extra
(`pip install -e ".[box2d]"`). Spec: [`lunarlander.yaml`](lunarlander.yaml).

```bash
rlens bench benchmarks/lunarlander.yaml --runs-dir runs_lander
rlens report runs_lander --targets benchmarks/lunarlander.yaml --best
```

| algorithm | env | seeds | eval return (best policy) | reference | status |
|-----------|-----|-------|---------------------------|-----------|--------|
| DQN | LunarLander-v3 | 3 | 251.5 ± 9.7 | ≥ 200 | ✅ |
| PPO | LunarLander-v3 | 3 | 229.9 ± 9.4 | ≥ 200 | ✅ |

Both algorithms solve the task. Because the decisive reward arrives at the end of a long
episode, PPO uses a long-horizon configuration (`gamma=0.999`, `gae_lambda=0.98`, 16 parallel
envs with 1024-step rollouts, 256 minibatches — see the spec) rather than the classic-control
defaults.

Measured on CPU; 3 seeds; 20 evaluation episodes per run.
