# Benchmarks

Reproducibility benchmarks: do rlens's algorithms actually reach the returns everyone else
reports? Each cell trains from scratch over several seeds; the headline number is a fresh
**deterministic** evaluation of the saved policy (exploration-free), averaged across seeds.

## Classic control (CPU, no MuJoCo/Box2D)

Spec: [`classic_control.yaml`](classic_control.yaml). Reproduce with:

```bash
rlens bench benchmarks/classic_control.yaml --runs-dir runs_bench
rlens report runs_bench --targets benchmarks/classic_control.yaml --out benchmarks/results.md
```

References are commonly reported "good policy" numbers: CartPole-v1 is *solved* at ≥ 475
(max 500); a good Acrobot-v1 policy sits around −100; SAC on Pendulum-v1 reaches roughly
−150 to −200 (random ≈ −1200). "Pass" means the across-seed mean clears the reference.

<!-- RESULTS:START -->
**Best policy** (`rlens report runs_bench --best`, highest-eval checkpoint) — all pass:

| algo | env | seeds | eval return (mean ± std) | reference | status |
|------|-----|-------|--------------------------|-----------|--------|
| dqn | Acrobot-v1 | 3 | -81.1 ± 0.6 | ≥ -100 | ✅ pass |
| dqn | CartPole-v1 | 3 | 500.0 ± 0.0 | ≥ 475 | ✅ pass |
| ppo | Acrobot-v1 | 3 | -86.4 ± 4.9 | ≥ -100 | ✅ pass |
| ppo | CartPole-v1 | 3 | 500.0 ± 0.0 | ≥ 475 | ✅ pass |
| sac | Pendulum-v1 | 3 | -131.4 ± 0.3 | ≥ -250 | ✅ pass |

**Final policy** (last checkpoint): identical except **ppo / CartPole-v1 = 463.1 ± 41.7**.
All three seeds *reach* 500 during training, but PPO's final policy oscillates once CartPole
is solved (a well-known behavior); `rlens` keeps the best-eval checkpoint so the deployed
policy reflects peak performance. Full table: [`results.md`](results.md).

Measured 2026-06; CPU; 3 seeds; 20 deterministic eval episodes per run via `rlens report`.
<!-- RESULTS:END -->

Numbers are produced by `rlens report` (20 eval episodes/run). Training budgets and seeds
are in the spec. See the live curves and per-run rollout videos in `rlens dashboard`.
