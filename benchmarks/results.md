## Final policy (last checkpoint)

| algo | env | seeds | eval return (mean ± std) | reference | status |
|------|-----|-------|--------------------------|-----------|--------|
| dqn | Acrobot-v1 | 3 | -81.1 ± 0.6 | ≥ -100 | ✅ pass |
| dqn | CartPole-v1 | 3 | 500.0 ± 0.0 | ≥ 475 | ✅ pass |
| ppo | Acrobot-v1 | 3 | -86.4 ± 4.9 | ≥ -100 | ✅ pass |
| ppo | CartPole-v1 | 3 | 463.1 ± 41.7 | ≥ 475 | ❌ miss |
| sac | Pendulum-v1 | 3 | -131.4 ± 0.3 | ≥ -250 | ✅ pass |

## Best policy (`--best`, highest-eval checkpoint)

| algo | env | seeds | eval return (mean ± std) | reference | status |
|------|-----|-------|--------------------------|-----------|--------|
| dqn | Acrobot-v1 | 3 | -81.1 ± 0.6 | ≥ -100 | ✅ pass |
| dqn | CartPole-v1 | 3 | 500.0 ± 0.0 | ≥ 475 | ✅ pass |
| ppo | Acrobot-v1 | 3 | -86.4 ± 4.9 | ≥ -100 | ✅ pass |
| ppo | CartPole-v1 | 3 | 500.0 ± 0.0 | ≥ 475 | ✅ pass |
| sac | Pendulum-v1 | 3 | -131.4 ± 0.3 | ≥ -250 | ✅ pass |
