# LunarLander-v3

## Best policy (`--best`)

| algo | env | seeds | eval return (mean ± std) | reference | status |
|------|-----|-------|--------------------------|-----------|--------|
| dqn | LunarLander-v3 | 3 | 251.5 ± 9.7 | ≥ 200 | ✅ pass |
| ppo | LunarLander-v3 | 3 | 50.5 ± 68.7 | ≥ 200 | ❌ miss |

## Final policy

| algo | env | seeds | eval return (mean ± std) | reference | status |
|------|-----|-------|--------------------------|-----------|--------|
| dqn | LunarLander-v3 | 3 | 224.0 ± 46.8 | ≥ 200 | ✅ pass |
| ppo | LunarLander-v3 | 3 | 32.3 ± 78.6 | ≥ 200 | ❌ miss |
