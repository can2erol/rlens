from rlens.experiment.bench import _expand_cells
from rlens.experiment.report import format_markdown, summarize_runs
from rlens.experiment.run import train_single


def test_expand_cells_runs_format():
    spec = {
        "seeds": [0, 1],
        "algo_overrides": {"ppo": {"lr": 0.1}},
        "runs": [
            {"algo": "ppo", "env": "CartPole-v1", "steps": 5000, "overrides": {"gamma": 0.9}},
            {"algo": "dqn", "env": "Acrobot-v1"},
        ],
    }
    cells = _expand_cells(spec)
    assert len(cells) == 4  # 2 runs x 2 seeds
    ppo = next(c for c in cells if c["algo"] == "ppo")
    assert ppo["steps"] == 5000
    assert ppo["overrides"] == {"lr": 0.1, "gamma": 0.9}  # algo + per-run merge
    dqn = next(c for c in cells if c["algo"] == "dqn")
    assert dqn["steps"] == 100_000  # default_steps fallback


def test_expand_cells_grid_format():
    spec = {"grid": {"algo": ["ppo", "dqn"], "env": ["CartPole-v1"], "seed": [0, 1, 2]}}
    cells = _expand_cells(spec)
    assert len(cells) == 6  # 2 x 1 x 3


def test_expand_cells_run_level_fields():
    spec = {
        "seeds": [0],
        "runs": [
            {"algo": "dqn", "env": "LunarLander-v3", "num_envs": 4,
             "update_every": 4, "gradient_steps": 1, "learning_starts": 10000},
        ],
    }
    cell = _expand_cells(spec)[0]
    assert cell["train"] == {
        "num_envs": 4, "update_every": 4, "gradient_steps": 1, "learning_starts": 10000,
    }


def test_summarize_and_markdown(tmp_path):
    for seed in (0, 1):
        train_single(
            algo="ppo", env_id="CartPole-v1", total_steps=1200, seed=seed,
            device="cpu", runs_dir=tmp_path, name=f"ppo-s{seed}", progress=False,
        )

    targets = {"ppo": {"CartPole-v1": -1e9}}  # trivially passes
    summary = summarize_runs(tmp_path, episodes=3, targets=targets)
    assert len(summary["rows"]) == 1
    row = summary["rows"][0]
    assert row["algo"] == "ppo" and row["env"] == "CartPole-v1"
    assert row["seeds"] == 2
    assert row["passed"] is True

    # an impossible target flips the verdict
    summary2 = summarize_runs(tmp_path, episodes=3, targets={"ppo": {"CartPole-v1": 1e9}})
    assert summary2["rows"][0]["passed"] is False

    md = format_markdown(summary)
    assert "CartPole-v1" in md and "ppo" in md and "✅ pass" in md
