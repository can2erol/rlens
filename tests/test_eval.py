import math

import torch

from rlens.algos.ppo import PPO
from rlens.core.env import EnvManager
from rlens.experiment.eval import evaluate, load_trained_algo
from rlens.experiment.run import train_single
from rlens.telemetry.store import TelemetryStore


def test_evaluate_structure():
    """evaluate() returns well-formed stats over the requested number of episodes."""
    device = torch.device("cpu")
    env = EnvManager("CartPole-v1", num_envs=1, seed=0)
    algo = PPO(env, device)
    env.close()

    res = evaluate(algo, "CartPole-v1", device, episodes=3, seed=0)

    assert res["episodes"] == 3
    assert len(res["returns"]) == 3
    assert math.isfinite(res["return_mean"])
    assert res["return_min"] <= res["return_mean"] <= res["return_max"]
    assert res["length_mean"] > 0


def test_evaluate_is_deterministic():
    """Greedy eval with the same seeds is reproducible run-to-run."""
    device = torch.device("cpu")
    env = EnvManager("CartPole-v1", num_envs=1, seed=0)
    algo = PPO(env, device)
    env.close()

    a = evaluate(algo, "CartPole-v1", device, episodes=4, seed=7, deterministic=True)
    b = evaluate(algo, "CartPole-v1", device, episodes=4, seed=7, deterministic=True)
    assert a["returns"] == b["returns"]


def test_eval_roundtrip_and_periodic(tmp_path):
    """A trained run can be reloaded and scored, and periodic eval writes an eval/ curve."""
    run = train_single(
        algo="ppo",
        env_id="CartPole-v1",
        total_steps=4_000,
        seed=1,
        device="cpu",
        runs_dir=tmp_path,
        name="ppo-eval",
        progress=False,
        eval_interval=2_000,
        eval_episodes=2,
    )

    # periodic eval logged a clean eval/ curve distinct from rollout/
    store = TelemetryStore(run)
    eval_pts = store.scalars("eval/return_mean")
    tags = store.tags()
    store.close()
    assert len(eval_pts) >= 1, "no periodic eval points logged"
    assert "eval/return_mean" in tags

    # the saved policy reloads and scores without error
    algo, env_id, meta = load_trained_algo(run, device="cpu")
    assert env_id == "CartPole-v1"
    assert meta["config"]["algo"] == "ppo"
    res = evaluate(algo, env_id, algo.device, episodes=2, seed=0)
    assert math.isfinite(res["return_mean"])
