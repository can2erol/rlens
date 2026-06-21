from rlens.experiment.eval import load_trained_algo
from rlens.experiment.run import train_single
from rlens.telemetry.store import read_meta


def test_best_policy_saved_and_loadable(tmp_path):
    """With eval enabled, training writes best_policy.pt and records the best in meta."""
    run = train_single(
        algo="ppo", env_id="CartPole-v1", total_steps=6000, seed=0,
        device="cpu", runs_dir=tmp_path, name="best", progress=False,
        eval_interval=2000, eval_episodes=5,
    )

    assert (run / "best_policy.pt").exists()
    assert (run / "policy.pt").exists()

    meta = read_meta(run)
    assert meta["best_return"] is not None
    assert meta["best_step"] is not None

    # prefer_best loads the best checkpoint; default loads the final one — both work
    best_algo, env_id, _ = load_trained_algo(run, device="cpu", prefer_best=True)
    final_algo, _, _ = load_trained_algo(run, device="cpu", prefer_best=False)
    assert env_id == "CartPole-v1"
    assert best_algo is not None and final_algo is not None


def test_no_best_policy_without_eval(tmp_path):
    """No eval cadence -> no best_policy.pt, and prefer_best falls back to final."""
    run = train_single(
        algo="ppo", env_id="CartPole-v1", total_steps=2000, seed=0,
        device="cpu", runs_dir=tmp_path, name="noeval", progress=False,
    )
    assert not (run / "best_policy.pt").exists()
    meta = read_meta(run)
    assert meta["best_return"] is None

    # prefer_best gracefully falls back to policy.pt
    algo, env_id, _ = load_trained_algo(run, device="cpu", prefer_best=True)
    assert env_id == "CartPole-v1"
