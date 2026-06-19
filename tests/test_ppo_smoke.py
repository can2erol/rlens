import numpy as np

from rlens.experiment.run import train_single
from rlens.telemetry.store import TelemetryStore


def test_ppo_learns_cartpole(tmp_path):
    """PPO should measurably improve on CartPole within a short budget."""
    run = train_single(
        algo="ppo",
        env_id="CartPole-v1",
        total_steps=30_000,
        seed=1,
        device="cpu",
        runs_dir=tmp_path,
        name="ppo-smoke",
        progress=False,
    )
    store = TelemetryStore(run)
    rets = [e["ret"] for e in store.episodes()]
    store.close()

    assert len(rets) > 20
    first = np.mean(rets[:10])
    last = np.mean(rets[-10:])
    # CartPole starts ~20; a learning agent comfortably clears 80 by 30k steps
    assert last > first + 40, f"no learning: first={first:.1f} last={last:.1f}"
    assert last > 80, f"final return too low: {last:.1f}"
