import numpy as np

from rlens.experiment.run import train_single
from rlens.telemetry.store import TelemetryStore


def _returns(run):
    s = TelemetryStore(run)
    rets = [e["ret"] for e in s.episodes()]
    s.close()
    return rets


def test_dqn_learns_cartpole(tmp_path):
    run = train_single(
        "dqn", "CartPole-v1", total_steps=30_000, seed=1, device="cpu",
        runs_dir=tmp_path, name="dqn-smoke", progress=False,
    )
    rets = _returns(run)
    assert len(rets) > 20
    first, last = np.mean(rets[:10]), np.mean(rets[-15:])
    assert last > first + 30, f"DQN did not learn: first={first:.1f} last={last:.1f}"


def test_sac_improves_pendulum(tmp_path):
    run = train_single(
        "sac", "Pendulum-v1", total_steps=10_000, seed=1, device="cpu",
        runs_dir=tmp_path, name="sac-smoke", progress=False,
    )
    rets = _returns(run)
    assert len(rets) > 10
    first, last = np.mean(rets[:5]), np.mean(rets[-5:])
    # Pendulum returns are negative; learning moves them sharply upward
    assert last > first + 300, f"SAC did not improve: first={first:.1f} last={last:.1f}"
