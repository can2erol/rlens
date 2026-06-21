from rlens.experiment.run import train_single
from rlens.telemetry.store import TelemetryStore


def _lr_curve(run):
    st = TelemetryStore(run)
    pts = st.scalars("ppo/lr")
    st.close()
    return [p["value"] for p in pts]


def test_ppo_lr_anneals_when_enabled(tmp_path):
    """With anneal_lr=True, ppo/lr is logged and decreases toward 0 over training."""
    run = train_single(
        algo="ppo", env_id="CartPole-v1", total_steps=6000, seed=0,
        device="cpu", runs_dir=tmp_path, name="anneal", progress=False,
        algo_overrides={"anneal_lr": True},
    )
    lrs = _lr_curve(run)
    assert len(lrs) >= 2
    assert lrs[0] > lrs[-1]                       # decreasing
    assert abs(lrs[0] - 3e-4) < 3e-4              # starts near the base LR
    assert lrs[-1] < lrs[0] * 0.5                 # well down by the end


def test_ppo_lr_constant_by_default(tmp_path):
    """Annealing is opt-in: by default the logged LR stays at the base value."""
    run = train_single(
        algo="ppo", env_id="CartPole-v1", total_steps=6000, seed=0,
        device="cpu", runs_dir=tmp_path, name="noanneal", progress=False,
    )
    lrs = _lr_curve(run)
    assert len(lrs) >= 2
    assert max(lrs) - min(lrs) < 1e-9             # constant
    assert abs(lrs[0] - 3e-4) < 1e-9
