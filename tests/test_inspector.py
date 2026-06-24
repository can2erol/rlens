"""Policy inspector: per-layer weight/gradient distribution capture and serving.

Covers the three new pieces — the grad/param distribution helpers, the Trainer logging
weight & gradient histograms over training on a coarse interval, and the
``histogram_series`` store read + dashboard endpoint that the heatmap view consumes.
"""

import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from rlens.dashboard.server import create_app
from rlens.experiment.config import TrainConfig
from rlens.experiment.run import run_config
from rlens.telemetry.grads import grad_distributions, param_distributions
from rlens.telemetry.recorder import Recorder
from rlens.telemetry.store import TelemetryStore


# ---- unit: distribution helpers -------------------------------------------
def test_param_distributions_one_entry_per_parameter_and_capped():
    net = nn.Sequential(nn.Linear(40, 60), nn.ReLU(), nn.Linear(60, 3))
    dists = param_distributions(net, prefix="weights/net", sample_cap=128)
    # one tag per named parameter (4: two weights + two biases), all under the prefix
    expected = {f"weights/net/{name}" for name, _ in net.named_parameters()}
    assert set(dists) == expected
    # the 40x60 = 2400-element weight tensor is downsampled to the cap
    assert dists["weights/net/0.weight"].shape == (128,)
    # the small bias (60 < cap) is left intact
    assert dists["weights/net/0.bias"].shape == (60,)


def test_grad_distributions_skip_params_without_grad():
    net = nn.Linear(4, 2)
    # no backward yet -> no gradients -> nothing captured
    assert grad_distributions(net) == {}
    net(torch.randn(8, 4)).sum().backward()
    grads = grad_distributions(net, prefix="grads/q")
    assert set(grads) == {"grads/q/weight", "grads/q/bias"}


# ---- store + endpoint: histogram time-series ------------------------------
def test_histogram_series_returns_ordered_snapshots(tmp_path):
    with Recorder(tmp_path / "r") as rec:
        rec.meta({"name": "r", "config": {"algo": "ppo"}})
        for step in range(5):
            rec.histogram("weights/actor/net.0.weight", [step, step + 1, step + 2], step=step)
        rec.flush()
    s = TelemetryStore(tmp_path / "r")
    try:
        snaps = s.histogram_series("weights/actor/net.0.weight")
        assert [snap["step"] for snap in snaps] == [0, 1, 2, 3, 4]
        assert all("counts" in snap and "edges" in snap for snap in snaps)
    finally:
        s.close()


def test_histogram_series_downsamples(tmp_path):
    with Recorder(tmp_path / "big") as rec:
        rec.meta({"name": "big", "config": {"algo": "ppo"}})
        for step in range(500):
            rec.histogram("grads/q/weight", [step, -step], step=step)
        rec.flush()
    client = TestClient(create_app(tmp_path))
    resp = client.get(
        "/api/runs/big/histogram_series",
        params={"tag": "grads/q/weight", "max_snapshots": 50},
    ).json()
    assert 0 < len(resp["snapshots"]) <= 50  # downsampled from 500


# ---- integration: Trainer captures weight & grad distributions ------------
def test_training_logs_weight_and_grad_histograms(tmp_path):
    cfg = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        total_steps=3072,
        seed=0,
        device="cpu",
        num_envs=4,
        rollout_len=128,
        inspect_interval_steps=1,  # snapshot every cycle
        algo_overrides={"num_minibatches": 2},
    )
    run = run_config(cfg, runs_dir=tmp_path, name="ppo-inspect", progress=False)

    s = TelemetryStore(run)
    try:
        hist_tags = s.tags("histograms")
        weight_tags = [t for t in hist_tags if t.startswith("weights/")]
        grad_tags = [t for t in hist_tags if t.startswith("grads/")]
        # both actor and critic modules contribute per-layer weight + grad histograms
        assert any(t.startswith("weights/actor/") for t in weight_tags)
        assert any(t.startswith("weights/critic/") for t in weight_tags)
        assert grad_tags, "no gradient distributions captured"
        # several snapshots accumulate over the run, not just one
        assert len(s.histogram_series(weight_tags[0])) >= 2
    finally:
        s.close()
