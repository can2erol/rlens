import numpy as np

from rlens.telemetry.recorder import Recorder
from rlens.telemetry.store import TelemetryStore, read_meta


def test_scalar_roundtrip(tmp_path):
    run_dir = tmp_path / "run0"
    with Recorder(run_dir, flush_every=1000, flush_interval_s=1000) as rec:
        for step in range(10):
            rec.scalar("reward", float(step), step=step)
        rec.flush()

    store = TelemetryStore(run_dir)
    rows = store.scalars("reward")
    assert len(rows) == 10
    assert [r["value"] for r in rows] == [float(i) for i in range(10)]
    assert [r["step"] for r in rows] == list(range(10))
    store.close()


def test_scalars_dict_and_tags(tmp_path):
    run_dir = tmp_path / "run1"
    with Recorder(run_dir) as rec:
        rec.scalars({"loss/policy": 0.1, "loss/value": 0.2}, step=0)
        rec.flush()

    store = TelemetryStore(run_dir)
    assert set(store.tags()) == {"loss/policy", "loss/value"}
    store.close()


def test_incremental_cursor(tmp_path):
    run_dir = tmp_path / "run2"
    rec = Recorder(run_dir, flush_every=1)
    rec.scalar("r", 1.0, step=0)
    rec.flush()

    store = TelemetryStore(run_dir)
    first = store.scalars("r")
    assert len(first) == 1
    cursor = first[-1]["id"]

    rec.scalar("r", 2.0, step=1)
    rec.flush()
    new = store.scalars("r", after_id=cursor)
    assert len(new) == 1
    assert new[0]["value"] == 2.0
    rec.close()
    store.close()


def test_histogram_roundtrip(tmp_path):
    run_dir = tmp_path / "run3"
    rng = np.random.default_rng(0)
    data = rng.normal(size=1000)
    with Recorder(run_dir) as rec:
        rec.histogram("acts", data, step=0, bins=20)
        rec.flush()

    store = TelemetryStore(run_dir)
    h = store.latest_histogram("acts")
    assert h is not None
    assert len(h["counts"]) == 20
    assert len(h["edges"]) == 21
    assert sum(h["counts"]) == 1000
    store.close()


def test_episodes_and_meta(tmp_path):
    run_dir = tmp_path / "run4"
    with Recorder(run_dir) as rec:
        rec.episode(ret=100.0, length=200, step=200)
        rec.meta({"algo": "ppo", "env": "CartPole-v1", "seed": 0})
        rec.flush()

    store = TelemetryStore(run_dir)
    eps = store.episodes()
    assert len(eps) == 1
    assert eps[0]["ret"] == 100.0
    assert eps[0]["length"] == 200
    store.close()

    meta = read_meta(run_dir)
    assert meta["algo"] == "ppo"
    assert "updated_at" in meta
