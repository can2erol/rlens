from fastapi.testclient import TestClient

from rlens.dashboard.server import create_app
from rlens.telemetry.recorder import Recorder


def _make_run(runs_dir, name="r0"):
    with Recorder(runs_dir / name) as rec:
        rec.meta({"name": name, "status": "completed", "config": {"algo": "ppo", "env_id": "CartPole-v1", "seed": 0}})
        for step in range(5):
            rec.scalar("rollout/episodic_return", float(step * 10), step=step)
        rec.histogram("actions", [0, 0, 1, 1, 1], step=4, bins=2)
        rec.flush()


def test_dashboard_endpoints(tmp_path):
    _make_run(tmp_path, "r0")
    app = create_app(tmp_path)
    client = TestClient(app)

    runs = client.get("/api/runs").json()
    assert len(runs) == 1
    assert runs[0]["id"] == "r0"
    assert runs[0]["algo"] == "ppo"

    tags = client.get("/api/runs/r0/tags").json()
    assert "rollout/episodic_return" in tags["scalars"]
    assert "actions" in tags["histograms"]

    sc = client.get("/api/runs/r0/scalars", params={"tag": "rollout/episodic_return"}).json()
    assert sc["values"] == [0.0, 10.0, 20.0, 30.0, 40.0]

    hist = client.get("/api/runs/r0/histogram", params={"tag": "actions"}).json()
    assert sum(hist["counts"]) == 5

    vids = client.get("/api/runs/r0/videos").json()
    assert vids["videos"] == []

    # scalars now carry wall_time for the time x-axis
    assert len(sc["times"]) == len(sc["values"])

    assert client.get("/").status_code == 200


def test_meta_and_summary_endpoints(tmp_path):
    name = "r1"
    with Recorder(tmp_path / name) as rec:
        rec.meta({
            "name": name, "status": "completed",
            "config": {"algo": "ppo", "env_id": "CartPole-v1", "seed": 2, "lr": 0.0003},
            "final_step": 5000, "best_return": 480.0, "best_step": 4000,
        })
        for step in range(5):
            rec.scalar("rollout/episodic_return", float(step * 100), step=step)  # max 400
            rec.scalar("eval/return_mean", float(step * 90), step=step)           # max 360
            rec.scalar("perf/steps_per_sec", 1234.0, step=step)
        rec.flush()

    client = TestClient(create_app(tmp_path))

    meta = client.get(f"/api/runs/{name}/meta").json()
    assert meta["config"]["lr"] == 0.0003
    assert meta["best_return"] == 480.0

    s = client.get(f"/api/runs/{name}/summary").json()
    assert s["algo"] == "ppo" and s["env"] == "CartPole-v1" and s["seed"] == 2
    assert s["status"] == "completed" and s["steps"] == 5000
    assert s["return_best"] == 400.0   # max of episodic_return
    assert s["return_last"] == 400.0   # last logged
    assert s["eval_best"] == 360.0
    assert s["fps"] == 1234.0
