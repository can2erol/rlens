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

    assert client.get("/").status_code == 200
