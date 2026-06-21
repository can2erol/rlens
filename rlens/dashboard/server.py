"""Dashboard server.

A FastAPI app that *tails* run directories and serves their telemetry to the browser SPA.
It is fully decoupled from training: it reads the same SQLite stores the trainer writes
(WAL mode → safe concurrent reads), so it can attach to a live run, a finished run, or a
whole benchmark grid, with no coordination beyond the shared ``runs/`` directory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rlens.telemetry.store import TelemetryStore, read_meta

STATIC_DIR = Path(__file__).parent / "static"


def _list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    if not runs_dir.exists():
        return runs
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not (d / "telemetry.db").exists():
            continue
        meta = read_meta(d)
        cfg = meta.get("config", {})
        runs.append(
            {
                "id": d.name,
                "status": meta.get("status", "unknown"),
                "algo": cfg.get("algo"),
                "env": cfg.get("env_id"),
                "seed": cfg.get("seed"),
                "final_step": meta.get("final_step"),
            }
        )
    return runs


def create_app(runs_dir: Path) -> FastAPI:
    runs_dir = Path(runs_dir)
    app = FastAPI(title="rlens dashboard")

    def store_for(run_id: str) -> TelemetryStore:
        return TelemetryStore(runs_dir / run_id)

    @app.get("/api/runs")
    def api_runs() -> Any:
        return _list_runs(runs_dir)

    @app.get("/api/runs/{run_id}/tags")
    def api_tags(run_id: str) -> Any:
        s = store_for(run_id)
        try:
            return {"scalars": s.tags("scalars"), "histograms": s.tags("histograms")}
        finally:
            s.close()

    @app.get("/api/runs/{run_id}/scalars")
    def api_scalars(run_id: str, tag: str, after_id: int = 0) -> Any:
        s = store_for(run_id)
        try:
            rows = s.scalars(tag, after_id=after_id)
            return {
                "tag": tag,
                "steps": [r["step"] for r in rows],
                "values": [r["value"] for r in rows],
                "times": [r["wall_time"] for r in rows],
                "last_id": rows[-1]["id"] if rows else after_id,
            }
        finally:
            s.close()

    @app.get("/api/runs/{run_id}/meta")
    def api_meta(run_id: str) -> Any:
        return read_meta(runs_dir / run_id)

    @app.get("/api/runs/{run_id}/summary")
    def api_summary(run_id: str) -> Any:
        """Headline metrics for the comparison table: best/last return, eval, FPS."""
        meta = read_meta(runs_dir / run_id)
        cfg = meta.get("config", {})
        s = store_for(run_id)
        try:
            ret = s.scalar_summary("rollout/episodic_return")
            ev = s.scalar_summary("eval/return_mean")
            fps = s.scalar_summary("perf/steps_per_sec")
        finally:
            s.close()
        return {
            "id": run_id,
            "algo": cfg.get("algo"),
            "env": cfg.get("env_id"),
            "seed": cfg.get("seed"),
            "status": meta.get("status", "unknown"),
            "steps": meta.get("final_step"),
            "return_last": ret["last"],
            "return_best": ret["best"],
            "eval_last": ev["last"],
            "eval_best": ev["best"],
            "best_return": meta.get("best_return"),
            "fps": fps["last"],
        }

    @app.get("/api/runs/{run_id}/histogram")
    def api_histogram(run_id: str, tag: str) -> Any:
        s = store_for(run_id)
        try:
            h = s.latest_histogram(tag)
            return h or JSONResponse({"error": "no data"}, status_code=404)
        finally:
            s.close()

    @app.get("/api/runs/{run_id}/episodes")
    def api_episodes(run_id: str, after_id: int = 0) -> Any:
        s = store_for(run_id)
        try:
            return {"episodes": s.episodes(after_id=after_id)}
        finally:
            s.close()

    @app.get("/api/runs/{run_id}/videos")
    def api_videos(run_id: str) -> Any:
        vdir = runs_dir / run_id / "videos"
        if not vdir.exists():
            return {"videos": []}
        vids = sorted(p.name for p in vdir.glob("*.mp4"))
        # parse the step out of step_XXXXXXXX.mp4 for labelling
        out = []
        for name in vids:
            stem = name.removesuffix(".mp4").removeprefix("step_")
            try:
                step = int(stem)
            except ValueError:
                step = None
            out.append({"name": name, "step": step, "url": f"/api/runs/{run_id}/video/{name}"})
        return {"videos": out}

    @app.get("/api/runs/{run_id}/video/{name}")
    def api_video_file(run_id: str, name: str) -> Any:
        path = (runs_dir / run_id / "videos" / name).resolve()
        # guard against path traversal
        if not str(path).startswith(str((runs_dir / run_id / "videos").resolve())) or not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, media_type="video/mp4")

    @app.websocket("/api/runs/{run_id}/stream")
    async def stream(ws: WebSocket, run_id: str) -> None:
        """Push new scalar rows (all tags) as they are written."""
        await ws.accept()
        s = store_for(run_id)
        cursors: dict[str, int] = {}
        try:
            while True:
                payload: dict[str, Any] = {"scalars": {}}
                for tag in s.tags("scalars"):
                    rows = s.scalars(tag, after_id=cursors.get(tag, 0))
                    if rows:
                        cursors[tag] = rows[-1]["id"]
                        payload["scalars"][tag] = {
                            "steps": [r["step"] for r in rows],
                            "values": [r["value"] for r in rows],
                        }
                if payload["scalars"]:
                    await ws.send_json(payload)
                await asyncio.sleep(0.75)
        except WebSocketDisconnect:
            pass
        finally:
            s.close()

    # static SPA -----------------------------------------------------------
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def serve(runs_dir: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    app = create_app(Path(runs_dir))
    print(f"rlens dashboard → http://{host}:{port}  (serving {Path(runs_dir).resolve()})")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve(Path("runs"))
