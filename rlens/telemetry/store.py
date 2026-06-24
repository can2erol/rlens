"""Per-run telemetry store backed by SQLite.

One run == one directory:

    runs/<run_id>/
        telemetry.db   scalars / histograms / episodes / frames tables
        run.json       config + library versions + status
        frames/        captured PNG frames (phase 6)
        videos/        assembled MP4 clips (phase 6)

The DB is written by the training process and *read* by the dashboard while training is
still in flight. SQLite WAL mode makes that concurrent reader/writer pattern safe. Every
table has an autoincrement ``id`` that the dashboard uses as a monotonic cursor to tail
only rows it has not seen yet.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS scalars (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    value     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scalars_tag ON scalars(tag, id);

CREATE TABLE IF NOT EXISTS histograms (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    counts    TEXT    NOT NULL,  -- json list[int]
    edges     TEXT    NOT NULL   -- json list[float], len = len(counts)+1
);
CREATE INDEX IF NOT EXISTS idx_hist_tag ON histograms(tag, id);

CREATE TABLE IF NOT EXISTS episodes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    ret       REAL    NOT NULL,
    length    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    step      INTEGER NOT NULL,
    episode   INTEGER NOT NULL,
    path      TEXT    NOT NULL
);
"""


@dataclass
class Histogram:
    counts: list[int]
    edges: list[float]

    @classmethod
    def from_values(cls, values: np.ndarray, bins: int = 30) -> Histogram:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return cls(counts=[], edges=[])
        counts, edges = np.histogram(values, bins=bins)
        return cls(counts=counts.astype(int).tolist(), edges=edges.astype(float).tolist())


class TelemetryStore:
    """Low-level SQLite read/write for a single run directory."""

    def __init__(self, run_dir: Path, create: bool = False):
        self.run_dir = Path(run_dir)
        self.db_path = self.run_dir / "telemetry.db"
        if create:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        if create:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---- writes -----------------------------------------------------------
    def insert_scalars(self, rows: Iterable[tuple[str, int, float, float]]) -> None:
        self._conn.executemany(
            "INSERT INTO scalars(tag, step, wall_time, value) VALUES (?,?,?,?)", rows
        )

    def insert_histograms(
        self, rows: Iterable[tuple[str, int, float, Histogram]]
    ) -> None:
        payload = [
            (tag, step, wt, json.dumps(h.counts), json.dumps(h.edges))
            for tag, step, wt, h in rows
        ]
        self._conn.executemany(
            "INSERT INTO histograms(tag, step, wall_time, counts, edges) VALUES (?,?,?,?,?)",
            payload,
        )

    def insert_episodes(self, rows: Iterable[tuple[int, float, float, int]]) -> None:
        self._conn.executemany(
            "INSERT INTO episodes(step, wall_time, ret, length) VALUES (?,?,?,?)", rows
        )

    def insert_frames(self, rows: Iterable[tuple[int, int, str]]) -> None:
        self._conn.executemany(
            "INSERT INTO frames(step, episode, path) VALUES (?,?,?)", rows
        )

    def commit(self) -> None:
        self._conn.commit()

    # ---- reads ------------------------------------------------------------
    def tags(self, table: str = "scalars") -> list[str]:
        rows = self._conn.execute(f"SELECT DISTINCT tag FROM {table} ORDER BY tag").fetchall()
        return [r["tag"] for r in rows]

    def scalars(self, tag: str, after_id: int = 0) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, step, wall_time, value FROM scalars WHERE tag=? AND id>? ORDER BY id",
            (tag, after_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def histogram_series(self, tag: str, max_snapshots: int = 200) -> list[dict[str, Any]]:
        """All histogram snapshots for a tag, oldest→newest, downsampled to ``max_snapshots``.

        Powers the dashboard's distribution-over-time view: each snapshot is one column of
        the heatmap. Downsampling keeps the payload bounded for long runs.
        """
        rows = self._conn.execute(
            "SELECT id, step, wall_time, counts, edges FROM histograms WHERE tag=? ORDER BY id",
            (tag,),
        ).fetchall()
        n = len(rows)
        if max_snapshots and n > max_snapshots:
            rows = rows[:: max(1, n // max_snapshots)]
        out = []
        for r in rows:
            d = dict(r)
            d["counts"] = json.loads(d["counts"])
            d["edges"] = json.loads(d["edges"])
            out.append(d)
        return out

    def latest_histogram(self, tag: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, step, wall_time, counts, edges FROM histograms "
            "WHERE tag=? ORDER BY id DESC LIMIT 1",
            (tag,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["counts"] = json.loads(d["counts"])
        d["edges"] = json.loads(d["edges"])
        return d

    def episodes(self, after_id: int = 0) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, step, wall_time, ret, length FROM episodes WHERE id>? ORDER BY id",
            (after_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def scalar_summary(self, tag: str) -> dict[str, Any]:
        """Best (max) and last value of a scalar tag — for the run comparison table."""
        agg = self._conn.execute(
            "SELECT COUNT(*) AS n, MAX(value) AS mx FROM scalars WHERE tag=?", (tag,)
        ).fetchone()
        last = self._conn.execute(
            "SELECT value FROM scalars WHERE tag=? ORDER BY id DESC LIMIT 1", (tag,)
        ).fetchone()
        return {
            "best": agg["mx"] if agg["n"] else None,
            "last": last["value"] if last else None,
        }

    def close(self) -> None:
        self._conn.close()


# ---- run metadata (run.json) ---------------------------------------------
def write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "updated_at": time.time()}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))


def read_meta(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "run.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())
