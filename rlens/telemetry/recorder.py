"""Recorder — the single telemetry sink that trainer and algos push to.

Design goals:
- *Near-zero boilerplate* in algorithm code: call ``rec.scalar(...)`` / ``rec.histogram(...)``
  and forget about it.
- Cheap on the hot path: writes are buffered in memory and flushed to SQLite in batches
  (by count or elapsed time) so logging never dominates the training step.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from rlens.telemetry.store import Histogram, TelemetryStore, write_meta


class Recorder:
    def __init__(
        self,
        run_dir: Path,
        flush_every: int = 200,
        flush_interval_s: float = 1.0,
    ):
        self.run_dir = Path(run_dir)
        self.store = TelemetryStore(self.run_dir, create=True)
        self.flush_every = flush_every
        self.flush_interval_s = flush_interval_s

        self._scalars: list[tuple[str, int, float, float]] = []
        self._hists: list[tuple[str, int, float, Histogram]] = []
        self._episodes: list[tuple[int, float, float, int]] = []
        self._frames: list[tuple[int, int, str]] = []
        self._last_flush = time.time()

    # ---- public API -------------------------------------------------------
    def scalar(self, tag: str, value: float, step: int) -> None:
        self._scalars.append((tag, int(step), time.time(), float(value)))
        self._maybe_flush()

    def scalars(self, values: dict[str, float], step: int) -> None:
        """Log a dict of tag -> value in one call (common for algo update metrics)."""
        wt = time.time()
        for tag, value in values.items():
            self._scalars.append((tag, int(step), wt, float(value)))
        self._maybe_flush()

    def histogram(self, tag: str, values: Any, step: int, bins: int = 30) -> None:
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 0:
            return
        self._hists.append((tag, int(step), time.time(), Histogram.from_values(arr, bins)))
        self._maybe_flush()

    def episode(self, ret: float, length: int, step: int) -> None:
        self._episodes.append((int(step), time.time(), float(ret), int(length)))
        self._maybe_flush()

    def frame(self, step: int, episode: int, path: str) -> None:
        self._frames.append((int(step), int(episode), str(path)))

    def meta(self, meta: dict[str, Any]) -> None:
        write_meta(self.run_dir, meta)

    # ---- flushing ---------------------------------------------------------
    def _pending(self) -> int:
        return len(self._scalars) + len(self._hists) + len(self._episodes) + len(self._frames)

    def _maybe_flush(self) -> None:
        if self._pending() >= self.flush_every or (
            time.time() - self._last_flush
        ) >= self.flush_interval_s:
            self.flush()

    def flush(self) -> None:
        if self._scalars:
            self.store.insert_scalars(self._scalars)
            self._scalars.clear()
        if self._hists:
            self.store.insert_histograms(self._hists)
            self._hists.clear()
        if self._episodes:
            self.store.insert_episodes(self._episodes)
            self._episodes.clear()
        if self._frames:
            self.store.insert_frames(self._frames)
            self._frames.clear()
        self.store.commit()
        self._last_flush = time.time()

    def close(self) -> None:
        self.flush()
        self.store.close()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
