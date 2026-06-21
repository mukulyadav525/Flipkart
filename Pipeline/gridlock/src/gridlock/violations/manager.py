"""Runs all violation engines for a frame, dedups, and persists events."""

from __future__ import annotations

import json
from pathlib import Path

from .base import FrameContext, ViolationEngine, ViolationEvent


class ViolationManager:
    def __init__(self, engines: list[ViolationEngine]):
        self.engines = engines
        self.events: list[ViolationEvent] = []
        self._seen: set[tuple[str, int]] = set()  # (type, track_id) dedup

    def update(self, ctx: FrameContext) -> list[ViolationEvent]:
        new: list[ViolationEvent] = []
        for engine in self.engines:
            for ev in engine.update(ctx):
                key = (ev.type, ev.track_id)
                if key in self._seen:
                    continue
                self._seen.add(key)
                self.events.append(ev)
                new.append(ev)
        return new

    def flagged_ids(self) -> dict[int, str]:
        """Map track_id -> violation type for any currently-flagged track."""
        out: dict[int, str] = {}
        for engine in self.engines:
            for tid in engine.flagged_ids():
                out[tid] = engine.name
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for ev in self.events:
                f.write(json.dumps(ev.to_dict()) + "\n")
