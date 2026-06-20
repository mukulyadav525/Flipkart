"""Violation engine interface + the event record they emit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..scene import CameraConfig
from ..tracking_state import TrackState
from ..types import Track


@dataclass
class FrameContext:
    """Everything an engine might need for one processed frame.

    Geometry engines use `states` + `config`; perception engines (helmet,
    seatbelt) additionally crop from `frame`. Phase 4 ANPR will use it too."""

    states: list[TrackState]
    tracks: list[Track]
    frame: Any  # np.ndarray (clean, un-preprocessed frame for crops)
    frame_idx: int
    timestamp: float
    config: CameraConfig


@dataclass
class ViolationEvent:
    type: str          # e.g. "triple_riding", "no_helmet", "illegal_parking"
    track_id: int
    class_name: str
    frame_idx: int
    timestamp: float   # seconds into the clip
    xyxy: tuple[float, float, float, float]
    zone: str | None = None
    detail: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ViolationEngine(Protocol):
    """Stateful per-frame engine.

    `update` returns any *newly* confirmed violations; the manager dedups by
    (type, track_id) across the run."""

    name: str

    def update(self, ctx: FrameContext) -> list[ViolationEvent]:
        ...

    def flagged_ids(self) -> set[int]:
        """Track ids currently considered in-violation (for drawing)."""
        ...
