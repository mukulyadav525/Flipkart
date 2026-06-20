"""Seatbelt non-compliance (hardest of the seven).

For each car/truck, crop the windshield region (upper-front of the vehicle box)
and run a seatbelt model that classifies driver belt presence. Flag when the
model reports "no seatbelt".

Reality check: at junction distance and with glare/occlusion this is the least
reliable violation. It works best on clear, close, daytime frames — scope it as
a stretch goal and demo it on good frames. Needs its own weights; disables
itself cleanly when absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..secondary import SecondaryDetector, crop_box
from .base import FrameContext, ViolationEvent

VEHICLES = {"car", "truck", "bus"}


@dataclass
class SeatbeltEngine:
    detector: SecondaryDetector | None = None
    windshield_top: float = 0.10       # crop the upper band of the vehicle box
    windshield_bottom: float = 0.55
    sustain_frames: int = 3
    name: str = "no_seatbelt"
    belt_names: tuple[str, ...] = (
        "seatbelt", "belt", "with seatbelt", "buckled", "person-seatbelt", "pakai")
    nobelt_names: tuple[str, ...] = (
        "no_seatbelt", "no-seatbelt", "without seatbelt", "no seatbelt", "unbuckled",
        "person-noseatbelt", "tidak-pakai")
    _streak: dict[int, int] = field(default_factory=dict)
    _flagged: set[int] = field(default_factory=set)
    _fired: set[int] = field(default_factory=set)

    @property
    def available(self) -> bool:
        return self.detector is not None and self.detector.available

    def update(self, ctx: FrameContext):
        events: list[ViolationEvent] = []
        if not self.available:
            return events

        for t in ctx.tracks:
            if t.class_name not in VEHICLES:
                continue
            verdict = self._lacks_belt(ctx, t.xyxy)
            if verdict is None:
                continue
            tid = t.track_id
            if verdict:
                self._streak[tid] = self._streak.get(tid, 0) + 1
            else:
                self._streak[tid] = 0
                self._flagged.discard(tid)
                continue

            if self._streak[tid] >= self.sustain_frames:
                self._flagged.add(tid)
                if tid not in self._fired:
                    self._fired.add(tid)
                    events.append(ViolationEvent(
                        type=self.name,
                        track_id=tid,
                        class_name=t.class_name,
                        frame_idx=ctx.frame_idx,
                        timestamp=ctx.timestamp,
                        xyxy=t.xyxy,
                        detail="driver without seatbelt",
                    ))
        return events

    def _lacks_belt(self, ctx: FrameContext, xyxy) -> bool | None:
        x1, y1, x2, y2 = xyxy
        h = y2 - y1
        windshield = (x1, y1 + h * self.windshield_top, x2, y1 + h * self.windshield_bottom)
        crop = crop_box(ctx.frame, windshield)
        dets = self.detector.detect(crop)
        if not dets:
            return None
        if any(d.class_name.lower() in self.belt_names for d in dets):
            return False
        if any(d.class_name.lower() in self.nobelt_names for d in dets):
            return True
        return None

    def flagged_ids(self) -> set[int]:
        return set(self._flagged)
