"""Helmet non-compliance.

The helmet model already localizes rider heads and classifies them as helmeted
or not, so we run it on the full frame and treat each "Without Helmet" / bare-head
detection as a candidate. To count only *riders* (helmet law applies to
two-wheelers, not pedestrians), we keep a bare head only when it sits on/above a
tracked motorcycle, and flag that motorcycle. Tracking the bike gives free dedup.

Needs a helmet model (see SecondaryDetector). Class-name matching is configurable;
this dataset uses {"With Helmet","Without Helmet","helmet"} which maps out of the
box. Without weights the engine disables itself cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..secondary import SecondaryDetector
from .base import FrameContext, ViolationEvent


@dataclass
class HelmetEngine:
    detector: SecondaryDetector | None = None
    sustain_frames: int = 2
    x_pad: float = 0.3          # widen the bike column when matching a head
    up_factor: float = 1.2      # how far above the bike a rider head may sit (× bike height)
    name: str = "no_helmet"
    nohelmet_names: tuple[str, ...] = (
        "no_helmet", "without helmet", "no-helmet", "no helmet", "head", "bare")
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

        dets = self.detector.detect(ctx.frame)  # absolute coords (full frame)
        bare = [d for d in dets if d.class_name.lower() in self.nohelmet_names]
        bikes = [t for t in ctx.tracks if t.class_name == "motorcycle"]

        for bike in bikes:
            bid = bike.track_id
            head = next((d for d in bare if self._head_on_bike(d.xyxy, bike.xyxy)), None)
            if head is None:
                self._streak[bid] = 0
                self._flagged.discard(bid)
                continue

            self._streak[bid] = self._streak.get(bid, 0) + 1
            if self._streak[bid] >= self.sustain_frames:
                self._flagged.add(bid)
                if bid not in self._fired:
                    self._fired.add(bid)
                    events.append(ViolationEvent(
                        type=self.name,
                        track_id=bid,
                        class_name="motorcycle",
                        frame_idx=ctx.frame_idx,
                        timestamp=ctx.timestamp,
                        xyxy=bike.xyxy,
                        detail="motorcycle rider without helmet",
                        extra={"head_conf": round(head.conf, 2)},
                    ))
        return events

    def _head_on_bike(self, head_xyxy, bike_xyxy) -> bool:
        hx = (head_xyxy[0] + head_xyxy[2]) / 2.0
        hy = (head_xyxy[1] + head_xyxy[3]) / 2.0
        bx1, by1, bx2, by2 = bike_xyxy
        pad = (bx2 - bx1) * self.x_pad
        if not (bx1 - pad <= hx <= bx2 + pad):
            return False
        # head should be within the bike's vertical span or just above it
        return (by1 - (by2 - by1) * self.up_factor) <= hy <= by2

    def flagged_ids(self) -> set[int]:
        return set(self._flagged)
