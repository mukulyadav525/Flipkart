"""Triple riding — 3+ people on a single two-wheeler.

Uses the base detector only (person + motorcycle), so it runs today with no
extra model. We count riders associated with each motorcycle track and require
the count to hold for a few frames to shrug off momentary mis-associations in
crowded scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..association import riders_per_bike
from .base import FrameContext, ViolationEvent


@dataclass
class TripleRidingEngine:
    min_riders: int = 3
    sustain_frames: int = 4
    name: str = "triple_riding"
    _streak: dict[int, int] = field(default_factory=dict)
    _flagged: set[int] = field(default_factory=set)
    _fired: set[int] = field(default_factory=set)

    def update(self, ctx: FrameContext):
        events: list[ViolationEvent] = []
        assoc = riders_per_bike(ctx.tracks)
        # Only motorcycles count for triple riding (bicycles excluded).
        bikes = {t.track_id: t for t in ctx.tracks if t.class_name == "motorcycle"}

        for bike_id, bike in bikes.items():
            n = len(assoc.get(bike_id, []))
            if n >= self.min_riders:
                self._streak[bike_id] = self._streak.get(bike_id, 0) + 1
            else:
                self._streak[bike_id] = 0
                self._flagged.discard(bike_id)
                continue

            if self._streak[bike_id] >= self.sustain_frames:
                self._flagged.add(bike_id)
                if bike_id not in self._fired:
                    self._fired.add(bike_id)
                    events.append(ViolationEvent(
                        type=self.name,
                        track_id=bike_id,
                        class_name="motorcycle",
                        frame_idx=ctx.frame_idx,
                        timestamp=ctx.timestamp,
                        xyxy=bike.xyxy,
                        detail=f"{n} riders on one two-wheeler",
                        extra={"rider_count": n},
                    ))
        return events

    def flagged_ids(self) -> set[int]:
        return set(self._flagged)
