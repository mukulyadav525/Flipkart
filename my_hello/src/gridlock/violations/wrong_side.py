"""Wrong-side (wrong-way) driving.

For a vehicle inside a calibrated lane, compare its travel direction with the
lane's allowed direction. If it moves clearly *against* the lane (cos < -thresh)
while moving fast enough to be real motion, and sustains that for a few samples,
flag it. The sustain requirement kills single-frame tracker jitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..geometry import cos_angle, point_in_polygon
from ..scene import CameraConfig, Lane
from ..tracking_state import TrackState, TrackTracker
from .base import FrameContext, ViolationEvent


@dataclass
class WrongSideEngine:
    # cos < -0.5  => heading more than 120 deg away from allowed direction.
    oppose_cos: float = -0.5
    min_speed: float = 30.0          # px/s; ignore near-stationary objects
    sustain_frames: int = 5          # consecutive wrong-way samples required
    name: str = "wrong_side"
    _streak: dict[int, int] = field(default_factory=dict)
    _flagged: set[int] = field(default_factory=set)
    _fired: set[int] = field(default_factory=set)

    def update(self, ctx: FrameContext):
        states, frame_idx, timestamp, config = (
            ctx.states, ctx.frame_idx, ctx.timestamp, ctx.config)
        events: list[ViolationEvent] = []
        if not config.lanes:
            return events

        for st in states:
            if not TrackTracker.is_vehicle(st):
                continue
            lane = self._lane_for(st, config)
            direction, speed = st.velocity(window_s=0.8)
            if lane is None or speed < self.min_speed:
                self._streak[st.track_id] = 0
                self._flagged.discard(st.track_id)
                continue

            if cos_angle(direction, lane.direction) <= self.oppose_cos:
                self._streak[st.track_id] = self._streak.get(st.track_id, 0) + 1
            else:
                self._streak[st.track_id] = 0
                self._flagged.discard(st.track_id)
                continue

            if self._streak[st.track_id] >= self.sustain_frames:
                self._flagged.add(st.track_id)
                if st.track_id not in self._fired:
                    self._fired.add(st.track_id)
                    events.append(ViolationEvent(
                        type=self.name,
                        track_id=st.track_id,
                        class_name=st.class_name,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        xyxy=st.last_xyxy,
                        zone=lane.name,
                        detail="moving against allowed lane direction",
                    ))
        return events

    def flagged_ids(self) -> set[int]:
        return set(self._flagged)

    @staticmethod
    def _lane_for(st: TrackState, config: CameraConfig) -> Lane | None:
        pt = st.position
        for lane in config.lanes:
            if point_in_polygon(pt, lane.polygon):
                return lane
        return None
