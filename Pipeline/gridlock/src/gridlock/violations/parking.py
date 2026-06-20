"""Illegal parking.

A vehicle is flagged when its ground-contact point stays inside a no-parking
polygon AND barely moves for longer than `min_seconds`. Tracking gives us the
persistent id; we accumulate dwell time and reset if it leaves or drives off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..geometry import point_in_polygon
from ..scene import CameraConfig
from ..tracking_state import TrackState, TrackTracker
from .base import FrameContext, ViolationEvent


@dataclass
class _Dwell:
    zone: str | None = None
    seconds: float = 0.0
    last_t: float | None = None
    fired: bool = False


@dataclass
class IllegalParkingEngine:
    # Demo-friendly default. Real deployments use ~minutes.
    min_seconds: float = 8.0
    # Max speed (px/s) to still count as "stopped". Scale to your frame.
    stationary_speed: float = 25.0
    name: str = "illegal_parking"
    _dwell: dict[int, _Dwell] = field(default_factory=dict)
    _flagged: set[int] = field(default_factory=set)

    def update(self, ctx: FrameContext):
        states, frame_idx, timestamp, config = (
            ctx.states, ctx.frame_idx, ctx.timestamp, ctx.config)
        events: list[ViolationEvent] = []
        if not config.no_parking:
            return events

        for st in states:
            if not TrackTracker.is_vehicle(st):
                continue
            d = self._dwell.setdefault(st.track_id, _Dwell())
            zone = self._zone_for(st, config)
            _, speed = st.velocity(window_s=1.0)

            if zone is None or speed > self.stationary_speed:
                # Left the zone or started moving -> reset dwell.
                d.zone, d.seconds, d.last_t = None, 0.0, None
                self._flagged.discard(st.track_id)
                continue

            # Inside a no-parking zone and effectively stopped.
            if d.last_t is not None and d.zone == zone:
                d.seconds += max(0.0, timestamp - d.last_t)
            d.zone, d.last_t = zone, timestamp

            if d.seconds >= self.min_seconds:
                self._flagged.add(st.track_id)
                if not d.fired:
                    d.fired = True
                    events.append(ViolationEvent(
                        type=self.name,
                        track_id=st.track_id,
                        class_name=st.class_name,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        xyxy=st.last_xyxy,
                        zone=zone,
                        detail=f"stopped {d.seconds:.1f}s in no-parking zone",
                    ))
        return events

    def flagged_ids(self) -> set[int]:
        return set(self._flagged)

    @staticmethod
    def _zone_for(st: TrackState, config: CameraConfig) -> str | None:
        pt = st.position
        for z in config.no_parking:
            if point_in_polygon(pt, z.polygon):
                return z.name
        return None
