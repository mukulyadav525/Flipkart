"""Phase 2 — stop-line and red-light violations.

Both hinge on the same event: a vehicle's ground-contact point crossing the
calibrated stop-line segment while the signal is RED. They differ in intent:

  StopLineEngine : ANY crossing of the stop line on red — the vehicle failed to
                   stop behind the line (covers creeping over it).
  RedLightEngine : a crossing on red while still moving fast — the vehicle ran
                   the light rather than nudging over it.

Signal state comes from a SignalProvider (scheduled cycle or ROI colour), so the
engines work whether or not a physical signal head is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..geometry import segments_intersect
from ..signal import RED, SignalProvider
from ..tracking_state import TrackTracker
from .base import FrameContext, ViolationEvent


@dataclass
class _SignalCrossingEngine:
    signal: SignalProvider
    name: str = "signal_crossing"
    require_speed: float = 0.0          # px/s; 0 = any crossing counts
    _fired: set[int] = field(default_factory=set)
    _flagged: set[int] = field(default_factory=set)

    def update(self, ctx: FrameContext):
        events: list[ViolationEvent] = []
        stop_line = ctx.config.stop_line
        if stop_line is None:
            return events
        state = self.signal.state_at(ctx.timestamp, ctx.frame)
        if state != RED:
            self._flagged.clear()
            return events

        for st in ctx.states:
            if not TrackTracker.is_vehicle(st) or st.track_id in self._fired:
                continue
            seg = st.recent_segment()
            if seg is None:
                continue
            if not segments_intersect(seg[0], seg[1], stop_line.p1, stop_line.p2):
                continue
            _, speed = st.velocity(window_s=0.5)
            if speed < self.require_speed:
                continue

            self._fired.add(st.track_id)
            self._flagged.add(st.track_id)
            events.append(ViolationEvent(
                type=self.name,
                track_id=st.track_id,
                class_name=st.class_name,
                frame_idx=ctx.frame_idx,
                timestamp=ctx.timestamp,
                xyxy=st.last_xyxy,
                detail=f"crossed stop line on RED (speed {speed:.0f}px/s)",
                extra={"signal": state},
            ))
        return events

    def flagged_ids(self) -> set[int]:
        return set(self._flagged)


@dataclass
class StopLineEngine(_SignalCrossingEngine):
    name: str = "stop_line"
    require_speed: float = 0.0          # any crossing on red


@dataclass
class RedLightEngine(_SignalCrossingEngine):
    name: str = "red_light"
    require_speed: float = 40.0         # only vehicles actually running the light
