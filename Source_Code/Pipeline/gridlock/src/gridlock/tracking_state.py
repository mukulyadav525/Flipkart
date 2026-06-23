"""Per-track temporal history.

Violation engines need motion over time (is it stationary? which way is it
heading?). This keeps a short rolling history of each track's ground-contact
point (bottom-center of the box) with timestamps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .geometry import Point, unit
from .types import Track


@dataclass
class TrackState:
    track_id: int
    class_name: str
    # (timestamp_s, x, y) of the ground-contact point, newest last.
    history: deque = field(default_factory=lambda: deque(maxlen=90))
    last_xyxy: tuple = (0, 0, 0, 0)

    @property
    def position(self) -> Point:
        _, x, y = self.history[-1]
        return (x, y)

    def recent_segment(self) -> tuple[Point, Point] | None:
        """(previous_pos, current_pos) from the last two samples, or None."""
        if len(self.history) < 2:
            return None
        _, px, py = self.history[-2]
        _, cx, cy = self.history[-1]
        return ((px, py), (cx, cy))

    def velocity(self, window_s: float = 1.0) -> tuple[Point, float]:
        """Return (unit_direction, speed_px_per_s) over the last `window_s`.

        Uses the oldest sample within the window vs the newest. Speed is on the
        net displacement, so a vehicle jittering in place reads ~0."""
        if len(self.history) < 2:
            return ((0.0, 0.0), 0.0)
        t_new, x_new, y_new = self.history[-1]
        t_old, x_old, y_old = self.history[0]
        for t, x, y in self.history:
            if t_new - t <= window_s:
                t_old, x_old, y_old = t, x, y
                break
        dt = t_new - t_old
        if dt <= 1e-6:
            return ((0.0, 0.0), 0.0)
        dx, dy = x_new - x_old, y_new - y_old
        speed = (dx * dx + dy * dy) ** 0.5 / dt
        return (unit((dx, dy)), speed)


class TrackTracker:
    """Maintains TrackState for every active track id."""

    VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}

    def __init__(self):
        self.states: dict[int, TrackState] = {}

    def update(self, tracks: list[Track], timestamp: float) -> list[TrackState]:
        active: list[TrackState] = []
        for t in tracks:
            st = self.states.get(t.track_id)
            if st is None:
                st = TrackState(track_id=t.track_id, class_name=t.class_name)
                self.states[t.track_id] = st
            bx, by = t.bottom_center
            st.history.append((timestamp, bx, by))
            st.last_xyxy = t.xyxy
            st.class_name = t.class_name
            active.append(st)
        return active

    @classmethod
    def is_vehicle(cls, state: TrackState) -> bool:
        return state.class_name in cls.VEHICLE_CLASSES
