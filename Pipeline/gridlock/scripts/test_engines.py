#!/usr/bin/env python3
"""Deterministic sanity test for the Phase 1 violation engines.

Feeds synthetic tracks (no YOLO) so the geometry/timing logic is verifiable in
isolation: parking must fire only after the dwell threshold; wrong-side must
fire only for sustained motion against the lane direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridlock.scene import CameraConfig, Lane, NoParkingZone  # noqa: E402
from gridlock.tracking_state import TrackTracker  # noqa: E402
from gridlock.types import Track  # noqa: E402
from gridlock.violations import (  # noqa: E402
    FrameContext, IllegalParkingEngine, RedLightEngine, StopLineEngine,
    TripleRidingEngine, WrongSideEngine,
)
from gridlock.scene import StopLine  # noqa: E402
from gridlock.signal import ScheduledSignal  # noqa: E402
import numpy as np  # noqa: E402

FPS = 30.0
DUMMY_FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def box_at(x, y, w=40, h=30):
    return (x - w / 2, y - h, x + w / 2, y)  # bottom-center at (x, y)


def feed(engine, frames, config):
    """Replay frames (list of track lists) through tracker+engine; return events.

    Each frame is either {id: (x,y)} (defaults to 'car') or a list of Tracks."""
    tracker = TrackTracker()
    events = []
    for i, item in enumerate(frames):
        ts = i / FPS
        if isinstance(item, dict):
            tracks = [
                Track(track_id=tid, class_id=2, class_name="car", conf=0.9, xyxy=box_at(x, y))
                for tid, (x, y) in item.items()
            ]
        else:
            tracks = item
        states = tracker.update(tracks, ts)
        ctx = FrameContext(states=states, tracks=tracks, frame=DUMMY_FRAME,
                           frame_idx=i, timestamp=ts, config=config)
        events += engine.update(ctx)
    return events


def test_parking():
    cfg = CameraConfig(name="t", no_parking=[
        NoParkingZone(name="zoneA", polygon=[(100, 100), (300, 100), (300, 300), (100, 300)])
    ])
    eng = IllegalParkingEngine(min_seconds=3.0, stationary_speed=25.0)
    # car #1 parks at (200,200) for 5s; car #2 drives straight through the zone.
    frames = []
    for i in range(int(5 * FPS)):
        frames.append({1: (200, 200), 2: (50 + i * 6, 200)})
    events = feed(eng, frames, cfg)
    park = [e for e in events if e.track_id == 1]
    drive_through = [e for e in events if e.track_id == 2]
    assert len(park) == 1, f"expected 1 parking event, got {len(park)}"
    assert park[0].timestamp >= 3.0, f"fired too early at {park[0].timestamp:.2f}s"
    assert not drive_through, "moving car should NOT be flagged for parking"
    print(f"  parking: OK (fired at {park[0].timestamp:.1f}s, '{park[0].detail}')")


def test_wrong_side():
    # Lane allows travel to the RIGHT (+x).
    cfg = CameraConfig(name="t", lanes=[
        Lane(name="laneA", polygon=[(0, 100), (640, 100), (640, 300), (0, 300)], direction=(1.0, 0.0))
    ])
    eng = WrongSideEngine(min_speed=30.0, sustain_frames=5)
    # car #1 goes right (correct), car #2 goes left (wrong way).
    frames = []
    for i in range(int(3 * FPS)):
        frames.append({1: (100 + i * 5, 200), 2: (600 - i * 5, 200)})
    events = feed(eng, frames, cfg)
    correct = [e for e in events if e.track_id == 1]
    wrong = [e for e in events if e.track_id == 2]
    assert not correct, "correct-direction car should NOT be flagged"
    assert len(wrong) == 1, f"expected 1 wrong-side event, got {len(wrong)}"
    print(f"  wrong_side: OK (fired at {wrong[0].timestamp:.1f}s for track 2)")


def test_triple_riding():
    cfg = CameraConfig(name="t")
    eng = TripleRidingEngine(min_riders=3, sustain_frames=4)

    def bike(tid, x, y, w=80, h=60):
        return Track(tid, 3, "motorcycle", 0.9, (x - w/2, y - h, x + w/2, y))

    def person(tid, x, y, w=30, h=70):
        return Track(tid, 0, "person", 0.9, (x - w/2, y - h, x + w/2, y))

    frames = []
    for _ in range(10):
        # bike #1 with 3 overlapping riders -> triple. bike #2 with 1 rider -> ok.
        frames.append([
            bike(1, 300, 400), person(11, 290, 380), person(12, 305, 380), person(13, 315, 380),
            bike(2, 800, 400), person(21, 800, 380),
        ])
    events = feed(eng, frames, cfg)
    triple = [e for e in events if e.track_id == 1]
    single = [e for e in events if e.track_id == 2]
    assert len(triple) == 1, f"expected 1 triple-riding event, got {len(triple)}"
    assert triple[0].extra["rider_count"] >= 3
    assert not single, "two-rider bike should NOT be flagged"
    print(f"  triple_riding: OK (bike 1 flagged, {triple[0].extra['rider_count']} riders)")


def test_stop_line_red_light():
    # Vertical stop line at x=100; car drives right through it, fast.
    cfg = CameraConfig(name="t", stop_line=StopLine((100.0, 0.0), (100.0, 200.0)))
    frames = [{1: (60 + i * 8, 100)} for i in range(12)]  # ~240 px/s

    # On RED: both stop_line and red_light should fire once.
    red = ScheduledSignal([("red", 100)])
    ev_stop = feed(StopLineEngine(signal=red), frames, cfg)
    ev_red = feed(RedLightEngine(signal=red), frames, cfg)
    assert len(ev_stop) == 1, f"expected 1 stop_line, got {len(ev_stop)}"
    assert len(ev_red) == 1, f"expected 1 red_light, got {len(ev_red)}"

    # On GREEN: nothing fires (legal crossing).
    green = ScheduledSignal([("green", 100)])
    assert not feed(StopLineEngine(signal=green), frames, cfg)
    assert not feed(RedLightEngine(signal=green), frames, cfg)
    print("  stop_line/red_light: OK (fire on RED, silent on GREEN)")


if __name__ == "__main__":
    print("running engine logic tests...")
    test_parking()
    test_wrong_side()
    test_triple_riding()
    test_stop_line_red_light()
    print("ALL PASSED")
