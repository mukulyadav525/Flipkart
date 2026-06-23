"""
Lightweight multi-object tracker (IoU / greedy association).

Why this exists
---------------
Streaming a real camera means the same vehicle appears in hundreds of frames.
Without identity we would:
  * issue the same challan hundreds of times, and
  * have no way to measure motion (needed for the wrong-side rule).

This tracker assigns a stable integer id to each detection across frames using
greedy IoU matching — no extra dependencies (no lap / scipy / ByteTrack), so it
runs anywhere cv2 + numpy run and is unit-testable in pure Python.

Each Track carries the per-vehicle state the stream pipeline needs:
  * centroid history          → motion vector for wrong-side detection
  * best plate reading so far  → one good ANPR result beats 200 blurry ones
  * the set of violation types already emitted → de-duplication
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from shared.schemas import BBox, DetectionRecord, Point2D, PlateRecord, VehicleClass, ViolationType


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((a.x2 - a.x1) * (a.y2 - a.y1), 1e-6)
    area_b = max((b.x2 - b.x1) * (b.y2 - b.y1), 1e-6)
    return inter / (area_a + area_b - inter)


def centroid(b: BBox) -> Point2D:
    return Point2D(x=(b.x1 + b.x2) / 2.0, y=(b.y1 + b.y2) / 2.0)


def detection_id(image_id: str, det: DetectionRecord) -> str:
    """Mirror violations.rules._detection_id so motion keys line up exactly."""
    return f"{image_id}:{det.class_label.value}:{det.bbox.x1:.0f},{det.bbox.y1:.0f}"


@dataclass
class Track:
    track_id: int
    class_label: VehicleClass
    bbox: BBox
    last_frame: int
    hits: int = 1
    history: list[tuple[int, Point2D]] = field(default_factory=list)   # (frame, centroid)
    best_plate: Optional[PlateRecord] = None
    fired: set[ViolationType] = field(default_factory=set)

    def motion(self, lookback: int = 8) -> Optional[Point2D]:
        """Displacement of the centroid over the last `lookback` frames."""
        if len(self.history) < 2:
            return None
        _, now = self.history[-1]
        # earliest sample within the lookback window
        ref = self.history[max(0, len(self.history) - 1 - lookback)][1]
        return Point2D(x=now.x - ref.x, y=now.y - ref.y)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30,
                 history_len: int = 32):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.history_len = history_len
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(
        self, detections: list[DetectionRecord], frame_idx: int
    ) -> list[tuple[Track, DetectionRecord]]:
        """
        Associate detections to existing tracks, age out stale ones, and return
        the (track, detection) pairs observed in THIS frame.
        """
        pairs: list[tuple[Track, DetectionRecord]] = []
        unmatched = list(range(len(detections)))

        # Greedy matching: highest IoU first, one detection per track.
        candidates: list[tuple[float, int, int]] = []
        track_ids = list(self.tracks.keys())
        for ti, tid in enumerate(track_ids):
            t = self.tracks[tid]
            for di in unmatched:
                d = detections[di]
                if d.class_label != t.class_label:
                    continue
                score = iou(t.bbox, d.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, tid, di))
        candidates.sort(reverse=True)

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for score, tid, di in candidates:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            t = self.tracks[tid]
            d = detections[di]
            t.bbox = d.bbox
            t.last_frame = frame_idx
            t.hits += 1
            t.history.append((frame_idx, centroid(d.bbox)))
            t.history = t.history[-self.history_len:]
            pairs.append((t, d))

        # New tracks for unmatched detections.
        for di, d in enumerate(detections):
            if di in used_dets:
                continue
            t = Track(
                track_id=self._next_id,
                class_label=d.class_label,
                bbox=d.bbox,
                last_frame=frame_idx,
                history=[(frame_idx, centroid(d.bbox))],
            )
            self._next_id += 1
            self.tracks[t.track_id] = t
            pairs.append((t, d))

        # Age out tracks unseen for too long.
        stale = [tid for tid, t in self.tracks.items()
                 if frame_idx - t.last_frame > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        return pairs
