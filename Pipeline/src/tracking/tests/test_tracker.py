"""Unit tests for the IoU tracker — pure Python, no cv2 / no model."""

from __future__ import annotations

from shared.schemas import BBox, DetectionRecord, VehicleClass
from tracking.tracker import IoUTracker, detection_id, iou


def _det(x1, y1, x2, y2, cls=VehicleClass.car, conf=0.9, image_id="cam_f0"):
    return DetectionRecord(image_id=image_id, bbox=BBox(x1, y1, x2, y2),
                           class_label=cls, track_confidence=conf)


class TestIoU:
    def test_identical_boxes(self):
        b = BBox(0, 0, 10, 10)
        assert iou(b, b) == 1.0

    def test_disjoint_boxes(self):
        assert iou(BBox(0, 0, 10, 10), BBox(50, 50, 60, 60)) == 0.0


class TestTracking:
    def test_same_vehicle_keeps_id_across_frames(self):
        tr = IoUTracker(iou_threshold=0.3)
        # frame 0
        pairs0 = tr.update([_det(100, 100, 200, 200, image_id="cam_f0")], 0)
        id0 = pairs0[0][0].track_id
        # frame 1 — vehicle moved a little (high overlap)
        pairs1 = tr.update([_det(108, 92, 208, 192, image_id="cam_f1")], 1)
        id1 = pairs1[0][0].track_id
        assert id0 == id1
        assert len(tr.tracks) == 1

    def test_new_vehicle_gets_new_id(self):
        tr = IoUTracker(iou_threshold=0.3)
        tr.update([_det(100, 100, 200, 200)], 0)
        pairs = tr.update([
            _det(108, 92, 208, 192),       # same car
            _det(600, 600, 700, 700),      # new car, far away
        ], 1)
        ids = {t.track_id for t, _ in pairs}
        assert len(ids) == 2
        assert len(tr.tracks) == 2

    def test_different_class_not_matched(self):
        tr = IoUTracker(iou_threshold=0.3)
        tr.update([_det(100, 100, 200, 200, cls=VehicleClass.car)], 0)
        pairs = tr.update([_det(100, 100, 200, 200, cls=VehicleClass.bike)], 1)
        # overlapping but different class → new track
        assert len(tr.tracks) == 2
        assert pairs[0][0].class_label == VehicleClass.bike

    def test_stale_track_is_aged_out(self):
        tr = IoUTracker(iou_threshold=0.3, max_age=5)
        tr.update([_det(100, 100, 200, 200)], 0)
        tr.update([], 10)   # 10 frames later, nothing seen
        assert len(tr.tracks) == 0

    def test_motion_vector_points_in_travel_direction(self):
        tr = IoUTracker(iou_threshold=0.1)
        t = None
        for i in range(6):
            # moving up the frame (decreasing y)
            y = 400 - i * 20
            pairs = tr.update([_det(100, y, 200, y + 100, image_id=f"cam_f{i}")], i)
            t = pairs[0][0]
        mv = t.motion()
        assert mv is not None
        assert mv.y < 0          # moved up
        assert abs(mv.x) < 1e-6  # no horizontal drift

    def test_detection_id_matches_rules_format(self):
        d = _det(120.4, 95.6, 400, 450, cls=VehicleClass.bike, image_id="cam_f3")
        # must equal violations.rules._detection_id format
        assert detection_id("cam_f3", d) == "cam_f3:bike:120,96"
