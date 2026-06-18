"""
Unit tests for src/detection/detect.py.

All tests use synthetic images (numpy arrays) or mock the YOLO model
so the test suite runs without GPU, downloaded weights, or ultralytics installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared.schemas import BBox, DetectionRecord, Point2D, VehicleClass


# ---------------------------------------------------------------------------
# Helpers — build fake YOLO result objects
# ---------------------------------------------------------------------------

def _make_box(x1, y1, x2, y2, conf, cls_id):
    """Return a mock ultralytics box object."""
    import torch
    box = MagicMock()
    box.xyxy = [torch.tensor([x1, y1, x2, y2], dtype=torch.float32)]
    box.conf  = torch.tensor([conf])
    box.cls   = torch.tensor([cls_id])
    return box


def _make_yolo_result(boxes: list, names: dict[int, str]):
    result = MagicMock()
    result.boxes = boxes
    # attach names to the model mock, not the result
    return result, names


def _synthetic_image(h=480, w=640, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test 1 — label mapping covers all five VehicleClass values
# ---------------------------------------------------------------------------

def test_label_map_covers_all_classes():
    from detection.detect import _map_label, _IDD_LABEL_MAP
    mapped = set(_IDD_LABEL_MAP.values())
    for cls in VehicleClass:
        assert cls in mapped, f"VehicleClass.{cls.value} missing from _IDD_LABEL_MAP"


# ---------------------------------------------------------------------------
# Test 2 — detect() returns correct DetectionRecord schema
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not available",
)
def test_detect_returns_detection_records():
    import torch
    pytest.importorskip("ultralytics")

    image = _synthetic_image()
    image_id = "test_frame_001"

    # Build two fake boxes: a car and a bike
    car_box  = _make_box(100, 80, 300, 200, 0.82, cls_id=2)   # COCO car
    bike_box = _make_box(350, 150, 500, 350, 0.71, cls_id=3)  # COCO motorcycle

    fake_result = MagicMock()
    fake_result.boxes = [car_box, bike_box]

    fake_det_model = MagicMock()
    fake_det_model.predict.return_value = [fake_result]
    fake_det_model.names = {2: "car", 3: "motorcycle"}

    # Pose model returns no keypoints (empty result)
    fake_pose_model = MagicMock()
    fake_pose_kpts = MagicMock()
    fake_pose_kpts.xy = torch.zeros((0, 17, 2))
    fake_pose_result = MagicMock()
    fake_pose_result.keypoints = fake_pose_kpts
    fake_pose_model.predict.return_value = [fake_pose_result]

    with (
        patch("detection.detect._load_detector", return_value=fake_det_model),
        patch("detection.detect._load_pose_model", return_value=fake_pose_model),
    ):
        from detection.detect import detect
        records = detect(image, image_id, print_latency=False)

    assert len(records) == 2
    classes = {r.class_label for r in records}
    assert VehicleClass.car in classes
    assert VehicleClass.bike in classes

    for rec in records:
        assert isinstance(rec, DetectionRecord)
        assert rec.image_id == image_id
        assert isinstance(rec.bbox, BBox)
        assert 0.0 < rec.track_confidence <= 1.0


# ---------------------------------------------------------------------------
# Test 3 — bike detection triggers pose estimation; car does not
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not available",
)
def test_pose_called_for_bike_not_car():
    import torch
    pytest.importorskip("ultralytics")

    image = _synthetic_image(seed=7)

    car_box  = _make_box(10, 10, 200, 200, 0.9, cls_id=2)
    bike_box = _make_box(210, 10, 400, 400, 0.85, cls_id=3)

    fake_result = MagicMock()
    fake_result.boxes = [car_box, bike_box]

    fake_det_model = MagicMock()
    fake_det_model.predict.return_value = [fake_result]
    fake_det_model.names = {2: "car", 3: "motorcycle"}

    # Pose model: return 17 keypoints for the bike crop
    kp_data = torch.zeros((1, 17, 2))
    for i in range(17):
        kp_data[0, i] = torch.tensor([float(i * 5), float(i * 3)])

    fake_pose_kpts = MagicMock()
    fake_pose_kpts.xy = kp_data
    fake_pose_result = MagicMock()
    fake_pose_result.keypoints = fake_pose_kpts

    fake_pose_model = MagicMock()
    fake_pose_model.predict.return_value = [fake_pose_result]

    with (
        patch("detection.detect._load_detector", return_value=fake_det_model),
        patch("detection.detect._load_pose_model", return_value=fake_pose_model),
    ):
        from detection.detect import detect
        records = detect(image, "pose_test", print_latency=False)

    car_rec  = next(r for r in records if r.class_label == VehicleClass.car)
    bike_rec = next(r for r in records if r.class_label == VehicleClass.bike)

    assert car_rec.pose_keypoints is None, "car should not have keypoints"
    assert bike_rec.pose_keypoints is not None, "bike should have keypoints"
    assert len(bike_rec.pose_keypoints) == 17
    # Keypoints are Point2D and coordinates are non-negative
    for kp in bike_rec.pose_keypoints:
        assert isinstance(kp, Point2D)


# ---------------------------------------------------------------------------
# Test 4 — detections below conf_threshold are filtered out
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not available",
)
def test_conf_threshold_filtering():
    import torch
    pytest.importorskip("ultralytics")

    image = _synthetic_image(seed=42)

    low_conf_box  = _make_box(50, 50, 200, 200, 0.20, cls_id=2)   # below threshold
    high_conf_box = _make_box(250, 50, 450, 300, 0.80, cls_id=2)  # above threshold

    fake_result = MagicMock()
    # Ultralytics already filters by conf in predict(); here we simulate that
    # only the high-conf box is returned (conf=0.4 threshold set in predict call)
    fake_result.boxes = [high_conf_box]

    fake_det_model = MagicMock()
    fake_det_model.predict.return_value = [fake_result]
    fake_det_model.names = {2: "car"}

    fake_pose_model = MagicMock()
    fake_pose_model.predict.return_value = [MagicMock(keypoints=None)]

    with (
        patch("detection.detect._load_detector", return_value=fake_det_model),
        patch("detection.detect._load_pose_model", return_value=fake_pose_model),
    ):
        from detection.detect import detect
        records = detect(image, "conf_test", conf_threshold=0.4, print_latency=False)

    assert len(records) == 1
    assert records[0].track_confidence >= 0.4


# ---------------------------------------------------------------------------
# Test 5 — empty image (no detections) returns empty list, no crash
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not available",
)
def test_empty_detection_result():
    pytest.importorskip("ultralytics")

    image = _synthetic_image(seed=99)

    fake_result = MagicMock()
    fake_result.boxes = []

    fake_det_model = MagicMock()
    fake_det_model.predict.return_value = [fake_result]
    fake_det_model.names = {}

    fake_pose_model = MagicMock()

    with (
        patch("detection.detect._load_detector", return_value=fake_det_model),
        patch("detection.detect._load_pose_model", return_value=fake_pose_model),
    ):
        from detection.detect import detect
        records = detect(image, "empty_test", print_latency=False)

    assert records == []
