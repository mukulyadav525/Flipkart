"""Unit tests for the stream scene loader and de-duplication logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.schemas import (
    BBox, DetectionRecord, PlateRecord, SignalState, VehicleClass,
    ViolationRecord, ViolationType,
)
from stream.scene import scene_from_dict, load_camera_config, CameraConfig
from stream.runner import (
    filter_new_violations, assign_plates_to_tracks, plate_for_emission,
)
from tracking.tracker import IoUTracker, detection_id


# ---------------------------------------------------------------------------
# Scene loader
# ---------------------------------------------------------------------------

class TestSceneLoader:
    def test_parses_full_geometry(self):
        d = {
            "name": "cam1",
            "signal_state": "red",
            "lane_direction_vector": {"x": 0.0, "y": -1.0},
            "stop_line_coords": [{"x": 0, "y": 600}, {"x": 1280, "y": 600}],
            "no_parking_zone_polygon": [
                {"x": 0, "y": 400}, {"x": 200, "y": 400}, {"x": 200, "y": 700}, {"x": 0, "y": 700},
            ],
            "no_parking_sign_visible": True,
        }
        s = scene_from_dict(d)
        assert s.signal_state == SignalState.red
        assert s.lane_direction_vector.y == -1.0
        assert len(s.stop_line_coords) == 2
        assert len(s.no_parking_zone_polygon) == 4
        assert s.no_parking_sign_visible is True

    def test_empty_dict_yields_blank_scene(self):
        s = scene_from_dict({}, image_id="x")
        assert s.image_id == "x"
        assert s.lane_direction_vector is None
        assert s.stop_line_coords is None
        assert s.no_parking_sign_visible is False

    def test_load_camera_config_roundtrip(self, tmp_path: Path):
        cfg = {"name": "demo", "source": "clip.mp4", "stride": 3,
               "signal_state": "green",
               "stop_line_coords": [{"x": 0, "y": 100}, {"x": 100, "y": 100}]}
        p = tmp_path / "demo.json"
        p.write_text(json.dumps(cfg))
        cam = load_camera_config(p)
        assert isinstance(cam, CameraConfig)
        assert cam.source == "clip.mp4"
        assert cam.stride == 3
        assert cam.has_geometry is True
        assert cam.scene.signal_state == SignalState.green

    def test_missing_source_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"name": "x"}))
        with pytest.raises(ValueError):
            load_camera_config(p)


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------

def _violation(primary_id: str, vtype=ViolationType.helmet) -> ViolationRecord:
    return ViolationRecord(image_id="cam_f0", violation_type=vtype, confidence=0.9,
                           rule_trace="t", related_detection_ids=[primary_id])


class TestDedup:
    def _track(self, image_id="cam_f0"):
        tr = IoUTracker(iou_threshold=0.1)
        det = DetectionRecord(image_id=image_id, bbox=BBox(100, 100, 200, 460),
                              class_label=VehicleClass.bike, track_confidence=0.9)
        pairs = tr.update([det], 0)
        return tr, pairs, det

    def test_first_violation_is_emitted(self):
        _, pairs, det = self._track()
        d2t = {detection_id("cam_f0", det): pairs[0][0]}
        pid = detection_id("cam_f0", det)
        out = filter_new_violations([_violation(pid)], d2t)
        assert len(out) == 1
        assert out[0][1].fired == {ViolationType.helmet}

    def test_same_violation_same_track_not_re_emitted(self):
        _, pairs, det = self._track()
        track = pairs[0][0]
        d2t = {detection_id("cam_f0", det): track}
        pid = detection_id("cam_f0", det)
        filter_new_violations([_violation(pid)], d2t)          # frame 1
        out = filter_new_violations([_violation(pid)], d2t)    # frame 2 — duplicate
        assert out == []                                       # suppressed

    def test_different_violation_type_same_track_emitted(self):
        _, pairs, det = self._track()
        track = pairs[0][0]
        d2t = {detection_id("cam_f0", det): track}
        pid = detection_id("cam_f0", det)
        filter_new_violations([_violation(pid, ViolationType.helmet)], d2t)
        out = filter_new_violations([_violation(pid, ViolationType.triple_riding)], d2t)
        assert len(out) == 1
        assert track.fired == {ViolationType.helmet, ViolationType.triple_riding}


# ---------------------------------------------------------------------------
# Plate aggregation
# ---------------------------------------------------------------------------

class TestPlateAggregation:
    def _plate(self, bbox, text, conf):
        return PlateRecord(image_id="cam", vehicle_bbox=bbox,
                           plate_bbox=BBox(bbox.x1, bbox.y2 - 20, bbox.x1 + 60, bbox.y2),
                           plate_text=text, ocr_confidence=conf)

    def test_best_plate_kept(self):
        tr = IoUTracker(iou_threshold=0.1)
        det = DetectionRecord(image_id="cam_f0", bbox=BBox(100, 100, 300, 300),
                              class_label=VehicleClass.car, track_confidence=0.9)
        pairs = tr.update([det], 0)
        assign_plates_to_tracks([self._plate(BBox(100, 100, 300, 300), "MH12AB1234", 0.6)], pairs)
        assign_plates_to_tracks([self._plate(BBox(100, 100, 300, 300), "MH12AB1234", 0.9)], pairs)
        assign_plates_to_tracks([self._plate(BBox(100, 100, 300, 300), "MH12AB0000", 0.4)], pairs)
        assert pairs[0][0].best_plate.ocr_confidence == 0.9

    def test_plate_for_emission_restamps_current_bbox(self):
        tr = IoUTracker(iou_threshold=0.1)
        det = DetectionRecord(image_id="cam_f0", bbox=BBox(100, 100, 300, 300),
                              class_label=VehicleClass.car, track_confidence=0.9)
        pairs = tr.update([det], 0)
        assign_plates_to_tracks([self._plate(BBox(100, 100, 300, 300), "MH12AB1234", 0.8)], pairs)
        current = BBox(150, 120, 350, 320)
        p = plate_for_emission(pairs[0][0], current)
        assert p.vehicle_bbox == current
        assert p.plate_text == "MH12AB1234"

    def test_plate_for_emission_none_when_no_plate(self):
        tr = IoUTracker(iou_threshold=0.1)
        det = DetectionRecord(image_id="cam_f0", bbox=BBox(0, 0, 10, 10),
                              class_label=VehicleClass.car, track_confidence=0.9)
        pairs = tr.update([det], 0)
        assert plate_for_emission(pairs[0][0], BBox(0, 0, 10, 10)) is None
