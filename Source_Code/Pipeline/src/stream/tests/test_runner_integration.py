"""
Integration test for StreamProcessor.process_frame.

Mocks only the YOLO detector (and OCR); everything else is the real pipeline —
preprocessing, tracker, violation rules, evidence packaging and JSONL writing —
exercised with real numpy + cv2.  Skipped automatically when cv2 is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from shared.schemas import BBox, DetectionRecord, Point2D, VehicleClass
from stream.scene import CameraConfig, scene_from_dict
from stream.runner import StreamProcessor


def _bike_with_visible_head(image_id):
    kp = ([Point2D(150, 120)] * 5            # head keypoints visible
          + [Point2D(130, 180), Point2D(170, 180)]
          + [Point2D(0, 0)] * 4
          + [Point2D(130, 300), Point2D(170, 300)]
          + [Point2D(0, 0)] * 4)
    return DetectionRecord(image_id=image_id, bbox=BBox(100, 90, 200, 460),
                           class_label=VehicleClass.bike, track_confidence=0.92,
                           pose_keypoints=kp)


@pytest.fixture
def patched_detect(monkeypatch):
    import detection.detect as dd

    def fake_detect(image, image_id, **kw):
        return [_bike_with_visible_head(image_id)]

    monkeypatch.setattr(dd, "detect", fake_detect)
    # OCR off — keep it deterministic
    return fake_detect


def _make_proc(tmp_path: Path) -> StreamProcessor:
    cam = CameraConfig(name="cam", source="x", scene=scene_from_dict({}, image_id="cam"))
    return StreamProcessor(cam, tmp_path, run_anpr=False, reset=True)


def test_emits_helmet_evidence(tmp_path, patched_detect):
    proc = _make_proc(tmp_path)
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    res = proc.process_frame(frame, 0)
    assert res.n_new_violations == 1
    assert "helmet" in res.emitted[0]
    # confirmed.jsonl written (conf 0.92 ≥ 0.85 cutoff)
    lines = proc.confirmed_jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["violation_record"]["violation_type"] == "helmet"
    # annotated image produced
    assert len(list(proc.ann_dir.glob("*.jpg"))) == 1


def test_same_vehicle_not_double_charged(tmp_path, patched_detect):
    proc = _make_proc(tmp_path)
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    r0 = proc.process_frame(frame, 0)
    r1 = proc.process_frame(frame, 2)   # same bike, next processed frame
    assert r0.n_new_violations == 1
    assert r1.n_new_violations == 0     # de-duplicated by track
    assert len(proc.confirmed_jsonl.read_text().strip().splitlines()) == 1


def test_latency_logged(tmp_path, patched_detect):
    proc = _make_proc(tmp_path)
    frame = (np.random.rand(360, 640, 3) * 255).astype(np.uint8)
    proc.process_frame(frame, 0)
    lat = proc.latency_jsonl.read_text().strip().splitlines()
    assert len(lat) == 1
    assert "total_ms" in json.loads(lat[0])
