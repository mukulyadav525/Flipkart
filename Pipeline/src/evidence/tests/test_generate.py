"""
Unit tests for src/evidence/generate.py.

No real images, no disk I/O.  cv2.imwrite and _append_jsonl are patched
so tests are hermetic.  Tests verify:
  - EvidenceRecord schema correctness
  - Confirmed vs review routing by confidence
  - Per-type threshold gate (below min_conf → not even queued for review)
  - Plate matching logic (IoU-based)
  - Annotated image array integrity (shape, dtype)
  - JSONL serialisation round-trip
  - iter_confirmed / iter_review_queue readers
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shared.schemas import (
    BBox,
    DetectionRecord,
    EvidenceRecord,
    PlateRecord,
    ViolationRecord,
    VehicleClass,
    ViolationType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _image(h=480, w=640) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _bike_det(x1=100, y1=100, x2=400, y2=450, conf=0.85) -> DetectionRecord:
    return DetectionRecord(
        image_id="frame_001",
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_label=VehicleClass.bike,
        track_confidence=conf,
    )


def _plate(vehicle_bbox: BBox, plate_text="MH12AB1234", conf=0.92) -> PlateRecord:
    return PlateRecord(
        image_id="frame_001",
        vehicle_bbox=vehicle_bbox,
        plate_bbox=BBox(x1=150, y1=420, x2=350, y2=450),
        plate_text=plate_text,
        ocr_confidence=conf,
    )


def _violation(vtype=ViolationType.helmet, conf=0.88,
               related_ids: list[str] | None = None) -> ViolationRecord:
    return ViolationRecord(
        image_id="frame_001",
        violation_type=vtype,
        confidence=conf,
        rule_trace="Bike detected. No helmet overlapping head region.",
        related_detection_ids=related_ids or [],
    )


def _det_id(det: DetectionRecord) -> str:
    return f"{det.image_id}:{det.class_label.value}:{det.bbox.x1:.0f},{det.bbox.y1:.0f}"


# ---------------------------------------------------------------------------
# Test 1 — EvidenceRecord schema: correct fields, correct types
# ---------------------------------------------------------------------------

def test_evidence_record_fields():
    from evidence.generate import generate_evidence

    img = _image()
    det = _bike_det()
    viol = _violation(conf=0.90, related_ids=[_det_id(det)])

    with (
        patch("cv2.imwrite", return_value=True),
        patch("evidence.generate._append_jsonl"),
    ):
        records = generate_evidence(
            img, "frame_001", [det], [], [viol],
            annotated_dir=Path("/tmp/ann"),
            confirmed_jsonl=Path("/tmp/confirmed.jsonl"),
            review_jsonl=Path("/tmp/review.jsonl"),
            timestamp="2025-01-01T00:00:00+00:00",
        )

    assert len(records) == 1
    r = records[0]
    assert isinstance(r, EvidenceRecord)
    assert r.violation_record is viol
    assert r.timestamp == "2025-01-01T00:00:00+00:00"
    assert r.plate_text == ""
    assert r.plate_confidence == 0.0
    assert "frame_001" in r.annotated_image_path
    assert "helmet" in r.annotated_image_path


# ---------------------------------------------------------------------------
# Test 2 — Routing: high confidence → confirmed JSONL
# ---------------------------------------------------------------------------

def test_high_confidence_routes_to_confirmed():
    from evidence.generate import generate_evidence, AUTO_PROCESS_CUTOFF

    img = _image()
    det = _bike_det()
    conf = AUTO_PROCESS_CUTOFF + 0.05   # above cutoff
    viol = _violation(conf=conf, related_ids=[_det_id(det)])

    confirmed_calls = []
    review_calls = []

    def fake_append(path, record):
        if "confirmed" in str(path):
            confirmed_calls.append(record)
        else:
            review_calls.append(record)

    with (
        patch("cv2.imwrite", return_value=True),
        patch("evidence.generate._append_jsonl", side_effect=fake_append),
    ):
        generate_evidence(
            img, "frame_001", [det], [], [viol],
            annotated_dir=Path("/tmp/ann"),
            confirmed_jsonl=Path("/tmp/confirmed.jsonl"),
            review_jsonl=Path("/tmp/review.jsonl"),
        )

    assert len(confirmed_calls) == 1
    assert len(review_calls) == 0


# ---------------------------------------------------------------------------
# Test 3 — Routing: low confidence → review JSONL
# ---------------------------------------------------------------------------

def test_low_confidence_routes_to_review():
    from evidence.generate import generate_evidence, AUTO_PROCESS_CUTOFF

    img = _image()
    det = _bike_det()
    conf = AUTO_PROCESS_CUTOFF - 0.10   # below cutoff but above per-type floor (0.6)
    viol = _violation(conf=conf, related_ids=[_det_id(det)])

    confirmed_calls = []
    review_calls = []

    def fake_append(path, record):
        if "confirmed" in str(path):
            confirmed_calls.append(record)
        else:
            review_calls.append(record)

    with (
        patch("cv2.imwrite", return_value=True),
        patch("evidence.generate._append_jsonl", side_effect=fake_append),
    ):
        generate_evidence(
            img, "frame_001", [det], [], [viol],
            annotated_dir=Path("/tmp/ann"),
            confirmed_jsonl=Path("/tmp/confirmed.jsonl"),
            review_jsonl=Path("/tmp/review.jsonl"),
        )

    assert len(review_calls) == 1
    assert len(confirmed_calls) == 0


# ---------------------------------------------------------------------------
# Test 4 — Below per-type min_conf threshold: record is silently discarded
# ---------------------------------------------------------------------------

def test_below_min_conf_discarded_entirely():
    from evidence.generate import generate_evidence

    img = _image()
    det = _bike_det()
    # helmet threshold is 0.6; send 0.3 — should not appear in either file
    viol = _violation(conf=0.30, related_ids=[_det_id(det)])

    append_calls = []

    with (
        patch("cv2.imwrite", return_value=True),
        patch("evidence.generate._append_jsonl", side_effect=lambda p, r: append_calls.append(r)),
    ):
        records = generate_evidence(
            img, "frame_001", [det], [], [viol],
            annotated_dir=Path("/tmp/ann"),
            confirmed_jsonl=Path("/tmp/confirmed.jsonl"),
            review_jsonl=Path("/tmp/review.jsonl"),
        )

    assert records == []
    assert append_calls == []


# ---------------------------------------------------------------------------
# Test 5 — Plate matching: overlapping plate is attached to EvidenceRecord
# ---------------------------------------------------------------------------

def test_plate_matched_when_overlapping():
    from evidence.generate import generate_evidence

    img = _image()
    det = _bike_det(x1=100, y1=100, x2=400, y2=450)
    plate = _plate(vehicle_bbox=BBox(x1=110, y1=110, x2=390, y2=440))
    viol = _violation(conf=0.90, related_ids=[_det_id(det)])

    with (
        patch("cv2.imwrite", return_value=True),
        patch("evidence.generate._append_jsonl"),
    ):
        records = generate_evidence(
            img, "frame_001", [det], [plate], [viol],
            annotated_dir=Path("/tmp/ann"),
            confirmed_jsonl=Path("/tmp/confirmed.jsonl"),
            review_jsonl=Path("/tmp/review.jsonl"),
        )

    assert records[0].plate_text == "MH12AB1234"
    assert records[0].plate_confidence == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# Test 6 — Plate not matched when far away (low IoU)
# ---------------------------------------------------------------------------

def test_plate_not_matched_when_far():
    from evidence.generate import generate_evidence

    img = _image()
    det = _bike_det(x1=100, y1=100, x2=400, y2=450)
    # Plate vehicle bbox is in a completely different region
    plate = _plate(vehicle_bbox=BBox(x1=500, y1=300, x2=620, y2=450))
    viol = _violation(conf=0.90, related_ids=[_det_id(det)])

    with (
        patch("cv2.imwrite", return_value=True),
        patch("evidence.generate._append_jsonl"),
    ):
        records = generate_evidence(
            img, "frame_001", [det], [plate], [viol],
            annotated_dir=Path("/tmp/ann"),
            confirmed_jsonl=Path("/tmp/confirmed.jsonl"),
            review_jsonl=Path("/tmp/review.jsonl"),
        )

    # No related_detection_ids → falls back to best plate by confidence
    # but since we set related_ids this time, let's use an empty related list
    # so the IoU path is exercised. Re-run with related_ids set:
    viol2 = _violation(conf=0.90, related_ids=[_det_id(det)])

    # The fallback (no related bboxes found) picks plate by confidence.
    # To exercise the IoU rejection path we need related_ids in the violation
    # but the plate's vehicle_bbox far from the detection bbox.
    # With current fixture related_ids=[det_id], plate vehicle IoU is low →
    # no candidates → returns None (plate falls back to highest-conf plate
    # only when related_bboxes list is empty). Verify via _match_plate directly.
    from evidence.generate import _match_plate, PLATE_MATCH_IOU
    result = _match_plate(viol2, [det], [plate])
    # IoU between det bbox and plate vehicle bbox should be near 0
    from evidence.generate import _iou
    iou = _iou(det.bbox, plate.vehicle_bbox)
    assert iou < PLATE_MATCH_IOU
    assert result is None


# ---------------------------------------------------------------------------
# Test 7 — Annotated image shape and dtype unchanged
# ---------------------------------------------------------------------------

def test_annotated_image_same_shape_and_dtype():
    from evidence.generate import _annotate

    img = _image(h=480, w=640)
    det = _bike_det()
    viol = _violation(conf=0.90, related_ids=[_det_id(det)])

    result = _annotate(img, viol, [det], plate=None, is_confirmed=True)

    assert result.shape == img.shape
    assert result.dtype == img.dtype
    assert result is not img   # must be a copy


# ---------------------------------------------------------------------------
# Test 8 — JSONL serialisation round-trip
# ---------------------------------------------------------------------------

def test_jsonl_roundtrip():
    from evidence.generate import _to_dict, _append_jsonl, iter_confirmed

    det = _bike_det()
    viol = _violation(conf=0.90, related_ids=[_det_id(det)])
    evidence = EvidenceRecord(
        violation_record=viol,
        annotated_image_path="/tmp/frame_001_helmet.jpg",
        timestamp="2025-06-01T12:00:00+00:00",
        plate_text="DL5SAB1234",
        plate_confidence=0.87,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "confirmed.jsonl"
        _append_jsonl(path, evidence)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

        d = json.loads(lines[0])
        assert d["plate_text"] == "DL5SAB1234"
        assert d["plate_confidence"] == pytest.approx(0.87)
        assert d["violation_record"]["violation_type"] == "helmet"
        assert d["violation_record"]["confidence"] == pytest.approx(0.90)

        # iter_confirmed must yield the same dict
        items = list(iter_confirmed(path))
        assert len(items) == 1
        assert items[0]["plate_text"] == "DL5SAB1234"


# ---------------------------------------------------------------------------
# Test 9 — Multiple violations produce multiple records, each in correct sink
# ---------------------------------------------------------------------------

def test_multiple_violations_routed_independently():
    from evidence.generate import generate_evidence, AUTO_PROCESS_CUTOFF

    img = _image()
    det = _bike_det()
    high_conf_viol = _violation(ViolationType.helmet,        conf=AUTO_PROCESS_CUTOFF + 0.05)
    low_conf_viol  = _violation(ViolationType.triple_riding, conf=AUTO_PROCESS_CUTOFF - 0.10)

    confirmed_paths = []
    review_paths    = []

    def fake_append(path, record):
        if "confirmed" in str(path):
            confirmed_paths.append(path)
        else:
            review_paths.append(path)

    with (
        patch("cv2.imwrite", return_value=True),
        patch("evidence.generate._append_jsonl", side_effect=fake_append),
    ):
        records = generate_evidence(
            img, "frame_001", [det], [],
            [high_conf_viol, low_conf_viol],
            annotated_dir=Path("/tmp/ann"),
            confirmed_jsonl=Path("/tmp/confirmed.jsonl"),
            review_jsonl=Path("/tmp/review.jsonl"),
        )

    assert len(records) == 2
    assert len(confirmed_paths) == 1
    assert len(review_paths) == 1


# ---------------------------------------------------------------------------
# Test 10 — Empty violations → empty return, no I/O
# ---------------------------------------------------------------------------

def test_empty_violations_no_io():
    from evidence.generate import generate_evidence

    img = _image()
    append_calls = []
    imwrite_calls = []

    with (
        patch("cv2.imwrite", side_effect=lambda *a: imwrite_calls.append(a)),
        patch("evidence.generate._append_jsonl", side_effect=lambda *a: append_calls.append(a)),
    ):
        records = generate_evidence(
            img, "frame_001", [], [], [],
            annotated_dir=Path("/tmp/ann"),
            confirmed_jsonl=Path("/tmp/confirmed.jsonl"),
            review_jsonl=Path("/tmp/review.jsonl"),
        )

    assert records == []
    assert imwrite_calls == []
    assert append_calls == []
