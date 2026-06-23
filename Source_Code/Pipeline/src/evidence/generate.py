"""
Evidence generation module.

Takes the outputs of detection, plate-OCR, and violation-rule stages
and produces:
  1. An annotated image saved to outputs/annotated_images/<image_id>_<violation_type>.jpg
     with bounding boxes, violation label, confidence score, and plate text drawn on.
  2. An EvidenceRecord (shared/schemas.py) returned to the caller.
  3. Persistence to one of two JSONL files (append, one record per line):
       outputs/violation_records/confirmed.jsonl   — confidence >= AUTO_PROCESS_CUTOFF
       outputs/human_review_queue.jsonl            — confidence < AUTO_PROCESS_CUTOFF
       (both files are the single source of truth; Phase 6 reads confirmed.jsonl)

Plate matching:
  A PlateRecord is associated with a ViolationRecord when its vehicle_bbox
  overlaps the detection bbox of any related detection by >= PLATE_MATCH_IOU.
  If multiple plates match, the one with the highest OCR confidence wins.
  If no plate matches, plate_text="" and plate_confidence=0.0.

Serialisation:
  EvidenceRecord is serialised to JSON via _to_dict(), which handles nested
  dataclasses and enums.  No third-party serialisation library is required.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from shared.schemas import (
    BBox,
    DetectionRecord,
    EvidenceRecord,
    PlateRecord,
    ViolationRecord,
    ViolationType,
)
from config import AUTO_PROCESS_CUTOFF, THRESHOLDS

# ---------------------------------------------------------------------------
# Output paths (relative to project root; callers may override)
# ---------------------------------------------------------------------------

ANNOTATED_DIR    = Path("outputs/annotated_images")
CONFIRMED_JSONL  = Path("outputs/violation_records/confirmed.jsonl")
REVIEW_JSONL     = Path("outputs/human_review_queue.jsonl")

# ---------------------------------------------------------------------------
# Visual style constants
# ---------------------------------------------------------------------------

# BGR colours
_COLOUR_CONFIRMED = (0, 60, 220)    # blue — auto-processed
_COLOUR_REVIEW    = (0, 165, 255)   # orange — human review
_COLOUR_PLATE     = (0, 200, 0)     # green — licence plate box
_COLOUR_DETECTION = (180, 180, 180) # grey  — related detection boxes

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS  = 2
_TAG_PAD    = 4   # pixels of padding inside label tag backgrounds

# Minimum IoU between a PlateRecord.vehicle_bbox and a detection bbox
# for the plate to be considered associated with that detection.
PLATE_MATCH_IOU: float = 0.30


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_dict(obj) -> object:
    """Recursively convert dataclasses / enums to JSON-serialisable types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, tuple):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    # Enum values are already str (str-Enum base class) — just return value
    if hasattr(obj, "value"):
        return obj.value
    return obj


def _append_jsonl(path: Path, record: EvidenceRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_dict(record), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((a.x2 - a.x1) * (a.y2 - a.y1), 1e-6)
    area_b = max((b.x2 - b.x1) * (b.y2 - b.y1), 1e-6)
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Plate matching
# ---------------------------------------------------------------------------

def _match_plate(
    violation: ViolationRecord,
    detections: list[DetectionRecord],
    plates: list[PlateRecord],
) -> Optional[PlateRecord]:
    """
    Find the best-matching PlateRecord for a given ViolationRecord.

    Matching strategy:
      1. Collect detection bboxes referenced by the violation's related_detection_ids.
      2. For each plate, compute IoU between plate.vehicle_bbox and each detection bbox.
      3. Keep plates whose max IoU >= PLATE_MATCH_IOU.
      4. Return the one with the highest ocr_confidence, or None if none qualify.
    """
    related_bboxes: list[BBox] = []
    related_ids = set(violation.related_detection_ids)
    for det in detections:
        det_id = f"{det.image_id}:{det.class_label.value}:{det.bbox.x1:.0f},{det.bbox.y1:.0f}"
        if det_id in related_ids:
            related_bboxes.append(det.bbox)

    if not related_bboxes:
        # Fall back: any plate in the image (first by confidence)
        if plates:
            return max(plates, key=lambda p: p.ocr_confidence)
        return None

    candidates: list[tuple[float, PlateRecord]] = []
    for plate in plates:
        best_iou = max(_iou(plate.vehicle_bbox, bbox) for bbox in related_bboxes)
        if best_iou >= PLATE_MATCH_IOU:
            candidates.append((plate.ocr_confidence, plate))

    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


# ---------------------------------------------------------------------------
# Annotation drawing
# ---------------------------------------------------------------------------

def _draw_tag(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
    *,
    above: bool = True,
) -> None:
    """Draw a filled rectangle label tag with `text` above or below (x, y)."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS - 1)
    if above:
        tag_y2 = y
        tag_y1 = y - th - _TAG_PAD * 2
    else:
        tag_y1 = y
        tag_y2 = y + th + _TAG_PAD * 2

    tag_x1 = x
    tag_x2 = x + tw + _TAG_PAD * 2

    # Clamp to image bounds
    h, w = image.shape[:2]
    tag_x1 = max(0, min(tag_x1, w - 1))
    tag_x2 = max(0, min(tag_x2, w - 1))
    tag_y1 = max(0, min(tag_y1, h - 1))
    tag_y2 = max(0, min(tag_y2, h - 1))

    cv2.rectangle(image, (tag_x1, tag_y1), (tag_x2, tag_y2), colour, cv2.FILLED)
    cv2.putText(
        image, text,
        (tag_x1 + _TAG_PAD, tag_y2 - _TAG_PAD - baseline // 2),
        _FONT, _FONT_SCALE, (255, 255, 255), _THICKNESS - 1, cv2.LINE_AA,
    )


def _annotate(
    image: np.ndarray,
    violation: ViolationRecord,
    detections: list[DetectionRecord],
    plate: Optional[PlateRecord],
    *,
    is_confirmed: bool,
) -> np.ndarray:
    """
    Draw all annotation layers onto a copy of the image and return it.

    Layer order (bottom → top):
      1. Grey boxes for all detection bboxes referenced by the violation
      2. Coloured primary violation box (blue=confirmed, orange=review)
      3. Green plate bbox + OCR text (if plate matched)
      4. Violation label tag (type + confidence) at top-left of violation box
      5. Plate text tag below the plate box
    """
    out = image.copy()
    colour = _COLOUR_CONFIRMED if is_confirmed else _COLOUR_REVIEW

    # --- Related detection boxes (grey, thin) ---
    related_ids = set(violation.related_detection_ids)
    primary_bbox: Optional[BBox] = None

    for det in detections:
        det_id = f"{det.image_id}:{det.class_label.value}:{det.bbox.x1:.0f},{det.bbox.y1:.0f}"
        if det_id not in related_ids:
            continue
        b = det.bbox
        cv2.rectangle(
            out,
            (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
            _COLOUR_DETECTION, 1,
        )
        if primary_bbox is None:
            primary_bbox = b

    # --- Primary violation box (coloured, thick) ---
    if primary_bbox is not None:
        b = primary_bbox
        cv2.rectangle(
            out,
            (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
            colour, _THICKNESS,
        )
        label = f"{violation.violation_type.value}  {violation.confidence:.0%}"
        _draw_tag(out, label, int(b.x1), int(b.y1), colour, above=True)

    # --- Plate box + text ---
    if plate is not None:
        pb = plate.plate_bbox
        cv2.rectangle(
            out,
            (int(pb.x1), int(pb.y1)), (int(pb.x2), int(pb.y2)),
            _COLOUR_PLATE, _THICKNESS,
        )
        plate_label = f"{plate.plate_text}  ({plate.ocr_confidence:.0%})"
        _draw_tag(out, plate_label, int(pb.x1), int(pb.y2), _COLOUR_PLATE, above=False)

    # --- Review watermark ---
    if not is_confirmed:
        cv2.putText(
            out, "REVIEW REQUIRED",
            (10, out.shape[0] - 12),
            _FONT, 0.65, _COLOUR_REVIEW, 2, cv2.LINE_AA,
        )

    return out


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def generate_evidence(
    image: np.ndarray,
    image_id: str,
    detections: list[DetectionRecord],
    plates: list[PlateRecord],
    violations: list[ViolationRecord],
    *,
    annotated_dir: Path = ANNOTATED_DIR,
    confirmed_jsonl: Path = CONFIRMED_JSONL,
    review_jsonl: Path = REVIEW_JSONL,
    timestamp: Optional[str] = None,
) -> list[EvidenceRecord]:
    """
    Generate one EvidenceRecord per ViolationRecord, annotate the image, and
    persist to the appropriate JSONL sink.

    Parameters
    ----------
    image        : Original (or preprocessed) BGR uint8 ndarray.
    image_id     : Stem of the source filename — used in output filenames.
    detections   : All DetectionRecords for this image (needed for box drawing
                   and plate matching).
    plates       : PlateRecords from the OCR stage (may be empty).
    violations   : ViolationRecords from the rule engine (may be empty).
    annotated_dir : Directory to write annotated JPEG images.
    confirmed_jsonl : JSONL path for auto-processed records.
    review_jsonl    : JSONL path for human-review queue.
    timestamp    : ISO-8601 string; defaults to current UTC time.

    Returns
    -------
    list[EvidenceRecord]
        One per input ViolationRecord; empty list if violations is empty.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    annotated_dir.mkdir(parents=True, exist_ok=True)

    evidence_records: list[EvidenceRecord] = []

    for violation in violations:
        # --- Threshold gate ---
        min_conf = THRESHOLDS.get(violation.violation_type)
        if violation.confidence < min_conf:
            # Below the per-type floor — discard entirely, not even for review
            continue

        is_confirmed = violation.confidence >= AUTO_PROCESS_CUTOFF

        # --- Plate matching ---
        plate = _match_plate(violation, detections, plates)
        plate_text  = plate.plate_text       if plate else ""
        plate_conf  = plate.ocr_confidence   if plate else 0.0

        # --- Annotated image ---
        annotated = _annotate(image, violation, detections, plate,
                              is_confirmed=is_confirmed)
        img_filename = f"{image_id}_{violation.violation_type.value}.jpg"
        img_path = annotated_dir / img_filename
        cv2.imwrite(str(img_path), annotated)

        # --- Build EvidenceRecord ---
        evidence = EvidenceRecord(
            violation_record=violation,
            annotated_image_path=str(img_path),
            timestamp=timestamp,
            plate_text=plate_text,
            plate_confidence=plate_conf,
        )
        evidence_records.append(evidence)

        # --- Persist ---
        sink = confirmed_jsonl if is_confirmed else review_jsonl
        _append_jsonl(sink, evidence)

    return evidence_records


def generate_evidence_from_file(
    image_path: str | Path,
    detections: list[DetectionRecord],
    plates: list[PlateRecord],
    violations: list[ViolationRecord],
    **kwargs,
) -> list[EvidenceRecord]:
    """Load image from disk then call generate_evidence()."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return generate_evidence(
        image, image_path.stem, detections, plates, violations, **kwargs
    )


# ---------------------------------------------------------------------------
# JSONL reader — used by Phase 6 analytics
# ---------------------------------------------------------------------------

def iter_confirmed(confirmed_jsonl: Path = CONFIRMED_JSONL):
    """
    Yield raw dicts from the confirmed JSONL file, one per line.

    Phase 6 reads this to build aggregate statistics.  Raw dicts are returned
    (not reconstructed EvidenceRecords) because analytics only needs field
    access, not the full dataclass hierarchy.
    """
    if not confirmed_jsonl.exists():
        return
    with confirmed_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_review_queue(review_jsonl: Path = REVIEW_JSONL):
    """Yield raw dicts from the human-review JSONL file."""
    if not review_jsonl.exists():
        return
    with review_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evidence.generate",
        description=(
            "Generate evidence records for a single image. "
            "Expects pre-computed JSON files for detections, plates, and violations."
        ),
    )
    p.add_argument("--image",      required=True, metavar="PATH",
                   help="Path to the source image")
    p.add_argument("--detections", required=True, metavar="JSON",
                   help="JSON file containing a list of DetectionRecord dicts")
    p.add_argument("--plates",     required=True, metavar="JSON",
                   help="JSON file containing a list of PlateRecord dicts (may be [])")
    p.add_argument("--violations", required=True, metavar="JSON",
                   help="JSON file containing a list of ViolationRecord dicts")
    p.add_argument("--annotated-dir", default=str(ANNOTATED_DIR), metavar="PATH")
    p.add_argument("--confirmed-out", default=str(CONFIRMED_JSONL), metavar="PATH")
    p.add_argument("--review-out",    default=str(REVIEW_JSONL),    metavar="PATH")
    return p


def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dict_to_bbox(d: dict) -> BBox:
    return BBox(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


def _dict_to_detection(d: dict) -> DetectionRecord:
    from shared.schemas import Point2D, VehicleClass
    kpts = None
    if d.get("pose_keypoints"):
        kpts = [Point2D(x=p["x"], y=p["y"]) for p in d["pose_keypoints"]]
    return DetectionRecord(
        image_id=d["image_id"],
        bbox=_dict_to_bbox(d["bbox"]),
        class_label=VehicleClass(d["class_label"]),
        track_confidence=d["track_confidence"],
        pose_keypoints=kpts,
    )


def _dict_to_plate(d: dict) -> PlateRecord:
    return PlateRecord(
        image_id=d["image_id"],
        vehicle_bbox=_dict_to_bbox(d["vehicle_bbox"]),
        plate_bbox=_dict_to_bbox(d["plate_bbox"]),
        plate_text=d["plate_text"],
        ocr_confidence=d["ocr_confidence"],
    )


def _dict_to_violation(d: dict) -> ViolationRecord:
    return ViolationRecord(
        image_id=d["image_id"],
        violation_type=ViolationType(d["violation_type"]),
        confidence=d["confidence"],
        rule_trace=d["rule_trace"],
        related_detection_ids=d.get("related_detection_ids", []),
        related_plate_text=d.get("related_plate_text"),
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    detections = [_dict_to_detection(d) for d in _load_json(args.detections)]
    plates     = [_dict_to_plate(d)     for d in _load_json(args.plates)]
    violations = [_dict_to_violation(d) for d in _load_json(args.violations)]

    records = generate_evidence_from_file(
        args.image,
        detections,
        plates,
        violations,
        annotated_dir=Path(args.annotated_dir),
        confirmed_jsonl=Path(args.confirmed_out),
        review_jsonl=Path(args.review_out),
    )

    confirmed = sum(1 for r in records if r.violation_record.confidence >= AUTO_PROCESS_CUTOFF)
    review    = len(records) - confirmed
    print(f"Generated {len(records)} evidence record(s): "
          f"{confirmed} confirmed, {review} queued for review.")
    for r in records:
        print(f"  [{r.violation_record.violation_type.value}] "
              f"conf={r.violation_record.confidence:.2f}  "
              f"plate='{r.plate_text}'  "
              f"img={r.annotated_image_path}")


if __name__ == "__main__":
    main()
