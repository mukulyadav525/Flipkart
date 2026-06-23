"""
Vehicle and pedestrian detection for the traffic violation pipeline.

Dataset: Indian Driving Dataset (IDD) — https://idd.insac.edu.in/
  - Fine-grained annotations for Indian road conditions (mixed traffic,
    autos, two-wheelers, pedestrians in dense urban scenes).
  - IDD class names are remapped to our five canonical VehicleClass labels.

Models:
  - Detection : YOLOv8n / YOLOv8s fine-tuned on IDD (weights loaded from
                DETECTOR_WEIGHTS, default 'weights/yolov8_idd.pt').
                Falls back to pretrained COCO weights so the file runs
                even before fine-tuning is complete.
  - Pose      : YOLOv8n-pose (COCO-pose) for bike riders and pedestrians.
                Keypoint ordering follows COCO 17-point skeleton:
                  0  nose          1  left_eye      2  right_eye
                  3  left_ear      4  right_ear     5  left_shoulder
                  6  right_shoulder 7 left_elbow    8  right_elbow
                  9  left_wrist   10  right_wrist  11  left_hip
                 12  right_hip    13  left_knee    14  right_knee
                 15  left_ankle   16  right_ankle
                Head keypoints (0-4) and shoulder keypoints (5-6) are most
                relevant for helmet / seatbelt checks in the violation module.

Inference latency is printed to stdout after each call to `detect()`.
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ultralytics import is deferred inside functions so the module can be imported
# in test environments where the package may not be installed.
try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

# Project root on sys.path is assumed (set by the caller or __main__).
from shared.schemas import BBox, DetectionRecord, Point2D, VehicleClass

# ---------------------------------------------------------------------------
# IDD → VehicleClass label map
# IDD uses fine-grained labels; we collapse them to our five classes.
# ---------------------------------------------------------------------------
_IDD_LABEL_MAP: dict[str, VehicleClass] = {
    # motorised two-wheelers (helmet law applies)
    "motorcycle":    VehicleClass.bike,
    "two wheeler":   VehicleClass.bike,
    "scooter":       VehicleClass.bike,
    # pedal cycles / cycle-rickshaws — no helmet requirement
    "bicycle":       VehicleClass.bicycle,
    "cycle":         VehicleClass.bicycle,
    "cycle rickshaw": VehicleClass.bicycle,
    # three-wheelers — their own class so helmet/triple-riding rules skip them
    "autorickshaw":  VehicleClass.auto,
    "auto rickshaw": VehicleClass.auto,
    "auto-rickshaw": VehicleClass.auto,
    "rickshaw":      VehicleClass.auto,
    "three wheeler": VehicleClass.auto,
    "three-wheeler": VehicleClass.auto,
    "tuk tuk":       VehicleClass.auto,
    # four-wheelers
    "car":           VehicleClass.car,
    "sedan":         VehicleClass.car,
    "suv":           VehicleClass.car,
    "vehicle fallback": VehicleClass.car,
    # heavy
    "bus":           VehicleClass.bus,
    "truck":         VehicleClass.truck,
    "vehicle":       VehicleClass.truck,  # IDD generic large vehicle
    # people
    "person":        VehicleClass.pedestrian,
    "pedestrian":    VehicleClass.pedestrian,
    # COCO fallback names (used when running pretrained COCO weights)
    "0":             VehicleClass.pedestrian,
    "1":             VehicleClass.bicycle,
    "2":             VehicleClass.car,
    "5":             VehicleClass.bus,
    "7":             VehicleClass.truck,
}

# COCO numeric class IDs we care about (used to filter COCO-pretrained model)
_COCO_KEEP_IDS: frozenset[int] = frozenset([
    0,   # person
    1,   # bicycle  → bike
    2,   # car
    3,   # motorcycle → bike
    5,   # bus
    7,   # truck
])

# Classes that trigger a pose estimation pass
_POSE_CLASSES: frozenset[VehicleClass] = frozenset([
    VehicleClass.bike,
    VehicleClass.pedestrian,
])

# Default weight paths (relative to project root)
DETECTOR_WEIGHTS: str = "weights/yolov8_idd.pt"
POSE_WEIGHTS: str = "weights/yolov8n-pose.pt"
# yolo11s (not yolov8n) — markedly better vehicle-type separation on this
# footage: it boxes three-wheelers wide enough for the auto-reclassification to
# catch them, while keeping genuine motorcycles narrow.
COCO_FALLBACK_WEIGHTS: str = "yolo11s.pt"   # ultralytics auto-downloads

# Confidence threshold for detection (separate from violation thresholds)
DETECT_CONF_THRESHOLD: float = 0.35
POSE_CONF_THRESHOLD: float = 0.35

# Three-wheelers (autorickshaws) aren't a COCO class, so the fallback model
# reports them as "motorcycle" -> bike.  A two-wheeler *with a rider* is clearly
# taller than wide; an autorickshaw/e-rickshaw box is roughly square or wider.
# Bike-mapped detections at/above this width:height ratio are reclassified as
# `auto` so helmet / triple-riding rules (which target genuine two-wheelers) skip
# them.  Calibrated on the Talaimari clip: rickshaw box w/h≈1.01, motorcycle≈0.64.
AUTO_ASPECT_RATIO: float = 0.9


# ---------------------------------------------------------------------------
# Model loader (cached singletons so repeated calls don't reload weights)
# ---------------------------------------------------------------------------

_detector: Optional["YOLO"] = None
_pose_model: Optional["YOLO"] = None


def _load_detector(weights: str = DETECTOR_WEIGHTS) -> "YOLO":
    global _detector
    if _detector is not None:
        return _detector

    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

    weights_path = Path(weights)
    if not weights_path.exists():
        print(
            f"[detect] Weights not found at '{weights}'. "
            f"Falling back to pretrained COCO weights ({COCO_FALLBACK_WEIGHTS}). "
            "Run finetune.py to create task-specific weights.",
            file=sys.stderr,
        )
        weights = COCO_FALLBACK_WEIGHTS

    _detector = YOLO(weights)
    return _detector


def _load_pose_model(weights: str = POSE_WEIGHTS) -> "YOLO":
    global _pose_model
    if _pose_model is not None:
        return _pose_model

    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

    weights_path = Path(weights)
    if not weights_path.exists():
        # Ultralytics auto-downloads official pose weights by name
        weights = "yolov8n-pose.pt"
        print(
            f"[detect] Pose weights not found. Using auto-download ({weights}).",
            file=sys.stderr,
        )

    _pose_model = YOLO(weights)
    return _pose_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_label(raw_name: str, class_id: int) -> Optional[VehicleClass]:
    """Return a VehicleClass or None if this detection should be skipped."""
    # Try name-based lookup first (IDD fine-tuned model)
    label = _IDD_LABEL_MAP.get(raw_name.lower())
    if label is not None:
        return label
    # Fall back to COCO numeric ID filtering
    if class_id in _COCO_KEEP_IDS:
        coco_map = {0: VehicleClass.pedestrian, 1: VehicleClass.bicycle,
                    2: VehicleClass.car, 3: VehicleClass.bike,
                    5: VehicleClass.bus, 7: VehicleClass.truck}
        return coco_map.get(class_id)
    return None


def _run_pose(
    image: np.ndarray,
    crop_bbox: BBox,
    pose_model: "YOLO",
) -> Optional[list[Point2D]]:
    """
    Crop the detection region, run pose estimation, return keypoints in
    original image coordinates.  Returns None if no person is detected
    in the crop.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(crop_bbox.x1))
    y1 = max(0, int(crop_bbox.y1))
    x2 = min(w, int(crop_bbox.x2))
    y2 = min(h, int(crop_bbox.y2))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    results = pose_model.predict(
        crop,
        conf=POSE_CONF_THRESHOLD,
        verbose=False,
        device="cpu",
    )

    if not results or results[0].keypoints is None:
        return None

    kpts_tensor = results[0].keypoints.xy  # shape (N, 17, 2)
    if kpts_tensor is None or len(kpts_tensor) == 0:
        return None

    # Take the highest-confidence person detection in the crop
    kpts = kpts_tensor[0].cpu().numpy()   # (17, 2) in crop coords

    # Translate back to original image coordinates
    keypoints = [
        Point2D(x=float(kp[0]) + x1, y=float(kp[1]) + y1)
        for kp in kpts
    ]
    return keypoints


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    image: np.ndarray,
    image_id: str,
    *,
    detector_weights: str = DETECTOR_WEIGHTS,
    pose_weights: str = POSE_WEIGHTS,
    conf_threshold: float = DETECT_CONF_THRESHOLD,
    run_pose: bool = True,
    device: str = "cpu",
    print_latency: bool = True,
    latency_log_path: Optional[str | Path] = None,
) -> list[DetectionRecord]:
    """
    Run detection (and optional pose estimation) on a preprocessed image.

    Parameters
    ----------
    image       : BGR uint8 ndarray from preprocessing stage.
    image_id    : Identifier string matching the source filename stem.
    detector_weights : Path to fine-tuned YOLOv8 weights.
    pose_weights     : Path to YOLOv8-pose weights.
    conf_threshold   : Minimum detection confidence to keep.
    run_pose    : If False, skip pose estimation (faster, loses keypoints).
    device      : 'cpu', 'cuda', or 'mps'.
    print_latency : Print per-image latency to stdout.
    latency_log_path : Optional path to a JSONL file where per-image latency
                       measurements are appended for evaluation (Phase 7).
                       Each line: {"image_id": ..., "total_ms": ..., "pose_ms": ...}

    Returns
    -------
    list[DetectionRecord]
        One record per detected object, strictly matching shared/schemas.py.
    """
    t_start = time.perf_counter()

    detector = _load_detector(detector_weights)
    pose_model = _load_pose_model(pose_weights) if run_pose else None

    # --- Detection pass ---
    det_results = detector.predict(
        image,
        conf=conf_threshold,
        verbose=False,
        device=device,
    )

    records: list[DetectionRecord] = []

    if not det_results:
        _report_latency(t_start, image_id, 0, print_latency)
        return records

    result = det_results[0]
    boxes = result.boxes

    t_pose_total = 0.0

    for box in boxes:
        class_id = int(box.cls[0].item())
        conf     = float(box.conf[0].item())
        raw_name = detector.names.get(class_id, str(class_id))

        vehicle_class = _map_label(raw_name, class_id)
        if vehicle_class is None:
            continue

        xyxy = box.xyxy[0].cpu().numpy()
        bbox = BBox(
            x1=float(xyxy[0]),
            y1=float(xyxy[1]),
            x2=float(xyxy[2]),
            y2=float(xyxy[3]),
        )

        # Shape-based recovery of three-wheelers when the detector lacks an
        # autorickshaw class (e.g. COCO fallback): a wide "bike" box is an auto.
        if vehicle_class is VehicleClass.bike:
            w = bbox.x2 - bbox.x1
            h = bbox.y2 - bbox.y1
            if h > 0 and (w / h) >= AUTO_ASPECT_RATIO:
                vehicle_class = VehicleClass.auto

        keypoints: Optional[list[Point2D]] = None
        if run_pose and pose_model is not None and vehicle_class in _POSE_CLASSES:
            t_pose_start = time.perf_counter()
            keypoints = _run_pose(image, bbox, pose_model)
            t_pose_total += time.perf_counter() - t_pose_start

        records.append(DetectionRecord(
            image_id=image_id,
            bbox=bbox,
            class_label=vehicle_class,
            track_confidence=conf,
            pose_keypoints=keypoints,
        ))

    _report_latency(
        t_start, image_id, len(records), print_latency,
        t_pose=t_pose_total if run_pose else None,
        latency_log_path=latency_log_path,
    )
    return records


def _report_latency(
    t_start: float,
    image_id: str,
    n_detections: int,
    enabled: bool,
    t_pose: Optional[float] = None,
    latency_log_path: Optional[str | Path] = None,
) -> None:
    total_ms = (time.perf_counter() - t_start) * 1000
    pose_ms  = t_pose * 1000 if t_pose is not None else None

    if enabled:
        pose_str = f"  (pose: {pose_ms:.1f} ms)" if pose_ms is not None else ""
        print(
            f"[detect] {image_id} | {n_detections} object(s) | "
            f"total: {total_ms:.1f} ms{pose_str}"
        )

    if latency_log_path is not None:
        import json as _json
        entry: dict = {"image_id": image_id, "total_ms": round(total_ms, 3)}
        if pose_ms is not None:
            entry["pose_ms"] = round(pose_ms, 3)
        log_path = Path(latency_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry) + "\n")


def detect_file(
    input_path: str | Path,
    *,
    detector_weights: str = DETECTOR_WEIGHTS,
    pose_weights: str = POSE_WEIGHTS,
    conf_threshold: float = DETECT_CONF_THRESHOLD,
    run_pose: bool = True,
    device: str = "cpu",
    print_latency: bool = True,
) -> list[DetectionRecord]:
    """Load an image from disk and run the detection pipeline."""
    input_path = Path(input_path)
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")
    return detect(
        image,
        image_id=input_path.stem,
        detector_weights=detector_weights,
        pose_weights=pose_weights,
        conf_threshold=conf_threshold,
        run_pose=run_pose,
        device=device,
        print_latency=print_latency,
    )


# ---------------------------------------------------------------------------
# CLI  —  python -m detection.detect --input path [options]
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="detection.detect",
        description="Run YOLOv8 detection on a single image and print DetectionRecords.",
    )
    p.add_argument("--input",   required=True,  metavar="PATH",
                   help="Path to input image (preprocessed BGR)")
    p.add_argument("--weights", default=DETECTOR_WEIGHTS, metavar="PATH",
                   help=f"Detector weights (default: {DETECTOR_WEIGHTS})")
    p.add_argument("--pose-weights", default=POSE_WEIGHTS, metavar="PATH",
                   help=f"Pose model weights (default: {POSE_WEIGHTS})")
    p.add_argument("--conf",    type=float, default=DETECT_CONF_THRESHOLD, metavar="F",
                   help=f"Detection confidence threshold (default: {DETECT_CONF_THRESHOLD})")
    p.add_argument("--no-pose", action="store_true",
                   help="Skip pose estimation (faster, no keypoints)")
    p.add_argument("--device",  default="cpu", metavar="DEV",
                   help="Inference device: cpu / cuda / mps (default: cpu)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    records = detect_file(
        args.input,
        detector_weights=args.weights,
        pose_weights=args.pose_weights,
        conf_threshold=args.conf,
        run_pose=not args.no_pose,
        device=args.device,
        print_latency=True,
    )
    print(f"\nDetected {len(records)} object(s):")
    for r in records:
        kp_count = len(r.pose_keypoints) if r.pose_keypoints else 0
        print(
            f"  [{r.class_label.value}] conf={r.track_confidence:.3f} "
            f"bbox=({r.bbox.x1:.0f},{r.bbox.y1:.0f},{r.bbox.x2:.0f},{r.bbox.y2:.0f}) "
            f"keypoints={kp_count}"
        )


if __name__ == "__main__":
    main()
