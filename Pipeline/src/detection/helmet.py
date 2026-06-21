"""
Helmet / no-helmet head detector.

Wraps a YOLO model trained to classify rider heads as *helmet* vs *no_helmet*
(e.g. the project's own ``gridlock/runs_helmet2/train/weights/best.pt``, whose
classes are ``{0: 'helmet', 1: 'no_helmet'}``).  It returns two lists of head
``DetectionRecord`` boxes so the helmet rule fires on an **actually detected
bare head** instead of assuming "no helmet" for every two-wheeler.

This is the precision fix for the COCO-fallback false positives (cyclists,
pedestrians, and — as far as a head classifier can — three-wheeler passengers):
no detected no-helmet head ⇒ no violation.

Heavy deps (ultralytics) are imported lazily, and if no trained model is found
the detector reports ``available() == False`` and the helmet rule simply skips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from shared.schemas import BBox, DetectionRecord, VehicleClass

try:
    from ultralytics import YOLO
    _ULTRA = True
except ImportError:
    _ULTRA = False

# Search order (relative to the Pipeline root) for a trained helmet model.
_DEFAULT_CANDIDATES = [
    "weights/helmet.pt",
    "gridlock/runs_helmet2/train/weights/best.pt",
    "gridlock/runs_helmet/train/weights/best.pt",
    "gridlock/models/helmet.pt",
]

_PIPELINE_ROOT = Path(__file__).resolve().parents[2]

_model = None
_model_key: Optional[str] = None


def _resolve(weights: Optional[str]) -> Optional[str]:
    if weights:
        return weights if Path(weights).exists() else None
    for c in _DEFAULT_CANDIDATES:
        p = _PIPELINE_ROOT / c
        if p.exists():
            return str(p)
    return None


def available(weights: Optional[str] = None) -> bool:
    """True when ultralytics is installed and a helmet model file is resolvable."""
    return _ULTRA and _resolve(weights) is not None


def resolved_weights(weights: Optional[str] = None) -> Optional[str]:
    return _resolve(weights)


def _load(weights: str):
    global _model, _model_key
    if _model is not None and _model_key == weights:
        return _model
    _model = YOLO(weights)
    _model_key = weights
    return _model


def _is_nohelmet(name: str) -> bool:
    n = name.lower().replace("-", " ").replace("_", " ")
    if "without" in n:
        return True
    if "helmet" in n and "no" in n.split():
        return True
    return n in {"head", "bare", "nohelmet"}


def detect_heads(
    image,
    image_id: str,
    *,
    weights: Optional[str] = None,
    conf: float = 0.3,
    device: str = "cpu",
    imgsz: int = 1280,   # heads are tiny; 736 misses ~4x the no-helmet heads
) -> tuple[list[DetectionRecord], list[DetectionRecord]]:
    """
    Run the helmet model on a full frame (whole-image pass).

    Returns ``(helmet_heads, nohelmet_heads)`` as DetectionRecord lists (only
    ``bbox`` and ``track_confidence`` are meaningful; class_label is a filler).
    Both lists are empty when no model is available.

    Prefer :func:`classify_riders` for the helmet rule — examining only the
    motorcycle regions is both more precise (no pedestrian/cyclist heads) and
    more sensitive (the crop upscales the rider's small head).
    """
    helmet: list[DetectionRecord] = []
    nohelmet: list[DetectionRecord] = []

    w = _resolve(weights)
    if not _ULTRA or w is None:
        return helmet, nohelmet

    model = _load(w)
    results = model.predict(image, conf=conf, verbose=False, device=device, imgsz=imgsz)
    if not results:
        return helmet, nohelmet

    for box in results[0].boxes:
        cid = int(box.cls[0].item())
        c = float(box.conf[0].item())
        name = model.names.get(cid, str(cid))
        xy = box.xyxy[0].cpu().numpy()
        rec = DetectionRecord(
            image_id=image_id,
            bbox=BBox(x1=float(xy[0]), y1=float(xy[1]), x2=float(xy[2]), y2=float(xy[3])),
            class_label=VehicleClass.pedestrian,   # filler; rule only uses bbox/conf
            track_confidence=c,
        )
        (nohelmet if _is_nohelmet(name) else helmet).append(rec)

    return helmet, nohelmet


def classify_riders(
    image,
    motorcycles: list[DetectionRecord],
    image_id: str,
    *,
    weights: Optional[str] = None,
    conf: float = 0.3,
    device: str = "cpu",
    imgsz: int = 640,
    up_factor: float = 1.0,   # expand box upward by this × box-height to catch the head
    side_pad: float = 0.15,
) -> tuple[list[DetectionRecord], list[DetectionRecord]]:
    """
    Motorcycle-first helmet check.

    For each detected **motorcycle**, crop the vehicle region (expanded upward to
    include the rider's head), run the helmet model on that crop, and return the
    helmet / no-helmet head boxes translated back to full-frame coordinates.

    Because only motorcycle regions are examined, pedestrians and cyclists are
    never considered; because the crop is upscaled to ``imgsz``, the rider's
    small head becomes large enough for the model to classify reliably.
    """
    helmet: list[DetectionRecord] = []
    nohelmet: list[DetectionRecord] = []

    w = _resolve(weights)
    if not _ULTRA or w is None or not motorcycles:
        return helmet, nohelmet

    model = _load(w)
    H, W = image.shape[:2]

    for v in motorcycles:
        b = v.bbox
        bw = b.x2 - b.x1
        bh = b.y2 - b.y1
        x1 = max(0, int(b.x1 - bw * side_pad))
        x2 = min(W, int(b.x2 + bw * side_pad))
        y1 = max(0, int(b.y1 - bh * up_factor))   # reach up for the rider's head
        y2 = min(H, int(b.y2 + bh * side_pad))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        results = model.predict(crop, conf=conf, verbose=False, device=device, imgsz=imgsz)
        if not results:
            continue
        for box in results[0].boxes:
            cid = int(box.cls[0].item())
            c = float(box.conf[0].item())
            name = model.names.get(cid, str(cid))
            xy = box.xyxy[0].cpu().numpy()
            rec = DetectionRecord(
                image_id=image_id,
                bbox=BBox(x1=float(xy[0]) + x1, y1=float(xy[1]) + y1,
                          x2=float(xy[2]) + x1, y2=float(xy[3]) + y1),
                class_label=VehicleClass.pedestrian,   # filler; rule uses bbox/conf
                track_confidence=c,
            )
            (nohelmet if _is_nohelmet(name) else helmet).append(rec)

    return helmet, nohelmet
