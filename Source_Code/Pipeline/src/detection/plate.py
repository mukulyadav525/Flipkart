"""
License-plate detector (localisation stage of ANPR).

Wraps a trained YOLO license-plate model and returns plate bounding boxes.  The
model is auto-discovered the same way as the helmet model — drop a file at
``Pipeline/weights/plate.pt`` (or train one and point ``--plate-weights`` at it)
and detection turns on automatically.

Design
------
* Detection-first: this module's job is to *localise* plates (boxes).  Reading
  the text is a separate, best-effort OCR step (``plate_ocr``).
* Graceful degradation: if ultralytics or a weights file is unavailable,
  ``available()`` is ``False`` and ``detect_plates`` returns ``[]`` so callers
  fall back to the classic-CV localiser without crashing.

A trained plate model is the accurate path; on this project's far-away,
motion-blurred footage no localiser reads much, but the system is correct and
turns on the moment good weights / closer footage are provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from shared.schemas import BBox

try:
    from ultralytics import YOLO
    _ULTRA = True
except ImportError:
    _ULTRA = False

# Search order (relative to the Pipeline root) for a trained plate model.
_DEFAULT_CANDIDATES = [
    "weights/plate.pt",
    "weights/license_plate.pt",
    "weights/anpr.pt",
    "gridlock/models/plate.pt",
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
    """True when ultralytics is installed and a plate model file is resolvable."""
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


def detect_plates(
    image,
    *,
    weights: Optional[str] = None,
    conf: float = 0.25,
    device: str = "cpu",
    imgsz: int = 1280,
) -> list[BBox]:
    """Detect plates over a whole frame. Returns plate BBoxes in image coords."""
    w = _resolve(weights)
    if not _ULTRA or w is None:
        return []
    model = _load(w)
    results = model.predict(image, conf=conf, verbose=False, device=device, imgsz=imgsz)
    if not results:
        return []
    out: list[BBox] = []
    for box in results[0].boxes:
        xy = box.xyxy[0].cpu().numpy()
        out.append(BBox(x1=float(xy[0]), y1=float(xy[1]), x2=float(xy[2]), y2=float(xy[3])))
    return out


def detect_plate_in_vehicle(
    image,
    vehicle_bbox: BBox,
    *,
    weights: Optional[str] = None,
    conf: float = 0.25,
    device: str = "cpu",
    imgsz: int = 640,
) -> list[BBox]:
    """
    Detect plates inside a single vehicle crop and return boxes in full-image
    coordinates.  Cropping to the vehicle upscales the plate, improving recall.
    """
    w = _resolve(weights)
    if not _ULTRA or w is None:
        return []
    H, W = image.shape[:2]
    x1 = max(0, int(vehicle_bbox.x1)); y1 = max(0, int(vehicle_bbox.y1))
    x2 = min(W, int(vehicle_bbox.x2)); y2 = min(H, int(vehicle_bbox.y2))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    model = _load(w)
    results = model.predict(crop, conf=conf, verbose=False, device=device, imgsz=imgsz)
    if not results:
        return []
    out: list[BBox] = []
    for box in results[0].boxes:
        xy = box.xyxy[0].cpu().numpy()
        out.append(BBox(x1=float(xy[0]) + x1, y1=float(xy[1]) + y1,
                        x2=float(xy[2]) + x1, y2=float(xy[3]) + y1))
    return out
