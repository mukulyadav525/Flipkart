"""
License-plate recognition (ANPR) for the traffic-violation pipeline — Task 5.

Pipeline position
-----------------
    detection (vehicle bboxes)  ->  plate_ocr  ->  PlateRecord list
                                                   |
                                  evidence.generate matches plates to
                                  violations by vehicle_bbox IoU.

Strategy (no hardcoded plate strings, no per-image tuning)
----------------------------------------------------------
For each vehicle detection we:
  1. Crop the vehicle region from the full frame.
  2. Localise candidate plate rectangles inside the crop with a classic
     edge + contour + aspect-ratio detector (no extra trained model needed).
     When EasyOCR is available its own text detector is also used, so a plate
     is found even when contours fail (dirty / angled plates).
  3. OCR every candidate region, normalise the text, and score each reading by
        ocr_confidence  ×  plate_likeness  ×  format_bonus
     where the multipliers are computed from the *content* of the string —
     never from a lookup table of known plates.
  4. Emit one PlateRecord (best reading) per vehicle, in full-image pixel
     coordinates, strictly matching shared/schemas.py.

Graceful degradation
---------------------
EasyOCR is optional and heavy.  If it is not installed, ``PlateReader.available``
is ``False`` and ``read_plates`` returns an empty list instead of raising, so the
rest of the pipeline keeps running.  The pure-Python text helpers
(``normalize_plate``, ``is_valid_indian_plate``, ``plate_likeness``,
``score_reading``) have no third-party dependencies and are unit-tested directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from shared.schemas import BBox, DetectionRecord, PlateRecord, VehicleClass

# cv2 / numpy are imported lazily so this module imports in environments
# (CI, unit tests for the text helpers) where they are not installed.
try:  # pragma: no cover - exercised only when the dep is present
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None      # type: ignore
    np = None       # type: ignore
    _CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tunables — every value is a documented geometric/format constant, NOT data.
# ---------------------------------------------------------------------------

# Vehicle classes that can carry a registration plate (pedestrians cannot).
PLATE_BEARING_CLASSES: frozenset[VehicleClass] = frozenset({
    VehicleClass.car, VehicleClass.bus, VehicleClass.truck, VehicleClass.bike,
})

# A licence plate is a wide rectangle.  Indian single-row plates run ~2:1–6:1;
# we accept a generous band so angled / partial plates still pass.
PLATE_MIN_ASPECT: float = 1.8
PLATE_MAX_ASPECT: float = 7.0

# A plate occupies a small fraction of the vehicle crop.  Reject blobs that are
# implausibly large (whole vehicle) or tiny (specular noise).
PLATE_MIN_AREA_FRAC: float = 0.005
PLATE_MAX_AREA_FRAC: float = 0.30

# OCR readings below this confidence are discarded outright.
MIN_OCR_CONF: float = 0.20

# Plates upscaled so their shortest text dimension is at least this many px —
# EasyOCR is far more accurate above this size.
MIN_OCR_DIM: int = 240

# Indian registration formats (post-2001 BharatSeries + classic state series).
# Validation returns a *bonus*, it never hard-rejects a reading, so OCR
# confusions (O/0, I/1) are tolerated and corrected rather than dropped.
_PLATE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),  # MH12AB1234, DL5SAB0001
    re.compile(r"^[A-Z]{2}\d{1,2}\d{4}$"),            # older 1-3 letter omitted
    re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),          # 22BH1234AA  (Bharat series)
)

# Common OCR character confusions, applied position-aware in normalize_plate.
_DIGIT_TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
_ALPHA_TO_DIGIT = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
                   "S": "5", "B": "8", "G": "6", "D": "0"}


# ---------------------------------------------------------------------------
# Pure-Python text helpers  (no cv2 / no easyocr — directly unit-tested)
# ---------------------------------------------------------------------------

def normalize_plate(raw: str) -> str:
    """Uppercase and strip every non-alphanumeric character from a raw reading."""
    return re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()


def is_valid_indian_plate(text: str) -> bool:
    """True when ``text`` matches any known Indian registration format exactly."""
    return any(p.match(text) for p in _PLATE_PATTERNS)


def is_acceptable_plate(text: str) -> bool:
    """
    Gate a normalised reading before it is accepted as a plate.

    Real registration plates always contain digits, so this rejects vehicle /
    shop / bus signage the OCR picks up (e.g. 'SPECIAL', 'MAHAZARA') which would
    otherwise masquerade as a plate.  Requires a plausible length, at least two
    digits, and at least one letter.
    """
    n = len(text)
    if not (6 <= n <= 11):
        return False
    digits = sum(c.isdigit() for c in text)
    alphas = sum(c.isalpha() for c in text)
    return digits >= 2 and alphas >= 1


def plate_likeness(text: str) -> float:
    """
    Heuristic [0, 1] score that a normalised string is a registration plate,
    derived purely from its composition (length + letter/digit mix).
    """
    n = len(text)
    if not (5 <= n <= 11):
        return 0.0
    has_digit = any(c.isdigit() for c in text)
    has_alpha = any(c.isalpha() for c in text)
    if not (has_digit and has_alpha):
        return 0.35  # all-letters or all-digits: unlikely but not impossible
    # Real plates are a balanced mix; reward 30–70 % digits.
    digit_frac = sum(c.isdigit() for c in text) / n
    balance = 1.0 - abs(0.5 - digit_frac) * 2.0      # 1.0 at 50 %, 0.0 at extremes
    return 0.6 + 0.4 * max(balance, 0.0)


def coerce_to_format(text: str) -> str:
    """
    Best-effort, position-aware correction of OCR confusions to the canonical
    Indian layout ``LL D[D] L[LL] DDDD`` (state, RTO, series, number).

    Only applied when the length is plausible; returns ``text`` unchanged if it
    cannot be confidently reshaped.  This corrects *characters*, never invents
    a different plate.
    """
    n = len(text)
    if not (8 <= n <= 11):
        return text
    # Layout: positions 0-1 = letters, 2-3 = digits, then series letters,
    # last 4 = digits.  Walk from both ends inward.
    chars = list(text)

    def to_alpha(c: str) -> str:
        return _DIGIT_TO_ALPHA.get(c, c) if c.isdigit() else c

    def to_digit(c: str) -> str:
        return _ALPHA_TO_DIGIT.get(c, c) if c.isalpha() else c

    # First two must be letters (state code).
    chars[0], chars[1] = to_alpha(chars[0]), to_alpha(chars[1])
    # Last four must be digits (serial number).
    for i in range(n - 4, n):
        chars[i] = to_digit(chars[i])
    # Positions 2.. up to RTO code: leading ones are digits.
    chars[2] = to_digit(chars[2])
    return "".join(chars)


def score_reading(text: str, ocr_conf: float) -> float:
    """
    Combine OCR confidence with content heuristics into a single ranking score.
    Exact-format matches get a strong bonus so a clean plate beats a noisier one.
    """
    base = ocr_conf * plate_likeness(text)
    if is_valid_indian_plate(text):
        return base * 1.5
    if is_valid_indian_plate(coerce_to_format(text)):
        return base * 1.2
    return base


# ---------------------------------------------------------------------------
# Reading container
# ---------------------------------------------------------------------------

@dataclass
class _Reading:
    text: str
    ocr_conf: float
    plate_bbox: BBox      # full-image coordinates
    score: float


# ---------------------------------------------------------------------------
# Plate-region localisation (classic CV, no extra model)
# ---------------------------------------------------------------------------

def _find_plate_regions(crop) -> list[tuple[int, int, int, int]]:
    """
    Return candidate plate rectangles (x, y, w, h) inside a BGR vehicle crop,
    ordered most-plate-like first.  Uses gradient + morphology + contour
    aspect-ratio filtering — a standard ANPR localisation, no learned weights.
    """
    if not _CV2_AVAILABLE or crop is None or crop.size == 0:
        return []

    h, w = crop.shape[:2]
    crop_area = float(h * w)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Horizontal gradient highlights the dense vertical strokes of plate glyphs.
    grad = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / ch
        area_frac = (cw * ch) / crop_area
        if not (PLATE_MIN_ASPECT <= aspect <= PLATE_MAX_ASPECT):
            continue
        if not (PLATE_MIN_AREA_FRAC <= area_frac <= PLATE_MAX_AREA_FRAC):
            continue
        # Plates sit on the lower 2/3 of a vehicle; prefer lower candidates.
        lower_bonus = (y + ch / 2) / h
        score = area_frac * lower_bonus
        candidates.append((score, (x, y, cw, ch)))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [rect for _, rect in candidates[:4]]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class PlateReader:
    """
    Wraps EasyOCR.  Construct once and reuse — the OCR model is loaded lazily on
    the first ``read_plates`` call so importing this module stays cheap.
    """

    def __init__(self, langs: Sequence[str] = ("en", "bn"), gpu: bool = False,
                 min_conf: float = MIN_OCR_CONF):
        self.langs = list(langs)
        self.gpu = gpu
        self.min_conf = min_conf
        self._reader = None
        try:  # pragma: no cover - depends on optional dep
            import easyocr  # noqa: F401
            self._easyocr = easyocr
        except Exception:  # pragma: no cover
            self._easyocr = None

    @property
    def available(self) -> bool:
        return self._easyocr is not None and _CV2_AVAILABLE

    def _ensure(self) -> None:  # pragma: no cover - needs the model
        if self._reader is None and self._easyocr is not None:
            self._reader = self._easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)

    # -- single crop ------------------------------------------------------

    def _ocr_region(self, region) -> list[tuple[str, float, "np.ndarray"]]:  # pragma: no cover
        """Run OCR on one BGR region, upscaling small crops first."""
        self._ensure()
        if region is None or region.size == 0:
            return []
        h, w = region.shape[:2]
        if max(h, w) < MIN_OCR_DIM:
            s = MIN_OCR_DIM / max(h, w)
            region = cv2.resize(region, (int(w * s), int(h * s)),
                                interpolation=cv2.INTER_CUBIC)
        out = []
        for box, text, conf in self._reader.readtext(region):
            out.append((text, float(conf), box))
        return out

    def read_box(self, image, plate_bbox: BBox) -> tuple[str, float]:  # pragma: no cover
        """
        OCR a pre-localised plate box (e.g. from the YOLO plate detector) and
        return ``(text, confidence)``.  Keeps the highest-confidence reading;
        prefers a normalised Indian-format match when one is present, otherwise
        returns the raw text (so Bengali plates still surface). ``("", 0.0)`` when
        nothing is read.
        """
        if not self.available:
            return "", 0.0
        H, W = image.shape[:2]
        x1 = max(0, int(plate_bbox.x1)); y1 = max(0, int(plate_bbox.y1))
        x2 = min(W, int(plate_bbox.x2)); y2 = min(H, int(plate_bbox.y2))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return "", 0.0

        best_text, best_conf, best_score = "", 0.0, -1.0
        for raw_text, conf, _ in self._ocr_region(crop):
            if conf < self.min_conf:
                continue
            norm = normalize_plate(raw_text)
            # Score normalised Indian-format readings; fall back to raw conf so
            # non-Latin (Bengali) text is still kept as a detection.
            score = score_reading(norm, conf) if norm else conf * 0.5
            display = raw_text.strip()
            if norm and is_acceptable_plate(norm):
                coerced = coerce_to_format(norm)
                display = coerced if (is_valid_indian_plate(coerced)
                                      and not is_valid_indian_plate(norm)) else norm
            if score > best_score:
                best_text, best_conf, best_score = display, conf, score
        return best_text, round(float(best_conf), 4)

    def read_vehicle(self, image, vehicle_bbox: BBox) -> Optional[PlateRecord]:  # pragma: no cover
        """OCR a single vehicle and return its best PlateRecord, or None."""
        if not self.available:
            return None

        H, W = image.shape[:2]
        vx1 = max(0, int(vehicle_bbox.x1)); vy1 = max(0, int(vehicle_bbox.y1))
        vx2 = min(W, int(vehicle_bbox.x2)); vy2 = min(H, int(vehicle_bbox.y2))
        crop = image[vy1:vy2, vx1:vx2]
        if crop.size == 0:
            return None

        # Candidate regions: contour proposals + the whole crop as a fallback so
        # EasyOCR's own detector still gets a chance on hard plates.
        regions = _find_plate_regions(crop)
        regions.append((0, 0, crop.shape[1], crop.shape[0]))

        best: Optional[_Reading] = None
        for (rx, ry, rw, rh) in regions:
            region = crop[ry:ry + rh, rx:rx + rw]
            for raw_text, conf, box in self._ocr_region(region):
                if conf < self.min_conf:
                    continue
                text = normalize_plate(raw_text)
                if not is_acceptable_plate(text):
                    continue
                score = score_reading(text, conf)
                if best is not None and score <= best.score:
                    continue
                # Map the OCR box (region coords) back to full-image coords.
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                pb = BBox(
                    x1=float(vx1 + rx + min(xs)), y1=float(vy1 + ry + min(ys)),
                    x2=float(vx1 + rx + max(xs)), y2=float(vy1 + ry + max(ys)),
                )
                best = _Reading(text=text, ocr_conf=conf, plate_bbox=pb, score=score)

        if best is None:
            return None
        # Apply format coercion only when it yields a valid plate.
        final_text = best.text
        coerced = coerce_to_format(best.text)
        if is_valid_indian_plate(coerced) and not is_valid_indian_plate(best.text):
            final_text = coerced
        return PlateRecord(
            image_id="",  # filled by caller
            vehicle_bbox=vehicle_bbox,
            plate_bbox=best.plate_bbox,
            plate_text=final_text,
            ocr_confidence=round(best.ocr_conf, 4),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_default_reader: Optional[PlateReader] = None


def _get_reader() -> PlateReader:
    global _default_reader
    if _default_reader is None:
        _default_reader = PlateReader()
    return _default_reader


def read_plates(
    image,
    detections: list[DetectionRecord],
    *,
    image_id: Optional[str] = None,
    reader: Optional[PlateReader] = None,
    plate_weights: Optional[str] = None,
    device: str = "cpu",
) -> list[PlateRecord]:
    """
    Detect and read licence plates for the plate-bearing vehicles in a frame.

    Localisation is detection-first:
      * If a trained YOLO plate model is available, each detected plate box is
        emitted as a PlateRecord — with OCR text when readable, or empty text
        (box only) when it isn't.  This is the "license-plate detection system".
      * Otherwise it falls back to the classic-CV localiser + OCR, which only
        emits a record when text is actually read (no speculative boxes).

    Parameters
    ----------
    image         : BGR uint8 frame (numpy array).
    detections    : DetectionRecords for this frame (from detect.py).
    image_id      : Overrides the image_id stamped on each PlateRecord.
    reader        : Optional shared PlateReader (reuse to avoid reloading OCR).
    plate_weights : Optional override for the YOLO plate-model path.
    device        : Inference device for the plate detector.

    Returns
    -------
    list[PlateRecord]   (never raises)
    """
    from detection import plate as plate_det

    rdr = reader or _get_reader()
    use_yolo = plate_det.available(plate_weights)

    if not use_yolo and not rdr.available:
        print("[plate_ocr] No plate model and EasyOCR/cv2 unavailable — skipping "
              "ANPR (add Pipeline/weights/plate.pt or pip install easyocr).",
              file=sys.stderr)
        return []

    plates: list[PlateRecord] = []
    for det in detections:
        if det.class_label not in PLATE_BEARING_CLASSES:
            continue
        iid = image_id or det.image_id

        if use_yolo:
            # Detection-first: localise with the YOLO plate model, OCR best-effort.
            for pb in plate_det.detect_plate_in_vehicle(
                image, det.bbox, weights=plate_weights, device=device,
            ):
                text, conf = rdr.read_box(image, pb) if rdr.available else ("", 0.0)
                plates.append(PlateRecord(
                    image_id=iid, vehicle_bbox=det.bbox, plate_bbox=pb,
                    plate_text=text, ocr_confidence=conf,
                ))
        else:
            rec = rdr.read_vehicle(image, det.bbox)
            if rec is not None:
                rec.image_id = iid
                plates.append(rec)
    return plates


def read_plates_from_file(
    image_path: str | Path,
    detections: list[DetectionRecord],
    **kwargs,
) -> list[PlateRecord]:  # pragma: no cover - thin I/O wrapper
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python is required to read images from disk")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return read_plates(image, detections, image_id=Path(image_path).stem, **kwargs)


# ---------------------------------------------------------------------------
# CLI  —  python -m plate_ocr --image frame.jpg --detections dets.json
# ---------------------------------------------------------------------------

def _dict_to_detection(d: dict) -> DetectionRecord:
    return DetectionRecord(
        image_id=d["image_id"],
        bbox=BBox(**d["bbox"]),
        class_label=VehicleClass(d["class_label"]),
        track_confidence=d.get("track_confidence", 1.0),
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI glue
    p = argparse.ArgumentParser(
        prog="plate_ocr",
        description="Read licence plates for the vehicles detected in an image.",
    )
    p.add_argument("--image", required=True, metavar="PATH", help="Input frame")
    p.add_argument("--detections", required=True, metavar="JSON",
                   help="JSON file: list of DetectionRecord dicts")
    p.add_argument("--out", default=None, metavar="JSON",
                   help="Write PlateRecord list as JSON (default: stdout)")
    args = p.parse_args(argv)

    with open(args.detections, encoding="utf-8") as f:
        detections = [_dict_to_detection(d) for d in json.load(f)]

    plates = read_plates_from_file(args.image, detections)
    payload = [
        {
            "image_id": pr.image_id,
            "vehicle_bbox": pr.vehicle_bbox.__dict__,
            "plate_bbox": pr.plate_bbox.__dict__,
            "plate_text": pr.plate_text,
            "ocr_confidence": pr.ocr_confidence,
        }
        for pr in plates
    ]
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {len(plates)} plate record(s) to {args.out}")
    else:
        print(text)


if __name__ == "__main__":  # pragma: no cover
    main()
