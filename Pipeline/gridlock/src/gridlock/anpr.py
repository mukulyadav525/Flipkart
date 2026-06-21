"""ANPR — read a licence plate from a vehicle crop.

Uses EasyOCR, whose built-in text *detector* localises the plate inside the crop,
so we don't need a separate plate-detection model. We then keep the most
plate-like string (alphanumeric, right length, mix of letters+digits).

Optional: if EasyOCR isn't installed, `available` is False and callers skip ANPR.
Plates in distant CCTV are often too small to read — treat results as best-effort.
"""

from __future__ import annotations

import re

import cv2
import numpy as np


class PlateReader:
    def __init__(self, langs=("en",), gpu: bool = False, min_conf: float = 0.2,
                 plate_detector=None):
        self.langs = list(langs)
        self.gpu = gpu
        self.min_conf = min_conf
        # Optional SecondaryDetector with licence-plate weights. When given, we
        # localise the plate inside the vehicle crop and OCR only that region —
        # this stops OCR from grabbing unrelated text (bus/shop signage).
        self.plate_detector = plate_detector
        self._reader = None
        try:
            import easyocr  # noqa: F401
            self._easyocr = easyocr
        except Exception:
            self._easyocr = None

    @property
    def available(self) -> bool:
        return self._easyocr is not None

    def _ensure(self):
        if self._reader is None and self._easyocr is not None:
            # verbose=False keeps the model-download chatter down on first use.
            self._reader = self._easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)

    def read_plate(self, crop: np.ndarray) -> tuple[str, float] | None:
        if not self.available or crop is None or crop.size == 0:
            return None
        self._ensure()

        # If a plate detector is available, narrow the crop to the plate first.
        if self.plate_detector is not None and self.plate_detector.available:
            dets = self.plate_detector.detect(crop)
            if dets:
                best = max(dets, key=lambda d: d.conf)
                x1, y1, x2, y2 = (int(v) for v in best.xyxy)
                sub = crop[max(0, y1):y2, max(0, x1):x2]
                if sub.size:
                    crop = sub

        # Upscale small crops — plates are tiny and OCR needs the detail.
        h, w = crop.shape[:2]
        if max(h, w) < 240:
            s = 240 / max(h, w)
            crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)

        best = None
        for _box, text, conf in self._reader.readtext(crop):
            if conf < self.min_conf:
                continue
            cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
            if not (5 <= len(cleaned) <= 11):
                continue
            score = conf * self._plate_likeness(cleaned)
            if best is None or score > best[2]:
                best = (cleaned, float(conf), score)
        return (best[0], best[1]) if best else None

    @staticmethod
    def _plate_likeness(s: str) -> float:
        has_digit = any(c.isdigit() for c in s)
        has_alpha = any(c.isalpha() for c in s)
        return 1.0 if (has_digit and has_alpha) else 0.5  # real plates mix both
