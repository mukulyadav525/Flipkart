"""Secondary detector — a YOLO model run on cropped regions.

Helmet and seatbelt aren't COCO classes, so they need their own weights. This
wrapper loads a YOLO `.pt` (e.g. a model exported from Roboflow Universe) and
runs it on a crop. It is *optional*: if no weights path is given, `available`
is False and the owning engine no-ops gracefully instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    class_name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # relative to the crop


class SecondaryDetector:
    def __init__(
        self,
        weights: str | None,
        conf: float = 0.35,
        imgsz: int = 256,
        device: str | None = None,
    ):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.model = None
        self.names: dict[int, str] = {}
        if weights:
            self._load(device)

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load(self, device: str | None):
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.weights)
            self.names = self.model.names
            self._device = device
        except Exception as e:  # missing file / bad weights -> stay disabled
            print(f"[secondary] could not load '{self.weights}': {e} (engine disabled)")
            self.model = None

    def detect(self, crop: np.ndarray) -> list[Detection]:
        if self.model is None or crop.size == 0:
            return []
        res = self.model.predict(
            crop, conf=self.conf, imgsz=self.imgsz, verbose=False,
            device=getattr(self, "_device", None),
        )[0]
        out: list[Detection] = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls = int(b.cls[0])
            xy = b.xyxy[0].cpu().numpy()
            out.append(Detection(
                class_name=self.names.get(cls, str(cls)),
                conf=float(b.conf[0]),
                xyxy=(float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])),
            ))
        return out


def crop_box(frame: np.ndarray, xyxy, pad: float = 0.0) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * pad)); y1 = int(max(0, y1 - bh * pad))
    x2 = int(min(w, x2 + bw * pad)); y2 = int(min(h, y2 + bh * pad))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]
