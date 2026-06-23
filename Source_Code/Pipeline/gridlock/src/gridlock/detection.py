"""YOLO detection + ByteTrack tracking wrapper.

We use Ultralytics' built-in tracker so ByteTrack state persists across frames
via `persist=True`. This wrapper keeps the rest of the codebase decoupled from
the Ultralytics result format and emits plain `Track` objects.
"""

from __future__ import annotations

import numpy as np

from .config import PipelineConfig
from .types import Track


def _auto_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Detector:
    def __init__(self, cfg: PipelineConfig):
        from ultralytics import YOLO  # imported lazily so config import stays cheap

        self.cfg = cfg
        self.device = cfg.device or _auto_device()
        self.model = YOLO(cfg.model_weights)
        self.names: dict[int, str] = self.model.names

    def track(self, frame: np.ndarray) -> list[Track]:
        """Run detection + tracking on one frame, return tracked objects."""
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.cfg.tracker,
            classes=self.cfg.classes,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._to_tracks(results[0])

    def _to_tracks(self, result) -> list[Track]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            return []  # nothing tracked this frame

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        cls = boxes.cls.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        tracks: list[Track] = []
        for box, tid, c, cf in zip(xyxy, ids, cls, confs):
            tracks.append(
                Track(
                    track_id=int(tid),
                    class_id=int(c),
                    class_name=self.names.get(int(c), str(c)),
                    conf=float(cf),
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        return tracks
