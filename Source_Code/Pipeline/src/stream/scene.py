"""
Per-camera calibration loader.

A fixed camera is calibrated once; its geometry (stop line, lane direction,
no-parking polygon) and signal state live in a JSON config under
``configs/cameras/<name>.json``.  This module parses that file into a
``CameraConfig`` (source + weights + a SceneContext the violation rules consume).

Config schema (all geometry fields optional — a rule that needs a missing field
simply does not fire):

    {
      "name": "demo",
      "source": "samples/demo.mp4",          // file path, rtsp:// url, or "0" (webcam)
      "stride": 2,                              // process every Nth frame
      "detector_weights": "weights/yolov8_idd.pt",
      "pose_weights": "weights/yolov8n-pose.pt",
      "signal_state": "red",                   // red|yellow|green|null (or a signal feed)
      "lane_direction_vector": { "x": 0.0, "y": -1.0 },
      "stop_line_coords": [ {"x":120,"y":610}, {"x":980,"y":610} ],
      "no_parking_zone_polygon": [ {"x":50,"y":400}, ... ],
      "no_parking_sign_visible": false
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from shared.schemas import Point2D, SceneContext, SignalState


def _point(d: Any) -> Point2D:
    return Point2D(x=float(d["x"]), y=float(d["y"]))


def scene_from_dict(d: dict, *, image_id: Optional[str] = None) -> SceneContext:
    """Build a SceneContext from a parsed config / sidecar dict."""
    stop_line = None
    if d.get("stop_line_coords"):
        pts = [_point(p) for p in d["stop_line_coords"]]
        if len(pts) >= 2:
            stop_line = (pts[0], pts[1])

    signal = None
    if d.get("signal_state"):
        signal = SignalState(str(d["signal_state"]).lower())

    polygon = None
    if d.get("no_parking_zone_polygon"):
        polygon = [_point(p) for p in d["no_parking_zone_polygon"]]

    lane = _point(d["lane_direction_vector"]) if d.get("lane_direction_vector") else None
    lane_zone = ([_point(p) for p in d["lane_zone_polygon"]]
                 if d.get("lane_zone_polygon") else None)

    return SceneContext(
        image_id=image_id or d.get("name") or d.get("image_id") or "camera",
        lane_direction_vector=lane,
        lane_zone_polygon=lane_zone,
        stop_line_coords=stop_line,
        signal_state=signal,
        no_parking_zone_polygon=polygon,
        no_parking_sign_visible=bool(d.get("no_parking_sign_visible", False)),
    )


@dataclass
class CameraConfig:
    name: str
    source: str
    scene: SceneContext
    stride: int = 2
    detector_weights: Optional[str] = None
    pose_weights: Optional[str] = None

    @property
    def has_geometry(self) -> bool:
        s = self.scene
        return any([
            s.lane_direction_vector, s.stop_line_coords,
            s.no_parking_zone_polygon, s.no_parking_sign_visible,
        ])


def load_camera_config(path: str | Path) -> CameraConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Camera config not found: {path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    if "source" not in d:
        raise ValueError(f"Camera config {path} is missing required 'source' field")
    name = d.get("name", path.stem)
    return CameraConfig(
        name=name,
        source=str(d["source"]),
        scene=scene_from_dict(d, image_id=name),
        stride=int(d.get("stride", 2)),
        detector_weights=d.get("detector_weights"),
        pose_weights=d.get("pose_weights"),
    )
