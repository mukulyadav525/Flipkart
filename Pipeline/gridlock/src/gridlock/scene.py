"""Per-camera scene calibration.

Because the cameras are static, the scene geometry (no-parking zones, lane
directions, stop line, traffic-light ROI) is defined once per camera and stored
as JSON in `configs/cameras/<name>.json`. Every violation engine reads from this.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

Point = tuple[float, float]


@dataclass
class NoParkingZone:
    name: str
    polygon: list[Point]


@dataclass
class Lane:
    """A drivable region plus the allowed direction of travel.

    `direction` is a unit vector (dx, dy) in image space pointing the way traffic
    is *supposed* to move. Captured during calibration by clicking tail -> head.
    """

    name: str
    polygon: list[Point]
    direction: Point


@dataclass
class StopLine:
    p1: Point
    p2: Point


@dataclass
class CameraConfig:
    name: str
    frame_size: tuple[int, int] = (1280, 720)
    no_parking: list[NoParkingZone] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    stop_line: StopLine | None = None
    light_roi: tuple[int, int, int, int] | None = None  # x, y, w, h (Phase 2)
    # Scheduled signal cycle for clips with no visible signal head, e.g.
    # [["green", 8], ["yellow", 2], ["red", 8]] (state, seconds). Phase 2.
    signal_cycle: list | None = None

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CameraConfig":
        data = json.loads(Path(path).read_text())
        no_parking = [NoParkingZone(**z) for z in data.get("no_parking", [])]
        lanes = [
            Lane(name=l["name"],
                 polygon=[tuple(p) for p in l["polygon"]],
                 direction=tuple(l["direction"]))
            for l in data.get("lanes", [])
        ]
        for z in no_parking:
            z.polygon = [tuple(p) for p in z.polygon]
        sl = data.get("stop_line")
        stop_line = StopLine(tuple(sl["p1"]), tuple(sl["p2"])) if sl else None
        roi = data.get("light_roi")
        return cls(
            name=data["name"],
            frame_size=tuple(data.get("frame_size", (1280, 720))),
            no_parking=no_parking,
            lanes=lanes,
            stop_line=stop_line,
            light_roi=tuple(roi) if roi else None,
            signal_cycle=data.get("signal_cycle"),
        )
