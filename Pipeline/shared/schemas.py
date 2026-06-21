from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Primitive geometry types
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Point2D:
    x: float
    y: float


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VehicleClass(str, Enum):
    car = "car"
    bike = "bike"          # motorcycle / scooter (helmet law applies)
    bicycle = "bicycle"    # pedal cycle / cycle-rickshaw (no helmet requirement)
    auto = "auto"          # three-wheeler / autorickshaw
    bus = "bus"
    truck = "truck"
    pedestrian = "pedestrian"


class SignalState(str, Enum):
    red = "red"
    yellow = "yellow"
    green = "green"
    unknown = "unknown"


class ViolationType(str, Enum):
    helmet = "helmet"
    seatbelt = "seatbelt"
    triple_riding = "triple_riding"
    wrong_side = "wrong_side"
    stop_line = "stop_line"
    red_light = "red_light"
    illegal_parking = "illegal_parking"


# ---------------------------------------------------------------------------
# Core interchange records
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    """Output of the detection module for a single detected object."""
    image_id: str
    bbox: BBox
    class_label: VehicleClass
    track_confidence: float
    # Optional: 2-D skeleton keypoints used for helmet / seatbelt checks.
    # List of (x, y) points in pixel space; ordering follows the model's
    # keypoint convention (document in detection/__init__.py).
    pose_keypoints: Optional[list[Point2D]] = None


@dataclass
class PlateRecord:
    """Output of the plate-OCR module for a single licence plate."""
    image_id: str
    vehicle_bbox: BBox
    plate_bbox: BBox
    plate_text: str
    ocr_confidence: float


@dataclass
class SceneContext:
    """
    Per-image geometry and signal annotations loaded from a JSON sidecar file.

    All fields except image_id are optional because manual annotation is
    partial — only the geometry relevant to a given scene is filled in.
    Missing fields must be treated as "unknown / not applicable" by all
    violation modules; they must NOT be treated as "violation confirmed."
    """
    image_id: str

    # Unit vector pointing in the legal direction of travel for the lane
    # the camera is watching.  Used by the wrong-side module.
    lane_direction_vector: Optional[Point2D] = None

    # Optional polygon bounding the lane that lane_direction_vector applies to.
    # On a two-way road a single global direction would wrongly flag the
    # opposing (legal) lane; scoping wrong-side to this polygon prevents that.
    lane_zone_polygon: Optional[list[Point2D]] = None

    # Two points defining the stop line in pixel space.
    stop_line_coords: Optional[tuple[Point2D, Point2D]] = None

    # Traffic-signal state at capture time.
    signal_state: Optional[SignalState] = None

    # Polygon (≥3 points) marking a no-parking zone in the image.
    no_parking_zone_polygon: Optional[list[Point2D]] = None

    # True when a "No Parking" sign is visible in the frame.
    no_parking_sign_visible: bool = False


@dataclass
class ViolationRecord:
    """A single confirmed or candidate traffic violation."""
    image_id: str
    violation_type: ViolationType
    confidence: float
    # Human-readable explanation of the rule and evidence that triggered it.
    rule_trace: str
    # image_ids or track IDs of DetectionRecords that contributed.
    related_detection_ids: list[str] = field(default_factory=list)
    related_plate_text: Optional[str] = None


@dataclass
class EvidenceRecord:
    """Packaged evidence bundle ready for export or human review."""
    violation_record: ViolationRecord
    annotated_image_path: str
    timestamp: str          # ISO-8601
    plate_text: str
    plate_confidence: float
