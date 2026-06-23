"""
Violation rule engine.

Each public function takes DetectionRecord list (and optionally SceneContext)
and returns a list of ViolationRecord objects — never raises, never returns None.

Implemented rules
-----------------
  check_helmet        — bike rider + absent helmet-class detection near head
  check_seatbelt      — car occupant + absent seatbelt-class detection at torso
  check_triple_riding — bike bbox + >2 persons overlapping it

Geometry rules (returns "insufficient_scene_context" when required fields absent)
-----------------
  check_wrong_side    — needs SceneContext.lane_direction_vector + per-vehicle
                        motion vectors; fires when travel opposes the lane.
  check_stop_line     — needs SceneContext.stop_line_coords; fires when a vehicle's
                        ground point has crossed the line and the signal is not green.
  check_red_light     — needs SceneContext.signal_state==red + stop_line_coords;
                        fires when a vehicle crossed the line on red.
  check_illegal_parking — needs SceneContext.no_parking_zone_polygon (point-in-polygon)
                        or no_parking_sign_visible (any vehicle in frame).

COCO keypoint indices used (documented in detection/__init__.py):
  Head    : 0 (nose), 1 (left_eye), 2 (right_eye), 3 (left_ear), 4 (right_ear)
  Torso   : 5 (left_shoulder), 6 (right_shoulder), 11 (left_hip), 12 (right_hip)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from shared.schemas import (
    BBox,
    DetectionRecord,
    Point2D,
    SceneContext,
    SignalState,
    VehicleClass,
    ViolationRecord,
    ViolationType,
)
from config import THRESHOLDS

# ---------------------------------------------------------------------------
# Internal result type — carries a status so stubs can signal missing context
# ---------------------------------------------------------------------------

class RuleStatus(str, Enum):
    fired    = "fired"               # violation detected, ViolationRecord emitted
    clear    = "clear"               # rule evaluated, no violation
    insufficient_scene_context = "insufficient_scene_context"  # stub: data absent
    skipped  = "skipped"             # detection precondition not met (e.g. no bikes)


@dataclass
class RuleResult:
    status: RuleStatus
    violation: Optional[ViolationRecord] = None
    # Human-readable reason for clear / insufficient / skipped outcomes
    reason: str = ""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two bounding boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner` bbox area that lies inside `outer`."""
    ix1 = max(inner.x1, outer.x1)
    iy1 = max(inner.y1, outer.y1)
    ix2 = min(inner.x2, outer.x2)
    iy2 = min(inner.y2, outer.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    inner_area = (inner.x2 - inner.x1) * (inner.y2 - inner.y1)
    return inter / inner_area if inner_area > 0 else 0.0


def _keypoint_bbox(keypoints: list[Point2D], indices: list[int], padding: float = 20.0) -> Optional[BBox]:
    """
    Build a bounding box around a subset of keypoints.

    Returns None if all requested keypoints are at (0, 0) — which YOLOv8-pose
    emits for invisible / occluded keypoints.
    """
    pts = [keypoints[i] for i in indices if i < len(keypoints)]
    visible = [p for p in pts if not (p.x == 0.0 and p.y == 0.0)]
    if not visible:
        return None
    xs = [p.x for p in visible]
    ys = [p.y for p in visible]
    return BBox(
        x1=min(xs) - padding,
        y1=min(ys) - padding,
        x2=max(xs) + padding,
        y2=max(ys) + padding,
    )


def _detection_id(rec: DetectionRecord) -> str:
    """Stable ID string for a detection — used in related_detection_ids."""
    return f"{rec.image_id}:{rec.class_label.value}:{rec.bbox.x1:.0f},{rec.bbox.y1:.0f}"


# COCO keypoint index groups
_HEAD_KP_INDICES   = [0, 1, 2, 3, 4]          # nose, eyes, ears
_TORSO_KP_INDICES  = [5, 6, 11, 12]           # shoulders, hips


# ---------------------------------------------------------------------------
# Rule 1 — Helmet
# ---------------------------------------------------------------------------

# How much of the head-bbox must a candidate helmet detection overlap to count
_HELMET_OVERLAP_THRESHOLD = 0.30
# Minimum IoU between two bike detections to consider them the same vehicle
_SAME_VEHICLE_IOU = 0.50


def _head_on_bike(bike_bbox: BBox, head_bbox: BBox, x_pad: float = 0.3, up_factor: float = 1.0) -> bool:
    """True when a head box's centre sits on/above a two-wheeler box.

    Riders sit on top of the bike, so a rider head lies within the bike's
    horizontal span (widened by x_pad) and from just above the box top down to
    its bottom.  This filters out heads belonging to pedestrians beside the bike.
    """
    hx = (head_bbox.x1 + head_bbox.x2) / 2.0
    hy = (head_bbox.y1 + head_bbox.y2) / 2.0
    bw = bike_bbox.x2 - bike_bbox.x1
    bh = bike_bbox.y2 - bike_bbox.y1
    pad = bw * x_pad
    if not (bike_bbox.x1 - pad <= hx <= bike_bbox.x2 + pad):
        return False
    return (bike_bbox.y1 - bh * up_factor) <= hy <= bike_bbox.y2


# Proxy fires (no helmet-model evidence either way) are capped below the
# auto-process cutoff so they always land in the human-review queue, never
# auto-confirmed without an actual no-helmet detection.
_HELMET_PROXY_CONF_CAP = 0.84


def check_helmet(
    detections: list[DetectionRecord],
    *,
    helmet_detections: list[DetectionRecord] | None = None,
    nohelmet_detections: list[DetectionRecord] | None = None,
    assume_nohelmet_on_motorcycle: bool = False,
) -> list[RuleResult]:
    """
    Hybrid helmet check — applies to motorcycles (``VehicleClass.bike``) only;
    bicycles and three-wheelers are separate classes and never reach this rule.

    Per motorcycle:
      - a model-detected NO-HELMET head sits on it -> fired (conf = head conf)
      - else a model-detected HELMET head sits on it -> clear
      - else (helmet model gave no verdict):
          * assume_nohelmet_on_motorcycle -> fired as a review candidate
            (confidence capped < AUTO_PROCESS_CUTOFF, so a human confirms)
          * otherwise -> skipped

    The fallback exists because the trained helmet model under-detects no-helmet
    heads on out-of-domain footage; scoping it to motorcycles keeps cyclists and
    rickshaws clear while still surfacing real bare-headed riders for review.

    Parameters
    ----------
    detections          : All DetectionRecords for the image.
    helmet_detections   : Heads the model classified as wearing a helmet.
    nohelmet_detections : Heads the model classified as *not* wearing a helmet.
    assume_nohelmet_on_motorcycle : enable the review-candidate fallback.
    """
    results: list[RuleResult] = []
    threshold = THRESHOLDS.get(ViolationType.helmet)
    helmet_detections = helmet_detections or []
    nohelmet_detections = nohelmet_detections or []

    bikes = [d for d in detections if d.class_label == VehicleClass.bike]
    if not bikes:
        return [RuleResult(status=RuleStatus.skipped, reason="no motorcycle detections in frame")]

    for bike in bikes:
        nh = next((h for h in nohelmet_detections if _head_on_bike(bike.bbox, h.bbox)), None)
        if nh is not None:
            conf = min(nh.track_confidence, 1.0)
            if conf < threshold:
                results.append(RuleResult(
                    status=RuleStatus.clear,
                    reason=f"no-helmet head conf {conf:.2f} below threshold {threshold:.2f} — skipped",
                ))
                continue
            results.append(RuleResult(
                status=RuleStatus.fired,
                violation=ViolationRecord(
                    image_id=bike.image_id,
                    violation_type=ViolationType.helmet,
                    confidence=conf,
                    rule_trace=(
                        f"Motorcycle detected at bbox=({bike.bbox.x1:.0f},{bike.bbox.y1:.0f},"
                        f"{bike.bbox.x2:.0f},{bike.bbox.y2:.0f}). "
                        f"Helmet model detected a NO-HELMET head (conf={conf:.2f}) at "
                        f"({nh.bbox.x1:.0f},{nh.bbox.y1:.0f}) on the rider. "
                        f"Rule: a rider on a motorcycle must wear a helmet."
                    ),
                    related_detection_ids=[_detection_id(bike)],
                ),
            ))
            continue

        hh = next((h for h in helmet_detections if _head_on_bike(bike.bbox, h.bbox)), None)
        if hh is not None:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"helmet head detected on rider of motorcycle at "
                       f"({bike.bbox.x1:.0f},{bike.bbox.y1:.0f})",
            ))
            continue

        # No helmet-model verdict for this motorcycle.
        if not assume_nohelmet_on_motorcycle:
            results.append(RuleResult(
                status=RuleStatus.skipped,
                reason=f"motorcycle at ({bike.bbox.x1:.0f},{bike.bbox.y1:.0f}): "
                       "no helmet-model verdict — cannot assess",
            ))
            continue

        conf = min(bike.track_confidence, _HELMET_PROXY_CONF_CAP)
        if conf < threshold:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"motorcycle conf {conf:.2f} below threshold {threshold:.2f} — skipped",
            ))
            continue
        results.append(RuleResult(
            status=RuleStatus.fired,
            violation=ViolationRecord(
                image_id=bike.image_id,
                violation_type=ViolationType.helmet,
                confidence=conf,
                rule_trace=(
                    f"Motorcycle detected (conf={bike.track_confidence:.2f}) at "
                    f"bbox=({bike.bbox.x1:.0f},{bike.bbox.y1:.0f},"
                    f"{bike.bbox.x2:.0f},{bike.bbox.y2:.0f}). "
                    f"Helmet model gave no helmet/no-helmet verdict for the rider; "
                    f"flagged as a REVIEW CANDIDATE (confidence capped at "
                    f"{_HELMET_PROXY_CONF_CAP}). A human reviewer confirms before action."
                ),
                related_detection_ids=[_detection_id(bike)],
            ),
        ))

    return results


# ---------------------------------------------------------------------------
# Rule 2 — Seatbelt
# ---------------------------------------------------------------------------

_SEATBELT_OVERLAP_THRESHOLD = 0.25
# Fraction of a person's bbox that must overlap a car bbox to count as occupant
_OCCUPANT_OVERLAP_THRESHOLD = 0.40


def check_seatbelt(
    detections: list[DetectionRecord],
    *,
    seatbelt_detections: list[DetectionRecord] | None = None,
) -> list[RuleResult]:
    """
    For every car detection, find pedestrian/person detections whose bounding
    box largely overlaps the car (i.e. occupants visible through windscreen /
    side window).  For each occupant with torso keypoints, check that a
    seatbelt-class detection covers the torso region.

    Parameters
    ----------
    detections          : All DetectionRecords for the image.
    seatbelt_detections : Optional seatbelt-class detections from an attribute
                          classifier.  If None or empty, no seatbelt assumed.
    """
    results: list[RuleResult] = []
    threshold = THRESHOLDS.get(ViolationType.seatbelt)

    cars = [d for d in detections if d.class_label == VehicleClass.car]
    people = [d for d in detections if d.class_label == VehicleClass.pedestrian]

    if not cars:
        return [RuleResult(status=RuleStatus.skipped, reason="no car detections in frame")]

    for car in cars:
        # Find occupants: pedestrian detections that overlap significantly with the car
        occupants = [
            p for p in people
            if _overlap_ratio(p.bbox, car.bbox) >= _OCCUPANT_OVERLAP_THRESHOLD
        ]

        if not occupants:
            results.append(RuleResult(
                status=RuleStatus.skipped,
                reason=f"car at ({car.bbox.x1:.0f},{car.bbox.y1:.0f}): "
                       "no overlapping person detections — occupants not visible",
            ))
            continue

        for occupant in occupants:
            if occupant.pose_keypoints is None:
                results.append(RuleResult(
                    status=RuleStatus.skipped,
                    reason=f"occupant at ({occupant.bbox.x1:.0f},{occupant.bbox.y1:.0f}) "
                           "has no pose keypoints — cannot assess seatbelt",
                ))
                continue

            torso_bbox = _keypoint_bbox(occupant.pose_keypoints, _TORSO_KP_INDICES, padding=10.0)
            if torso_bbox is None:
                results.append(RuleResult(
                    status=RuleStatus.skipped,
                    reason=f"occupant at ({occupant.bbox.x1:.0f},{occupant.bbox.y1:.0f}): "
                           "torso keypoints all occluded — cannot assess seatbelt",
                ))
                continue

            seatbelt_found = False
            if seatbelt_detections:
                for sb in seatbelt_detections:
                    if _overlap_ratio(sb.bbox, torso_bbox) >= _SEATBELT_OVERLAP_THRESHOLD:
                        seatbelt_found = True
                        break

            if seatbelt_found:
                results.append(RuleResult(
                    status=RuleStatus.clear,
                    reason=f"seatbelt detected at torso of occupant "
                           f"({occupant.bbox.x1:.0f},{occupant.bbox.y1:.0f}) "
                           f"in car ({car.bbox.x1:.0f},{car.bbox.y1:.0f})",
                ))
            else:
                conf = min(
                    car.track_confidence * occupant.track_confidence, 1.0
                )
                if conf < threshold:
                    results.append(RuleResult(
                        status=RuleStatus.clear,
                        reason=f"combined conf {conf:.2f} below seatbelt threshold "
                               f"{threshold:.2f} — skipped",
                    ))
                    continue

                results.append(RuleResult(
                    status=RuleStatus.fired,
                    violation=ViolationRecord(
                        image_id=car.image_id,
                        violation_type=ViolationType.seatbelt,
                        confidence=conf,
                        rule_trace=(
                            f"Car detected (conf={car.track_confidence:.2f}) at "
                            f"bbox=({car.bbox.x1:.0f},{car.bbox.y1:.0f},"
                            f"{car.bbox.x2:.0f},{car.bbox.y2:.0f}). "
                            f"Occupant detected (conf={occupant.track_confidence:.2f}) with "
                            f"{_occupant_overlap_pct(occupant.bbox, car.bbox):.0f}% overlap inside car. "
                            f"Torso keypoints visible at approx "
                            f"({torso_bbox.x1+10:.0f},{torso_bbox.y1+10:.0f}). "
                            f"No seatbelt-class detection overlaps torso region "
                            f"(overlap threshold={_SEATBELT_OVERLAP_THRESHOLD}). "
                            f"Rule: seatbelt diagonal strap must be detectable across "
                            f"shoulder-to-hip torso keypoint region."
                        ),
                        related_detection_ids=[_detection_id(car), _detection_id(occupant)],
                    ),
                ))

    return results


def _occupant_overlap_pct(person_bbox: BBox, car_bbox: BBox) -> float:
    return _overlap_ratio(person_bbox, car_bbox) * 100


# ---------------------------------------------------------------------------
# Rule 3 — Triple riding
# ---------------------------------------------------------------------------

_TRIPLE_RIDING_PERSON_OVERLAP = 0.35   # fraction of person bbox inside bike bbox
_MAX_RIDERS = 2                        # legal limit on a two-wheeler


def check_triple_riding(detections: list[DetectionRecord]) -> list[RuleResult]:
    """
    For each bike detection, count how many person detections overlap its
    bounding box.  Fire a violation if the count exceeds _MAX_RIDERS.

    People sitting on the bike are identified by requiring that at least
    _TRIPLE_RIDING_PERSON_OVERLAP fraction of their bbox falls inside the
    bike bbox — this filters bystanders standing next to the vehicle.
    """
    results: list[RuleResult] = []
    threshold = THRESHOLDS.get(ViolationType.triple_riding)

    bikes = [d for d in detections if d.class_label == VehicleClass.bike]
    people = [d for d in detections if d.class_label == VehicleClass.pedestrian]

    if not bikes:
        return [RuleResult(status=RuleStatus.skipped, reason="no bike detections in frame")]

    for bike in bikes:
        riders = [
            p for p in people
            if _overlap_ratio(p.bbox, bike.bbox) >= _TRIPLE_RIDING_PERSON_OVERLAP
        ]
        rider_count = len(riders)

        if rider_count <= _MAX_RIDERS:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"bike at ({bike.bbox.x1:.0f},{bike.bbox.y1:.0f}): "
                       f"{rider_count} rider(s) detected — within legal limit of {_MAX_RIDERS}",
            ))
            continue

        conf = min(bike.track_confidence, 1.0)
        if conf < threshold:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"triple-riding candidate: bike conf {conf:.2f} below "
                       f"threshold {threshold:.2f} — skipped",
            ))
            continue

        rider_ids = [_detection_id(r) for r in riders]
        results.append(RuleResult(
            status=RuleStatus.fired,
            violation=ViolationRecord(
                image_id=bike.image_id,
                violation_type=ViolationType.triple_riding,
                confidence=conf,
                rule_trace=(
                    f"Bike detected (conf={bike.track_confidence:.2f}) at "
                    f"bbox=({bike.bbox.x1:.0f},{bike.bbox.y1:.0f},"
                    f"{bike.bbox.x2:.0f},{bike.bbox.y2:.0f}). "
                    f"{rider_count} person detection(s) overlap the bike bbox by "
                    f">={_TRIPLE_RIDING_PERSON_OVERLAP*100:.0f}% (rider overlap threshold). "
                    f"Legal maximum is {_MAX_RIDERS} rider(s) on a two-wheeler. "
                    f"Rule fired: {rider_count} > {_MAX_RIDERS}."
                ),
                related_detection_ids=[_detection_id(bike)] + rider_ids,
            ),
        ))

    return results


# ---------------------------------------------------------------------------
# Geometry primitives shared by the scene-context rules
# ---------------------------------------------------------------------------

# Vehicle classes that can commit road-position violations (people excluded).
_VEHICLE_CLASSES: frozenset[VehicleClass] = frozenset({
    VehicleClass.car, VehicleClass.bus, VehicleClass.truck, VehicleClass.bike,
    VehicleClass.auto, VehicleClass.bicycle,
})

# A point must be at least this many pixels past the stop line (in the travel
# direction) to count as "crossed" — a small tolerance against bbox jitter.
_STOP_LINE_MARGIN_PX = 2.0

# Wrong-side fires only when travel clearly opposes the lane vector: the cosine
# similarity between motion and lane direction must be below this (negative).
_WRONG_SIDE_COS_THRESHOLD = -0.30
# Ignore near-stationary vehicles whose motion magnitude is below this (px).
_MIN_MOTION_MAGNITUDE = 1.0
# A vehicle counts as "parked" only if its motion over the tracker lookback
# window stays below this (px); above it the vehicle is moving, not parked.
_PARKED_MAX_MOTION = 15.0


def _bottom_center(b: BBox) -> Point2D:
    """Point where the vehicle meets the road — robust ground reference."""
    return Point2D(x=(b.x1 + b.x2) / 2.0, y=b.y2)


def _centroid(b: BBox) -> Point2D:
    return Point2D(x=(b.x1 + b.x2) / 2.0, y=(b.y1 + b.y2) / 2.0)


def _point_in_polygon(p: Point2D, polygon: list[Point2D]) -> bool:
    """Ray-casting point-in-polygon test (handles concave polygons)."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        if ((yi > p.y) != (yj > p.y)) and \
           (p.x < (xj - xi) * (p.y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _downstream_unit(
    a: Point2D, b: Point2D, lane_dir: Optional[Point2D]
) -> tuple[float, float]:
    """
    Unit vector pointing across the stop line in the legal direction of travel.

    Uses lane_direction_vector when available; otherwise falls back to the line
    normal oriented toward the top of the frame (decreasing y) — the usual
    "into the intersection" direction for a camera facing oncoming traffic.
    """
    if lane_dir is not None and (lane_dir.x or lane_dir.y):
        norm = math.hypot(lane_dir.x, lane_dir.y) or 1.0
        return lane_dir.x / norm, lane_dir.y / norm
    # Normal to the line segment, flipped to point upward (−y).
    lx, ly = (b.x - a.x), (b.y - a.y)
    nx, ny = -ly, lx                       # rotate +90°
    if ny > 0:                             # ensure it points toward smaller y
        nx, ny = -nx, -ny
    norm = math.hypot(nx, ny) or 1.0
    return nx / norm, ny / norm


def _signed_crossing(
    p: Point2D, a: Point2D, b: Point2D, lane_dir: Optional[Point2D]
) -> float:
    """
    Signed distance (px) of point ``p`` past the stop line in the downstream
    direction.  Positive ⇒ the point has crossed into the intersection.
    """
    mx, my = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
    dx, dy = _downstream_unit(a, b, lane_dir)
    return (p.x - mx) * dx + (p.y - my) * dy


# ---------------------------------------------------------------------------
# Rule 4 — Wrong-side driving
# ---------------------------------------------------------------------------

def check_wrong_side(
    detections: list[DetectionRecord],
    scene: SceneContext,
    *,
    motion: Optional[dict[str, Point2D]] = None,
) -> list[RuleResult]:
    """
    Fire when a vehicle's motion vector opposes the lane's legal travel direction.

    Requires SceneContext.lane_direction_vector AND a per-vehicle ``motion``
    map (detection-id → displacement Point2D, e.g. from an inter-frame tracker
    or optical flow).  Direction of travel genuinely cannot be recovered from a
    single still frame, so when motion is unavailable the rule reports
    ``skipped`` rather than guessing — it never fabricates a violation.
    """
    if scene.lane_direction_vector is None:
        return [RuleResult(
            status=RuleStatus.insufficient_scene_context,
            reason=(
                "wrong_side rule requires SceneContext.lane_direction_vector "
                "but it is None for image_id='{}'. Annotate the sidecar JSON "
                "with a unit vector for the legal direction of travel."
                .format(scene.image_id)
            ),
        )]

    if not motion:
        return [RuleResult(
            status=RuleStatus.skipped,
            reason=("wrong_side needs per-vehicle motion vectors (inter-frame "
                    "displacement); none supplied for this frame."),
        )]

    lane = scene.lane_direction_vector
    lane_norm = math.hypot(lane.x, lane.y) or 1.0
    lx, ly = lane.x / lane_norm, lane.y / lane_norm
    threshold = THRESHOLDS.get(ViolationType.wrong_side)

    results: list[RuleResult] = []
    vehicles = [d for d in detections if d.class_label in _VEHICLE_CLASSES]
    if not vehicles:
        return [RuleResult(status=RuleStatus.skipped,
                           reason="no vehicle detections in frame")]

    for v in vehicles:
        # Scope to the monitored lane: a single lane direction must not be
        # applied to vehicles in the opposing (legal) lane of a two-way road.
        if scene.lane_zone_polygon is not None and \
           not _point_in_polygon(_bottom_center(v.bbox), scene.lane_zone_polygon):
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"vehicle at ({v.bbox.x1:.0f},{v.bbox.y1:.0f}) is outside "
                       "the monitored lane zone — direction rule not applicable",
            ))
            continue

        mv = motion.get(_detection_id(v))
        if mv is None:
            results.append(RuleResult(
                status=RuleStatus.skipped,
                reason=f"no motion vector for vehicle at "
                       f"({v.bbox.x1:.0f},{v.bbox.y1:.0f})",
            ))
            continue
        mag = math.hypot(mv.x, mv.y)
        if mag < _MIN_MOTION_MAGNITUDE:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"vehicle near-stationary (|motion|={mag:.1f}px) — "
                       "direction undetermined",
            ))
            continue
        cos_sim = (mv.x * lx + mv.y * ly) / mag
        if cos_sim > _WRONG_SIDE_COS_THRESHOLD:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"vehicle travel aligns with lane "
                       f"(cosine={cos_sim:+.2f})",
            ))
            continue

        conf = min(v.track_confidence, 1.0)
        if conf < threshold:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"wrong-side candidate below threshold "
                       f"({conf:.2f} < {threshold:.2f})",
            ))
            continue

        results.append(RuleResult(
            status=RuleStatus.fired,
            violation=ViolationRecord(
                image_id=v.image_id,
                violation_type=ViolationType.wrong_side,
                confidence=conf,
                rule_trace=(
                    f"Vehicle ({v.class_label.value}, conf={v.track_confidence:.2f}) "
                    f"at bbox=({v.bbox.x1:.0f},{v.bbox.y1:.0f},"
                    f"{v.bbox.x2:.0f},{v.bbox.y2:.0f}). "
                    f"Motion vector ({mv.x:+.1f},{mv.y:+.1f}) opposes lane "
                    f"direction ({lx:+.2f},{ly:+.2f}): cosine similarity "
                    f"{cos_sim:+.2f} < {_WRONG_SIDE_COS_THRESHOLD}. "
                    "Rule fired: vehicle travelling against legal flow."
                ),
                related_detection_ids=[_detection_id(v)],
            ),
        ))
    return results


# ---------------------------------------------------------------------------
# Rule 5 — Stop-line crossing
# ---------------------------------------------------------------------------

def check_stop_line(
    detections: list[DetectionRecord],
    scene: SceneContext,
) -> list[RuleResult]:
    """
    Fire when a vehicle's ground point has crossed the stop line while the
    signal is *not* green.

    Requires SceneContext.stop_line_coords.  signal_state is optional: a known
    green signal clears the violation; red / yellow / unknown leave it active
    (the painted stop line is binding whenever traffic must hold).
    """
    if scene.stop_line_coords is None:
        return [RuleResult(
            status=RuleStatus.insufficient_scene_context,
            reason=(
                "stop_line rule requires SceneContext.stop_line_coords but it "
                "is None for image_id='{}'. Annotate the sidecar JSON with two "
                "pixel-space points defining the stop line.".format(scene.image_id)
            ),
        )]

    if scene.signal_state == SignalState.green:
        return [RuleResult(
            status=RuleStatus.clear,
            reason="signal is green — crossing the stop line is legal",
        )]

    a, b = scene.stop_line_coords
    threshold = THRESHOLDS.get(ViolationType.stop_line)
    results: list[RuleResult] = []
    vehicles = [d for d in detections if d.class_label in _VEHICLE_CLASSES]
    if not vehicles:
        return [RuleResult(status=RuleStatus.skipped,
                           reason="no vehicle detections in frame")]

    for v in vehicles:
        ref = _bottom_center(v.bbox)
        crossing = _signed_crossing(ref, a, b, scene.lane_direction_vector)
        if crossing <= _STOP_LINE_MARGIN_PX:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"vehicle at ({v.bbox.x1:.0f},{v.bbox.y1:.0f}) is "
                       f"{-crossing:.0f}px short of / on the stop line",
            ))
            continue

        conf = min(v.track_confidence, 1.0)
        if conf < threshold:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"stop-line candidate below threshold "
                       f"({conf:.2f} < {threshold:.2f})",
            ))
            continue

        sig = scene.signal_state.value if scene.signal_state else "unknown"
        results.append(RuleResult(
            status=RuleStatus.fired,
            violation=ViolationRecord(
                image_id=v.image_id,
                violation_type=ViolationType.stop_line,
                confidence=conf,
                rule_trace=(
                    f"Vehicle ({v.class_label.value}, conf={v.track_confidence:.2f}) "
                    f"ground point ({ref.x:.0f},{ref.y:.0f}) is {crossing:.0f}px "
                    f"past the stop line "
                    f"({a.x:.0f},{a.y:.0f})→({b.x:.0f},{b.y:.0f}) in the travel "
                    f"direction while signal='{sig}'. "
                    "Rule fired: stop line crossed when not green."
                ),
                related_detection_ids=[_detection_id(v)],
            ),
        ))
    return results


# ---------------------------------------------------------------------------
# Rule 6 — Red-light running
# ---------------------------------------------------------------------------

def check_red_light(
    detections: list[DetectionRecord],
    scene: SceneContext,
) -> list[RuleResult]:
    """
    Fire when a vehicle crossed the stop line while the signal is red.

    Requires BOTH SceneContext.signal_state and stop_line_coords.  Signal state
    alone is insufficient — the vehicle must also be past the line.
    """
    missing: list[str] = []
    if scene.signal_state is None:
        missing.append("signal_state")
    if scene.stop_line_coords is None:
        missing.append("stop_line_coords")
    if missing:
        return [RuleResult(
            status=RuleStatus.insufficient_scene_context,
            reason=(
                "red_light rule requires SceneContext fields {} but they are "
                "absent for image_id='{}'. Annotate signal state and stop-line "
                "coordinates.".format(missing, scene.image_id)
            ),
        )]

    if scene.signal_state != SignalState.red:
        return [RuleResult(
            status=RuleStatus.clear,
            reason=f"signal is '{scene.signal_state.value}', not red — "
                   "no red-light violation possible",
        )]

    a, b = scene.stop_line_coords
    threshold = THRESHOLDS.get(ViolationType.red_light)
    results: list[RuleResult] = []
    vehicles = [d for d in detections if d.class_label in _VEHICLE_CLASSES]
    if not vehicles:
        return [RuleResult(status=RuleStatus.skipped,
                           reason="no vehicle detections in frame")]

    for v in vehicles:
        ref = _bottom_center(v.bbox)
        crossing = _signed_crossing(ref, a, b, scene.lane_direction_vector)
        if crossing <= _STOP_LINE_MARGIN_PX:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"vehicle at ({v.bbox.x1:.0f},{v.bbox.y1:.0f}) has not "
                       "crossed the stop line",
            ))
            continue

        conf = min(v.track_confidence, 1.0)
        if conf < threshold:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"red-light candidate below threshold "
                       f"({conf:.2f} < {threshold:.2f})",
            ))
            continue

        results.append(RuleResult(
            status=RuleStatus.fired,
            violation=ViolationRecord(
                image_id=v.image_id,
                violation_type=ViolationType.red_light,
                confidence=conf,
                rule_trace=(
                    f"Signal state=red (from scene sidecar). Vehicle "
                    f"({v.class_label.value}, conf={v.track_confidence:.2f}) "
                    f"ground point ({ref.x:.0f},{ref.y:.0f}) is {crossing:.0f}px "
                    f"past the stop line "
                    f"({a.x:.0f},{a.y:.0f})→({b.x:.0f},{b.y:.0f}). "
                    "Rule fired: vehicle crossed the stop line on red."
                ),
                related_detection_ids=[_detection_id(v)],
            ),
        ))
    return results


# ---------------------------------------------------------------------------
# Rule 7 — Illegal parking
# ---------------------------------------------------------------------------

def check_illegal_parking(
    detections: list[DetectionRecord],
    scene: SceneContext,
    *,
    motion: Optional[dict[str, Point2D]] = None,
) -> list[RuleResult]:
    """
    Fire when a vehicle stands inside a no-parking zone.

    Two evidence sources (at least one required):
      - no_parking_zone_polygon : vehicle ground point inside the polygon
                                  (point-in-polygon, primary and most specific).
      - no_parking_sign_visible : a "No Parking" sign in frame makes any vehicle
                                  present a candidate (lower confidence).

    When per-vehicle ``motion`` is supplied (streaming), a vehicle that is
    clearly moving is treated as passing through, not parked, and cleared —
    this stops moving traffic in the zone being flagged.
    """
    has_polygon = scene.no_parking_zone_polygon is not None and \
        len(scene.no_parking_zone_polygon) >= 3
    has_sign = scene.no_parking_sign_visible

    if not has_polygon and not has_sign:
        return [RuleResult(
            status=RuleStatus.insufficient_scene_context,
            reason=(
                "illegal_parking rule requires SceneContext.no_parking_zone_polygon "
                "and/or no_parking_sign_visible=True, but both are absent/False "
                "for image_id='{}'.".format(scene.image_id)
            ),
        )]

    threshold = THRESHOLDS.get(ViolationType.illegal_parking)
    results: list[RuleResult] = []
    vehicles = [d for d in detections if d.class_label in _VEHICLE_CLASSES]
    if not vehicles:
        return [RuleResult(status=RuleStatus.skipped,
                           reason="no vehicle detections in frame")]

    for v in vehicles:
        ref = _bottom_center(v.bbox)

        # Streaming mode: only a vehicle confirmed stationary over time is
        # "parked".  Without enough track history (mv is None) we wait rather
        # than fire — this prevents flagging a vehicle the instant it appears.
        if motion is not None:                       # streaming (dict supplied)
            mv = motion.get(_detection_id(v))
            if mv is None:
                results.append(RuleResult(
                    status=RuleStatus.skipped,
                    reason=f"vehicle at ({v.bbox.x1:.0f},{v.bbox.y1:.0f}): "
                           "awaiting motion history to confirm it is parked",
                ))
                continue
            if math.hypot(mv.x, mv.y) >= _PARKED_MAX_MOTION:
                results.append(RuleResult(
                    status=RuleStatus.clear,
                    reason=f"vehicle at ({v.bbox.x1:.0f},{v.bbox.y1:.0f}) is moving "
                           f"(|motion|={math.hypot(mv.x, mv.y):.0f}px) — not parked",
                ))
                continue

        if has_polygon:
            inside = _point_in_polygon(ref, scene.no_parking_zone_polygon)
            if not inside:
                results.append(RuleResult(
                    status=RuleStatus.clear,
                    reason=f"vehicle ground point ({ref.x:.0f},{ref.y:.0f}) is "
                           "outside the no-parking polygon",
                ))
                continue
            evidence = (f"ground point ({ref.x:.0f},{ref.y:.0f}) falls inside the "
                        f"{len(scene.no_parking_zone_polygon)}-vertex no-parking "
                        "zone polygon")
        else:
            # Sign-only evidence: weaker, so we don't claim polygon containment.
            evidence = "a 'No Parking' sign is visible in the frame and a "\
                       "vehicle is present in it"

        conf = min(v.track_confidence, 1.0)
        if conf < threshold:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"parking candidate below threshold "
                       f"({conf:.2f} < {threshold:.2f})",
            ))
            continue

        results.append(RuleResult(
            status=RuleStatus.fired,
            violation=ViolationRecord(
                image_id=v.image_id,
                violation_type=ViolationType.illegal_parking,
                confidence=conf,
                rule_trace=(
                    f"Vehicle ({v.class_label.value}, conf={v.track_confidence:.2f}) "
                    f"at bbox=({v.bbox.x1:.0f},{v.bbox.y1:.0f},"
                    f"{v.bbox.x2:.0f},{v.bbox.y2:.0f}): {evidence}. "
                    "Rule fired: vehicle stationary in a no-parking area."
                ),
                related_detection_ids=[_detection_id(v)],
            ),
        ))
    return results


# ---------------------------------------------------------------------------
# Convenience runner — evaluates all rules for a single image
# ---------------------------------------------------------------------------

def evaluate_all(
    detections: list[DetectionRecord],
    scene: Optional[SceneContext] = None,
    *,
    helmet_detections: list[DetectionRecord] | None = None,
    nohelmet_detections: list[DetectionRecord] | None = None,
    seatbelt_detections: list[DetectionRecord] | None = None,
    motion: Optional[dict[str, Point2D]] = None,
) -> list[ViolationRecord]:
    """
    Run every rule against a single image's detections.

    Returns only fired ViolationRecord objects — clear, skipped, and
    insufficient_scene_context results are not returned (they are debug info).

    Parameters
    ----------
    detections          : DetectionRecord list from detect.py.
    scene               : SceneContext loaded from sidecar JSON, or None if no
                          sidecar exists (the geometry rules then return
                          insufficient_scene_context and fire nothing).
    helmet_detections   : Helmet attribute classifier output (optional).
    seatbelt_detections : Seatbelt attribute classifier output (optional).
    motion              : Optional per-vehicle displacement vectors keyed by
                          detection id; enables the wrong_side rule.
    """
    violations: list[ViolationRecord] = []

    def _collect(results: list[RuleResult]) -> None:
        for r in results:
            if r.status == RuleStatus.fired and r.violation is not None:
                violations.append(r.violation)

    _collect(check_helmet(detections, helmet_detections=helmet_detections,
                          nohelmet_detections=nohelmet_detections))
    _collect(check_seatbelt(detections, seatbelt_detections=seatbelt_detections))
    _collect(check_triple_riding(detections))

    if scene is not None:
        _collect(check_wrong_side(detections, scene, motion=motion))
        _collect(check_stop_line(detections, scene))
        _collect(check_red_light(detections, scene))
        _collect(check_illegal_parking(detections, scene, motion=motion))

    return violations
