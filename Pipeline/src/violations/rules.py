"""
Violation rule engine.

Each public function takes DetectionRecord list (and optionally SceneContext)
and returns a list of ViolationRecord objects — never raises, never returns None.

Implemented rules
-----------------
  check_helmet        — bike rider + absent helmet-class detection near head
  check_seatbelt      — car occupant + absent seatbelt-class detection at torso
  check_triple_riding — bike bbox + >2 persons overlapping it

Geometry-dependent stubs (returns "insufficient_scene_context" when fields absent)
-----------------
  check_wrong_side    — needs SceneContext.lane_direction_vector
  check_stop_line     — needs SceneContext.stop_line_coords
  check_red_light     — needs SceneContext.signal_state + stop_line_coords
  check_illegal_parking — needs SceneContext.no_parking_zone_polygon or no_parking_sign_visible

COCO keypoint indices used (documented in detection/__init__.py):
  Head    : 0 (nose), 1 (left_eye), 2 (right_eye), 3 (left_ear), 4 (right_ear)
  Torso   : 5 (left_shoulder), 6 (right_shoulder), 11 (left_hip), 12 (right_hip)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from shared.schemas import (
    BBox,
    DetectionRecord,
    Point2D,
    SceneContext,
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


def check_helmet(
    detections: list[DetectionRecord],
    *,
    # "helmet" class detections come from a separate attribute classifier
    # run over each crop; pass an empty list if that stage hasn't run yet.
    helmet_detections: list[DetectionRecord] | None = None,
) -> list[RuleResult]:
    """
    For every bike detection that has head keypoints, verify a helmet detection
    overlaps the head region.  Fire a violation if none does.

    Parameters
    ----------
    detections         : All DetectionRecords for the image.
    helmet_detections  : Optional list of helmet-class detections from an
                         attribute classifier.  If None or empty, the rule
                         assumes no helmet is present (conservative).
    """
    results: list[RuleResult] = []
    threshold = THRESHOLDS.get(ViolationType.helmet)

    bikes = [d for d in detections if d.class_label == VehicleClass.bike]
    if not bikes:
        return [RuleResult(status=RuleStatus.skipped, reason="no bike detections in frame")]

    for bike in bikes:
        if bike.pose_keypoints is None:
            # Can't evaluate without keypoints — skip this bike, don't fire
            results.append(RuleResult(
                status=RuleStatus.skipped,
                reason=f"bike at ({bike.bbox.x1:.0f},{bike.bbox.y1:.0f}) has no pose keypoints",
            ))
            continue

        head_bbox = _keypoint_bbox(bike.pose_keypoints, _HEAD_KP_INDICES, padding=15.0)
        if head_bbox is None:
            results.append(RuleResult(
                status=RuleStatus.skipped,
                reason=f"bike at ({bike.bbox.x1:.0f},{bike.bbox.y1:.0f}): "
                       "all head keypoints occluded (0,0) — cannot assess helmet",
            ))
            continue

        # Check whether any helmet detection overlaps the head region
        helmet_found = False
        if helmet_detections:
            for h in helmet_detections:
                if _overlap_ratio(h.bbox, head_bbox) >= _HELMET_OVERLAP_THRESHOLD:
                    helmet_found = True
                    break

        if helmet_found:
            results.append(RuleResult(
                status=RuleStatus.clear,
                reason=f"helmet detected overlapping head region of bike "
                       f"at ({bike.bbox.x1:.0f},{bike.bbox.y1:.0f})",
            ))
        else:
            # Violation confidence is the bike's detection confidence, capped
            # at 1.0, because certainty that this is a bike is the main
            # source of uncertainty — we conservatively assume no helmet.
            conf = min(bike.track_confidence, 1.0)
            if conf < threshold:
                results.append(RuleResult(
                    status=RuleStatus.clear,
                    reason=f"bike conf {conf:.2f} below helmet threshold {threshold:.2f} — skipped",
                ))
                continue

            results.append(RuleResult(
                status=RuleStatus.fired,
                violation=ViolationRecord(
                    image_id=bike.image_id,
                    violation_type=ViolationType.helmet,
                    confidence=conf,
                    rule_trace=(
                        f"Bike detected (conf={bike.track_confidence:.2f}) at "
                        f"bbox=({bike.bbox.x1:.0f},{bike.bbox.y1:.0f},"
                        f"{bike.bbox.x2:.0f},{bike.bbox.y2:.0f}). "
                        f"Head keypoints visible at approx "
                        f"({head_bbox.x1+15:.0f},{head_bbox.y1+15:.0f}). "
                        f"No helmet-class detection overlaps head region "
                        f"(overlap threshold={_HELMET_OVERLAP_THRESHOLD}). "
                        f"Rule: helmet must be present within {_HELMET_OVERLAP_THRESHOLD*100:.0f}% "
                        f"overlap of head keypoint bounding box."
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
# Geometry-dependent stubs — wire ready, logic not yet implemented
# ---------------------------------------------------------------------------

def check_wrong_side(
    detections: list[DetectionRecord],
    scene: SceneContext,
) -> list[RuleResult]:
    """
    Stub: wrong-side driving.

    Requires SceneContext.lane_direction_vector — a unit vector indicating
    the legal direction of travel in the camera's lane.  The detection logic
    will compare each vehicle's optical-flow or inter-frame displacement
    vector against this reference.

    Returns insufficient_scene_context when the field is absent.
    """
    if scene.lane_direction_vector is None:
        return [RuleResult(
            status=RuleStatus.insufficient_scene_context,
            reason=(
                "wrong_side rule requires SceneContext.lane_direction_vector "
                "but it is None for image_id='{}'. "
                "Annotate the sidecar JSON with a unit vector pointing in the "
                "legal direction of travel for this camera's lane.".format(scene.image_id)
            ),
        )]

    # TODO: implement wrong-side detection using optical flow or homography
    # to estimate each vehicle's movement vector and compare against
    # lane_direction_vector with a dot-product threshold.
    return [RuleResult(
        status=RuleStatus.skipped,
        reason="wrong_side detection logic not yet implemented",
    )]


def check_stop_line(
    detections: list[DetectionRecord],
    scene: SceneContext,
) -> list[RuleResult]:
    """
    Stub: stop-line crossing.

    Requires SceneContext.stop_line_coords — two pixel-space points defining
    the stop line.  The detection logic will test whether a vehicle's front
    axle crosses the line while the signal is not green.

    Returns insufficient_scene_context when the field is absent.
    """
    if scene.stop_line_coords is None:
        return [RuleResult(
            status=RuleStatus.insufficient_scene_context,
            reason=(
                "stop_line rule requires SceneContext.stop_line_coords "
                "but it is None for image_id='{}'. "
                "Annotate the sidecar JSON with two pixel-space points "
                "defining the stop line from left to right.".format(scene.image_id)
            ),
        )]

    # TODO: implement stop-line crossing check using the front-axle keypoint
    # or bottom-edge of vehicle bbox projected against the line segment.
    return [RuleResult(
        status=RuleStatus.skipped,
        reason="stop_line detection logic not yet implemented",
    )]


def check_red_light(
    detections: list[DetectionRecord],
    scene: SceneContext,
) -> list[RuleResult]:
    """
    Stub: red-light running.

    Requires BOTH SceneContext.signal_state == 'red' AND
    SceneContext.stop_line_coords to be populated.  Signal state alone is not
    sufficient — the rule must also confirm the vehicle crossed the stop line.

    Returns insufficient_scene_context when either field is absent.
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
                "red_light rule requires SceneContext fields {} "
                "but they are absent for image_id='{}'. "
                "Annotate the sidecar JSON with the traffic signal state "
                "and stop-line coordinates.".format(missing, scene.image_id)
            ),
        )]

    # TODO: implement red-light check: confirm signal_state == red AND
    # vehicle front bbox edge has crossed stop_line_coords.
    return [RuleResult(
        status=RuleStatus.skipped,
        reason="red_light detection logic not yet implemented",
    )]


def check_illegal_parking(
    detections: list[DetectionRecord],
    scene: SceneContext,
) -> list[RuleResult]:
    """
    Stub: illegal parking.

    Requires at least ONE of:
      - SceneContext.no_parking_zone_polygon  (geometry)
      - SceneContext.no_parking_sign_visible  (sign visible in frame)

    If no_parking_sign_visible is True but no polygon is provided, the rule
    can still fire (the sign is evidence enough).  If both are absent the
    rule cannot proceed.

    Returns insufficient_scene_context when neither indicator is present.
    """
    has_polygon = scene.no_parking_zone_polygon is not None
    has_sign    = scene.no_parking_sign_visible

    if not has_polygon and not has_sign:
        return [RuleResult(
            status=RuleStatus.insufficient_scene_context,
            reason=(
                "illegal_parking rule requires SceneContext.no_parking_zone_polygon "
                "and/or SceneContext.no_parking_sign_visible=True, "
                "but both are absent/False for image_id='{}'. "
                "Annotate the sidecar JSON with the no-parking polygon "
                "or set no_parking_sign_visible=true.".format(scene.image_id)
            ),
        )]

    # TODO: implement parking check: test whether any vehicle bbox centroid
    # falls inside no_parking_zone_polygon (point-in-polygon) or
    # is spatially associated with the visible sign region.
    return [RuleResult(
        status=RuleStatus.skipped,
        reason="illegal_parking detection logic not yet implemented",
    )]


# ---------------------------------------------------------------------------
# Convenience runner — evaluates all rules for a single image
# ---------------------------------------------------------------------------

def evaluate_all(
    detections: list[DetectionRecord],
    scene: Optional[SceneContext] = None,
    *,
    helmet_detections: list[DetectionRecord] | None = None,
    seatbelt_detections: list[DetectionRecord] | None = None,
) -> list[ViolationRecord]:
    """
    Run every implemented rule and all stubs against a single image's detections.

    Returns only fired ViolationRecord objects — clear, skipped, and
    insufficient_scene_context results are not returned (they are debug info).

    Parameters
    ----------
    detections          : DetectionRecord list from detect.py.
    scene               : SceneContext loaded from sidecar JSON, or None if no
                          sidecar exists (all geometry-dependent stubs will return
                          insufficient_scene_context).
    helmet_detections   : Helmet attribute classifier output (optional).
    seatbelt_detections : Seatbelt attribute classifier output (optional).
    """
    violations: list[ViolationRecord] = []

    def _collect(results: list[RuleResult]) -> None:
        for r in results:
            if r.status == RuleStatus.fired and r.violation is not None:
                violations.append(r.violation)

    _collect(check_helmet(detections, helmet_detections=helmet_detections))
    _collect(check_seatbelt(detections, seatbelt_detections=seatbelt_detections))
    _collect(check_triple_riding(detections))

    if scene is not None:
        _collect(check_wrong_side(detections, scene))
        _collect(check_stop_line(detections, scene))
        _collect(check_red_light(detections, scene))
        _collect(check_illegal_parking(detections, scene))

    return violations
