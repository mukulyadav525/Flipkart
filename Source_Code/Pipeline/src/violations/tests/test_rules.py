"""
Unit tests for src/violations/rules.py.

All tests use hand-crafted DetectionRecord / SceneContext objects — no models,
no images, no I/O.  Tests verify rule logic and rule_trace population only.
"""

from __future__ import annotations

import pytest

from shared.schemas import (
    BBox,
    DetectionRecord,
    Point2D,
    SceneContext,
    SignalState,
    VehicleClass,
    ViolationType,
)
from violations.rules import (
    RuleStatus,
    check_helmet,
    check_illegal_parking,
    check_red_light,
    check_seatbelt,
    check_stop_line,
    check_triple_riding,
    check_wrong_side,
    evaluate_all,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _bike(x1=100, y1=100, x2=400, y2=450, conf=0.85,
          keypoints: list[Point2D] | None = None) -> DetectionRecord:
    return DetectionRecord(
        image_id="test",
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_label=VehicleClass.bike,
        track_confidence=conf,
        pose_keypoints=keypoints,
    )


def _car(x1=50, y1=50, x2=500, y2=350, conf=0.90,
         keypoints: list[Point2D] | None = None) -> DetectionRecord:
    return DetectionRecord(
        image_id="test",
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_label=VehicleClass.car,
        track_confidence=conf,
        pose_keypoints=keypoints,
    )


def _person(x1=150, y1=120, x2=350, y2=420, conf=0.80,
            keypoints: list[Point2D] | None = None) -> DetectionRecord:
    return DetectionRecord(
        image_id="test",
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_label=VehicleClass.pedestrian,
        track_confidence=conf,
        pose_keypoints=keypoints,
    )


def _coco_keypoints_17(
    nose=(200, 130),
    l_eye=(190, 120), r_eye=(210, 120),
    l_ear=(185, 125), r_ear=(215, 125),
    l_shoulder=(180, 180), r_shoulder=(220, 180),
    l_elbow=(170, 240), r_elbow=(230, 240),
    l_wrist=(165, 300), r_wrist=(235, 300),
    l_hip=(185, 320), r_hip=(215, 320),
    l_knee=(185, 380), r_knee=(215, 380),
    l_ankle=(185, 430), r_ankle=(215, 430),
) -> list[Point2D]:
    """Return a full 17-point COCO skeleton as Point2D objects."""
    coords = [nose, l_eye, r_eye, l_ear, r_ear,
              l_shoulder, r_shoulder, l_elbow, r_elbow,
              l_wrist, r_wrist, l_hip, r_hip,
              l_knee, r_knee, l_ankle, r_ankle]
    return [Point2D(x=float(x), y=float(y)) for x, y in coords]


def _head_region_detection(x1=170, y1=100, x2=230, y2=145) -> DetectionRecord:
    """A small detection overlapping the head keypoints."""
    return DetectionRecord(
        image_id="test",
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_label=VehicleClass.pedestrian,  # label doesn't matter for helmet check
        track_confidence=0.9,
    )


def _torso_region_detection(x1=170, y1=160, x2=230, y2=335) -> DetectionRecord:
    return DetectionRecord(
        image_id="test",
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_label=VehicleClass.pedestrian,
        track_confidence=0.88,
    )


# ---------------------------------------------------------------------------
# Helmet tests
# ---------------------------------------------------------------------------

class TestHelmet:

    def test_fires_when_nohelmet_head_on_bike(self):
        # The helmet model reports a no-helmet head sitting on the two-wheeler.
        nohelmet = _head_region_detection()
        results = check_helmet([_bike()], nohelmet_detections=[nohelmet])
        assert len(results) == 1
        r = results[0]
        assert r.status == RuleStatus.fired
        assert r.violation is not None
        assert r.violation.violation_type == ViolationType.helmet
        assert "NO-HELMET head" in r.violation.rule_trace

    def test_clear_when_helmet_head_on_bike(self):
        helmet_det = _head_region_detection()  # head wearing a helmet, on the bike
        results = check_helmet([_bike()], helmet_detections=[helmet_det])
        assert results[0].status == RuleStatus.clear

    def test_skipped_when_no_bikes(self):
        results = check_helmet([_car()], nohelmet_detections=[_head_region_detection()])
        assert len(results) == 1
        assert results[0].status == RuleStatus.skipped

    def test_candidate_fires_when_no_verdict_and_fallback_enabled(self):
        # With the opt-in fallback, a motorcycle with no verdict -> review candidate.
        results = check_helmet([_bike()], assume_nohelmet_on_motorcycle=True)
        assert results[0].status == RuleStatus.fired
        assert results[0].violation.confidence <= 0.84

    def test_no_verdict_skipped_by_default(self):
        # Precise default: no helmet-model verdict -> skipped, never assumed.
        results = check_helmet([_bike()])
        assert results[0].status == RuleStatus.skipped

    def test_nohelmet_head_off_bike_not_model_confirmed(self):
        # A bare head far from the bike must not produce a model-confirmed fire.
        off = _head_region_detection(x1=900, y1=900, x2=960, y2=945)
        results = check_helmet([_bike()], nohelmet_detections=[off],
                               assume_nohelmet_on_motorcycle=False)
        assert results[0].status == RuleStatus.skipped

    def test_below_threshold_does_not_fire(self):
        faint = _head_region_detection()
        faint.track_confidence = 0.2   # below helmet threshold
        results = check_helmet([_bike()], nohelmet_detections=[faint])
        assert results[0].status == RuleStatus.clear

    def test_rule_trace_populated(self):
        results = check_helmet([_bike()], nohelmet_detections=[_head_region_detection()])
        assert results[0].status == RuleStatus.fired
        trace = results[0].violation.rule_trace
        assert "Motorcycle detected" in trace
        assert "helmet" in trace.lower()

    def test_related_detection_ids_populated(self):
        results = check_helmet([_bike()], nohelmet_detections=[_head_region_detection()])
        assert len(results[0].violation.related_detection_ids) == 1


# ---------------------------------------------------------------------------
# Seatbelt tests
# ---------------------------------------------------------------------------

class TestSeatbelt:

    def test_fires_when_no_seatbelt_detected(self):
        kpts = _coco_keypoints_17()
        # Person bbox largely inside car bbox
        occupant = _person(x1=100, y1=60, x2=300, y2=300, keypoints=kpts)
        results = check_seatbelt([_car(), occupant], seatbelt_detections=[])
        fired = [r for r in results if r.status == RuleStatus.fired]
        assert len(fired) == 1
        assert fired[0].violation.violation_type == ViolationType.seatbelt
        assert "No seatbelt-class detection" in fired[0].violation.rule_trace

    def test_clear_when_seatbelt_present(self):
        kpts = _coco_keypoints_17()
        occupant = _person(x1=100, y1=60, x2=300, y2=300, keypoints=kpts)
        seatbelt = _torso_region_detection()
        results = check_seatbelt([_car(), occupant], seatbelt_detections=[seatbelt])
        assert any(r.status == RuleStatus.clear for r in results)

    def test_skipped_when_no_cars(self):
        results = check_seatbelt([_bike()])
        assert results[0].status == RuleStatus.skipped

    def test_skipped_when_no_occupants_overlap(self):
        # Person far outside car bbox
        far_person = _person(x1=600, y1=600, x2=800, y2=900)
        results = check_seatbelt([_car(), far_person])
        assert any(r.status == RuleStatus.skipped for r in results)

    def test_rule_trace_contains_car_and_occupant_info(self):
        kpts = _coco_keypoints_17()
        occupant = _person(x1=100, y1=60, x2=300, y2=300, keypoints=kpts)
        results = check_seatbelt([_car(), occupant], seatbelt_detections=[])
        fired = next(r for r in results if r.status == RuleStatus.fired)
        trace = fired.violation.rule_trace
        assert "Car detected" in trace
        assert "Occupant detected" in trace
        assert "torso" in trace.lower()

    def test_related_detection_ids_includes_car_and_occupant(self):
        kpts = _coco_keypoints_17()
        occupant = _person(x1=100, y1=60, x2=300, y2=300, keypoints=kpts)
        results = check_seatbelt([_car(), occupant], seatbelt_detections=[])
        fired = next(r for r in results if r.status == RuleStatus.fired)
        assert len(fired.violation.related_detection_ids) == 2


# ---------------------------------------------------------------------------
# Triple riding tests
# ---------------------------------------------------------------------------

class TestTripleRiding:

    def _riders_on_bike(self, count: int) -> list[DetectionRecord]:
        """Return a bike + `count` persons all overlapping it."""
        bike = _bike(x1=100, y1=100, x2=400, y2=450)
        people = [
            _person(x1=120 + i*10, y1=120, x2=380 + i*10, y2=420)
            for i in range(count)
        ]
        return [bike] + people

    def test_fires_for_three_riders(self):
        detections = self._riders_on_bike(3)
        results = check_triple_riding(detections)
        fired = [r for r in results if r.status == RuleStatus.fired]
        assert len(fired) == 1
        assert fired[0].violation.violation_type == ViolationType.triple_riding

    def test_clear_for_two_riders(self):
        detections = self._riders_on_bike(2)
        results = check_triple_riding(detections)
        assert all(r.status == RuleStatus.clear for r in results)

    def test_clear_for_one_rider(self):
        detections = self._riders_on_bike(1)
        results = check_triple_riding(detections)
        assert all(r.status == RuleStatus.clear for r in results)

    def test_skipped_when_no_bikes(self):
        results = check_triple_riding([_car()])
        assert results[0].status == RuleStatus.skipped

    def test_bystander_not_counted(self):
        """Person far from bike bbox should not be counted as a rider."""
        bike = _bike(x1=100, y1=100, x2=400, y2=450)
        rider1 = _person(x1=120, y1=120, x2=380, y2=420)
        rider2 = _person(x1=130, y1=120, x2=390, y2=420)
        bystander = _person(x1=500, y1=100, x2=700, y2=400)  # outside bike bbox
        results = check_triple_riding([bike, rider1, rider2, bystander])
        assert all(r.status == RuleStatus.clear for r in results)

    def test_rule_trace_contains_rider_count(self):
        detections = self._riders_on_bike(3)
        results = check_triple_riding(detections)
        fired = next(r for r in results if r.status == RuleStatus.fired)
        assert "3" in fired.violation.rule_trace
        assert "2" in fired.violation.rule_trace   # legal max

    def test_related_ids_includes_bike_and_all_riders(self):
        detections = self._riders_on_bike(3)
        results = check_triple_riding(detections)
        fired = next(r for r in results if r.status == RuleStatus.fired)
        # bike + 3 persons = 4 IDs
        assert len(fired.violation.related_detection_ids) == 4


# ---------------------------------------------------------------------------
# Geometry rule tests — real logic (insufficient / clear / fired)
# ---------------------------------------------------------------------------

def _detection_id_of(det: DetectionRecord) -> str:
    return f"{det.image_id}:{det.class_label.value}:{det.bbox.x1:.0f},{det.bbox.y1:.0f}"


class TestWrongSide:

    def _scene(self) -> SceneContext:
        # Legal travel is "up" the frame (decreasing y).
        return SceneContext(image_id="test", lane_direction_vector=Point2D(0.0, -1.0))

    def test_insufficient_when_no_vector(self):
        results = check_wrong_side([_car()], SceneContext(image_id="test"))
        assert results[0].status == RuleStatus.insufficient_scene_context
        assert "lane_direction_vector" in results[0].reason

    def test_skipped_when_no_motion_supplied(self):
        results = check_wrong_side([_car()], self._scene())
        assert results[0].status == RuleStatus.skipped

    def test_fires_when_motion_opposes_lane(self):
        car = _car()
        motion = {_detection_id_of(car): Point2D(0.0, +8.0)}  # moving down vs legal up
        results = check_wrong_side([car], self._scene(), motion=motion)
        fired = [r for r in results if r.status == RuleStatus.fired]
        assert len(fired) == 1
        assert fired[0].violation.violation_type == ViolationType.wrong_side
        assert "opposes lane" in fired[0].violation.rule_trace

    def test_clear_when_motion_aligns_with_lane(self):
        car = _car()
        motion = {_detection_id_of(car): Point2D(0.0, -8.0)}  # moving up = legal
        results = check_wrong_side([car], self._scene(), motion=motion)
        assert all(r.status == RuleStatus.clear for r in results)

    def test_clear_when_near_stationary(self):
        car = _car()
        motion = {_detection_id_of(car): Point2D(0.2, 0.1)}
        results = check_wrong_side([car], self._scene(), motion=motion)
        assert all(r.status == RuleStatus.clear for r in results)

    def test_vehicle_outside_lane_zone_is_cleared(self):
        # opposing motion, but the vehicle sits outside the monitored lane zone
        scene = self._scene()
        scene.lane_zone_polygon = [
            Point2D(0, 0), Point2D(100, 0), Point2D(100, 100), Point2D(0, 100),
        ]
        car = _car(x1=500, y1=500, x2=700, y2=650)   # ground point outside zone
        motion = {_detection_id_of(car): Point2D(0.0, +8.0)}
        results = check_wrong_side([car], scene, motion=motion)
        assert all(r.status == RuleStatus.clear for r in results)
        assert "outside the monitored lane zone" in results[0].reason


class TestStopLine:

    def _scene(self, signal=None) -> SceneContext:
        return SceneContext(
            image_id="test",
            lane_direction_vector=Point2D(0.0, -1.0),       # legal travel = up
            stop_line_coords=(Point2D(0, 300), Point2D(1280, 300)),
            signal_state=signal,
        )

    def test_insufficient_when_no_coords(self):
        results = check_stop_line([_car()], SceneContext(image_id="test"))
        assert results[0].status == RuleStatus.insufficient_scene_context
        assert "stop_line_coords" in results[0].reason

    def test_fires_when_vehicle_past_line(self):
        # car bottom edge y=350; line y=300; downstream = up ⇒ crossed if y<300?
        # car ground point y=350 is *below* the line (larger y) -> NOT crossed.
        # Use a car fully above the line so its ground point is past it.
        car = _car(y1=50, y2=250)  # ground point y=250 < 300 -> crossed (downstream up)
        results = check_stop_line([car], self._scene(signal=SignalState.red))
        fired = [r for r in results if r.status == RuleStatus.fired]
        assert len(fired) == 1
        assert fired[0].violation.violation_type == ViolationType.stop_line

    def test_clear_when_vehicle_before_line(self):
        car = _car(y1=350, y2=550)  # ground point y=550 > 300 -> not crossed
        results = check_stop_line([car], self._scene(signal=SignalState.red))
        assert all(r.status == RuleStatus.clear for r in results)

    def test_clear_when_signal_green(self):
        car = _car(y1=50, y2=250)  # crossed geometrically
        results = check_stop_line([car], self._scene(signal=SignalState.green))
        assert all(r.status == RuleStatus.clear for r in results)


class TestRedLight:

    def _scene(self, signal) -> SceneContext:
        return SceneContext(
            image_id="test",
            lane_direction_vector=Point2D(0.0, -1.0),
            stop_line_coords=(Point2D(0, 300), Point2D(1280, 300)),
            signal_state=signal,
        )

    def test_insufficient_when_both_missing(self):
        results = check_red_light([_car()], SceneContext(image_id="test"))
        assert results[0].status == RuleStatus.insufficient_scene_context
        assert "signal_state" in results[0].reason
        assert "stop_line_coords" in results[0].reason

    def test_insufficient_when_only_stop_line_missing(self):
        scene = SceneContext(image_id="test", signal_state=SignalState.red)
        results = check_red_light([_car()], scene)
        assert results[0].status == RuleStatus.insufficient_scene_context
        assert "stop_line_coords" in results[0].reason

    def test_clear_when_signal_not_red(self):
        car = _car(y1=50, y2=250)  # crossed geometrically
        results = check_red_light([car], self._scene(SignalState.yellow))
        assert all(r.status == RuleStatus.clear for r in results)

    def test_fires_when_crossed_on_red(self):
        car = _car(y1=50, y2=250)  # ground point past the line
        results = check_red_light([car], self._scene(SignalState.red))
        fired = [r for r in results if r.status == RuleStatus.fired]
        assert len(fired) == 1
        assert fired[0].violation.violation_type == ViolationType.red_light
        assert "red" in fired[0].violation.rule_trace.lower()

    def test_clear_when_not_crossed_on_red(self):
        car = _car(y1=350, y2=550)  # before the line
        results = check_red_light([car], self._scene(SignalState.red))
        assert all(r.status == RuleStatus.clear for r in results)


class TestIllegalParking:

    def test_insufficient_when_no_polygon_no_sign(self):
        results = check_illegal_parking([_car()], SceneContext(image_id="test"))
        assert results[0].status == RuleStatus.insufficient_scene_context
        assert "no_parking_zone_polygon" in results[0].reason

    def test_fires_when_inside_polygon(self):
        # car ground point (bottom-center). _car() bbox=(50,50,500,350) ->
        # ground point (275, 350). Polygon covers x:0-600, y:300-720.
        scene = SceneContext(
            image_id="test",
            no_parking_zone_polygon=[
                Point2D(0, 300), Point2D(600, 300),
                Point2D(600, 720), Point2D(0, 720),
            ],
        )
        results = check_illegal_parking([_car()], scene)
        fired = [r for r in results if r.status == RuleStatus.fired]
        assert len(fired) == 1
        assert fired[0].violation.violation_type == ViolationType.illegal_parking
        assert "polygon" in fired[0].violation.rule_trace

    def test_clear_when_outside_polygon(self):
        # Polygon far from the car's ground point (275, 350).
        scene = SceneContext(
            image_id="test",
            no_parking_zone_polygon=[
                Point2D(800, 600), Point2D(1000, 600),
                Point2D(1000, 720), Point2D(800, 720),
            ],
        )
        results = check_illegal_parking([_car()], scene)
        assert all(r.status == RuleStatus.clear for r in results)

    def test_fires_when_sign_visible(self):
        scene = SceneContext(image_id="test", no_parking_sign_visible=True)
        results = check_illegal_parking([_car()], scene)
        fired = [r for r in results if r.status == RuleStatus.fired]
        assert len(fired) == 1
        assert "No Parking" in fired[0].violation.rule_trace

    def test_skipped_when_no_vehicles(self):
        scene = SceneContext(image_id="test", no_parking_sign_visible=True)
        results = check_illegal_parking([_person()], scene)
        assert results[0].status == RuleStatus.skipped

    def test_moving_vehicle_in_zone_is_cleared(self):
        scene = SceneContext(
            image_id="test",
            no_parking_zone_polygon=[
                Point2D(0, 300), Point2D(600, 300),
                Point2D(600, 720), Point2D(0, 720),
            ],
        )
        car = _car()  # ground point (275, 350) is inside the polygon
        motion = {_detection_id_of(car): Point2D(40.0, 5.0)}   # clearly moving
        results = check_illegal_parking([car], scene, motion=motion)
        assert all(r.status == RuleStatus.clear for r in results)
        assert "moving" in results[0].reason

    def test_stationary_vehicle_in_zone_still_fires(self):
        scene = SceneContext(
            image_id="test",
            no_parking_zone_polygon=[
                Point2D(0, 300), Point2D(600, 300),
                Point2D(600, 720), Point2D(0, 720),
            ],
        )
        car = _car()
        motion = {_detection_id_of(car): Point2D(1.0, 0.5)}    # parked (jitter)
        results = check_illegal_parking([car], scene, motion=motion)
        assert any(r.status == RuleStatus.fired for r in results)

    def test_streaming_without_history_waits(self):
        # streaming mode (motion dict supplied) but no entry for this vehicle yet
        scene = SceneContext(
            image_id="test",
            no_parking_zone_polygon=[
                Point2D(0, 300), Point2D(600, 300),
                Point2D(600, 720), Point2D(0, 720),
            ],
        )
        car = _car()
        results = check_illegal_parking([car], scene, motion={})   # no history
        assert all(r.status != RuleStatus.fired for r in results)
        assert results[0].status == RuleStatus.skipped


# ---------------------------------------------------------------------------
# evaluate_all integration
# ---------------------------------------------------------------------------

class TestEvaluateAll:

    def test_returns_only_violations(self):
        kpts = _coco_keypoints_17()
        bike = _bike(keypoints=kpts)
        violations = evaluate_all([bike], scene=None, helmet_detections=[])
        # All fired rules produce ViolationRecord objects, not RuleResult
        for v in violations:
            assert hasattr(v, "violation_type")
            assert hasattr(v, "rule_trace")

    def test_no_scene_skips_geometry_rules(self):
        """Passing scene=None must not crash and must not produce geometry violations."""
        kpts = _coco_keypoints_17()
        bike = _bike(keypoints=kpts)
        violations = evaluate_all([bike], scene=None, helmet_detections=[])
        geo_types = {
            ViolationType.wrong_side, ViolationType.stop_line,
            ViolationType.red_light, ViolationType.illegal_parking,
        }
        for v in violations:
            assert v.violation_type not in geo_types

    def test_empty_detections_returns_empty(self):
        violations = evaluate_all([], scene=None)
        assert violations == []
