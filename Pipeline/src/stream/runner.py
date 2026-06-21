"""
Stream processor — drives the full violation pipeline over a video stream.

Per frame:
    preprocess → detect (+pose) → track → ANPR → violation rules
              → de-duplicate per vehicle → package evidence → write outputs

Outputs are written incrementally to the exact files the Backend serves, so the
three portals update live while the stream runs:

    <out>/violation_records/confirmed.jsonl   (confidence ≥ AUTO_PROCESS_CUTOFF)
    <out>/human_review_queue.jsonl            (below the cutoff)
    <out>/annotated_images/*.jpg              (evidence snapshots)
    <out>/latency_log.jsonl                   (per-frame total_ms)
    <out>/reports/*.csv                       (analytics, written on exit)

Heavy dependencies (ultralytics, easyocr, cv2) are imported lazily so this
module imports — and its pure orchestration logic stays unit-testable — even
when they are absent.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from shared.schemas import (
    BBox, DetectionRecord, PlateRecord, VehicleClass, ViolationRecord, ViolationType,
)
from stream.scene import CameraConfig
from tracking.tracker import IoUTracker, Track, detection_id, iou


# ---------------------------------------------------------------------------
# Pure orchestration helpers (no cv2 / no model) — unit-tested directly
# ---------------------------------------------------------------------------

def filter_new_violations(
    violations: list[ViolationRecord],
    det_to_track: dict[str, Track],
) -> list[tuple[ViolationRecord, Optional[Track]]]:
    """
    Keep only violations not already emitted for their vehicle's track, and mark
    them as emitted.  This is what stops one vehicle generating a fresh challan
    on every single frame it is visible.
    """
    selected: list[tuple[ViolationRecord, Optional[Track]]] = []
    for v in violations:
        primary = v.related_detection_ids[0] if v.related_detection_ids else None
        track = det_to_track.get(primary) if primary else None
        if track is None:
            selected.append((v, None))      # untracked edge case: emit once
            continue
        if v.violation_type in track.fired:
            continue                          # already issued for this vehicle
        track.fired.add(v.violation_type)
        selected.append((v, track))
    return selected


def assign_plates_to_tracks(
    plates: list[PlateRecord],
    pairs: list[tuple[Track, DetectionRecord]],
    *,
    min_iou: float = 0.3,
) -> None:
    """Attach each plate reading to its vehicle track, keeping the best OCR."""
    for plate in plates:
        best_track, best_iou = None, min_iou
        for track, _det in pairs:
            score = iou(plate.vehicle_bbox, track.bbox)
            if score >= best_iou:
                best_track, best_iou = track, score
        if best_track is None:
            continue
        if best_track.best_plate is None or \
           plate.ocr_confidence > best_track.best_plate.ocr_confidence:
            best_track.best_plate = plate


def plate_for_emission(track: Optional[Track], vehicle_bbox: BBox) -> Optional[PlateRecord]:
    """
    The track's best plate, re-stamped with the vehicle's *current* bbox so the
    evidence module's IoU plate-matching succeeds even if the plate was read in
    an earlier frame.
    """
    if track is None or track.best_plate is None:
        return None
    p = track.best_plate
    return PlateRecord(
        image_id=p.image_id,
        vehicle_bbox=vehicle_bbox,
        plate_bbox=p.plate_bbox,
        plate_text=p.plate_text,
        ocr_confidence=p.ocr_confidence,
    )


@dataclass
class FrameResult:
    frame_idx: int
    n_detections: int
    n_new_violations: int
    latency_ms: float
    emitted: list[str] = field(default_factory=list)   # "vtype:plate"


# ---------------------------------------------------------------------------
# Stream processor
# ---------------------------------------------------------------------------

class StreamProcessor:
    def __init__(
        self,
        camera: CameraConfig,
        output_dir: str | Path,
        *,
        run_pose: bool = True,
        run_anpr: bool = True,
        device: str = "cpu",
        conf_threshold: float = 0.35,
        helmet_weights: Optional[str] = None,
        use_helmet_model: bool = True,
        reset: bool = False,
    ):
        self.camera = camera
        self.out = Path(output_dir)
        self.run_pose = run_pose
        self.run_anpr = run_anpr
        self.device = device
        self.conf_threshold = conf_threshold
        self.helmet_weights = helmet_weights
        # Helmet violations come from a trained no-helmet head model, not a guess.
        # Disabled automatically when no model file is available.
        self.use_helmet_model = use_helmet_model
        if self.use_helmet_model:
            from detection.helmet import available as _helmet_available
            if not _helmet_available(helmet_weights):
                self.use_helmet_model = False

        self.ann_dir       = self.out / "annotated_images"
        self.confirmed_jsonl = self.out / "violation_records" / "confirmed.jsonl"
        self.review_jsonl  = self.out / "human_review_queue.jsonl"
        self.latency_jsonl = self.out / "latency_log.jsonl"
        self.reports_dir   = self.out / "reports"

        self.tracker = IoUTracker()
        self.total_confirmed = 0
        self.total_review = 0
        self._reader = None     # lazy PlateReader

        for p in (self.ann_dir, self.confirmed_jsonl.parent, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
        if reset:
            self._reset_outputs()

    def _reset_outputs(self) -> None:
        for f in (self.confirmed_jsonl, self.review_jsonl, self.latency_jsonl):
            if f.exists():
                f.unlink()
        for img in self.ann_dir.glob("*.jpg"):
            img.unlink()

    # -- per-frame ---------------------------------------------------------

    def process_frame(self, frame, frame_idx: int) -> FrameResult:
        from preprocessing.preprocess import preprocess
        from detection.detect import detect
        from violations.rules import evaluate_all
        from evidence.generate import generate_evidence

        t0 = time.perf_counter()
        image_id = f"{self.camera.name}_f{frame_idx:06d}"

        proc = preprocess(frame)
        dets: list[DetectionRecord] = detect(
            proc, image_id,
            detector_weights=self.camera.detector_weights or "weights/yolov8_idd.pt",
            pose_weights=self.camera.pose_weights or "weights/yolov8n-pose.pt",
            conf_threshold=self.conf_threshold,
            run_pose=self.run_pose,
            device=self.device,
            print_latency=False,
        )

        # Track → stable ids, motion, dedup state.
        pairs = self.tracker.update(dets, frame_idx)
        det_to_track = {detection_id(image_id, d): t for t, d in pairs}
        id_to_det = {detection_id(image_id, d): d for d in dets}

        # Motion vectors keyed exactly like rules._detection_id.
        motion = {
            detection_id(image_id, d): t.motion()
            for t, d in pairs if t.motion() is not None
        }
        motion = {k: v for k, v in motion.items() if v is not None}

        # ANPR → accumulate the best plate per track.
        if self.run_anpr:
            from plate_ocr import read_plates
            if self._reader is None:
                from plate_ocr import PlateReader
                self._reader = PlateReader()
            plates = read_plates(proc, dets, image_id=image_id, reader=self._reader,
                                 device=self.device)
            assign_plates_to_tracks(plates, pairs)

        # Motorcycle-first helmet check: crop each detected motorcycle and run the
        # helmet model on that region only. Excludes pedestrians/cyclists by
        # construction, and upscales the rider's small head so it's detectable.
        helmet_heads: list[DetectionRecord] = []
        nohelmet_heads: list[DetectionRecord] = []
        if self.use_helmet_model:
            from detection.helmet import classify_riders
            motorcycles = [d for d in dets if d.class_label == VehicleClass.bike]
            helmet_heads, nohelmet_heads = classify_riders(
                frame, motorcycles, image_id, weights=self.helmet_weights,
                conf=0.3, device=self.device,
            )

        # Violation rules (scene geometry + motion).
        scene = dataclasses.replace(self.camera.scene, image_id=image_id)
        violations = evaluate_all(
            dets, scene,
            helmet_detections=helmet_heads,
            nohelmet_detections=nohelmet_heads,
            motion=motion,
        )

        # De-duplicate per vehicle, then package evidence for the new ones.
        selected = filter_new_violations(violations, det_to_track)
        new_violations: list[ViolationRecord] = []
        frame_plates: list[PlateRecord] = []
        emitted: list[str] = []
        for v, track in selected:
            primary_id = v.related_detection_ids[0] if v.related_detection_ids else None
            primary_det = id_to_det.get(primary_id) if primary_id else None
            bbox = primary_det.bbox if primary_det else (track.bbox if track else None)
            plate = plate_for_emission(track, bbox) if bbox else None
            if plate:
                frame_plates.append(plate)
            new_violations.append(v)
            emitted.append(f"{v.violation_type.value}:{plate.plate_text if plate else '—'}")

        if new_violations:
            from config import AUTO_PROCESS_CUTOFF
            records = generate_evidence(
                frame, image_id, dets, frame_plates, new_violations,
                annotated_dir=self.ann_dir,
                confirmed_jsonl=self.confirmed_jsonl,
                review_jsonl=self.review_jsonl,
            )
            for r in records:
                if r.violation_record.confidence >= AUTO_PROCESS_CUTOFF:
                    self.total_confirmed += 1
                else:
                    self.total_review += 1

        latency_ms = (time.perf_counter() - t0) * 1000
        self._log_latency(image_id, latency_ms)
        return FrameResult(frame_idx, len(dets), len(new_violations), latency_ms, emitted)

    def _log_latency(self, image_id: str, total_ms: float) -> None:
        import json
        with self.latency_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"image_id": image_id, "total_ms": round(total_ms, 2)}) + "\n")

    # -- analytics on exit -------------------------------------------------

    def write_reports(self) -> None:
        try:
            from analytics.report import load_records, export_csv, build_summary
        except Exception:
            return
        records = load_records(self.confirmed_jsonl)
        if not records:
            return
        export_csv(records, reports_dir=self.reports_dir)
        print(build_summary(records))

    # -- driver ------------------------------------------------------------

    def run(self, *, max_frames: Optional[int] = None, progress_every: int = 30) -> dict:
        """
        Open the camera source and process frames until the stream ends, a frame
        cap is hit, or Ctrl-C.  Returns a small run summary.
        """
        import cv2

        src = self.camera.source
        cap_arg: object = int(src) if src.isdigit() else src
        cap = cv2.VideoCapture(cap_arg)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {src!r}")

        stride = max(1, self.camera.stride)
        print(f"[stream] camera='{self.camera.name}' source={src!r} stride={stride} "
              f"geometry={'yes' if self.camera.has_geometry else 'NONE (geometry rules off)'}")

        frame_idx = processed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    res = self.process_frame(frame, frame_idx)
                    processed += 1
                    if res.emitted:
                        print(f"[stream] f{frame_idx:>6} · {res.n_detections} obj · "
                              f"NEW: {', '.join(res.emitted)}")
                    elif processed % progress_every == 0:
                        print(f"[stream] f{frame_idx:>6} · {res.n_detections} obj · "
                              f"{res.latency_ms:.0f} ms · "
                              f"confirmed={self.total_confirmed} review={self.total_review}")
                frame_idx += 1
                if max_frames and processed >= max_frames:
                    break
        except KeyboardInterrupt:
            print("\n[stream] interrupted — flushing reports…")
        finally:
            cap.release()
            self.write_reports()

        summary = {
            "frames_read": frame_idx,
            "frames_processed": processed,
            "confirmed": self.total_confirmed,
            "review": self.total_review,
            "outputs_dir": str(self.out.resolve()),
        }
        print(f"\n[stream] done · processed {processed} frames · "
              f"confirmed={self.total_confirmed} review={self.total_review}")
        print(f"[stream] outputs → {self.out.resolve()}")
        return summary
