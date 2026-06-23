#!/usr/bin/env python3
"""
License-plate detection demo.

Runs:  vehicle detection -> plate detection (YOLO if weights present, else the
classic-CV localiser) -> best-effort OCR (English + Bengali) -> draws plate
boxes (green = text read, amber = plate detected but unreadable) and writes an
annotated image / video plus a JSON + CSV ledger.

Examples
--------
    # a single frame
    python scripts/detect_plates.py --image frame.jpg

    # a video, every 3rd frame, on Apple GPU, first 300 frames
    python scripts/detect_plates.py --source clip.mp4 --stride 3 --device mps --max-frames 300

    # with a trained plate model (accurate localisation)
    python scripts/detect_plates.py --source clip.mp4 --plate-weights weights/plate.pt

Drop a trained YOLO plate model at Pipeline/weights/plate.pt (or pass
--plate-weights) to switch from the CV fallback to accurate detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from detection.detect import detect            # noqa: E402
from detection import plate as plate_det        # noqa: E402
from plate_ocr import read_plates, PlateReader  # noqa: E402

_GREEN = (0, 200, 0)
_AMBER = (0, 165, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw(frame, plates) -> int:
    """Draw plate boxes + text; return how many had readable text."""
    read = 0
    for pr in plates:
        b = pr.plate_bbox
        has_text = bool(pr.plate_text.strip())
        colour = _GREEN if has_text else _AMBER
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, 2)
        label = f"{pr.plate_text}  {pr.ocr_confidence:.0%}" if has_text else "plate"
        ytxt = max(0, int(b.y1) - 6)
        cv2.putText(frame, label, (int(b.x1), ytxt), _FONT, 0.6, colour, 2, cv2.LINE_AA)
        if has_text:
            read += 1
    return read


def _process(frame, image_id, reader, args):
    dets = detect(frame, image_id, run_pose=False, device=args.device, print_latency=False)
    plates = read_plates(frame, dets, image_id=image_id, reader=reader,
                         plate_weights=args.plate_weights, device=args.device)
    return plates


def _ledger_rows(plates):
    return [{
        "image_id": p.image_id,
        "plate_text": p.plate_text,
        "ocr_confidence": p.ocr_confidence,
        "plate_bbox": f"{p.plate_bbox.x1:.0f},{p.plate_bbox.y1:.0f},{p.plate_bbox.x2:.0f},{p.plate_bbox.y2:.0f}",
    } for p in plates]


def main() -> None:
    ap = argparse.ArgumentParser(description="License-plate detection demo")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="A single image to process")
    g.add_argument("--source", help="A video file / rtsp url / webcam index")
    ap.add_argument("--out-dir", default=str(PIPELINE_ROOT / "outputs" / "plates"))
    ap.add_argument("--plate-weights", default=None, help="Trained YOLO plate model")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    reader = PlateReader()
    mode = "YOLO plate model" if plate_det.available(args.plate_weights) else "classic-CV localiser"
    print(f"[plates] localisation: {mode}  |  OCR: {'on' if reader.available else 'OFF'} (en+bn)")
    if not plate_det.available(args.plate_weights):
        print("[plates] tip: drop a trained model at Pipeline/weights/plate.pt for accurate detection")

    all_plates = []

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read image: {args.image}")
        stem = Path(args.image).stem
        plates = _process(frame, stem, reader, args)
        read = _draw(frame, plates)
        cv2.imwrite(str(out / f"{stem}_plates.jpg"), frame)
        all_plates += plates
        print(f"[plates] {stem}: {len(plates)} plate(s) detected, {read} read")
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"could not open source: {args.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = None
        idx = processed = total_read = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride == 0:
                iid = f"f{idx:06d}"
                plates = _process(frame, iid, reader, args)
                total_read += _draw(frame, plates)
                all_plates += plates
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out / "plates_annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (w, h))
                writer.write(frame)
                processed += 1
                if plates:
                    print(f"[plates] {iid}: {len(plates)} detected "
                          f"({sum(1 for p in plates if p.plate_text.strip())} read)")
                if args.max_frames and processed >= args.max_frames:
                    break
            idx += 1
        cap.release()
        if writer:
            writer.release()
        print(f"[plates] processed {processed} frames; "
              f"{len(all_plates)} plate detections, {total_read} read")
        print(f"[plates] annotated video -> {out/'plates_annotated.mp4'}")

    rows = _ledger_rows(all_plates)
    (out / "plates.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    with (out / "plates.csv").open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["image_id", "plate_text", "ocr_confidence", "plate_bbox"])
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"[plates] ledger -> {out/'plates.json'} , {out/'plates.csv'}")


if __name__ == "__main__":
    main()
