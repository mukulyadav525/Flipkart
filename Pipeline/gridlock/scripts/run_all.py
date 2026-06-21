#!/usr/bin/env python3
"""Gridlock — run ALL phases on one video in a single pass.

Phase 0  preprocessing + detection + tracking            (always)
Phase 1  illegal parking + wrong-side  (geometry)        (needs a camera config)
Phase 3  triple riding                                   (always)
         helmet / seatbelt                               (need weights)
Phase 2  stop-line + red-light                           (not implemented yet)

One detection+tracking pass feeds every engine, so this is no slower than a
single phase. Produces an annotated video, a combined violations JSONL, and a
JSON summary.

Examples
--------
    python scripts/run_all.py archive/Vodra/North.mp4
    python scripts/run_all.py archive/Vodra/South.mp4 --config configs/cameras/South.json
    python scripts/run_all.py archive/Vodra/North.mp4 \
        --helmet-weights models/helmet.pt --seatbelt-weights models/seatbelt.pt

If --config is omitted, a matching configs/cameras/<video-stem>.json is used
automatically when present (otherwise geometry violations are skipped).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridlock.config import PipelineConfig  # noqa: E402
from gridlock.pipeline import Pipeline  # noqa: E402
from gridlock.scene import CameraConfig  # noqa: E402
from gridlock.secondary import SecondaryDetector  # noqa: E402
from gridlock.signal import build_signal  # noqa: E402
from gridlock.violations import (  # noqa: E402
    HelmetEngine, IllegalParkingEngine, RedLightEngine, SeatbeltEngine,
    StopLineEngine, TripleRidingEngine, WrongSideEngine,
)

# Human-readable labels for the summary.
VIOLATION_LABELS = {
    "illegal_parking": "Illegal parking",
    "wrong_side": "Wrong-side driving",
    "triple_riding": "Triple riding",
    "no_helmet": "Helmet non-compliance",
    "no_seatbelt": "Seatbelt non-compliance",
    "stop_line": "Stop-line violation",
    "red_light": "Red-light violation",
}


def resolve_config(source: str, explicit: str | None) -> CameraConfig | None:
    if explicit:
        return CameraConfig.load(explicit)
    guess = ROOT / "configs" / "cameras" / f"{Path(source).stem}.json"
    if guess.exists():
        print(f"[gridlock] auto-loaded camera config: {guess.name}")
        return CameraConfig.load(guess)
    return None


def main():
    p = argparse.ArgumentParser(description="Gridlock — run all phases")
    p.add_argument("source")
    p.add_argument("--config", default=None, help="camera config JSON (geometry violations)")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--events", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--model", default="yolo11s.pt", help="base detector (n/s/m)")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--helmet-weights", default=None)
    p.add_argument("--seatbelt-weights", default=None)
    p.add_argument("--min-riders", type=int, default=3)
    p.add_argument("--park-seconds", type=float, default=8.0)
    p.add_argument("--no-video", action="store_true")
    # Phase 4 evidence
    p.add_argument("--evidence", action="store_true", help="cut a 5s clip + snapshot per violation")
    p.add_argument("--anpr", action="store_true", help="read plates (needs easyocr); implies --evidence")
    p.add_argument("--plate-weights", default=None, help="licence-plate YOLO to localise plates before OCR")
    p.add_argument("--clip-seconds", type=float, default=5.0)
    args = p.parse_args()

    stem = Path(args.source).stem
    out_video = None if args.no_video else (args.output or str(ROOT / "outputs" / f"{stem}_all.mp4"))
    events_path = args.events or str(ROOT / "outputs" / f"{stem}_all_violations.jsonl")
    summary_path = args.summary or str(ROOT / "outputs" / f"{stem}_all_summary.json")

    camera = resolve_config(args.source, args.config)

    # Assemble every available engine. Helmet runs on the full frame (small,
    # distant heads) so it needs a larger inference size than the crop default.
    helmet = HelmetEngine(detector=SecondaryDetector(args.helmet_weights, conf=0.4, imgsz=736))
    seatbelt = SeatbeltEngine(detector=SecondaryDetector(args.seatbelt_weights))
    engines = [TripleRidingEngine(min_riders=args.min_riders), helmet, seatbelt]
    if camera is not None:
        engines += [IllegalParkingEngine(min_seconds=args.park_seconds), WrongSideEngine()]

    # Phase 2: stop-line + red-light need a stop line + a signal source.
    signal = build_signal(camera) if camera is not None else None
    has_signal_setup = camera is not None and camera.stop_line is not None and signal is not None
    if has_signal_setup:
        engines += [StopLineEngine(signal=signal), RedLightEngine(signal=signal)]

    enabled = {
        "triple_riding": True,
        "no_helmet": helmet.available,
        "no_seatbelt": seatbelt.available,
        "illegal_parking": camera is not None,
        "wrong_side": camera is not None,
        "stop_line": has_signal_setup,
        "red_light": has_signal_setup,
    }

    print(f"\n[gridlock] running ALL phases on {args.source}")
    print(f"  model={args.model}  stride={args.stride}  camera={'yes' if camera else 'none'}")
    off_reason = {
        "no_helmet": "OFF (no --helmet-weights)",
        "no_seatbelt": "OFF (no --seatbelt-weights)",
        "illegal_parking": "OFF (no camera config)",
        "wrong_side": "OFF (no camera config)",
        "stop_line": "OFF (config needs stop_line + signal)",
        "red_light": "OFF (config needs stop_line + signal)",
    }
    for k, v in enabled.items():
        state = "on" if v else off_reason.get(k, "OFF")
        print(f"    {VIOLATION_LABELS[k]:24s} : {state}")

    cfg = PipelineConfig(model_weights=args.model)
    cfg.preprocess.frame_stride = args.stride
    cfg.write_video = not args.no_video

    pipe = Pipeline(cfg, camera_config=camera, engines=engines, signal_provider=signal)
    stats = pipe.run(args.source, output=out_video, max_frames=args.max_frames,
                     events_path=events_path)

    # Build summary from the manager's full event list.
    events = pipe.manager.events if pipe.manager else []
    counts = {t: sum(1 for e in events if e.type == t) for t in VIOLATION_LABELS}
    summary = {
        "source": args.source,
        "model": args.model,
        "device": pipe.detector.device,
        "frames_processed": stats.frames_processed,
        "fps": round(stats.processed_fps, 1),
        "camera_config": camera.name if camera else None,
        "enabled": enabled,
        "violation_counts": counts,
        "total_violations": len(events),
        "events": [e.to_dict() for e in events],
    }
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2))

    # Console report.
    print("\n" + "=" * 48)
    print("GRIDLOCK — ALL PHASES SUMMARY")
    print("=" * 48)
    print(f"frames processed : {stats.frames_processed}  ({stats.processed_fps:.1f} fps, {pipe.detector.device})")
    print(f"objects tracked  : " + ", ".join(f"{k}={v}" for k, v in sorted(stats.class_counts.items())))
    print("\nviolations detected:")
    for t, label in VIOLATION_LABELS.items():
        mark = f"{counts[t]}" if enabled[t] else "— (disabled)"
        print(f"  {label:24s} : {mark}")
    print(f"\n  TOTAL violations : {len(events)}")
    print("=" * 48)
    if out_video:
        print(f"annotated video  : {out_video}")
    print(f"violations log   : {events_path}")
    print(f"summary json     : {summary_path}")

    # Phase 4 — evidence clips + ANPR.
    if (args.evidence or args.anpr) and events:
        from gridlock.evidence import build_evidence
        plate_reader = None
        if args.anpr:
            from gridlock.anpr import PlateReader
            plate_det = SecondaryDetector(args.plate_weights, conf=0.3) if args.plate_weights else None
            plate_reader = PlateReader(plate_detector=plate_det)
            if not plate_reader.available:
                print("[warn] easyocr not installed — plates will be blank (pip install easyocr)")
        ev_dir = str(ROOT / "outputs" / "evidence" / stem)
        clip_src = out_video if out_video else args.source
        recs = build_evidence([e.to_dict() for e in events], args.source, clip_src,
                              ev_dir, window=args.clip_seconds, plate_reader=plate_reader)
        print(f"\nevidence ({len(recs)} clips) : {ev_dir}/  (ledger.csv + ledger.json)")
        if args.anpr:
            read = sum(1 for r in recs if r["plate"])
            print(f"plates read      : {read}/{len(recs)}")


if __name__ == "__main__":
    main()
