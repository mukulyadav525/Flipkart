#!/usr/bin/env python3
"""Phase 1 runner: detection + tracking + violation engines (parking, wrong-side).

Requires a calibrated camera config (see scripts/calibrate.py).

Example
-------
    python scripts/run_phase1.py archive/Vodra/North.mp4 \
        --config configs/cameras/North.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridlock.config import PipelineConfig  # noqa: E402
from gridlock.pipeline import Pipeline  # noqa: E402
from gridlock.scene import CameraConfig  # noqa: E402
from gridlock.violations import IllegalParkingEngine, WrongSideEngine  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Gridlock Phase 1 (parking + wrong-side)")
    p.add_argument("source")
    p.add_argument("--config", required=True, help="camera config JSON")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--events", default=None, help="violations JSONL output")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--park-seconds", type=float, default=8.0,
                   help="dwell time before illegal-parking fires")
    args = p.parse_args()

    stem = Path(args.source).stem
    out = args.output or str(ROOT / "outputs" / f"{stem}_phase1.mp4")
    events = args.events or str(ROOT / "outputs" / f"{stem}_violations.jsonl")

    cam = CameraConfig.load(args.config)
    engines = [
        IllegalParkingEngine(min_seconds=args.park_seconds),
        WrongSideEngine(),
    ]

    cfg = PipelineConfig(model_weights=args.model)
    cfg.preprocess.frame_stride = args.stride

    print(f"[gridlock] phase1 source={args.source} camera={cam.name} "
          f"zones={len(cam.no_parking)} lanes={len(cam.lanes)}")
    pipe = Pipeline(cfg, camera_config=cam, engines=engines)
    stats = pipe.run(args.source, output=out, max_frames=args.max_frames, events_path=events)

    print("\n=== run stats ===")
    print(f"frames processed : {stats.frames_processed}")
    print(f"elapsed          : {stats.elapsed_s:.1f}s ({stats.processed_fps:.1f} fps)")
    print(f"violations       : {dict(stats.violation_counts) or 'none'}")
    print(f"annotated video  : {out}")
    print(f"events log       : {events}")


if __name__ == "__main__":
    main()
