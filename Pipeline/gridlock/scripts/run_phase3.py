#!/usr/bin/env python3
"""Phase 3 runner: perception violations.

  - triple riding  : works now, no extra model (uses base person+motorcycle)
  - helmet         : needs a helmet YOLO (--helmet-weights)
  - seatbelt       : needs a seatbelt YOLO (--seatbelt-weights)

Examples
--------
    # triple riding only (runs today)
    python scripts/run_phase3.py archive/Vodra/South.mp4

    # add helmet detection with a Roboflow-exported model
    python scripts/run_phase3.py archive/Vodra/South.mp4 \
        --helmet-weights models/helmet.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridlock.config import PipelineConfig  # noqa: E402
from gridlock.pipeline import Pipeline  # noqa: E402
from gridlock.secondary import SecondaryDetector  # noqa: E402
from gridlock.violations import (  # noqa: E402
    HelmetEngine, SeatbeltEngine, TripleRidingEngine,
)


def main():
    p = argparse.ArgumentParser(description="Gridlock Phase 3 (perception)")
    p.add_argument("source")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--events", default=None)
    p.add_argument("--model", default="yolo11n.pt", help="base detector (n/s/m)")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--helmet-weights", default=None, help="helmet YOLO .pt")
    p.add_argument("--seatbelt-weights", default=None, help="seatbelt YOLO .pt")
    p.add_argument("--min-riders", type=int, default=3, help="triple-riding threshold")
    args = p.parse_args()

    stem = Path(args.source).stem
    out = args.output or str(ROOT / "outputs" / f"{stem}_phase3.mp4")
    events = args.events or str(ROOT / "outputs" / f"{stem}_phase3_violations.jsonl")

    engines = [TripleRidingEngine(min_riders=args.min_riders)]

    helmet = HelmetEngine(detector=SecondaryDetector(args.helmet_weights))
    seatbelt = SeatbeltEngine(detector=SecondaryDetector(args.seatbelt_weights))
    engines += [helmet, seatbelt]

    print(f"[gridlock] phase3 source={args.source}")
    print(f"  triple_riding : on (>= {args.min_riders} riders)")
    print(f"  helmet        : {'on' if helmet.available else 'OFF (no weights)'}")
    print(f"  seatbelt      : {'on' if seatbelt.available else 'OFF (no weights)'}")

    cfg = PipelineConfig(model_weights=args.model)
    cfg.preprocess.frame_stride = args.stride

    pipe = Pipeline(cfg, engines=engines)
    stats = pipe.run(args.source, output=out, max_frames=args.max_frames, events_path=events)

    print("\n=== run stats ===")
    print(f"frames processed : {stats.frames_processed}")
    print(f"elapsed          : {stats.elapsed_s:.1f}s ({stats.processed_fps:.1f} fps)")
    print(f"violations       : {dict(stats.violation_counts) or 'none'}")
    print(f"annotated video  : {out}")
    print(f"events log       : {events}")


if __name__ == "__main__":
    main()
