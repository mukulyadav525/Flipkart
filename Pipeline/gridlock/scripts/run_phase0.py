#!/usr/bin/env python3
"""Phase 0 runner: preprocess + detect + track + write annotated video.

Examples
--------
    python scripts/run_phase0.py archive/Vodra/North.mp4
    python scripts/run_phase0.py archive/Talaimari/east.mp4 -o outputs/east.mp4 --max-frames 150
    python scripts/run_phase0.py archive/Vodra/North.mp4 --no-preprocess --model yolo11s.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridlock.config import PipelineConfig  # noqa: E402
from gridlock.pipeline import Pipeline  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Gridlock Phase 0 pipeline")
    p.add_argument("source", help="input video path")
    p.add_argument("-o", "--output", default=None, help="annotated output mp4 (default: outputs/<name>_phase0.mp4)")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (n/s/m...)")
    p.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--max-frames", type=int, default=None, help="cap processed frames (quick test)")
    p.add_argument("--no-preprocess", action="store_true", help="disable preprocessing chain")
    p.add_argument("--no-video", action="store_true", help="skip writing annotated video")
    args = p.parse_args()

    out = args.output
    if out is None and not args.no_video:
        out = str(ROOT / "outputs" / (Path(args.source).stem + "_phase0.mp4"))

    cfg = PipelineConfig(model_weights=args.model, conf_threshold=args.conf, imgsz=args.imgsz)
    cfg.preprocess.frame_stride = args.stride
    cfg.preprocess.enabled = not args.no_preprocess
    cfg.write_video = not args.no_video

    print(f"[gridlock] source={args.source} model={args.model} "
          f"preprocess={'on' if cfg.preprocess.enabled else 'off'}")
    pipe = Pipeline(cfg)
    stats = pipe.run(args.source, output=out, max_frames=args.max_frames)

    print("\n=== run stats ===")
    print(f"frames read      : {stats.frames_read}")
    print(f"frames processed : {stats.frames_processed}")
    print(f"elapsed          : {stats.elapsed_s:.1f}s  ({stats.processed_fps:.1f} fps)")
    print(f"device           : {pipe.detector.device}")
    print("unique objects (by class):")
    for name, n in sorted(stats.class_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {n}")
    if out:
        print(f"\nannotated video  : {out}")


if __name__ == "__main__":
    main()
