#!/usr/bin/env python3
"""Build evidence (5s clips + snapshots + plate read) from a violations JSONL.

Run after run_all.py. Example:
    python scripts/make_evidence.py \
        --source archive/cctv/20230707_17_CY23_T1_Camera1_0.mp4 \
        --events outputs/20230707_17_CY23_T1_Camera1_0_all_violations.jsonl \
        --anpr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridlock.evidence import build_evidence  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Gridlock evidence builder")
    p.add_argument("--source", required=True, help="original video (for plate crops)")
    p.add_argument("--events", required=True, help="violations JSONL from run_all")
    p.add_argument("--clip-video", default=None,
                   help="video to cut clips from (default: the annotated <stem>_all.mp4)")
    p.add_argument("--out", default=None, help="evidence output dir")
    p.add_argument("--window", type=float, default=5.0, help="clip length in seconds")
    p.add_argument("--anpr", action="store_true", help="read plates with EasyOCR")
    args = p.parse_args()

    events = [json.loads(l) for l in Path(args.events).read_text().splitlines() if l.strip()]
    if not events:
        print("no violations in", args.events)
        return

    stem = Path(args.source).stem
    clip_video = args.clip_video or str(ROOT / "outputs" / f"{stem}_all.mp4")
    if not Path(clip_video).exists():
        print(f"[warn] annotated clip {clip_video} not found — cutting from source instead")
        clip_video = args.source
    out_dir = args.out or str(ROOT / "outputs" / "evidence" / stem)

    plate_reader = None
    if args.anpr:
        from gridlock.anpr import PlateReader
        plate_reader = PlateReader()
        if not plate_reader.available:
            print("[warn] EasyOCR not installed — skipping plate reading (pip install easyocr)")

    print(f"building evidence for {len(events)} violations -> {out_dir}")
    records = build_evidence(events, args.source, clip_video, out_dir,
                             window=args.window, plate_reader=plate_reader)

    print(f"\n{'type':16s} {'track':6s} {'time':>6s}  {'plate':12s} clip")
    for r in records:
        print(f"{r['type']:16s} {r['track_id']:<6} {r['timestamp']:6.1f}  "
              f"{(r['plate'] or '—'):12s} {Path(r['clip']).name}")
    print(f"\nledger: {out_dir}/ledger.csv  +  ledger.json")


if __name__ == "__main__":
    main()
