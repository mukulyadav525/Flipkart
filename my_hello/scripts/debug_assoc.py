#!/usr/bin/env python3
"""Diagnostic: what classes are detected, and how many riders per motorcycle."""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from gridlock.association import riders_per_bike  # noqa: E402
from gridlock.config import PipelineConfig  # noqa: E402
from gridlock.detection import Detector  # noqa: E402

src = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "yolo11s.pt"
nframes = int(sys.argv[3]) if len(sys.argv) > 3 else 120

det = Detector(PipelineConfig(model_weights=model))
cap = cv2.VideoCapture(src)
cls_counter = Counter()
max_riders = Counter()  # rider_count -> how many bike-frames had it
i = 0
while i < nframes:
    ok, frame = cap.read()
    if not ok:
        break
    if i % 2 == 0:
        tracks = det.track(frame)
        for t in tracks:
            cls_counter[t.class_name] += 1
        assoc = riders_per_bike(tracks)
        for riders in assoc.values():
            max_riders[len(riders)] += 1
    i += 1
cap.release()

print("detections by class (per processed frame, summed):")
for k, v in cls_counter.most_common():
    print(f"  {k:12s} {v}")
print("\nriders-per-motorcycle distribution (bike-frames):")
for k in sorted(max_riders):
    print(f"  {k} riders: {max_riders[k]}")
