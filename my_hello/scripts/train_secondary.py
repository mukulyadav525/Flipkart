#!/usr/bin/env python3
"""Train a helmet / seatbelt YOLO from a Roboflow dataset, ready for Gridlock.

Produces a `.pt` you point `--helmet-weights` / `--seatbelt-weights` at.
Run on a GPU (Google Colab / Kaggle) — Apple Silicon has no CUDA and will be slow.

Setup:
    pip install roboflow ultralytics

Example:
    python scripts/train_secondary.py \
        --api-key YOUR_KEY --workspace some-ws --project helmet-detection \
        --version 3 --out models/helmet.pt --epochs 50

Find WORKSPACE / PROJECT / VERSION on the dataset's Roboflow Universe page
(the "Download this Dataset" → "show download code" snippet lists them).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Train a Gridlock secondary model from Roboflow")
    p.add_argument("--api-key", required=True, help="Roboflow API key")
    p.add_argument("--workspace", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--out", required=True, help="where to copy best.pt, e.g. models/helmet.pt")
    p.add_argument("--base", default="yolo11n.pt", help="base weights to fine-tune")
    p.add_argument("--format", default="yolov11",
                   help="Roboflow export format (yolov11 and yolov8 are identical Ultralytics layout)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    from roboflow import Roboflow
    from ultralytics import YOLO

    print("[1/3] downloading dataset from Roboflow...")
    rf = Roboflow(api_key=args.api_key)
    ds = (rf.workspace(args.workspace).project(args.project)
            .version(args.version).download(args.format))
    data_yaml = Path(ds.location) / "data.yaml"
    print(f"      dataset at {ds.location}")

    print(f"[2/3] training {args.base} for {args.epochs} epochs...")
    model = YOLO(args.base)
    results = model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz)

    best = Path(results.save_dir) / "weights" / "best.pt"
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)

    print(f"[3/3] done. weights -> {out}")
    print("      class names:", YOLO(out).names)
    print("\nNext:")
    print(f"  python scripts/run_all.py <video> --helmet-weights {args.out}")
    print("  (if class names differ from helmet/no-helmet, adjust violations/helmet.py)")


if __name__ == "__main__":
    main()
