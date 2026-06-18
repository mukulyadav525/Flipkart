"""
Fine-tuning script — YOLOv8 on the Indian Driving Dataset (IDD).

Dataset
-------
IDD (Indian Driving Dataset) — https://idd.insac.edu.in/
  • ~10,000 frames captured from a moving vehicle on Indian roads.
  • Covers dense mixed traffic: two-wheelers, autos, pedestrians, cattle.
  • We use the Detection subset (bounding-box annotations in IDD format).

Before running this script:
  1. Download IDD Detection from the official site and extract to data/idd/.
  2. Run `python -m detection.finetune --prepare` once to convert IDD annotations
     to YOLO format and write data/idd_yolo.yaml.
  3. Then run without --prepare to launch training.

Typical command (single GPU):
    python -m detection.finetune \
        --data data/idd_yolo.yaml \
        --base-weights yolov8s.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 16 \
        --output weights/yolov8_idd.pt

Outputs
-------
  weights/yolov8_idd.pt   — best checkpoint (also at runs/detect/idd_*/weights/best.pt)
  runs/detect/idd_*/      — Ultralytics training artefacts (metrics, confusion matrix, etc.)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

# IDD fine-grained class names → our five canonical labels
# Keys are lowercase IDD label strings; values are YOLO integer class IDs
# in our dataset (0-4 corresponding to VehicleClass order).
_IDD_TO_YOLO_ID: dict[str, int] = {
    # pedestrian = 0
    "person":          0,
    "pedestrian":      0,
    "rider":           0,   # rider counts as pedestrian at detection stage;
                            # the pose / helmet check distinguishes rider vs pillion
    # bike = 1
    "bicycle":         1,
    "motorcycle":      1,
    "autorickshaw":    1,
    "two wheeler":     1,
    # car = 2
    "car":             2,
    "sedan":           2,
    "suv":             2,
    # bus = 3
    "bus":             3,
    # truck = 4
    "truck":           4,
    "vehicle":         4,
}

_YOLO_CLASS_NAMES = ["pedestrian", "bike", "car", "bus", "truck"]


# ---------------------------------------------------------------------------
# Dataset preparation — IDD JSON → YOLO txt labels
# ---------------------------------------------------------------------------

def prepare_idd(
    idd_root: Path,
    output_root: Path,
    splits: tuple[str, ...] = ("train", "val"),
) -> Path:
    """
    Convert IDD Detection annotations (JSON in COCO format) to YOLO format.

    IDD provides one JSON per split:
        idd_root/annotations/instance_<split>.json

    Writes:
        output_root/images/{split}/*.jpg   (symlinks to originals)
        output_root/labels/{split}/*.txt   (YOLO format)
        output_root/idd_yolo.yaml

    Returns the path to the generated YAML file.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    yaml_data: dict = {
        "path": str(output_root.resolve()),
        "nc":   len(_YOLO_CLASS_NAMES),
        "names": _YOLO_CLASS_NAMES,
    }

    for split in splits:
        ann_file = idd_root / "annotations" / f"instance_{split}.json"
        if not ann_file.exists():
            print(f"[finetune] Warning: annotation file not found: {ann_file}", file=sys.stderr)
            continue

        with open(ann_file) as f:
            coco = json.load(f)

        images_by_id = {img["id"]: img for img in coco["images"]}
        label_dir = output_root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        img_dir = output_root / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)

        # Group annotations by image
        ann_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        # Build category_id → YOLO class_id mapping from IDD category names
        cat_to_yolo: dict[int, int] = {}
        for cat in coco["categories"]:
            yolo_id = _IDD_TO_YOLO_ID.get(cat["name"].lower())
            if yolo_id is not None:
                cat_to_yolo[cat["id"]] = yolo_id

        written = 0
        for img_info in coco["images"]:
            img_id   = img_info["id"]
            img_w    = img_info["width"]
            img_h    = img_info["height"]
            img_file = Path(img_info["file_name"])

            # Symlink image
            src = idd_root / img_file
            dst = img_dir / img_file.name
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())

            annotations = ann_by_image.get(img_id, [])
            lines: list[str] = []
            for ann in annotations:
                cat_id = ann["category_id"]
                yolo_cls = cat_to_yolo.get(cat_id)
                if yolo_cls is None:
                    continue
                x, y, w, h = ann["bbox"]   # COCO: top-left x,y + width,height
                # Convert to YOLO normalised cx, cy, w, h
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path = label_dir / (img_file.stem + ".txt")
            label_path.write_text("\n".join(lines))
            written += 1

        yaml_data[split] = str((output_root / "images" / split).resolve())
        print(f"[finetune] Prepared {written} images for split '{split}'")

    yaml_path = output_root / "idd_yolo.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print(f"[finetune] Dataset YAML written to {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_yaml: str | Path,
    base_weights: str = "yolov8s.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    output_weights: str | Path = "weights/yolov8_idd.pt",
    project: str = "runs/detect",
    name: str = "idd",
    resume: bool = False,
) -> None:
    """
    Fine-tune YOLOv8 on the prepared IDD dataset.

    Uses Ultralytics' standard training API. Cosine LR schedule,
    mosaic + mixup augmentation, and close-mosaic-on-final-10-epochs
    are all enabled by default in Ultralytics — no extra config needed.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(base_weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        # Augmentation — keep mosaic; disable for last 10 epochs (default)
        mosaic=1.0,
        mixup=0.1,
        # Indian roads have a lot of small objects — favour small object recall
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    # Copy best checkpoint to the canonical output path
    best_ckpt = Path(project) / name / "weights" / "best.pt"
    if best_ckpt.exists():
        output_weights = Path(output_weights)
        output_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, output_weights)
        print(f"[finetune] Best weights copied to {output_weights}")
    else:
        print(f"[finetune] Warning: best.pt not found at {best_ckpt}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="detection.finetune",
        description="Prepare IDD dataset and/or fine-tune YOLOv8.",
    )
    p.add_argument("--prepare", action="store_true",
                   help="Convert IDD annotations to YOLO format (run once)")
    p.add_argument("--idd-root", default="data/idd", metavar="PATH",
                   help="Root of the extracted IDD dataset (default: data/idd)")
    p.add_argument("--yolo-data", default="data/idd_yolo", metavar="PATH",
                   help="Output directory for YOLO-format dataset (default: data/idd_yolo)")
    p.add_argument("--data", default="data/idd_yolo/idd_yolo.yaml", metavar="PATH",
                   help="Path to dataset YAML for training")
    p.add_argument("--base-weights", default="yolov8s.pt", metavar="PATH",
                   help="Starting weights (default: yolov8s.pt — auto-downloaded)")
    p.add_argument("--epochs",  type=int,   default=50,    metavar="N")
    p.add_argument("--imgsz",   type=int,   default=640,   metavar="N")
    p.add_argument("--batch",   type=int,   default=16,    metavar="N")
    p.add_argument("--device",  default="0", metavar="DEV",
                   help="Training device: 0 (first GPU), cpu, 0,1 (multi-GPU)")
    p.add_argument("--output",  default="weights/yolov8_idd.pt", metavar="PATH",
                   help="Where to save the best checkpoint after training")
    p.add_argument("--resume",  action="store_true",
                   help="Resume an interrupted training run")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.prepare:
        prepare_idd(
            idd_root=Path(args.idd_root),
            output_root=Path(args.yolo_data),
        )

    if not args.prepare or args.data:
        data_yaml = Path(args.data)
        if not data_yaml.exists():
            print(
                f"[finetune] Dataset YAML not found: {data_yaml}\n"
                "Run with --prepare first to generate it.",
                file=sys.stderr,
            )
            sys.exit(1)
        train(
            data_yaml=data_yaml,
            base_weights=args.base_weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            output_weights=args.output,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
