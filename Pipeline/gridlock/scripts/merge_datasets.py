#!/usr/bin/env python3
"""Merge multiple Roboflow YOLO datasets into one with a unified class schema.

Different Roboflow projects use different class names/orders. This remaps each
source's labels to a shared class list (dropping classes mapped to None) and
builds a single dataset Ultralytics can train on. Images are symlinked (no copy),
labels are rewritten with the unified class ids.

Usage:
    python scripts/merge_datasets.py helmet
    python scripts/merge_datasets.py seatbelt
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"

# unified class list + per-source {source_class_name: unified_name | None(drop)}
SPECS = {
    "helmet": {
        "classes": ["helmet", "no_helmet"],
        "sources": {
            "motorcycle helmet.v1i.yolov11": {
                "With Helmet": "helmet", "Without Helmet": "no_helmet", "helmet": "helmet"},
            "Motorcycle helmet.v1-helmet_detection_v1.yolov11": {
                "helmet": "helmet", "no helmet": "no_helmet"},
        },
    },
    "seatbelt": {
        "classes": ["seatbelt", "no_seatbelt"],
        "sources": {
            "CCTV Seatbelt Detection.v2i.yolov11": {
                "pakai": "seatbelt", "tidak-pakai": "no_seatbelt", "windshield": None},
            "Seatbelt Detection.v4i.yolov11": {
                "person-seatbelt": "seatbelt", "seatbelt": "seatbelt",
                "person-noseatbelt": "no_seatbelt", "1": None, "2": None},
        },
    },
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SPECS:
        sys.exit(f"usage: merge_datasets.py [{'|'.join(SPECS)}]")
    name = sys.argv[1]
    spec = SPECS[name]
    classes = spec["classes"]
    uni_idx = {c: i for i, c in enumerate(classes)}

    out = ARCHIVE / f"merged_{name}"
    for split in ("train", "valid", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    totals = {"images": 0, "labels": 0, "boxes": 0, "dropped": 0}
    for src_name, mapping in spec["sources"].items():
        src = ARCHIVE / src_name
        names = yaml.safe_load((src / "data.yaml").read_text())["names"]
        # source class index -> unified index (or None)
        idx_map = {}
        for i, nm in enumerate(names):
            tgt = mapping.get(nm, None)
            idx_map[i] = uni_idx[tgt] if tgt is not None else None
        prefix = "".join(ch for ch in src_name if ch.isalnum())[:12]

        for split in ("train", "valid", "test"):
            img_dir = src / split / "images"
            lbl_dir = src / split / "labels"
            if not img_dir.exists():
                continue
            for img in img_dir.iterdir():
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                new_lines = []
                if lbl.exists():
                    for line in lbl.read_text().splitlines():
                        parts = line.split()
                        if not parts:
                            continue
                        mapped = idx_map.get(int(parts[0]))
                        if mapped is None:
                            totals["dropped"] += 1
                            continue
                        new_lines.append(" ".join([str(mapped)] + parts[1:]))
                        totals["boxes"] += 1
                stem = f"{prefix}_{img.stem}"
                link = out / split / "images" / (stem + img.suffix)
                if not link.exists():
                    link.symlink_to(img.resolve())
                (out / split / "labels" / (stem + ".txt")).write_text("\n".join(new_lines))
                totals["images"] += 1
                if new_lines:
                    totals["labels"] += 1

    data_yaml = {
        "path": str(out), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": len(classes), "names": classes,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    cfg = ROOT / "configs" / f"merged_{name}_data.yaml"
    cfg.write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print(f"merged '{name}' -> {out}")
    print(f"  classes: {classes}")
    print(f"  images: {totals['images']}  labeled: {totals['labels']}  "
          f"boxes kept: {totals['boxes']}  boxes dropped: {totals['dropped']}")
    print(f"  data.yaml -> {cfg}")


if __name__ == "__main__":
    main()
