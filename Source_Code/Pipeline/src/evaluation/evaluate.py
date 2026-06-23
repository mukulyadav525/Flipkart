"""
Evaluation module for the traffic violation detection pipeline.

Computes per-violation-type Precision, Recall, F1, Accuracy, and mAP
by comparing predicted ViolationRecords against manually labelled ground
truth stored in data/ground_truth/.

Ground truth format
-------------------
One or more JSONL files under data/ground_truth/ (e.g. ground_truth.jsonl).
Each line is a JSON object with at minimum:

  {
    "image_id":      "frame_001",
    "violation_type": "helmet",
    "confidence":     1.0,           // always 1.0 for GT
    "bbox": {"x1": 100, "y1": 80, "x2": 400, "y2": 450}  // optional
  }

The "bbox" field is used for IoU-based matching when present; otherwise
matching is image_id + violation_type only (sufficient for Phase 3 rule-level
evaluation before dense spatial annotations exist).

Prediction format
-----------------
ViolationRecords serialised as JSON dicts — identical to the lines written
by evidence/generate.py into outputs/violation_records/confirmed.jsonl PLUS
the human_review_queue.jsonl (evaluation needs all predictions, not just the
auto-processed ones).

Latency log format
------------------
Optional JSONL file (default: outputs/latency_log.jsonl) written by the
detection pipeline.  Each line:

  {"image_id": "frame_001", "total_ms": 42.3, "pose_ms": 18.1}

detect.py writes these when `latency_log_path` is provided (see
_report_latency in detect.py — callers may pass the path).
The evaluate.py latency section is silently skipped if the file is absent.

Blocking note
-------------
Full evaluation requires a labelled validation set in data/ground_truth/.
Until that set exists:
  - run with --dry-run to see the expected output schema
  - the script exits with a descriptive error if no GT files are found
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from shared.schemas import ViolationType

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GROUND_TRUTH_DIR   = Path("data/ground_truth")
CONFIRMED_JSONL    = Path("outputs/violation_records/confirmed.jsonl")
REVIEW_JSONL       = Path("outputs/human_review_queue.jsonl")
LATENCY_LOG_JSONL  = Path("outputs/latency_log.jsonl")
REPORTS_DIR        = Path("outputs/reports")

# IoU threshold for a prediction bbox to count as a true positive.
# Used only when GT records carry a "bbox" field.
IOU_THRESHOLD: float = 0.50

# Violation types that depend on SceneContext being populated.
# Reported separately in the output so reviewers can interpret low recall.
SCENE_CONTEXT_DEPENDENT: frozenset[str] = frozenset([
    ViolationType.wrong_side.value,
    ViolationType.stop_line.value,
    ViolationType.red_light.value,
    ViolationType.illegal_parking.value,
])


# ---------------------------------------------------------------------------
# Internal record types
# ---------------------------------------------------------------------------

@dataclass
class GTRecord:
    image_id:       str
    violation_type: str
    bbox:           Optional[dict] = None   # raw {x1,y1,x2,y2} or None


@dataclass
class PredRecord:
    image_id:       str
    violation_type: str
    confidence:     float
    bbox:           Optional[dict] = None


@dataclass
class PerTypeMetrics:
    violation_type:  str
    n_gt:            int
    n_pred:          int
    tp:              int
    fp:              int
    fn:              int
    precision:       float
    recall:          float
    f1:              float
    accuracy:        float          # TP / (TP + FP + FN) — Jaccard for single class
    average_precision: float        # area under P-R curve (101-point interpolation)
    scene_context_dependent: bool


@dataclass
class LatencyStats:
    n_images:       int
    mean_ms:        float
    median_ms:      float
    p95_ms:         float
    min_ms:         float
    max_ms:         float
    throughput_fps: float           # 1000 / mean_ms


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[eval] Skipping malformed line {lineno} in {path}: {exc}",
                      file=sys.stderr)


def load_ground_truth(gt_dir: Path = GROUND_TRUTH_DIR) -> list[GTRecord]:
    """Load all JSONL files under gt_dir into GTRecord objects."""
    records: list[GTRecord] = []
    jsonl_files = sorted(gt_dir.glob("*.jsonl")) if gt_dir.exists() else []
    for path in jsonl_files:
        for d in _iter_jsonl(path):
            records.append(GTRecord(
                image_id=d["image_id"],
                violation_type=d["violation_type"],
                bbox=d.get("bbox"),
            ))
    return records


def load_predictions(
    confirmed_jsonl: Path = CONFIRMED_JSONL,
    review_jsonl:    Path = REVIEW_JSONL,
) -> list[PredRecord]:
    """
    Load predictions from both confirmed and review sinks.

    Evaluation must see all predictions regardless of the auto-process cutoff,
    otherwise the cutoff inflates precision by silently dropping uncertain ones.
    """
    records: list[PredRecord] = []
    for path in (confirmed_jsonl, review_jsonl):
        for d in _iter_jsonl(path):
            vr = d.get("violation_record", {})
            records.append(PredRecord(
                image_id=vr.get("image_id", ""),
                violation_type=vr.get("violation_type", ""),
                confidence=vr.get("confidence", 0.0),
                bbox=d.get("pred_bbox"),   # optional future field
            ))
    return records


def load_latency(latency_log: Path = LATENCY_LOG_JSONL) -> list[float]:
    """Return list of per-image total_ms values. Empty list if file absent."""
    return [
        d["total_ms"]
        for d in _iter_jsonl(latency_log)
        if "total_ms" in d
    ]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((a["x2"] - a["x1"]) * (a["y2"] - a["y1"]), 1e-6)
    area_b = max((b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), 1e-6)
    return inter / (area_a + area_b - inter)


def _pred_matches_gt(pred: PredRecord, gt: GTRecord) -> bool:
    """
    A prediction matches a GT record when:
      - image_id and violation_type agree, AND
      - if both carry bbox, IoU >= IOU_THRESHOLD;
        if either lacks bbox, spatial check is skipped.
    """
    if pred.image_id != gt.image_id:
        return False
    if pred.violation_type != gt.violation_type:
        return False
    if pred.bbox is not None and gt.bbox is not None:
        return _iou(pred.bbox, gt.bbox) >= IOU_THRESHOLD
    return True


# ---------------------------------------------------------------------------
# AP computation (101-point interpolation, COCO-style)
# ---------------------------------------------------------------------------

def _compute_ap(
    sorted_preds: list[PredRecord],
    gt_records:   list[GTRecord],
) -> float:
    """
    Compute Average Precision for a single class using 101-point interpolation.

    Parameters
    ----------
    sorted_preds : predictions for this class, sorted by confidence descending.
    gt_records   : ground truth records for this class.

    Returns float in [0, 1].
    """
    n_gt = len(gt_records)
    if n_gt == 0:
        return 0.0

    # Track which GT records have been matched (one-to-one greedy matching)
    gt_matched = [False] * n_gt

    tp_list: list[int] = []
    fp_list: list[int] = []

    for pred in sorted_preds:
        best_iou = -1.0
        best_idx = -1

        for i, gt in enumerate(gt_records):
            if gt_matched[i]:
                continue
            if pred.image_id != gt.image_id or pred.violation_type != gt.violation_type:
                continue
            if pred.bbox is not None and gt.bbox is not None:
                iou = _iou(pred.bbox, gt.bbox)
            else:
                # No spatial info — treat as a match when IDs and type agree
                iou = 1.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= IOU_THRESHOLD:
            gt_matched[best_idx] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Cumulative TP/FP → precision/recall pairs
    cum_tp = 0
    cum_fp = 0
    precisions: list[float] = []
    recalls:    list[float] = []

    for tp, fp in zip(tp_list, fp_list):
        cum_tp += tp
        cum_fp += fp
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / n_gt)

    # 101-point interpolation (recall thresholds 0.00, 0.01, ..., 1.00)
    ap = 0.0
    for t in [i / 100 for i in range(101)]:
        p_interp = max(
            (p for p, r in zip(precisions, recalls) if r >= t),
            default=0.0,
        )
        ap += p_interp / 101

    return ap


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    predictions: list[PredRecord],
    ground_truth: list[GTRecord],
) -> list[PerTypeMetrics]:
    """
    Compute per-violation-type metrics.

    Matching is greedy (highest-confidence prediction matched first).
    Each GT record can be matched at most once.
    """
    all_types = sorted({
        *{g.violation_type for g in ground_truth},
        *{p.violation_type for p in predictions},
    })

    metrics: list[PerTypeMetrics] = []

    for vtype in all_types:
        gt_for_type   = [g for g in ground_truth if g.violation_type == vtype]
        pred_for_type = sorted(
            [p for p in predictions if p.violation_type == vtype],
            key=lambda p: p.confidence,
            reverse=True,
        )

        n_gt   = len(gt_for_type)
        n_pred = len(pred_for_type)

        gt_matched   = [False] * n_gt
        pred_matched = [False] * n_pred

        for pi, pred in enumerate(pred_for_type):
            for gi, gt in enumerate(gt_for_type):
                if gt_matched[gi]:
                    continue
                if _pred_matches_gt(pred, gt):
                    gt_matched[gi]   = True
                    pred_matched[pi] = True
                    break

        tp = sum(pred_matched)
        fp = n_pred - tp
        fn = n_gt - sum(gt_matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        # Per-class Jaccard / intersection-over-union: TP / (TP + FP + FN)
        accuracy  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        ap        = _compute_ap(pred_for_type, gt_for_type)

        metrics.append(PerTypeMetrics(
            violation_type=vtype,
            n_gt=n_gt,
            n_pred=n_pred,
            tp=tp, fp=fp, fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            average_precision=ap,
            scene_context_dependent=(vtype in SCENE_CONTEXT_DEPENDENT),
        ))

    return metrics


def compute_latency_stats(latencies_ms: list[float]) -> Optional[LatencyStats]:
    if not latencies_ms:
        return None
    n   = len(latencies_ms)
    s   = sorted(latencies_ms)
    mean_ms   = sum(s) / n
    median_ms = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    p95_ms    = s[min(int(math.ceil(0.95 * n)) - 1, n - 1)]
    return LatencyStats(
        n_images=n,
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        min_ms=s[0],
        max_ms=s[-1],
        throughput_fps=1000.0 / mean_ms if mean_ms > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _flag(m: PerTypeMetrics) -> str:
    return " *" if m.scene_context_dependent else "  "


_CONTEXT_NOTE = (
    "* Scene-context-dependent rule: recall is expected to be low until "
    "data/ground_truth/ includes scene-annotated images with populated "
    "SceneContext sidecar files."
)

_BLOCKED_NOTE = (
    "EVALUATION BLOCKED: No ground truth files found in data/ground_truth/.\n"
    "Label your validation set and save one JSONL per split:\n"
    "  data/ground_truth/val.jsonl\n"
    "Each line: "
    '{"image_id":"frame_001","violation_type":"helmet","bbox":{...}}\n'
    "Re-run evaluate.py once labelling is complete."
)

_DRY_RUN_RECORD = '{"image_id":"frame_001","violation_type":"helmet","confidence":1.0}'


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def build_console_report(
    metrics: list[PerTypeMetrics],
    latency: Optional[LatencyStats],
    *,
    n_pred_total: int,
    n_gt_total: int,
) -> str:
    lines: list[str] = []
    sep  = "=" * 80
    sep2 = "-" * 80

    lines += [
        sep,
        "  TRAFFIC VIOLATION DETECTION — EVALUATION RESULTS",
        f"  Ground truth records: {n_gt_total}   |   Predictions evaluated: {n_pred_total}",
        sep,
        "",
    ]

    # Per-type table
    header = (
        f"  {'Type':<20}  {'GT':>4}  {'Pred':>4}  {'TP':>4}  {'FP':>4}  {'FN':>4}"
        f"  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Acc':>7}  {'AP':>7}"
    )
    lines.append("PER-VIOLATION-TYPE METRICS")
    lines.append(sep2)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    has_context_dep = False
    for m in metrics:
        flag = _flag(m)
        if m.scene_context_dependent:
            has_context_dep = True
        lines.append(
            f"  {m.violation_type:<20}{flag}"
            f"  {m.n_gt:>4}  {m.n_pred:>4}  {m.tp:>4}  {m.fp:>4}  {m.fn:>4}"
            f"  {_pct(m.precision):>7}  {_pct(m.recall):>7}  {_pct(m.f1):>7}"
            f"  {_pct(m.accuracy):>7}  {_pct(m.average_precision):>7}"
        )

    if metrics:
        # Macro-average (unweighted) across all types
        n = len(metrics)
        lines.append("  " + "-" * (len(header) - 2))
        lines.append(
            f"  {'macro-average':<22}"
            f"  {'':>4}  {'':>4}  {'':>4}  {'':>4}  {'':>4}"
            f"  {_pct(sum(m.precision for m in metrics)/n):>7}"
            f"  {_pct(sum(m.recall    for m in metrics)/n):>7}"
            f"  {_pct(sum(m.f1        for m in metrics)/n):>7}"
            f"  {_pct(sum(m.accuracy  for m in metrics)/n):>7}"
            f"  {_pct(sum(m.average_precision for m in metrics)/n):>7}"
        )
        mAP = sum(m.average_precision for m in metrics) / n
        lines.append(f"\n  mAP@IoU={IOU_THRESHOLD:.2f}  =  {_pct(mAP)}")

    if has_context_dep:
        lines += ["", f"  {_CONTEXT_NOTE}"]

    lines.append("")

    # Latency section
    lines.append("INFERENCE LATENCY & THROUGHPUT")
    lines.append(sep2)
    if latency:
        lines += [
            f"  Images measured     : {latency.n_images}",
            f"  Mean latency        : {latency.mean_ms:.1f} ms",
            f"  Median latency      : {latency.median_ms:.1f} ms",
            f"  P95 latency         : {latency.p95_ms:.1f} ms",
            f"  Min / Max           : {latency.min_ms:.1f} ms / {latency.max_ms:.1f} ms",
            f"  Throughput (1/mean) : {latency.throughput_fps:.2f} images/sec",
            "",
            "  Note: latency includes detection + pose estimation stages.",
            "  Plate OCR and violation rule engine are not timed here.",
            "  For GPU throughput, re-run with --device cuda and a fresh latency log.",
        ]
    else:
        lines += [
            "  No latency log found at outputs/latency_log.jsonl.",
            "  To capture latency, pass latency_log_path='outputs/latency_log.jsonl'",
            "  to detect() in src/detection/detect.py.",
        ]

    lines += ["", sep]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def build_markdown_report(
    metrics: list[PerTypeMetrics],
    latency: Optional[LatencyStats],
    *,
    n_pred_total: int,
    n_gt_total: int,
) -> str:
    lines: list[str] = []

    lines += [
        "# Traffic Violation Detection — Evaluation Results",
        "",
        f"**Ground truth records:** {n_gt_total}  |  "
        f"**Predictions evaluated:** {n_pred_total}  |  "
        f"**IoU threshold:** {IOU_THRESHOLD:.2f}",
        "",
    ]

    # Per-type table
    lines += [
        "## Per-Violation-Type Metrics",
        "",
        "| Violation Type | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Accuracy | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        suffix = " †" if m.scene_context_dependent else ""
        lines.append(
            f"| {m.violation_type}{suffix} "
            f"| {m.n_gt} | {m.n_pred} | {m.tp} | {m.fp} | {m.fn} "
            f"| {_pct(m.precision)} | {_pct(m.recall)} | {_pct(m.f1)} "
            f"| {_pct(m.accuracy)} | {_pct(m.average_precision)} |"
        )

    if metrics:
        n   = len(metrics)
        mAP = sum(m.average_precision for m in metrics) / n
        lines += [
            f"| **macro-average** | | | | | | "
            f"**{_pct(sum(m.precision for m in metrics)/n)}** | "
            f"**{_pct(sum(m.recall    for m in metrics)/n)}** | "
            f"**{_pct(sum(m.f1        for m in metrics)/n)}** | "
            f"**{_pct(sum(m.accuracy  for m in metrics)/n)}** | "
            f"**{_pct(mAP)}** |",
            "",
            f"**mAP@IoU={IOU_THRESHOLD:.2f} = {_pct(mAP)}**",
        ]

    lines += [
        "",
        "> † Scene-context-dependent rule. Recall is expected to be low until "
        "validation images have populated SceneContext sidecar files "
        "(`lane_direction_vector`, `stop_line_coords`, `signal_state`, "
        "`no_parking_zone_polygon`).",
        "",
    ]

    # Latency section
    lines += ["## Inference Latency & Throughput", ""]
    if latency:
        lines += [
            f"| Metric | Value |",
            f"|---|---|",
            f"| Images measured | {latency.n_images} |",
            f"| Mean latency | {latency.mean_ms:.1f} ms |",
            f"| Median latency | {latency.median_ms:.1f} ms |",
            f"| P95 latency | {latency.p95_ms:.1f} ms |",
            f"| Min / Max | {latency.min_ms:.1f} ms / {latency.max_ms:.1f} ms |",
            f"| **Throughput** | **{latency.throughput_fps:.2f} images/sec** |",
            "",
            "_Latency covers detection + pose estimation. "
            "Plate OCR and violation rules are not included in these timings. "
            "Re-run with `--device cuda` for GPU throughput._",
        ]
    else:
        lines += [
            "_No latency data available. See `outputs/latency_log.jsonl`._", ""
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    metrics: list[PerTypeMetrics],
    latency: Optional[LatencyStats],
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # Metrics CSV
    path = reports_dir / "eval_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "violation_type", "n_gt", "n_pred", "tp", "fp", "fn",
            "precision", "recall", "f1", "accuracy", "average_precision",
            "scene_context_dependent",
        ])
        for m in metrics:
            w.writerow([
                m.violation_type, m.n_gt, m.n_pred, m.tp, m.fp, m.fn,
                round(m.precision, 4), round(m.recall, 4), round(m.f1, 4),
                round(m.accuracy, 4), round(m.average_precision, 4),
                m.scene_context_dependent,
            ])
    written["eval_metrics"] = path

    # Latency CSV
    path = reports_dir / "eval_latency.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        if latency:
            w.writerows([
                ["n_images",       latency.n_images],
                ["mean_ms",        round(latency.mean_ms, 2)],
                ["median_ms",      round(latency.median_ms, 2)],
                ["p95_ms",         round(latency.p95_ms, 2)],
                ["min_ms",         round(latency.min_ms, 2)],
                ["max_ms",         round(latency.max_ms, 2)],
                ["throughput_fps", round(latency.throughput_fps, 3)],
            ])
        else:
            w.writerow(["note", "no latency data"])
    written["eval_latency"] = path

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluation.evaluate",
        description="Evaluate violation detection against ground truth.",
    )
    p.add_argument(
        "--gt-dir", default=str(GROUND_TRUTH_DIR), metavar="PATH",
        help=f"Ground truth JSONL directory (default: {GROUND_TRUTH_DIR})",
    )
    p.add_argument(
        "--confirmed", default=str(CONFIRMED_JSONL), metavar="PATH",
        help="Confirmed predictions JSONL",
    )
    p.add_argument(
        "--review", default=str(REVIEW_JSONL), metavar="PATH",
        help="Human-review-queue JSONL (included in evaluation)",
    )
    p.add_argument(
        "--latency-log", default=str(LATENCY_LOG_JSONL), metavar="PATH",
        help="Per-image latency log JSONL written by detect.py",
    )
    p.add_argument(
        "--reports-dir", default=str(REPORTS_DIR), metavar="PATH",
        help=f"Output directory for CSV and markdown (default: {REPORTS_DIR})",
    )
    p.add_argument(
        "--iou-threshold", type=float, default=IOU_THRESHOLD, metavar="F",
        help=f"IoU threshold for bbox matching (default: {IOU_THRESHOLD})",
    )
    p.add_argument(
        "--no-csv", action="store_true",
        help="Skip CSV and markdown output, print console table only",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show expected GT record format and exit without running evaluation",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    global IOU_THRESHOLD

    args = _build_parser().parse_args(argv)

    if args.dry_run:
        print("Expected ground truth record format (one JSON per line):")
        print(_DRY_RUN_RECORD)
        print("\nSave labelled records to:", args.gt_dir)
        print("Then re-run without --dry-run.")
        return

    IOU_THRESHOLD = args.iou_threshold

    gt = load_ground_truth(Path(args.gt_dir))
    if not gt:
        print(_BLOCKED_NOTE, file=sys.stderr)
        sys.exit(1)

    preds = load_predictions(
        confirmed_jsonl=Path(args.confirmed),
        review_jsonl=Path(args.review),
    )
    latencies = load_latency(Path(args.latency_log))

    metrics = evaluate(preds, gt)
    latency = compute_latency_stats(latencies)

    n_gt_total   = len(gt)
    n_pred_total = len(preds)

    console = build_console_report(
        metrics, latency,
        n_pred_total=n_pred_total,
        n_gt_total=n_gt_total,
    )
    print(console)

    if not args.no_csv:
        reports_dir = Path(args.reports_dir)
        written = export_csv(metrics, latency, reports_dir)

        md_path = reports_dir / "eval_results.md"
        md = build_markdown_report(
            metrics, latency,
            n_pred_total=n_pred_total,
            n_gt_total=n_gt_total,
        )
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        written["eval_results_md"] = md_path

        print(f"\nOutputs written to {reports_dir}/")
        for key, path in written.items():
            print(f"  {key:<25} {path}")


if __name__ == "__main__":
    main()
