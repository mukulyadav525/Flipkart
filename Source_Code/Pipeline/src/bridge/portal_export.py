"""
Gridlock  ->  TrafficEye portal exporter.

Reads a Gridlock run's *summary JSON* (produced by
``Pipeline/gridlock/scripts/run_all.py`` — it contains the full event list) and,
optionally, an *evidence ledger* (``ledger.json`` from ``--evidence``/``--anpr``,
which carries per-violation snapshots and plate reads).  It then writes the exact
files the Backend already consumes:

    outputs/violation_records/confirmed.jsonl   confidence >= AUTO_PROCESS_CUTOFF
    outputs/human_review_queue.jsonl            confidence <  AUTO_PROCESS_CUTOFF
    outputs/reports/analytics.json              dashboard summary
    outputs/annotated_images/<id>.jpg           copied violation snapshots
    outputs/latency_log.jsonl                   synthesised from the run's fps

Type mapping (Gridlock event ``type`` -> portal ``ViolationType``):
    no_helmet   -> helmet
    no_seatbelt -> seatbelt
    (all other types map unchanged)

Each Gridlock ``ViolationEvent`` carries a ``timestamp`` in seconds-into-the-clip.
We anchor those to a wall-clock capture time (``--captured-at``, default: now)
so the portal's time-of-day analytics and human-readable timestamps work.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# --- Make the Pipeline project importable (config + shared at root, packages
#     such as `analytics` under src/ — matching the repo's test convention). ---
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for _p in (PIPELINE_ROOT, PIPELINE_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import AUTO_PROCESS_CUTOFF                       # noqa: E402
from shared.schemas import ViolationType                     # noqa: E402
from analytics.report import (                               # noqa: E402
    counts_by_type as _report_counts_by_type,
    counts_by_time_of_day as _report_tod,
    repeat_plates as _report_repeat_plates,
    severity_ranking as _report_severity,
)

# ---------------------------------------------------------------------------
# Default output locations (relative to the Pipeline root)
# ---------------------------------------------------------------------------

OUTPUTS_DIR     = PIPELINE_ROOT / "outputs"
CONFIRMED_JSONL = OUTPUTS_DIR / "violation_records" / "confirmed.jsonl"
REVIEW_JSONL    = OUTPUTS_DIR / "human_review_queue.jsonl"
ANALYTICS_JSON  = OUTPUTS_DIR / "reports" / "analytics.json"
ANNOTATED_DIR   = OUTPUTS_DIR / "annotated_images"
LATENCY_JSONL   = OUTPUTS_DIR / "latency_log.jsonl"

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

# Gridlock emits "no_helmet"/"no_seatbelt"; the portal vocabulary uses the
# bare nouns.  Everything else already lines up with ViolationType.
_TYPE_MAP: dict[str, str] = {
    "no_helmet":   ViolationType.helmet.value,
    "no_seatbelt": ViolationType.seatbelt.value,
}

# Gridlock detector class names -> the portal's five VehicleClass labels.
_CLASS_MAP: dict[str, str] = {
    "motorcycle": "bike",
    "bicycle":    "bike",
    "car":        "car",
    "bus":        "bus",
    "truck":      "truck",
    "person":     "pedestrian",
}

# Fallback confidence per Gridlock type when no per-event confidence is present.
# Geometry/signal violations fire only after a sustained check, so they are
# assigned high confidence; perception ones inherit a moderate prior.
_DEFAULT_CONF: dict[str, float] = {
    "no_helmet":       0.82,
    "no_seatbelt":     0.80,
    "triple_riding":   0.85,
    "illegal_parking": 0.90,
    "wrong_side":      0.88,
    "stop_line":       0.90,
    "red_light":       0.93,
}


# ---------------------------------------------------------------------------
# Field derivation helpers
# ---------------------------------------------------------------------------

def _portal_type(gridlock_type: str) -> Optional[str]:
    """Map a Gridlock event type to a portal ViolationType value, or None."""
    mapped = _TYPE_MAP.get(gridlock_type, gridlock_type)
    try:
        return ViolationType(mapped).value
    except ValueError:
        return None


def _confidence(event: dict) -> float:
    """Best available confidence in [0, 1] for an event."""
    extra = event.get("extra") or {}
    # Any "*conf*" key the engine attached wins (e.g. helmet head_conf).
    for k, v in extra.items():
        if "conf" in k.lower():
            try:
                return max(0.0, min(1.0, round(float(v), 2)))
            except (TypeError, ValueError):
                pass
    gtype = event.get("type", "")
    if gtype == "triple_riding":
        riders = int(extra.get("rider_count", 3))
        return max(0.0, min(0.99, round(0.62 + 0.09 * (riders - 2), 2)))
    return _DEFAULT_CONF.get(gtype, 0.75)


def _iso_timestamp(base: datetime, seconds_into_clip: float) -> str:
    return (base + timedelta(seconds=float(seconds_into_clip))).isoformat()


def _image_id(stem: str, frame_idx: int) -> str:
    return f"{stem}_frame{int(frame_idx):06d}"


def _detection_id(image_id: str, event: dict) -> str:
    portal_class = _CLASS_MAP.get(event.get("class_name", ""), event.get("class_name", "object"))
    return f"{image_id}:{portal_class}:track{event.get('track_id', -1)}"


def _rule_trace(event: dict, portal_type: str, confidence: float) -> str:
    detail = event.get("detail", "").strip()
    extra = event.get("extra") or {}
    extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
    parts = [
        f"Gridlock engine '{event.get('type')}' fired on track "
        f"#{event.get('track_id')} ({event.get('class_name')}) "
        f"at t={event.get('timestamp', 0):.1f}s (frame {event.get('frame_idx')}).",
    ]
    if detail:
        parts.append(detail.capitalize() + ".")
    if extra_str:
        parts.append(f"Evidence: {extra_str}.")
    parts.append(f"Mapped to portal violation '{portal_type}' at confidence {confidence:.2f}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Evidence ledger (snapshots + plates) — optional
# ---------------------------------------------------------------------------

def _load_ledger(ledger_path: Optional[Path]) -> dict[tuple[str, int, str], dict]:
    """Index a Gridlock evidence ledger by (type, track_id, timestamp@0.1s)."""
    if ledger_path is None or not ledger_path.exists():
        return {}
    try:
        rows = json.loads(ledger_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    index: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        key = (row.get("type", ""), int(row.get("track_id", -1)),
               f"{float(row.get('timestamp', 0.0)):.1f}")
        index[key] = row
    return index


def _ledger_lookup(index: dict, event: dict) -> dict:
    key = (event.get("type", ""), int(event.get("track_id", -1)),
           f"{float(event.get('timestamp', 0.0)):.1f}")
    return index.get(key, {})


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def event_to_evidence(
    event: dict,
    *,
    stem: str,
    base_time: datetime,
    ledger_index: dict,
    annotated_dir: Path,
) -> Optional[dict]:
    """Convert one Gridlock event dict to a portal EvidenceRecord dict."""
    portal_type = _portal_type(event.get("type", ""))
    if portal_type is None:
        return None  # unknown/internal event type — skip

    confidence = _confidence(event)
    frame_idx = event.get("frame_idx", 0)
    image_id = _image_id(stem, frame_idx)
    timestamp = _iso_timestamp(base_time, event.get("timestamp", 0.0))

    ledger = _ledger_lookup(ledger_index, event)
    plate_text = str(ledger.get("plate", "") or "")
    try:
        plate_conf = float(ledger.get("plate_conf") or 0.0)
    except (TypeError, ValueError):
        plate_conf = 0.0

    # Copy the snapshot into the portal's image dir (served by Backend at /images).
    annotated_rel = ""
    snapshot = ledger.get("snapshot")
    if snapshot and Path(snapshot).exists():
        annotated_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{image_id}_{portal_type}.jpg"
        try:
            shutil.copyfile(snapshot, annotated_dir / dest_name)
            annotated_rel = f"outputs/annotated_images/{dest_name}"
        except OSError:
            annotated_rel = ""

    is_confirmed = confidence >= AUTO_PROCESS_CUTOFF
    record_id = f"{portal_type}_t{event.get('track_id')}_f{frame_idx}"

    return {
        "id": record_id,
        "status": "confirmed" if is_confirmed else "pending",
        "violation_record": {
            "image_id": image_id,
            "violation_type": portal_type,
            "confidence": confidence,
            "rule_trace": _rule_trace(event, portal_type, confidence),
            "related_detection_ids": [_detection_id(image_id, event)],
            "related_plate_text": plate_text or None,
        },
        "annotated_image_path": annotated_rel,
        "timestamp": timestamp,
        "plate_text": plate_text,
        "plate_confidence": round(plate_conf, 2),
    }


# ---------------------------------------------------------------------------
# Analytics assembly (reuses Pipeline/src/analytics/report.py logic)
# ---------------------------------------------------------------------------

_TOD_BUCKETS = ["Night", "Morning", "Afternoon", "Evening"]


def build_analytics(confirmed: list[dict], pending_count: int, *, now: datetime) -> dict:
    all_types = [vt.value for vt in ViolationType]

    raw_counts = _report_counts_by_type(confirmed)
    counts_by_type = {vt: int(raw_counts.get(vt, 0)) for vt in all_types}

    # Time windows (records carry ISO timestamps).
    today = now.date()
    week_ago = now - timedelta(days=7)
    total_today = total_week = 0
    for r in confirmed:
        ts = _parse_dt(r.get("timestamp", ""))
        if ts is None:
            continue
        if ts.date() == today:
            total_today += 1
        if ts >= week_ago:
            total_week += 1

    # Time-of-day breakdown, zero-filled so the dashboard never KeyErrors.
    tod_raw = _report_tod(confirmed)
    tod_breakdown: dict[str, dict[str, int]] = {}
    for bucket in _TOD_BUCKETS:
        bucket_counts = tod_raw.get(bucket, {})
        tod_breakdown[bucket] = {vt: int(bucket_counts.get(vt, 0)) for vt in all_types}

    return {
        "total_today": total_today,
        "total_this_week": total_week,
        "pending_review": pending_count,
        "counts_by_type": counts_by_type,
        "severity_ranking": _report_severity(confirmed),
        "repeat_offenders": _report_repeat_plates(confirmed, min_count=2),
        "tod_breakdown": tod_breakdown,
    }


def _parse_dt(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Latency log (synthesised from the run's measured fps)
# ---------------------------------------------------------------------------

def write_latency_log(summary: dict, path: Path, *, max_entries: int = 500) -> int:
    fps = float(summary.get("fps") or 0.0)
    frames = int(summary.get("frames_processed") or 0)
    if fps <= 0 or frames <= 0:
        return 0
    per_frame_ms = round(1000.0 / fps, 3)
    n = min(frames, max_entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"image_id": f"frame_{i:06d}", "total_ms": per_frame_ms}) + "\n")
    return n


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------

def export(
    summary_path: Path,
    *,
    ledger_path: Optional[Path] = None,
    captured_at: Optional[datetime] = None,
    outputs_dir: Path = OUTPUTS_DIR,
    append: bool = False,
) -> dict:
    """Convert a Gridlock summary (+ optional ledger) into portal outputs.

    Returns a small dict of counts for logging.
    """
    summary = json.loads(Path(summary_path).read_text())
    events = summary.get("events", [])
    stem = Path(summary.get("source", "clip")).stem
    base_time = captured_at or datetime.now(timezone.utc)
    ledger_index = _load_ledger(ledger_path)

    confirmed_path = outputs_dir / "violation_records" / "confirmed.jsonl"
    review_path    = outputs_dir / "human_review_queue.jsonl"
    analytics_path = outputs_dir / "reports" / "analytics.json"
    annotated_dir  = outputs_dir / "annotated_images"
    latency_path   = outputs_dir / "latency_log.jsonl"

    confirmed: list[dict] = []
    review: list[dict] = []
    skipped = 0
    for ev in events:
        rec = event_to_evidence(
            ev, stem=stem, base_time=base_time,
            ledger_index=ledger_index, annotated_dir=annotated_dir,
        )
        if rec is None:
            skipped += 1
            continue
        (confirmed if rec["status"] == "confirmed" else review).append(rec)

    # When appending, fold in records already on disk so analytics stays whole.
    if append:
        confirmed = _read_jsonl(confirmed_path) + confirmed
        review = _read_jsonl(review_path) + review

    _write_jsonl(confirmed_path, confirmed)
    _write_jsonl(review_path, review)

    analytics = build_analytics(confirmed, len(review), now=base_time)
    analytics_path.parent.mkdir(parents=True, exist_ok=True)
    analytics_path.write_text(json.dumps(analytics, indent=2))

    latency_n = write_latency_log(summary, latency_path)

    return {
        "events_in": len(events),
        "confirmed": len(confirmed),
        "review": len(review),
        "skipped": skipped,
        "latency_entries": latency_n,
        "outputs_dir": str(outputs_dir),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bridge.portal_export",
        description="Convert a Gridlock run summary into TrafficEye portal outputs.",
    )
    p.add_argument("summary", help="Path to Gridlock <stem>_all_summary.json")
    p.add_argument("--ledger", default=None,
                   help="Optional Gridlock evidence ledger.json (snapshots + plates)")
    p.add_argument("--captured-at", default=None,
                   help="ISO-8601 wall-clock time of clip start (default: now, UTC)")
    p.add_argument("--outputs-dir", default=str(OUTPUTS_DIR),
                   help=f"Portal outputs directory (default: {OUTPUTS_DIR})")
    p.add_argument("--append", action="store_true",
                   help="Append to existing portal records instead of overwriting")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    captured_at = _parse_dt(args.captured_at) if args.captured_at else None
    if args.captured_at and captured_at is None:
        raise SystemExit(f"--captured-at: not a valid ISO-8601 datetime: {args.captured_at!r}")

    result = export(
        Path(args.summary),
        ledger_path=Path(args.ledger) if args.ledger else None,
        captured_at=captured_at,
        outputs_dir=Path(args.outputs_dir),
        append=args.append,
    )
    print("[bridge] Gridlock -> portal export complete")
    print(f"  events read     : {result['events_in']}")
    print(f"  confirmed       : {result['confirmed']}  -> outputs/violation_records/confirmed.jsonl")
    print(f"  needs review    : {result['review']}  -> outputs/human_review_queue.jsonl")
    print(f"  skipped (type)  : {result['skipped']}")
    print(f"  latency entries : {result['latency_entries']}  -> outputs/latency_log.jsonl")
    print(f"  analytics       : outputs/reports/analytics.json")


if __name__ == "__main__":
    main()
