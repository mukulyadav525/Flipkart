"""
TrafficEye Portal — FastAPI backend.

Reads from Pipeline outputs when they exist; falls back to
Backend/mock_data/ automatically.  No auth for prototype.

Start with:
    cd Backend && uvicorn main:app --reload --port 8000

Environment variables:
    PIPELINE_ROOT   Path to the Pipeline directory (default: ../Pipeline)
    AUDIT_LOG_PATH  Where to persist review actions (default: Backend/data/audit_log.jsonl)
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent
PIPELINE_ROOT = Path(os.getenv("PIPELINE_ROOT", BASE_DIR / ".." / "Pipeline"))
MOCK_DIR      = BASE_DIR / "mock_data"
AUDIT_LOG     = Path(os.getenv("AUDIT_LOG_PATH", BASE_DIR / "data" / "audit_log.jsonl"))

# All seven violation classes, in a stable display order.
ALL_VIOLATION_TYPES: list[str] = [
    "helmet", "seatbelt", "triple_riding",
    "wrong_side", "stop_line", "red_light", "illegal_parking",
]

# Severity weights come from the Pipeline config (single source of truth) when
# importable; otherwise an equal-weight fallback keeps the endpoint working.
try:
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT.resolve()))
    from config import SEVERITY as _SEVERITY  # type: ignore
    from shared.schemas import ViolationType as _VT  # type: ignore

    SEVERITY_WEIGHTS: dict[str, float] = {
        vt: _SEVERITY.get(_VT(vt)) for vt in ALL_VIOLATION_TYPES
    }
except Exception:  # pragma: no cover - Pipeline not importable from this env
    SEVERITY_WEIGHTS = {vt: 1.0 for vt in ALL_VIOLATION_TYPES}

CONFIRMED_JSONL = PIPELINE_ROOT / "outputs" / "violation_records" / "confirmed.jsonl"
REVIEW_JSONL    = PIPELINE_ROOT / "outputs" / "human_review_queue.jsonl"
ANALYTICS_JSON  = PIPELINE_ROOT / "outputs" / "reports" / "analytics.json"
LATENCY_JSONL   = PIPELINE_ROOT / "outputs" / "latency_log.jsonl"
EVAL_CSV        = PIPELINE_ROOT / "outputs" / "reports" / "eval_metrics.csv"
ANNOTATED_DIR   = PIPELINE_ROOT / "outputs" / "annotated_images"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TrafficEye Portal API",
    version="0.1.0",
    description="Role-based traffic violation portal backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if ANNOTATED_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(ANNOTATED_DIR)), name="images")

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pipeline_has_output() -> bool:
    """True once the pipeline has produced any output (confirmed or review)."""
    return CONFIRMED_JSONL.exists() or REVIEW_JSONL.exists()


def _load_violations() -> list[dict]:
    """Real confirmed records once the pipeline has run (even if empty); else mock."""
    if _pipeline_has_output():
        records = _read_jsonl(CONFIRMED_JSONL)
        for i, r in enumerate(records):
            r.setdefault("id", f"ev_{i:04d}")
            r.setdefault("status", "confirmed")
        return records
    return _read_json(MOCK_DIR / "violations.json") or []


def _load_review_queue() -> list[dict]:
    if _pipeline_has_output():
        records = _read_jsonl(REVIEW_JSONL)
        for i, r in enumerate(records):
            r.setdefault("id", f"rq_{i:04d}")
            r.setdefault("status", "pending")
        return records
    return _read_json(MOCK_DIR / "review_queue.json") or []


def _vtype(rec: dict) -> str:
    return rec.get("violation_record", {}).get("violation_type", "")


def _vconf(rec: dict) -> float:
    return float(rec.get("violation_record", {}).get("confidence", 0.0) or 0.0)


_TOD_BUCKETS = [
    ("Night", range(0, 6)), ("Morning", range(6, 12)),
    ("Afternoon", range(12, 18)), ("Evening", range(18, 24)),
]


def _tod_bucket(hour: int) -> str:
    for name, hours in _TOD_BUCKETS:
        if hour in hours:
            return name
    return "Unknown"


def _parse_dt(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _build_analytics_summary() -> dict:
    """
    Compute the dashboard summary directly from the confirmed + review records.

    Nothing here is hardcoded — every figure is derived from the actual records
    on disk (or the mock fallback those loaders return).  Severity weights are
    pulled from Pipeline/config.py via SEVERITY_WEIGHTS.
    """
    confirmed = _load_violations()
    review    = _load_review_queue()

    now        = datetime.now(timezone.utc)
    today      = now.date()
    week_start = now - timedelta(days=7)

    counts: dict[str, int]       = {vt: 0 for vt in ALL_VIOLATION_TYPES}
    conf_sum: dict[str, float]   = defaultdict(float)
    tod: dict[str, dict[str, int]] = {name: {} for name, _ in _TOD_BUCKETS}
    plates: dict[str, dict]      = {}
    total_today = total_week = 0

    for r in confirmed:
        vt = _vtype(r)
        if vt in counts:
            counts[vt] += 1
            conf_sum[vt] += _vconf(r)

        dt = _parse_dt(r.get("timestamp", ""))
        if dt is not None:
            if dt.astimezone(timezone.utc).date() == today:
                total_today += 1
            if dt >= week_start:
                total_week += 1
            bucket = _tod_bucket(dt.astimezone(timezone.utc).hour)
            tod.setdefault(bucket, {})
            tod[bucket][vt] = tod[bucket].get(vt, 0) + 1

        plate = (r.get("plate_text") or "").strip().upper()
        if plate:
            entry = plates.setdefault(
                plate, {"count": 0, "violation_types": set(), "last_seen_timestamp": ""}
            )
            entry["count"] += 1
            entry["violation_types"].add(vt)
            ts = r.get("timestamp", "")
            if ts > entry["last_seen_timestamp"]:
                entry["last_seen_timestamp"] = ts

    severity_ranking = []
    for vt in ALL_VIOLATION_TYPES:
        c = counts[vt]
        mean_conf = conf_sum[vt] / c if c else 0.0
        weight = SEVERITY_WEIGHTS.get(vt, 1.0)
        severity_ranking.append({
            "violation_type":  vt,
            "count":           c,
            "severity_weight": weight,
            "mean_confidence": round(mean_conf, 4),
            "severity_score":  round(c * weight * mean_conf, 4),
        })
    severity_ranking.sort(key=lambda x: x["severity_score"], reverse=True)

    repeat_offenders = sorted(
        (
            {
                "plate_text":          plate,
                "count":               data["count"],
                "violation_types":     sorted(data["violation_types"]),
                "last_seen_timestamp": data["last_seen_timestamp"],
            }
            for plate, data in plates.items()
            if data["count"] >= 2
        ),
        key=lambda x: x["count"], reverse=True,
    )

    actioned = _actioned_record_ids()
    pending = len([r for r in review if r.get("status") == "pending" and r.get("id") not in actioned])

    return {
        "total_today":     total_today,
        "total_this_week": total_week or len(confirmed),
        "pending_review":  pending,
        "counts_by_type":  counts,
        "severity_ranking": severity_ranking,
        "repeat_offenders": repeat_offenders,
        "tod_breakdown":   {k: v for k, v in tod.items() if v},
    }


def _load_analytics() -> dict:
    """
    Prefer a live summary computed from the records on disk.  Fall back to a
    pre-baked analytics.json (e.g. produced by the Gridlock bridge) and finally
    to the bundled mock summary so the dashboard is never empty.
    """
    if CONFIRMED_JSONL.exists() or REVIEW_JSONL.exists():
        return _build_analytics_summary()
    data = _read_json(ANALYTICS_JSON)
    if data:
        return data
    return _read_json(MOCK_DIR / "analytics.json") or {}


def _load_audit_log() -> list[dict]:
    return _read_jsonl(AUDIT_LOG)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ReviewAction(BaseModel):
    action: Literal["approved", "rejected"]
    reviewer_id: str = "officer_stub"
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# CITIZEN endpoints
# ---------------------------------------------------------------------------

citizen_router_prefix = "/api/citizen"


@app.get(f"{citizen_router_prefix}/violations")
def citizen_violations(
    plate: str = Query(..., description="Plate number to query"),
) -> list[dict]:
    """
    Return confirmed violations for a single plate.

    Citizens may only look up their own vehicle, so the result is always scoped
    to the supplied plate.  The full violation_record (including rule_trace) is
    returned for transparency.
    """
    target = plate.strip().upper()
    return [
        r for r in _load_violations()
        if (r.get("plate_text") or "").strip().upper() == target
    ]


# ---------------------------------------------------------------------------
# POLICE endpoints
# ---------------------------------------------------------------------------

police_router_prefix = "/api/police"


@app.get(f"{police_router_prefix}/violations")
def police_violations(
    violation_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    records = _load_violations()

    if violation_type:
        records = [r for r in records if r.get("violation_record", {}).get("violation_type") == violation_type]
    if status:
        records = [r for r in records if r.get("status") == status]
    if date_from:
        records = [r for r in records if r.get("timestamp", "") >= date_from]
    if date_to:
        records = [r for r in records if r.get("timestamp", "") <= date_to]

    total = len(records)
    return {"total": total, "records": records[offset: offset + limit]}


def _actioned_record_ids() -> set[str]:
    """record_ids that have already been approved/rejected (from the audit log)."""
    return {e.get("record_id") for e in _load_audit_log()}


@app.get(f"{police_router_prefix}/review-queue")
def review_queue() -> list[dict]:
    """Pending review records that have not yet been actioned by an officer."""
    actioned = _actioned_record_ids()
    return [
        r for r in _load_review_queue()
        if r.get("status") == "pending" and r.get("id") not in actioned
    ]


@app.post(f"{police_router_prefix}/review/{{record_id}}")
def submit_review(record_id: str, body: ReviewAction) -> dict:
    """
    Approve or reject a record from the human review queue.

    Always appends to the immutable audit log.  On **approve**, the challan is
    *issued* — the reviewed record is promoted into confirmed.jsonl so it shows
    up for the citizen, in the confirmed log, and in analytics.  On **reject**
    it is simply dropped (logged as a false positive).
    """
    record = next((r for r in _load_review_queue() if r.get("id") == record_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found in review queue")

    audit_entry = {
        "id":             str(uuid.uuid4()),
        "record_id":      record_id,
        "action":         body.action,
        "reviewer_id":    body.reviewer_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "plate_text":     record.get("plate_text", ""),
        "violation_type": record.get("violation_record", {}).get("violation_type", ""),
        "notes":          body.notes,
    }
    _append_jsonl(AUDIT_LOG, audit_entry)

    issued = None
    if body.action == "approved":
        issued = {k: v for k, v in record.items() if k != "status"}
        issued["status"] = "approved"
        issued["reviewed_by"] = body.reviewer_id
        issued["reviewed_at"] = audit_entry["timestamp"]
        _append_jsonl(CONFIRMED_JSONL, issued)

    return {"status": "ok", "action": body.action, "issued": issued, "audit_entry": audit_entry}


@app.get(f"{police_router_prefix}/repeat-offenders")
def repeat_offenders(min_count: int = 2) -> list[dict]:
    analytics = _load_analytics()
    offenders = analytics.get("repeat_offenders", [])
    return [o for o in offenders if o.get("count", 0) >= min_count]


@app.get(f"{police_router_prefix}/summary")
def police_summary() -> dict:
    return _load_analytics()


# ---------------------------------------------------------------------------
# ADMIN endpoints
# ---------------------------------------------------------------------------

admin_router_prefix = "/api/admin"


@app.get(f"{admin_router_prefix}/metrics")
def admin_metrics() -> dict:
    """
    System performance metrics.
    eval_metrics: None until Phase 7 evaluation runs and produces outputs/reports/eval_metrics.csv.
    latency: None until Pipeline produces outputs/latency_log.jsonl.
    """
    latency_records = _read_jsonl(LATENCY_JSONL)
    latency_stats: Optional[dict] = None
    if latency_records:
        ms_values = [r["total_ms"] for r in latency_records if "total_ms" in r]
        if ms_values:
            s = sorted(ms_values)
            n = len(s)
            latency_stats = {
                "n_images":       n,
                "mean_ms":        round(sum(s) / n, 2),
                "median_ms":      s[n // 2],
                "p95_ms":         s[min(int(0.95 * n), n - 1)],
                "throughput_fps": round(1000 / (sum(s) / n), 2),
            }

    eval_metrics: Optional[list[dict]] = None
    if EVAL_CSV.exists():
        import csv
        with EVAL_CSV.open(encoding="utf-8") as f:
            eval_metrics = list(csv.DictReader(f))

    return {
        "latency":      latency_stats,
        "eval_metrics": eval_metrics,
        "data_available": {
            "latency":   latency_stats is not None,
            "eval":      eval_metrics is not None,
        },
    }


@app.get(f"{admin_router_prefix}/audit-log")
def audit_log_endpoint(limit: int = 50, offset: int = 0) -> dict:
    entries = _load_audit_log()
    entries_desc = list(reversed(entries))
    return {"total": len(entries), "entries": entries_desc[offset: offset + limit]}


@app.get(f"{admin_router_prefix}/system-info")
def system_info() -> dict:
    weights_path = PIPELINE_ROOT / "weights" / "yolov8_idd.pt"
    return {
        "model_name":        "YOLOv8s + YOLOv8n-pose",
        "model_weights":     str(weights_path),
        "weights_exist":     weights_path.exists(),
        "dataset_name":      "Indian Driving Dataset (IDD)",
        "dataset_version":   "IDD Detection v1.0",
        "last_trained":      "pending — run Pipeline/src/detection/finetune.py",
        "pipeline_version":  "0.1.0",
        "pipeline_root":     str(PIPELINE_ROOT.resolve()),
        "outputs_root":      str((PIPELINE_ROOT / "outputs").resolve()),
        "confirmed_records": len(_load_violations()),
        "pending_review":    len([r for r in _load_review_queue() if r.get("status") == "pending" and r.get("id") not in _actioned_record_ids()]),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_mode": not _pipeline_has_output()}
