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
import uuid
from datetime import datetime, timezone
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


def _load_violations() -> list[dict]:
    """Load from pipeline output if it exists, else mock data."""
    pipeline = _read_jsonl(CONFIRMED_JSONL)
    if pipeline:
        # Pipeline records don't have id/status — add them
        for i, r in enumerate(pipeline):
            r.setdefault("id", f"ev_{i:04d}")
            r.setdefault("status", "confirmed")
        return pipeline
    return _read_json(MOCK_DIR / "violations.json") or []


def _load_review_queue() -> list[dict]:
    pipeline = _read_jsonl(REVIEW_JSONL)
    if pipeline:
        for i, r in enumerate(pipeline):
            r.setdefault("id", f"rq_{i:04d}")
            r.setdefault("status", "pending")
        return pipeline
    return _read_json(MOCK_DIR / "review_queue.json") or []


def _load_analytics() -> dict:
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
def citizen_violations(plate: str = Query(..., description="Plate number to query")) -> list[dict]:
    """
    Return confirmed violations for a specific plate number only.
    Citizens can only query their own vehicle's records.
    """
    plate = plate.strip().upper()
    records = _load_violations()
    matched = [
        r for r in records
        if r.get("plate_text", "").upper() == plate
    ]
    # Strip internal fields citizens don't need
    for r in matched:
        r.pop("violation_record", {})  # keep only top-level fields for now
        # Re-add violation_record for transparency (rule_trace is the point)
    return _load_violations_for_plate(plate)


def _load_violations_for_plate(plate: str) -> list[dict]:
    return [
        r for r in _load_violations()
        if r.get("plate_text", "").upper() == plate.upper()
    ]


# Override the route registered above with the correct implementation
app.routes.pop()  # remove the duplicate


@app.get(f"{citizen_router_prefix}/violations")
def citizen_violations_clean(plate: str = Query(...)) -> list[dict]:
    """Violations scoped to a single plate — citizen self-lookup only."""
    return _load_violations_for_plate(plate.strip().upper())


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


@app.get(f"{police_router_prefix}/review-queue")
def review_queue() -> list[dict]:
    return [r for r in _load_review_queue() if r.get("status") == "pending"]


@app.post(f"{police_router_prefix}/review/{{record_id}}")
def submit_review(record_id: str, body: ReviewAction) -> dict:
    """
    Approve or reject a record from the human review queue.
    Writes to the audit log; does not mutate the source JSONL
    (immutable event log pattern — analytics re-reads all sources).
    """
    queue = _load_review_queue()
    record = next((r for r in queue if r.get("id") == record_id), None)
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
    return {"status": "ok", "audit_entry": audit_entry}


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
        "pending_review":    len([r for r in _load_review_queue() if r.get("status") == "pending"]),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_mode": not CONFIRMED_JSONL.exists()}
