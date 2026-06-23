"""
Analytics report for the traffic violation detection pipeline.

Reads from outputs/violation_records/confirmed.jsonl (the single source of
truth written by evidence/generate.py) and produces:

  1. Violation counts by type
  2. Counts by time-of-day bucket (Night / Morning / Afternoon / Evening)
     — only when ISO-8601 timestamps are present in the records
  3. Repeat-plate table: plates seen more than once, sorted by frequency
  4. Severity-weighted ranking: count × severity_weight × mean_confidence,
     weights configured in config.py (not here)

Outputs
-------
  CLI  : structured text summary printed to stdout
  CSV  : one file per report section written to outputs/reports/

No dashboard is built here — visualisation will be added once real data
from the live pipeline is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from config import SEVERITY
from shared.schemas import ViolationType

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIRMED_JSONL = Path("outputs/violation_records/confirmed.jsonl")
REPORTS_DIR     = Path("outputs/reports")

# ---------------------------------------------------------------------------
# Time-of-day buckets
# ---------------------------------------------------------------------------

_TOD_BUCKETS: list[tuple[str, range]] = [
    ("Night",     range(0,  6)),   # 00:00 – 05:59
    ("Morning",   range(6,  12)),  # 06:00 – 11:59
    ("Afternoon", range(12, 18)),  # 12:00 – 17:59
    ("Evening",   range(18, 24)),  # 18:00 – 23:59
]


def _hour_to_bucket(hour: int) -> str:
    for name, hours in _TOD_BUCKETS:
        if hour in hours:
            return name
    return "Unknown"


def _parse_hour(timestamp: str) -> int | None:
    """Return the UTC hour from an ISO-8601 string, or None on parse failure."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.astimezone(timezone.utc).hour
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _iter_records(jsonl_path: Path) -> Iterator[dict]:
    if not jsonl_path.exists():
        return
    with jsonl_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[report] Skipping malformed line {lineno}: {exc}", file=sys.stderr)


def load_records(jsonl_path: Path = CONFIRMED_JSONL) -> list[dict]:
    return list(_iter_records(jsonl_path))


# ---------------------------------------------------------------------------
# Analysis functions — each takes list[dict], returns a structured result
# ---------------------------------------------------------------------------

def counts_by_type(records: list[dict]) -> dict[str, int]:
    """Return {violation_type: count} sorted descending by count."""
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        vtype = r.get("violation_record", {}).get("violation_type", "unknown")
        counts[vtype] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def counts_by_time_of_day(records: list[dict]) -> dict[str, dict[str, int]]:
    """
    Return {bucket: {violation_type: count}}.

    Records without a parseable timestamp are collected under "Unknown".
    Returns an empty dict if no records have timestamps at all.
    """
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    has_any_timestamp = False

    for r in records:
        ts = r.get("timestamp", "")
        hour = _parse_hour(ts)
        bucket = _hour_to_bucket(hour) if hour is not None else "Unknown"
        vtype = r.get("violation_record", {}).get("violation_type", "unknown")
        result[bucket][vtype] += 1
        if hour is not None:
            has_any_timestamp = True

    if not has_any_timestamp:
        return {}

    # Convert inner defaultdicts and order buckets chronologically
    ordered: dict[str, dict[str, int]] = {}
    for name, _ in _TOD_BUCKETS:
        if name in result:
            ordered[name] = dict(result[name])
    if "Unknown" in result:
        ordered["Unknown"] = dict(result["Unknown"])
    return ordered


def repeat_plates(records: list[dict], min_count: int = 2) -> list[dict]:
    """
    Return records for plates seen >= min_count times, sorted by frequency desc.

    Each entry: {plate_text, count, violation_types, last_seen_timestamp}
    Plates with empty plate_text are excluded (no OCR result).
    """
    plate_data: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "violation_types": set(), "last_seen_timestamp": ""}
    )

    for r in records:
        plate = r.get("plate_text", "").strip()
        if not plate:
            continue
        entry = plate_data[plate]
        entry["count"] += 1
        vtype = r.get("violation_record", {}).get("violation_type", "unknown")
        entry["violation_types"].add(vtype)
        ts = r.get("timestamp", "")
        if ts > entry["last_seen_timestamp"]:
            entry["last_seen_timestamp"] = ts

    results = [
        {
            "plate_text":           plate,
            "count":                data["count"],
            "violation_types":      sorted(data["violation_types"]),
            "last_seen_timestamp":  data["last_seen_timestamp"],
        }
        for plate, data in plate_data.items()
        if data["count"] >= min_count
    ]
    return sorted(results, key=lambda x: x["count"], reverse=True)


def severity_ranking(records: list[dict]) -> list[dict]:
    """
    Compute a severity score per violation type:
        score = count × severity_weight × mean_confidence

    Returns list of dicts sorted by score descending.
    Weights come from config.SEVERITY — never hardcoded here.
    """
    accumulator: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "confidence_sum": 0.0}
    )

    for r in records:
        vtype = r.get("violation_record", {}).get("violation_type", "unknown")
        conf  = r.get("violation_record", {}).get("confidence", 0.0)
        accumulator[vtype]["count"]          += 1
        accumulator[vtype]["confidence_sum"] += conf

    ranked: list[dict] = []
    for vtype_str, data in accumulator.items():
        count      = data["count"]
        mean_conf  = data["confidence_sum"] / count if count else 0.0
        try:
            vtype_enum = ViolationType(vtype_str)
            weight = SEVERITY.get(vtype_enum)
        except ValueError:
            weight = 1.0
        score = count * weight * mean_conf
        ranked.append({
            "violation_type": vtype_str,
            "count":          count,
            "mean_confidence": round(mean_conf, 4),
            "severity_weight": weight,
            "severity_score":  round(score, 4),
        })

    return sorted(ranked, key=lambda x: x["severity_score"], reverse=True)


# ---------------------------------------------------------------------------
# Report summary — structured text for stdout
# ---------------------------------------------------------------------------

_SECTION = "=" * 60
_SUBSECT  = "-" * 40


def _fmt_pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "—"


def build_summary(
    records: list[dict],
    *,
    repeat_min_count: int = 2,
) -> str:
    total = len(records)
    lines: list[str] = []

    lines += [
        _SECTION,
        "  TRAFFIC VIOLATION ANALYTICS REPORT",
        f"  Records analysed: {total}",
        f"  Source: {CONFIRMED_JSONL}",
        _SECTION,
        "",
    ]

    # --- Section 1: counts by type ---
    lines += ["VIOLATION COUNTS BY TYPE", _SUBSECT]
    by_type = counts_by_type(records)
    if by_type:
        col_w = max(len(k) for k in by_type) + 2
        for vtype, count in by_type.items():
            lines.append(f"  {vtype:<{col_w}} {count:>5}   ({_fmt_pct(count, total)})")
    else:
        lines.append("  (no records)")
    lines.append("")

    # --- Section 2: time-of-day ---
    lines += ["COUNTS BY TIME OF DAY (UTC)", _SUBSECT]
    tod = counts_by_time_of_day(records)
    if tod:
        all_vtypes = sorted({vt for bucket in tod.values() for vt in bucket})
        # Header
        header = f"  {'Bucket':<12}" + "".join(f"  {vt[:10]:>10}" for vt in all_vtypes) + "   Total"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for bucket, type_counts in tod.items():
            bucket_total = sum(type_counts.values())
            row = f"  {bucket:<12}" + "".join(
                f"  {type_counts.get(vt, 0):>10}" for vt in all_vtypes
            ) + f"   {bucket_total:>5}"
            lines.append(row)
    else:
        lines.append("  (no timestamp data available)")
    lines.append("")

    # --- Section 3: repeat plates ---
    lines += [f"REPEAT OFFENDERS (plate seen >= {repeat_min_count}×)", _SUBSECT]
    repeats = repeat_plates(records, min_count=repeat_min_count)
    if repeats:
        for entry in repeats:
            vtypes_str = ", ".join(entry["violation_types"])
            lines.append(
                f"  {entry['plate_text']:<16}  {entry['count']:>3}×   "
                f"violations: {vtypes_str}"
            )
            if entry["last_seen_timestamp"]:
                lines.append(f"    last seen: {entry['last_seen_timestamp']}")
    else:
        lines.append("  (none)")
    lines.append("")

    # --- Section 4: severity ranking ---
    lines += ["SEVERITY-WEIGHTED RANKING", _SUBSECT]
    lines.append("  (score = count × weight × mean_confidence;  weights from config.SEVERITY)")
    lines.append(
        f"  {'Type':<18}  {'Count':>5}  {'Weight':>6}  {'MeanConf':>8}  {'Score':>8}"
    )
    lines.append("  " + "-" * 56)
    ranked = severity_ranking(records)
    if ranked:
        for row in ranked:
            lines.append(
                f"  {row['violation_type']:<18}  {row['count']:>5}  "
                f"{row['severity_weight']:>6.1f}  {row['mean_confidence']:>8.3f}  "
                f"{row['severity_score']:>8.3f}"
            )
    else:
        lines.append("  (no records)")
    lines.append("")
    lines.append(_SECTION)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(records: list[dict], reports_dir: Path = REPORTS_DIR) -> dict[str, Path]:
    """
    Write four CSV files to reports_dir.  Returns {section_name: path}.
    Creates the directory if absent.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # 1 — counts by type
    path = reports_dir / "counts_by_type.csv"
    by_type = counts_by_type(records)
    total = len(records)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["violation_type", "count", "pct_of_total"])
        for vtype, count in by_type.items():
            w.writerow([vtype, count, f"{count / total * 100:.2f}" if total else "0"])
    written["counts_by_type"] = path

    # 2 — counts by time of day (wide format: rows=buckets, cols=types)
    path = reports_dir / "counts_by_time_of_day.csv"
    tod = counts_by_time_of_day(records)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if tod:
            all_vtypes = sorted({vt for bucket in tod.values() for vt in bucket})
            w.writerow(["time_bucket"] + all_vtypes + ["total"])
            for bucket, type_counts in tod.items():
                row = [bucket] + [type_counts.get(vt, 0) for vt in all_vtypes]
                row.append(sum(type_counts.values()))
                w.writerow(row)
        else:
            w.writerow(["note"])
            w.writerow(["no timestamp data available"])
    written["counts_by_time_of_day"] = path

    # 3 — repeat plates
    path = reports_dir / "repeat_plates.csv"
    repeats = repeat_plates(records)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plate_text", "count", "violation_types", "last_seen_timestamp"])
        for entry in repeats:
            w.writerow([
                entry["plate_text"],
                entry["count"],
                "|".join(entry["violation_types"]),
                entry["last_seen_timestamp"],
            ])
    written["repeat_plates"] = path

    # 4 — severity ranking
    path = reports_dir / "severity_ranking.csv"
    ranked = severity_ranking(records)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["violation_type", "count", "severity_weight",
                    "mean_confidence", "severity_score"])
        for row in ranked:
            w.writerow([
                row["violation_type"],
                row["count"],
                row["severity_weight"],
                row["mean_confidence"],
                row["severity_score"],
            ])
    written["severity_ranking"] = path

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analytics.report",
        description="Generate analytics report from confirmed violation records.",
    )
    p.add_argument(
        "--input", default=str(CONFIRMED_JSONL), metavar="PATH",
        help=f"JSONL source file (default: {CONFIRMED_JSONL})",
    )
    p.add_argument(
        "--reports-dir", default=str(REPORTS_DIR), metavar="PATH",
        help=f"Directory for CSV output (default: {REPORTS_DIR})",
    )
    p.add_argument(
        "--repeat-min", type=int, default=2, metavar="N",
        help="Minimum appearances to flag a plate as repeat offender (default: 2)",
    )
    p.add_argument(
        "--no-csv", action="store_true",
        help="Print summary only, skip CSV export",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    records = load_records(Path(args.input))

    if not records:
        print(f"[report] No records found in {args.input}. "
              "Run the detection pipeline to generate data.", file=sys.stderr)
        # Still print an empty report so callers get clean output
        print(build_summary([], repeat_min_count=args.repeat_min))
        sys.exit(0)

    print(build_summary(records, repeat_min_count=args.repeat_min))

    if not args.no_csv:
        written = export_csv(records, reports_dir=Path(args.reports_dir))
        print(f"\nCSV files written to {args.reports_dir}/")
        for section, path in written.items():
            print(f"  {section:<25} {path}")


if __name__ == "__main__":
    main()
