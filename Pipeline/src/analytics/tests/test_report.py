"""
Unit tests for src/analytics/report.py.

All tests use hand-crafted record dicts that match the shape written by
evidence/generate.py.  No disk I/O except the CSV export tests which use
tempfile.TemporaryDirectory.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from analytics.report import (
    counts_by_time_of_day,
    counts_by_type,
    export_csv,
    repeat_plates,
    severity_ranking,
    build_summary,
)


# ---------------------------------------------------------------------------
# Helpers — minimal EvidenceRecord dicts matching the JSONL schema
# ---------------------------------------------------------------------------

def _rec(
    vtype: str = "helmet",
    conf: float = 0.90,
    plate: str = "MH12AB1234",
    plate_conf: float = 0.88,
    timestamp: str = "2025-06-01T08:30:00+00:00",
) -> dict:
    return {
        "violation_record": {
            "image_id":             "frame_001",
            "violation_type":       vtype,
            "confidence":           conf,
            "rule_trace":           "...",
            "related_detection_ids": [],
            "related_plate_text":   None,
        },
        "annotated_image_path": f"/tmp/{vtype}.jpg",
        "timestamp":            timestamp,
        "plate_text":           plate,
        "plate_confidence":     plate_conf,
    }


# ---------------------------------------------------------------------------
# counts_by_type
# ---------------------------------------------------------------------------

class TestCountsByType:

    def test_basic_counts(self):
        records = [
            _rec("helmet"),
            _rec("helmet"),
            _rec("red_light"),
        ]
        result = counts_by_type(records)
        assert result["helmet"]    == 2
        assert result["red_light"] == 1

    def test_sorted_descending(self):
        records = [_rec("helmet")] * 3 + [_rec("red_light")] * 5 + [_rec("seatbelt")]
        result = counts_by_type(records)
        counts = list(result.values())
        assert counts == sorted(counts, reverse=True)

    def test_empty_records(self):
        assert counts_by_type([]) == {}

    def test_unknown_type_still_counted(self):
        r = _rec()
        r["violation_record"]["violation_type"] = "future_violation"
        result = counts_by_type([r])
        assert result.get("future_violation") == 1


# ---------------------------------------------------------------------------
# counts_by_time_of_day
# ---------------------------------------------------------------------------

class TestCountsByTimeOfDay:

    def test_morning_bucket(self):
        records = [_rec(timestamp="2025-06-01T08:00:00+00:00")]
        tod = counts_by_time_of_day(records)
        assert "Morning" in tod
        assert tod["Morning"].get("helmet") == 1

    def test_night_bucket(self):
        records = [_rec(timestamp="2025-06-01T03:00:00+00:00")]
        tod = counts_by_time_of_day(records)
        assert "Night" in tod

    def test_afternoon_bucket(self):
        records = [_rec(timestamp="2025-06-01T14:00:00+00:00")]
        tod = counts_by_time_of_day(records)
        assert "Afternoon" in tod

    def test_evening_bucket(self):
        records = [_rec(timestamp="2025-06-01T20:00:00+00:00")]
        tod = counts_by_time_of_day(records)
        assert "Evening" in tod

    def test_empty_returns_empty_dict(self):
        assert counts_by_time_of_day([]) == {}

    def test_missing_timestamp_returns_empty(self):
        r = _rec()
        r["timestamp"] = ""
        result = counts_by_time_of_day([r])
        # All timestamps unparseable → no has_any_timestamp → empty dict
        assert result == {}

    def test_mixed_timestamps(self):
        records = [
            _rec("helmet",    timestamp="2025-06-01T08:00:00+00:00"),
            _rec("red_light", timestamp="2025-06-01T22:00:00+00:00"),
            _rec("seatbelt",  timestamp="2025-06-01T08:30:00+00:00"),
        ]
        tod = counts_by_time_of_day(records)
        assert tod["Morning"]["helmet"]    == 1
        assert tod["Morning"]["seatbelt"]  == 1
        assert tod["Evening"]["red_light"] == 1

    def test_timezone_normalised_to_utc(self):
        # 23:00 IST = 17:30 UTC → Afternoon
        records = [_rec(timestamp="2025-06-01T23:00:00+05:30")]
        tod = counts_by_time_of_day(records)
        assert "Afternoon" in tod


# ---------------------------------------------------------------------------
# repeat_plates
# ---------------------------------------------------------------------------

class TestRepeatPlates:

    def test_flags_plate_seen_twice(self):
        records = [_rec(plate="MH12AB1234")] * 2
        result = repeat_plates(records, min_count=2)
        assert len(result) == 1
        assert result[0]["plate_text"]  == "MH12AB1234"
        assert result[0]["count"]       == 2

    def test_single_appearance_not_flagged(self):
        records = [_rec(plate="DL5SAB0001")]
        result = repeat_plates(records, min_count=2)
        assert result == []

    def test_empty_plate_excluded(self):
        records = [_rec(plate="")] * 5
        result = repeat_plates(records, min_count=2)
        assert result == []

    def test_sorted_by_frequency_desc(self):
        records = (
            [_rec(plate="AAA")] * 5 +
            [_rec(plate="BBB")] * 2 +
            [_rec(plate="CCC")] * 8
        )
        result = repeat_plates(records, min_count=2)
        counts = [r["count"] for r in result]
        assert counts == sorted(counts, reverse=True)

    def test_violation_types_collected(self):
        records = [
            _rec(vtype="helmet",    plate="MH12AB1234"),
            _rec(vtype="red_light", plate="MH12AB1234"),
        ]
        result = repeat_plates(records, min_count=2)
        assert "helmet"    in result[0]["violation_types"]
        assert "red_light" in result[0]["violation_types"]

    def test_last_seen_is_latest_timestamp(self):
        records = [
            _rec(plate="MH12AB1234", timestamp="2025-06-01T08:00:00+00:00"),
            _rec(plate="MH12AB1234", timestamp="2025-06-02T14:00:00+00:00"),
        ]
        result = repeat_plates(records, min_count=2)
        assert result[0]["last_seen_timestamp"] == "2025-06-02T14:00:00+00:00"

    def test_custom_min_count(self):
        records = [_rec(plate="MH12AB1234")] * 3
        assert repeat_plates(records, min_count=3) != []
        assert repeat_plates(records, min_count=4) == []


# ---------------------------------------------------------------------------
# severity_ranking
# ---------------------------------------------------------------------------

class TestSeverityRanking:

    def test_red_light_outranks_helmet(self):
        records = [
            _rec("red_light", conf=0.88),
            _rec("helmet",    conf=0.88),
        ]
        ranked = severity_ranking(records)
        types_in_order = [r["violation_type"] for r in ranked]
        assert types_in_order.index("red_light") < types_in_order.index("helmet")

    def test_score_formula(self):
        from config import SEVERITY
        from shared.schemas import ViolationType
        records = [_rec("red_light", conf=0.80)]
        ranked = severity_ranking(records)
        row = ranked[0]
        expected_score = 1 * SEVERITY.get(ViolationType.red_light) * 0.80
        assert row["severity_score"] == pytest.approx(expected_score, rel=1e-4)

    def test_higher_count_same_type_increases_score(self):
        single = severity_ranking([_rec("helmet", conf=0.90)])
        double = severity_ranking([_rec("helmet", conf=0.90)] * 2)
        assert double[0]["severity_score"] > single[0]["severity_score"]

    def test_empty_records_returns_empty(self):
        assert severity_ranking([]) == []

    def test_sorted_descending_by_score(self):
        records = (
            [_rec("helmet",       conf=0.85)] * 3 +
            [_rec("red_light",    conf=0.85)] * 1 +
            [_rec("wrong_side",   conf=0.85)] * 1
        )
        ranked = severity_ranking(records)
        scores = [r["severity_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_type_gets_default_weight_1(self):
        r = _rec()
        r["violation_record"]["violation_type"] = "future_type"
        ranked = severity_ranking([r])
        row = next(x for x in ranked if x["violation_type"] == "future_type")
        assert row["severity_weight"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:

    def test_empty_records_no_crash(self):
        summary = build_summary([])
        assert "TRAFFIC VIOLATION ANALYTICS REPORT" in summary
        assert "Records analysed: 0" in summary

    def test_contains_all_sections(self):
        records = [_rec("helmet"), _rec("red_light", plate="DL5SAB1234")]
        summary = build_summary(records)
        assert "VIOLATION COUNTS BY TYPE"         in summary
        assert "COUNTS BY TIME OF DAY"            in summary
        assert "REPEAT OFFENDERS"                 in summary
        assert "SEVERITY-WEIGHTED RANKING"        in summary

    def test_repeat_offender_appears_in_summary(self):
        records = [_rec(plate="MH12AB1234")] * 3
        summary = build_summary(records, repeat_min_count=2)
        assert "MH12AB1234" in summary

    def test_violation_type_count_appears(self):
        records = [_rec("helmet")] * 4
        summary = build_summary(records)
        assert "helmet" in summary
        assert "4" in summary


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------

class TestExportCsv:

    def _sample_records(self) -> list[dict]:
        return [
            _rec("helmet",    conf=0.90, plate="MH12AB1234", timestamp="2025-06-01T08:00:00+00:00"),
            _rec("helmet",    conf=0.85, plate="MH12AB1234", timestamp="2025-06-01T09:00:00+00:00"),
            _rec("red_light", conf=0.92, plate="DL5SAB0001", timestamp="2025-06-01T22:00:00+00:00"),
            _rec("seatbelt",  conf=0.78, plate="KA01MN5678", timestamp="2025-06-01T14:00:00+00:00"),
        ]

    def test_creates_four_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = export_csv(self._sample_records(), reports_dir=Path(tmpdir))
        assert len(written) == 4

    def test_counts_csv_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = export_csv(self._sample_records(), reports_dir=Path(tmpdir))
            path = written["counts_by_type"]
            rows = list(csv.DictReader(path.open()))
        by_type = {r["violation_type"]: int(r["count"]) for r in rows}
        assert by_type["helmet"] == 2
        assert by_type["red_light"] == 1
        assert by_type["seatbelt"] == 1

    def test_repeat_plates_csv_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = export_csv(self._sample_records(), reports_dir=Path(tmpdir))
            rows = list(csv.DictReader(written["repeat_plates"].open()))
        # Only MH12AB1234 appears twice
        assert len(rows) == 1
        assert rows[0]["plate_text"] == "MH12AB1234"
        assert rows[0]["count"] == "2"

    def test_severity_csv_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = export_csv(self._sample_records(), reports_dir=Path(tmpdir))
            rows = list(csv.DictReader(written["severity_ranking"].open()))
        scores = [float(r["severity_score"]) for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_empty_records_writes_valid_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = export_csv([], reports_dir=Path(tmpdir))
            for path in written.values():
                assert path.exists()
                assert path.stat().st_size > 0   # at least a header row
