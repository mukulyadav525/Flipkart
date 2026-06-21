"""
Unit tests for src/evaluation/evaluate.py.

All tests use hand-crafted GT and prediction records — no disk reads,
no model calls, no real data required.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from evaluation.evaluate import (
    GTRecord,
    PredRecord,
    PerTypeMetrics,
    _compute_ap,
    _iou,
    _pred_matches_gt,
    compute_latency_stats,
    evaluate,
    export_csv,
    build_console_report,
    build_markdown_report,
    load_ground_truth,
    load_latency,
    load_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gt(vtype="helmet", image_id="img_001", bbox=None) -> GTRecord:
    return GTRecord(image_id=image_id, violation_type=vtype, bbox=bbox)


def _pred(vtype="helmet", conf=0.90, image_id="img_001", bbox=None) -> PredRecord:
    return PredRecord(image_id=image_id, violation_type=vtype,
                      confidence=conf, bbox=bbox)


def _bbox(x1=100, y1=100, x2=400, y2=400) -> dict:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

class TestIou:
    def test_perfect_overlap(self):
        b = _bbox()
        assert _iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = _bbox(0, 0, 100, 100)
        b = _bbox(200, 200, 300, 300)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _bbox(0, 0, 200, 200)
        b = _bbox(100, 100, 300, 300)
        # intersection 100×100=10000; union = 40000+40000-10000=70000; iou≈0.143
        assert 0.1 < _iou(a, b) < 0.2

    def test_adjacent_no_overlap(self):
        a = _bbox(0, 0, 100, 100)
        b = _bbox(100, 0, 200, 100)
        assert _iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

class TestPredMatchesGt:
    def test_same_image_and_type_no_bbox(self):
        assert _pred_matches_gt(_pred(), _gt())

    def test_different_image_no_match(self):
        assert not _pred_matches_gt(_pred(image_id="A"), _gt(image_id="B"))

    def test_different_type_no_match(self):
        assert not _pred_matches_gt(_pred(vtype="helmet"), _gt(vtype="red_light"))

    def test_bbox_high_iou_matches(self):
        b = _bbox()
        assert _pred_matches_gt(_pred(bbox=b), _gt(bbox=b))

    def test_bbox_low_iou_no_match(self):
        a = _bbox(0, 0, 100, 100)
        b = _bbox(500, 500, 600, 600)
        assert not _pred_matches_gt(_pred(bbox=a), _gt(bbox=b))

    def test_pred_has_bbox_gt_does_not_matches(self):
        # GT has no bbox — spatial check skipped, type+image match is enough
        assert _pred_matches_gt(_pred(bbox=_bbox()), _gt(bbox=None))


# ---------------------------------------------------------------------------
# evaluate() — core metric computation
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_perfect_prediction(self):
        gt   = [_gt("helmet")]
        pred = [_pred("helmet", conf=0.95)]
        metrics = evaluate(pred, gt)
        m = next(x for x in metrics if x.violation_type == "helmet")
        assert m.tp == 1
        assert m.fp == 0
        assert m.fn == 0
        assert m.precision == pytest.approx(1.0)
        assert m.recall    == pytest.approx(1.0)
        assert m.f1        == pytest.approx(1.0)

    def test_false_positive(self):
        gt   = []
        pred = [_pred("helmet", conf=0.90)]
        metrics = evaluate(pred, gt)
        m = next(x for x in metrics if x.violation_type == "helmet")
        assert m.fp == 1
        assert m.tp == 0
        assert m.precision == pytest.approx(0.0)
        assert m.recall    == pytest.approx(0.0)

    def test_false_negative(self):
        gt   = [_gt("helmet")]
        pred = []
        metrics = evaluate(pred, gt)
        m = next(x for x in metrics if x.violation_type == "helmet")
        assert m.fn == 1
        assert m.tp == 0
        assert m.recall    == pytest.approx(0.0)

    def test_per_type_isolation(self):
        gt   = [_gt("helmet"), _gt("red_light")]
        pred = [_pred("helmet", conf=0.90)]   # misses red_light
        metrics = evaluate(pred, gt)
        helmet_m    = next(x for x in metrics if x.violation_type == "helmet")
        redlight_m  = next(x for x in metrics if x.violation_type == "red_light")
        assert helmet_m.tp   == 1
        assert redlight_m.fn == 1
        assert redlight_m.tp == 0

    def test_duplicate_gt_one_match(self):
        # Two GT records, one prediction — only one TP (greedy one-to-one)
        gt   = [_gt("helmet"), _gt("helmet")]
        pred = [_pred("helmet", conf=0.90)]
        metrics = evaluate(pred, gt)
        m = next(x for x in metrics if x.violation_type == "helmet")
        assert m.tp == 1
        assert m.fn == 1

    def test_accuracy_formula(self):
        # Accuracy = TP / (TP + FP + FN) = Jaccard
        gt   = [_gt("helmet"), _gt("helmet")]
        pred = [_pred("helmet", conf=0.9), _pred("helmet", conf=0.8),
                _pred("helmet", conf=0.7)]   # 2 TP, 1 FP, 0 FN
        metrics = evaluate(pred, gt)
        m = next(x for x in metrics if x.violation_type == "helmet")
        # tp=2, fp=1, fn=0 → accuracy = 2/3
        assert m.accuracy == pytest.approx(2 / 3, rel=1e-4)

    def test_scene_context_flag_set(self):
        gt   = [_gt("wrong_side")]
        pred = [_pred("wrong_side", conf=0.80)]
        metrics = evaluate(pred, gt)
        m = next(x for x in metrics if x.violation_type == "wrong_side")
        assert m.scene_context_dependent is True

    def test_helmet_not_scene_context(self):
        gt   = [_gt("helmet")]
        pred = [_pred("helmet", conf=0.90)]
        metrics = evaluate(pred, gt)
        m = next(x for x in metrics if x.violation_type == "helmet")
        assert m.scene_context_dependent is False

    def test_empty_gt_and_pred(self):
        assert evaluate([], []) == []

    def test_f1_harmonic_mean(self):
        # 3 GT, 2 correct predictions → p=1.0, r=2/3, f1=0.8
        gt   = [_gt("helmet"), _gt("helmet"), _gt("helmet")]
        pred = [_pred("helmet", conf=0.9), _pred("helmet", conf=0.8)]
        metrics = evaluate(pred, gt)
        m = next(x for x in metrics if x.violation_type == "helmet")
        assert m.precision == pytest.approx(1.0)
        assert m.recall    == pytest.approx(2 / 3, rel=1e-4)
        expected_f1 = 2 * 1.0 * (2/3) / (1.0 + 2/3)
        assert m.f1        == pytest.approx(expected_f1, rel=1e-4)


# ---------------------------------------------------------------------------
# AP computation
# ---------------------------------------------------------------------------

class TestComputeAp:
    def test_perfect_ranking_ap_is_1(self):
        gt   = [_gt("helmet", image_id=f"img_{i}") for i in range(5)]
        pred = [_pred("helmet", conf=1.0 - i*0.1, image_id=f"img_{i}") for i in range(5)]
        ap = _compute_ap(pred, gt)
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_no_gt_ap_is_0(self):
        pred = [_pred("helmet", conf=0.9)]
        assert _compute_ap(pred, []) == pytest.approx(0.0)

    def test_no_pred_ap_is_0(self):
        gt = [_gt("helmet")]
        assert _compute_ap([], gt) == pytest.approx(0.0)

    def test_all_wrong_ap_near_0(self):
        gt   = [_gt("helmet", image_id="img_A")]
        pred = [_pred("helmet", conf=0.9, image_id="img_B")]   # wrong image
        ap = _compute_ap(pred, gt)
        assert ap == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Latency stats
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_none_on_empty(self):
        assert compute_latency_stats([]) is None

    def test_single_value(self):
        stats = compute_latency_stats([50.0])
        assert stats.mean_ms        == pytest.approx(50.0)
        assert stats.throughput_fps == pytest.approx(20.0)

    def test_throughput_formula(self):
        stats = compute_latency_stats([100.0, 200.0])   # mean=150
        assert stats.throughput_fps == pytest.approx(1000.0 / 150.0, rel=1e-4)

    def test_p95_index(self):
        latencies = list(range(1, 101))   # 1..100 ms
        stats = compute_latency_stats(latencies)
        assert stats.p95_ms == pytest.approx(95.0)

    def test_median_even_count(self):
        stats = compute_latency_stats([10.0, 20.0, 30.0, 40.0])
        # sorted: [10,20,30,40] → median = (20+30)/2 = 25
        assert stats.median_ms == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# File I/O — load_ground_truth, load_predictions, load_latency
# ---------------------------------------------------------------------------

class TestFileIO:

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_load_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_jsonl(
                Path(tmpdir) / "val.jsonl",
                [{"image_id": "img_001", "violation_type": "helmet"}],
            )
            gt = load_ground_truth(Path(tmpdir))
        assert len(gt) == 1
        assert gt[0].image_id == "img_001"
        assert gt[0].violation_type == "helmet"

    def test_load_ground_truth_missing_dir(self):
        gt = load_ground_truth(Path("/nonexistent/dir"))
        assert gt == []

    def test_load_latency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "latency.jsonl"
            self._write_jsonl(log, [
                {"image_id": "img_001", "total_ms": 42.0},
                {"image_id": "img_002", "total_ms": 55.5},
            ])
            latencies = load_latency(log)
        assert latencies == pytest.approx([42.0, 55.5])

    def test_load_latency_missing_file(self):
        assert load_latency(Path("/nonexistent.jsonl")) == []

    def test_load_predictions_from_confirmed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            confirmed = Path(tmpdir) / "confirmed.jsonl"
            self._write_jsonl(confirmed, [{
                "violation_record": {
                    "image_id": "img_001",
                    "violation_type": "helmet",
                    "confidence": 0.88,
                    "rule_trace": "...",
                },
                "plate_text": "MH12",
                "timestamp": "2025-01-01T00:00:00+00:00",
            }])
            preds = load_predictions(
                confirmed_jsonl=confirmed,
                review_jsonl=Path("/nonexistent.jsonl"),
            )
        assert len(preds) == 1
        assert preds[0].confidence == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Report formatting — smoke tests (no crash, key strings present)
# ---------------------------------------------------------------------------

class TestReportFormatting:

    def _sample_metrics(self) -> list[PerTypeMetrics]:
        return [
            PerTypeMetrics(
                violation_type="helmet", n_gt=10, n_pred=9,
                tp=8, fp=1, fn=2,
                precision=0.888, recall=0.800, f1=0.842,
                accuracy=0.727, average_precision=0.810,
                scene_context_dependent=False,
            ),
            PerTypeMetrics(
                violation_type="wrong_side", n_gt=5, n_pred=3,
                tp=2, fp=1, fn=3,
                precision=0.667, recall=0.400, f1=0.500,
                accuracy=0.333, average_precision=0.420,
                scene_context_dependent=True,
            ),
        ]

    def test_console_report_no_crash(self):
        report = build_console_report(
            self._sample_metrics(), latency=None,
            n_pred_total=12, n_gt_total=15,
        )
        assert "helmet" in report
        assert "wrong_side" in report
        assert "mAP" in report
        assert "INFERENCE LATENCY" in report

    def test_console_report_contains_context_note(self):
        report = build_console_report(
            self._sample_metrics(), latency=None,
            n_pred_total=12, n_gt_total=15,
        )
        assert "Scene-context" in report

    def test_markdown_report_has_table(self):
        md = build_markdown_report(
            self._sample_metrics(), latency=None,
            n_pred_total=12, n_gt_total=15,
        )
        assert "| helmet" in md
        assert "| wrong_side" in md
        assert "mAP" in md
        assert "†" in md   # scene-context footnote marker

    def test_export_csv_creates_files(self):
        from evaluation.evaluate import LatencyStats
        latency = LatencyStats(
            n_images=50, mean_ms=80.0, median_ms=75.0, p95_ms=120.0,
            min_ms=40.0, max_ms=200.0, throughput_fps=12.5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            written = export_csv(
                self._sample_metrics(), latency, reports_dir=Path(tmpdir)
            )
            assert "eval_metrics"  in written
            assert "eval_latency"  in written
            for path in written.values():
                assert path.exists()
                assert path.stat().st_size > 0

    def test_markdown_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "eval_results.md"
            md = build_markdown_report(
                self._sample_metrics(), latency=None,
                n_pred_total=12, n_gt_total=15,
            )
            md_path.write_text(md, encoding="utf-8")
            content = md_path.read_text()
        assert "# Traffic Violation Detection" in content
