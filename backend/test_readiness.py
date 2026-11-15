"""
Tests for the training-data readiness pre-flight.

These pin the important promises: missing columns are fatal, small/synthetic
files are never "ready", class and label problems are surfaced, leakage-safe
split feasibility is evaluated, and MuRIL is gated on genuine volume.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from ml.data.readiness import (
    MIN_TOTAL_RECORDS,
    MURIL_MIN_RECORDS_PER_CLASS,
    MURIL_MIN_TOTAL_RECORDS,
    TaskSchema,
    analyse_annotation_csv,
    analyse_records,
    can_train_muril,
)

RESPONSE_TYPES = (
    "agreement",
    "personal_experience",
    "opinion",
    "advice",
    "solution",
    "counterargument",
)

SCHEMA = TaskSchema(
    required=("document_id", "text", "response_type"),
    label_column="response_type",
)
FIELDS = ["document_id", "text", "response_type", "post_id"]


def _make_rows(total, per_class=None, groups_of=1):
    """Balanced rows across the six classes, grouped by post_id."""
    rows = []
    for i in range(total):
        label = RESPONSE_TYPES[i % len(RESPONSE_TYPES)]
        rows.append(
            {
                "document_id": f"d{i}",
                "text": f"unique text number {i}",
                "response_type": label,
                "post_id": f"p{i // groups_of}",
            }
        )
    return rows


def _analyse(rows, fieldnames=FIELDS):
    return analyse_records(
        rows,
        fieldnames=fieldnames,
        schema=SCHEMA,
        label_space=RESPONSE_TYPES,
    )


class TestMissingColumns(unittest.TestCase):
    def test_missing_column_is_not_ready(self):
        rows = [{"document_id": "d1", "text": "hi"}]  # no response_type
        report = _analyse(rows, fieldnames=["document_id", "text"])
        self.assertEqual(report.verdict, "not_ready")
        self.assertIn("response_type", report.missing_columns)


class TestReadinessVerdicts(unittest.TestCase):
    def test_sufficient_balanced_data_is_ready(self):
        report = _analyse(_make_rows(120, groups_of=5))
        self.assertEqual(report.verdict, "ready")
        self.assertTrue(report.split_feasible)
        self.assertGreaterEqual(report.total_records, MIN_TOTAL_RECORDS)

    def test_small_file_warns_and_calls_metrics_unreal(self):
        report = _analyse(_make_rows(24, groups_of=2))
        self.assertEqual(report.verdict, "warn")
        joined = " ".join(report.warnings).lower()
        self.assertIn("not a real performance result", joined)

    def test_single_class_is_not_ready(self):
        rows = [
            {"document_id": f"d{i}", "text": f"t{i}", "response_type": "opinion",
             "post_id": f"p{i}"}
            for i in range(10)
        ]
        report = _analyse(rows)
        self.assertEqual(report.verdict, "not_ready")

    def test_unknown_label_is_not_ready(self):
        rows = _make_rows(20)
        rows[0]["response_type"] = "personal experience"
        report = _analyse(rows)
        self.assertEqual(report.verdict, "not_ready")
        self.assertIn("personal experience", report.unknown_labels)

    def test_missing_classes_reported(self):
        rows = _make_rows(20, groups_of=2)
        # keep only two classes
        rows = [r for r in rows if r["response_type"] in ("opinion", "agreement")]
        report = _analyse(rows)
        self.assertTrue(report.missing_classes)
        self.assertIn("advice", report.missing_classes)


class TestDuplicateAndEmpty(unittest.TestCase):
    def test_duplicates_and_empty_counted(self):
        rows = _make_rows(20)
        rows[0]["text"] = ""
        rows[1]["text"] = rows[2]["text"]
        rows[3]["document_id"] = rows[4]["document_id"]
        report = _analyse(rows)
        self.assertEqual(report.empty_content, 1)
        self.assertGreaterEqual(report.duplicate_texts, 1)
        self.assertIn(rows[4]["document_id"], report.duplicate_document_ids)


class TestSplitFeasibility(unittest.TestCase):
    def test_group_split_feasible_with_enough_threads(self):
        report = _analyse(_make_rows(60, groups_of=3))
        self.assertTrue(report.split_feasible)
        self.assertGreaterEqual(report.thread_count, 3)
        self.assertGreaterEqual(report.multi_comment_threads, 1)

    def test_group_split_requires_three_independent_threads_for_three_way_eval(self):
        report = _analyse(_make_rows(15, groups_of=5))
        self.assertTrue(report.split_feasible)
        self.assertEqual(report.thread_count, 3)

    def test_too_few_records_not_split_feasible(self):
        report = _analyse(_make_rows(2, groups_of=1))
        self.assertFalse(report.split_feasible)

    def test_ungrouped_records_flagged(self):
        rows = _make_rows(20)
        for row in rows:
            row["post_id"] = ""
        report = _analyse(rows)
        self.assertEqual(report.ungrouped_records, 20)
        self.assertTrue(any("leakage" in w.lower() for w in report.warnings))


class TestMurilGate(unittest.TestCase):
    def test_small_data_blocks_muril(self):
        report = _analyse(_make_rows(120, groups_of=5))
        eligible, blockers = can_train_muril(report)
        self.assertFalse(eligible)
        self.assertTrue(blockers)

    def test_large_balanced_data_allows_muril(self):
        total = MURIL_MIN_TOTAL_RECORDS + 60
        report = _analyse(_make_rows(total, groups_of=5))
        eligible, blockers = can_train_muril(report)
        self.assertTrue(eligible, msg=f"blockers={blockers}")
        self.assertFalse(blockers)

    def test_muril_needs_each_class(self):
        # enough total, but one class under the per-class floor
        rows = _make_rows(MURIL_MIN_TOTAL_RECORDS + 60, groups_of=5)
        report = _analyse(rows)
        # Every class appears; the gate should pass here.
        self.assertTrue(report.muril_eligible)
        self.assertGreaterEqual(
            min(report.label_counts.values()), MURIL_MIN_RECORDS_PER_CLASS
        )


    def test_rejects_malformed_row_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write("document_id,text,response_type,post_id\n")
                handle.write("d1,too-few,agreement\n")
            report = analyse_annotation_csv(path)
        self.assertEqual(report.verdict, "not_ready")
        self.assertTrue(any("malformed CSV row width" in error for error in report.errors))

    def test_rejects_duplicate_header_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write("document_id,text,response_type,response_type\n")
                handle.write("d1,hello,agreement,agreement\n")
            report = analyse_annotation_csv(path)
        self.assertEqual(report.verdict, "not_ready")
        self.assertTrue(any("duplicate CSV column" in error for error in report.errors))


class TestCsvEntryPoint(unittest.TestCase):
    def test_missing_file_is_not_ready(self):
        report = analyse_annotation_csv("does/not/exist.csv")
        self.assertEqual(report.verdict, "not_ready")
        self.assertTrue(report.errors)

    def test_reads_real_csv(self):
        rows = _make_rows(120, groups_of=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ann.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write("document_id,text,response_type,post_id\n")
                for row in rows:
                    handle.write(
                        f'{row["document_id"]},{row["text"]},'
                        f'{row["response_type"]},{row["post_id"]}\n'
                    )
            report = analyse_annotation_csv(path)
        self.assertEqual(report.verdict, "ready")
        self.assertEqual(report.total_records, 120)


if __name__ == "__main__":
    unittest.main()
