from __future__ import annotations

import csv
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from convert_annotation_candidates import convert_file, main
from app.services.annotation.candidates import CANDIDATE_COLUMNS
from app.services.nlp.response_classification.models.annotation import CSV_COLUMNS


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = os.path.join(self.tmp.name, "candidates.csv")
        self.output = os.path.join(self.tmp.name, "training.csv")

    def write_rows(self, rows, columns=CANDIDATE_COLUMNS):
        with open(self.source, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def candidate(self, did="d1", label="agreement", normalized="norm", **kw):
        row = {c: "" for c in CANDIDATE_COLUMNS}
        comment_id = kw.pop("comment_id", f"comment-{did}")
        row.update(document_id=did, text="original", normalized_text=normalized,
                   response_type=label, post_id="p1", comment_id=comment_id,
                   language="en", script="latin", is_code_mixed="false",
                   detected_languages="en")
        row.update(kw)
        return row

    def test_converts_exact_training_schema_and_normalized_text(self):
        self.write_rows([self.candidate()])
        report = convert_file(self.source, self.output)
        with open(self.output, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, CSV_COLUMNS)
        self.assertEqual(rows[0]["text"], "norm")
        self.assertEqual(rows[0]["post_id"], "p1")
        self.assertEqual(report["label_counts"], {"agreement": 1})

    def test_skips_unreviewed_and_blank_normalized_with_counts(self):
        self.write_rows([self.candidate(label=""), self.candidate(did="d2", normalized="")])
        report = convert_file(self.source, self.output)
        self.assertEqual(report["skipped_unlabelled"], 1)
        self.assertEqual(report["skipped_blank_normalized_text"], 1)
        self.assertEqual(report["written"], 0)

    def test_rejects_invalid_label(self):
        self.write_rows([self.candidate(label="personal experience")])
        with self.assertRaisesRegex(ValueError, "invalid response_type"):
            convert_file(self.source, self.output)
        self.assertFalse(os.path.exists(self.output))

    def test_rejects_duplicate_and_conflicting_document_ids(self):
        for labels in [("agreement", "agreement"), ("agreement", "solution")]:
            with self.subTest(labels=labels):
                self.write_rows([self.candidate(label=labels[0]), self.candidate(label=labels[1])])
                with self.assertRaisesRegex(ValueError, "duplicate document_id"):
                    convert_file(self.source, self.output)

    def test_rejects_duplicate_comment_ids(self):
        self.write_rows(
            [
                self.candidate(did="d1", comment_id="same-comment"),
                self.candidate(did="d2", comment_id="same-comment"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate_comment_ids"):
            convert_file(self.source, self.output)

    def test_rejects_malformed_row_width(self):
        with open(self.source, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(CANDIDATE_COLUMNS) + "\n")
            f.write("d1,too,few\n")
        with self.assertRaisesRegex(ValueError, "row width"):
            convert_file(self.source, self.output)

    def test_no_overwrite_without_flag_and_source_unchanged(self):
        self.write_rows([self.candidate()])
        with open(self.source, "rb") as f:
            before = f.read()
        convert_file(self.source, self.output)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            convert_file(self.source, self.output)
        with open(self.source, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_failed_conversion_leaves_no_partial_output(self):
        self.write_rows([self.candidate(label="bad-label")])
        with self.assertRaisesRegex(ValueError, "invalid response_type"):
            convert_file(self.source, self.output)
        self.assertTrue(os.path.isfile(self.source))
        self.assertFalse(os.path.exists(self.output))
        self.assertEqual(len(os.listdir(self.tmp.name)), 1)

    def test_output_collision_requires_explicit_overwrite(self):
        self.write_rows([self.candidate()])
        with open(self.output, "w", encoding="utf-8") as f:
            f.write("preserve")
        with self.assertRaises(FileExistsError):
            convert_file(self.source, self.output)
        with open(self.output, encoding="utf-8") as f:
            self.assertEqual(f.read(), "preserve")

    def test_guide_covers_adjudication_boundaries(self):
        guide = os.path.join(os.path.dirname(__file__), "..", "docs", "annotation_guide_response.md")
        with open(guide, encoding="utf-8") as handle:
            text = handle.read()
        for rule in ("Operational decision procedure", "leave `response_type` blank", "Advice vs. solution", "Opinion vs. counterargument", "Agreement vs. personal_experience", "emoji-only", "truncated", "code-mixed"):
            with self.subTest(rule=rule):
                self.assertIn(rule, text)
        self.assertIn("agreement", text)
        self.assertIn("personal_experience", text)
        self.assertIn("counterargument", text)

    def test_cli_prints_counts_not_text_and_overwrite_flag(self):
        self.write_rows([self.candidate()])
        out = StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--input", self.source, "--output", self.output]), 0)
        self.assertIn("'agreement': 1", out.getvalue())
        self.assertNotIn("original", out.getvalue())
        self.assertNotIn("\n\nnorm\n", out.getvalue())

    def test_conversion_reports_excluded_rows(self):
        self.write_rows(
            [
                self.candidate(did="d1", label=""),
                self.candidate(did="d2", normalized=""),
            ]
        )
        report = convert_file(self.source, self.output)
        self.assertEqual(report["excluded"], 2)
        self.assertEqual(report["exclusion_report"]["blank_response_labels"], 1)
        self.assertEqual(report["exclusion_report"]["missing_normalized_text"], 1)


if __name__ == "__main__":
    unittest.main()
