"""
Tests for the response annotation-candidate export.

The point of these tests is the contract with the human annotator: original
text preserved, ``response_type`` blank, predictions never copied into ground
truth, and a validated CSV round-trip.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from app.services.annotation.candidates import (
    CANDIDATE_COLUMNS,
    CandidateValidationError,
    build_candidate_rows,
    audit_training_candidate_rows,
    candidate_context_warnings,
    read_candidates_csv,
    to_training_rows,
    validate_candidate_rows,
    write_candidates_csv,
)


class FakePreprocessor:
    def preprocess(self, text, language=None, script=None, is_code_mixed=False,
                   detected_languages=None):
        return SimpleNamespace(normalized_text=f"norm:{text}")


class FakeLanguageService:
    def analyze(self, text):
        return SimpleNamespace(
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
        )


class TestBuildCandidates(unittest.TestCase):
    def test_original_text_preserved_and_label_blank(self):
        rows = build_candidate_rows(
            [
                {
                    "document_id": "doc-c1",
                    "comment_id": "c1",
                    "text": "same bro!!",
                    "post_id": "p1",
                    "parent_text": "I got in",
                    "post_title": "placement",
                    "subreddit": "IIMSambalpur",
                    "score": 4,
                    "created_at": "2026-01-02T03:04:05+00:00",
                    "source_url": "https://www.reddit.com/r/IIMSambalpur/comments/p1/title/",
                    "source_provider": "approved-static-export",
                    "source_mode": "direct",
                    "parent_context_status": "available",
                    "parent_comment_id": "parent-1",
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["text"], "same bro!!")
        self.assertEqual(row["response_type"], "")
        self.assertEqual(row["document_id"], "doc-c1")
        self.assertEqual(row["parent_text"], "I got in")
        self.assertEqual(row["source_url"], "https://www.reddit.com/r/IIMSambalpur/comments/p1/title/")
        self.assertEqual(row["source_provider"], "approved-static-export")
        self.assertEqual(row["source_mode"], "direct")
        self.assertEqual(row["parent_context_status"], "available")

    def test_preprocessor_fills_normalized_without_touching_original(self):
        rows = build_candidate_rows(
            [{"document_id": "doc-c1", "comment_id": "c1", "text": "hello"}],
            preprocessor=FakePreprocessor(),
        )
        self.assertEqual(rows[0]["text"], "hello")
        self.assertEqual(rows[0]["normalized_text"], "norm:hello")

    def test_language_service_fills_metadata(self):
        rows = build_candidate_rows(
            [{"document_id": "doc-c1", "comment_id": "c1", "text": "hai"}],
            language_service=FakeLanguageService(),
        )
        self.assertEqual(rows[0]["language"], "hi")
        self.assertEqual(rows[0]["script"], "latin")
        self.assertEqual(rows[0]["is_code_mixed"], "true")
        self.assertEqual(rows[0]["detected_languages"], "hi|en")

    def test_missing_required_field_raises(self):
        with self.assertRaises(CandidateValidationError):
            build_candidate_rows([{"document_id": "doc-c1", "comment_id": "c1"}])


class TestValidation(unittest.TestCase):
    def _row(self, **kwargs):
        base = {
            "document_id": "cand-c1",
            "comment_id": "c1",
            "text": "hi",
            "response_type": "",
        }
        base.update(kwargs)
        return base

    def test_valid_row_no_problems(self):
        self.assertEqual(validate_candidate_rows([self._row()]), [])

    def test_duplicate_document_id_flagged(self):
        problems = validate_candidate_rows([self._row(), self._row()])
        self.assertTrue(any("duplicate document_id" in p for p in problems))

    def test_duplicate_comment_id_flagged(self):
        problems = validate_candidate_rows(
            [self._row(document_id="d1"), self._row(document_id="d2")]
        )
        self.assertTrue(any("duplicate comment_id" in p for p in problems))

    def test_shared_parent_id_is_allowed(self):
        rows = [
            self._row(document_id="d1", comment_id="c1", parent_comment_id="parent"),
            self._row(document_id="d2", comment_id="c2", parent_comment_id="parent"),
        ]
        self.assertEqual(validate_candidate_rows(rows), [])

    def test_empty_text_flagged(self):
        problems = validate_candidate_rows([self._row(text="")])
        self.assertTrue(any("empty original text" in p for p in problems))

    def test_prefilled_response_type_flagged(self):
        problems = validate_candidate_rows([self._row(response_type="opinion")])
        self.assertTrue(any("pre-filled" in p for p in problems))


class TestCsvRoundTrip(unittest.TestCase):
    def test_write_then_read(self):
        rows = build_candidate_rows(
            [{"document_id": "doc-c1", "comment_id": "c1", "text": "hello"}]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "candidates.csv")
            write_candidates_csv(path, rows)
            read_back = read_candidates_csv(path)
        self.assertEqual(len(read_back), 1)
        self.assertEqual(list(read_back[0].keys()), CANDIDATE_COLUMNS)
        self.assertEqual(read_back[0]["response_type"], "")

    def test_invalid_rows_not_written(self):
        bad = [{"document_id": "", "comment_id": "", "text": ""}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "candidates.csv")
            with self.assertRaises(CandidateValidationError):
                write_candidates_csv(path, bad)
            self.assertFalse(os.path.exists(path))

    def test_existing_destination_is_preserved(self):
        rows = build_candidate_rows(
            [{"document_id": "doc-c1", "comment_id": "c1", "text": "hello"}]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "candidates.csv")
            original = b"already approved content\n"
            with open(path, "wb") as handle:
                handle.write(original)
            with self.assertRaises(FileExistsError):
                write_candidates_csv(path, rows)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), original)

    def test_malformed_score_and_boolean_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            score_path = os.path.join(tmp, "score.csv")
            with self.assertRaises(CandidateValidationError):
                write_candidates_csv(
                    score_path,
                    [
                        {
                            "document_id": "d1",
                            "comment_id": "c1",
                            "text": "hello",
                            "score": "not-a-number",
                            "response_type": "",
                        }
                    ],
                )
            self.assertFalse(os.path.exists(score_path))

            bool_path = os.path.join(tmp, "bool.csv")
            with self.assertRaises(CandidateValidationError):
                write_candidates_csv(
                    bool_path,
                    [
                        {
                            "document_id": "d1",
                            "comment_id": "c1",
                            "text": "hello",
                            "is_code_mixed": "sometimes",
                            "response_type": "",
                        }
                    ],
                )
            self.assertFalse(os.path.exists(bool_path))

    def test_zero_score_and_missing_score_are_distinct(self):
        rows = build_candidate_rows(
            [
                {"document_id": "d1", "comment_id": "c1", "text": "zero", "score": 0},
                {"document_id": "d2", "comment_id": "c2", "text": "missing"},
            ]
        )
        self.assertEqual(rows[0]["score"], "0")
        self.assertEqual(rows[1]["score"], "")

    def test_missing_context_is_reported_without_fabrication(self):
        rows = build_candidate_rows(
            [{"document_id": "d1", "comment_id": "c1", "text": "hello"}]
        )
        warnings = candidate_context_warnings(rows)
        self.assertTrue(any("missing source_url" in warning for warning in warnings))
        self.assertTrue(any("missing parent_context_status" in warning for warning in warnings))
        self.assertEqual(rows[0]["source_url"], "")
        self.assertEqual(rows[0]["parent_context_status"], "")


class TestToTrainingRows(unittest.TestCase):
    def test_uses_normalized_text_and_skips_unlabelled(self):
        rows = [
            {
                "document_id": "c1",
                "text": "original",
                "normalized_text": "normalized",
                "response_type": "opinion",
                "post_id": "p1",
                "comment_id": "c1",
                "language": "en",
                "script": "latin",
                "is_code_mixed": "false",
                "detected_languages": "en",
            },
            {
                "document_id": "c2",
                "text": "original2",
                "normalized_text": "normalized2",
                "response_type": "",  # not annotated yet
            },
            {
                "document_id": "c3",
                "text": "original3",
                "normalized_text": "",  # nothing to train on
                "response_type": "opinion",
            },
            {
                "document_id": "c4",
                "text": "original4",
                "normalized_text": "normalized4",
                "response_type": "not_a_label",
            },
        ]
        training = to_training_rows(rows)
        self.assertEqual(len(training), 1)
        self.assertEqual(training[0]["text"], "normalized")
        self.assertEqual(training[0]["response_type"], "opinion")

    def test_training_audit_reports_exclusions_without_fallback(self):
        report = audit_training_candidate_rows(
            [
                {
                    "document_id": "c1",
                    "comment_id": "comment-1",
                    "text": "original",
                    "normalized_text": "normalized",
                    "response_type": "opinion",
                },
                {
                    "document_id": "c2",
                    "comment_id": "comment-2",
                    "text": "original2",
                    "normalized_text": "",
                    "response_type": "",
                },
            ]
        )
        self.assertEqual(report["accepted"], [0])
        self.assertEqual(report["blank_response_labels"], [1])
        self.assertEqual(report["missing_normalized_text"], [1])
        self.assertEqual(report["excluded"], 1)
        self.assertFalse(report["fatal"])


if __name__ == "__main__":
    unittest.main()
