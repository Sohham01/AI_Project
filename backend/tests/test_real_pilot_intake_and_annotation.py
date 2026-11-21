"""Comprehensive validation tests for the RCI First Real-Data Pilot workflow:
  - Authorized static export intake (all 4 formats)
  - Privacy & PII minimization (author stripping, email/phone/user-mention redaction)
  - Zero silent fabrication of missing fields & explicit rejection diagnostics
  - Approval gate enforcement (load_static_export -> prepare_authorized_static_pilot)
  - 5-task human annotation validation, explicit uncertain/abstain routing,
    forbidden-placeholder rejection, and training-readiness gating
  - Strict /api/demo boundary isolation
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
for p in (str(ROOT), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.api.demo import ARTIFACT_DIR, DATA_DIR, summary as demo_summary
from app.services.ingestion.static_export import load_static_export
from app.services.pilot import prepare_authorized_static_pilot
from backend.import_reddit_export import main as import_main
from backend.validate_pilot_annotations import (
    CANONICAL_RESPONSES,
    validate_and_split_pilot_csv,
)


class TestStaticExportPrivacyAndValidation(unittest.TestCase):
    def test_pii_redaction_and_academic_number_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            (raw / "format_v1.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "metadata": {
                            "dataset_id": "v1-input",
                            "source": {
                                "community": "IIMSambalpur",
                                "source_reference": "ref-1",
                                "source_provider": "supplier",
                                "source_mode": "direct",
                            },
                            "authorization": {
                                "authorization_status": "approved",
                                "authorization_reference": "auth-1",
                            },
                        },
                        "posts": [
                            {
                                "post_id": "p1",
                                "subreddit": "IIMSambalpur",
                                "author": "secret_user_1",
                                "title": "CAT 2026 fee 2100000 query from u/aspirant_99",
                                "body": "Reach me at student.name@iimsambalpur.ac.in or +91 9876543210.",
                                "created_at": "2026-10-01T10:00:00Z",
                                "score": 5,
                                "num_comments": 1,
                                "comments": [
                                    {
                                        "comment_id": "c1",
                                        "author": "secret_user_2",
                                        "body": "Ask /u/placement_lead or mail office@example.org for 2026 schedule.",
                                        "created_at": "2026-10-01T10:15:00Z",
                                        "score": 2,
                                        "parent_comment_id": None,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out_json = root / "sanitized.json"
            cand_dir = root / "candidates"
            rc = import_main(
                [
                    "--input", str(raw),
                    "--output", str(out_json),
                    "--dataset-id", "pilot-pii-test",
                    "--community", "IIMSambalpur",
                    "--authorization-status", "approved",
                    "--authorization-reference", "irb-ref-001",
                    "--export-candidates", str(cand_dir),
                ]
            )
            self.assertEqual(rc, 0)
            sanitized_text = out_json.read_text(encoding="utf-8")
            # Zero usernames, emails, or phones leak into sanitized export
            for secret in (
                "secret_user_1",
                "secret_user_2",
                "u/aspirant_99",
                "/u/placement_lead",
                "student.name@iimsambalpur.ac.in",
                "office@example.org",
                "9876543210",
            ):
                self.assertNotIn(secret, sanitized_text)
            # Academic/domain numbers stay intact
            self.assertIn("CAT 2026", sanitized_text)
            self.assertIn("<EMAIL>", sanitized_text)
            self.assertIn("<PHONE>", sanitized_text)
            self.assertIn("<USER_MENTION>", sanitized_text)

            # Verify approval gate works with static_export + pilot.py
            loaded = load_static_export(out_json)
            self.assertTrue(loaded.valid)
            self.assertTrue(loaded.usable_for_pilot)
            self.assertTrue(loaded.authorization_approved)

            with self.assertRaises(PermissionError):
                prepare_authorized_static_pilot(loaded, approval_confirmed=False)

            pilot_res = prepare_authorized_static_pilot(loaded, approval_confirmed=True)
            self.assertEqual(pilot_res.summary["status"], "ready_for_review")
            self.assertTrue(pilot_res.join_report.valid)

    def test_no_silent_fabrication_and_explicit_rejection_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jsonl"
            lines = [
                # Valid post missing score & num_comments (must NOT fabricate 0)
                json.dumps({
                    "id": "p10",
                    "subreddit": "MBAIndia",
                    "title": "IIM Sambalpur hostel query",
                    "created_utc": 1763500000,
                }),
                # Invalid post with naive (non-zoned) timestamp -> rejected
                json.dumps({
                    "id": "p11",
                    "subreddit": "MBAIndia",
                    "title": "Bad timestamp post",
                    "created_at": "2026-10-01T10:00:00",
                }),
                # Valid comment missing score
                json.dumps({
                    "id": "c10",
                    "link_id": "t3_p10",
                    "parent_id": "t3_p10",
                    "body": "Hostel rooms are single occupancy.",
                    "created_utc": 1763500300,
                }),
                # Deleted comment -> rejected
                json.dumps({
                    "id": "c11",
                    "link_id": "t3_p10",
                    "parent_id": "t3_p10",
                    "body": "[removed]",
                    "created_utc": 1763500400,
                }),
                # Orphaned comment whose post doesn't exist -> skipped & reported
                json.dumps({
                    "id": "c12",
                    "link_id": "t3_nonexistent",
                    "parent_id": "t3_nonexistent",
                    "body": "Orphan comment.",
                    "created_utc": 1763500500,
                }),
            ]
            raw.write_text("\n".join(lines) + "\n", encoding="utf-8")

            out_json = root / "sanitized.json"
            report_json = root / "quality.json"
            rc = import_main(
                [
                    "--input", str(raw),
                    "--output", str(out_json),
                    "--dataset-id", "pilot-missing-fields",
                    "--community", "MBAIndia",
                    "--quality-report", str(report_json),
                ]
            )
            self.assertEqual(rc, 0)
            rep = json.loads(report_json.read_text(encoding="utf-8"))
            diag = rep["import_stats"]["diagnostics_and_rejections"]
            self.assertEqual(diag["post_invalid_or_missing_timestamp"], 1)
            self.assertEqual(diag["comment_deleted_or_removed"], 1)
            self.assertEqual(diag["orphaned_comments_post_missing"], 1)
            self.assertEqual(diag["post_missing_score_unfabricated"], 1)
            self.assertEqual(diag["comment_missing_score_unfabricated"], 2)
            # static_export flags lossy_default_fields explicitly
            self.assertIn("post.score", rep["validation_summary"]["lossy_default_fields"])
            self.assertIn("comment.score", rep["validation_summary"]["lossy_default_fields"])
            self.assertFalse(rep["validation_summary"]["usable_for_pilot"])


class TestPilotAnnotationValidationAndAbstainWorkflow(unittest.TestCase):
    def test_uncertain_and_abstain_never_silently_converted_and_readiness_gated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_csv = root / "review.csv"
            out_dir = root / "converted"
            cols = [
                "document_id", "post_id", "comment_id", "subreddit", "post_title",
                "parent_comment_id", "parent_text", "parent_context_status",
                "created_at", "score", "original_text", "normalized_text",
                "language", "script", "is_code_mixed", "detected_languages",
                "sentiment", "emotions", "issues", "response_type", "is_solution",
                "review_status", "abstain_reason", "annotator_id", "annotator_notes",
            ]
            rows = [
                {
                    "document_id": "comment-1", "post_id": "p1", "comment_id": "1",
                    "subreddit": "IIMSambalpur", "post_title": "Hostel Wi-Fi",
                    "parent_comment_id": "", "parent_text": "", "parent_context_status": "top_level",
                    "created_at": "2026-10-01T10:00:00Z", "score": "4",
                    "original_text": "Raise an IT ticket.", "normalized_text": "Raise an IT ticket.",
                    "language": "en", "script": "latin", "is_code_mixed": "false", "detected_languages": "en",
                    "sentiment": "neutral", "emotions": "", "issues": "campus_infrastructure",
                    "response_type": "solution", "is_solution": "solution",
                    "review_status": "labeled", "abstain_reason": "", "annotator_id": "ann1", "annotator_notes": "Clear fix",
                },
                {
                    "document_id": "comment-2", "post_id": "p1", "comment_id": "2",
                    "subreddit": "IIMSambalpur", "post_title": "Hostel Wi-Fi",
                    "parent_comment_id": "missing99", "parent_text": "", "parent_context_status": "unavailable",
                    "created_at": "2026-10-01T10:05:00Z", "score": "1",
                    "original_text": "Lol sure.", "normalized_text": "Lol sure.",
                    "language": "en", "script": "latin", "is_code_mixed": "false", "detected_languages": "en",
                    "sentiment": "", "emotions": "", "issues": "",
                    "response_type": "", "is_solution": "",
                    "review_status": "uncertain", "abstain_reason": "missing_parent",
                    "annotator_id": "ann1", "annotator_notes": "Sarcastic? Needs parent",
                },
                {
                    "document_id": "comment-3", "post_id": "p2", "comment_id": "3",
                    "subreddit": "MBAIndia", "post_title": "Random post",
                    "parent_comment_id": "", "parent_text": "", "parent_context_status": "top_level",
                    "created_at": "2026-10-01T11:00:00Z", "score": "0",
                    "original_text": "RemindMe! 2 days", "normalized_text": "RemindMe! 2 days",
                    "language": "en", "script": "latin", "is_code_mixed": "false", "detected_languages": "en",
                    "sentiment": "", "emotions": "", "issues": "",
                    "response_type": "", "is_solution": "",
                    "review_status": "abstain", "abstain_reason": "off_topic_noise",
                    "annotator_id": "ann1", "annotator_notes": "Bot command",
                },
            ]
            with review_csv.open("w", encoding="utf-8-sig", newline="") as h:
                w = csv.DictWriter(h, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)

            summary = validate_and_split_pilot_csv(review_csv, out_dir=out_dir)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["labeled_comments"], 1)
            self.assertEqual(summary["adjudication_queue_count"], 2)
            # Insufficient data -> training readiness gate MUST block retraining
            self.assertFalse(summary["ready_for_training"])
            self.assertTrue(any("insufficient_labeled_comments" in b for b in summary["readiness_blockers"]))

            # Uncertain/abstain rows must ONLY appear in adjudication_queue.csv, never in training CSVs
            adj_rows = list(csv.DictReader((out_dir / "adjudication_queue.csv").open("r", encoding="utf-8-sig")))
            self.assertEqual([r["document_id"] for r in adj_rows], ["comment-2", "comment-3"])

            sol_rows = list(csv.DictReader((out_dir / "solution.csv").open("r", encoding="utf-8-sig")))
            self.assertEqual(len(sol_rows), 1)
            self.assertEqual(sol_rows[0]["document_id"], "comment-1")
            self.assertEqual(sol_rows[0]["is_solution"], "true")

            # Verify canonical response ordering is preserved verbatim
            self.assertEqual(
                CANONICAL_RESPONSES,
                ("agreement", "personal_experience", "opinion", "advice", "solution", "counterargument"),
            )

    def test_placeholder_in_label_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            review_csv = Path(tmp) / "bad.csv"
            cols = [
                "document_id", "post_id", "comment_id", "normalized_text",
                "sentiment", "emotions", "issues", "response_type", "is_solution",
                "review_status", "abstain_reason",
            ]
            with review_csv.open("w", encoding="utf-8-sig", newline="") as h:
                w = csv.DictWriter(h, fieldnames=cols)
                w.writeheader()
                w.writerow({
                    "document_id": "comment-bad", "post_id": "p1", "comment_id": "bad",
                    "normalized_text": "Some text",
                    "sentiment": "unclear", "emotions": "none", "issues": "",
                    "response_type": "opinion", "is_solution": "not_solution",
                    "review_status": "labeled", "abstain_reason": "",
                })
            summary = validate_and_split_pilot_csv(review_csv)
            self.assertFalse(summary["valid"])
            codes = {e["code"] for e in summary["errors"]}
            self.assertIn("forbidden_placeholder_in_label_cell", codes)


class TestDemoBoundaryIsolation(unittest.TestCase):
    def test_live_api_uses_sanitized_live_corpus_and_isolates_raw_pilot_files(self):
        import os
        from app.api import demo as demo_mod

        demo_mod._data.cache_clear()
        data_dir, art_dir = demo_mod._resolve_dirs()
        self.assertEqual(data_dir.name, "live_corpus")
        self.assertEqual(art_dir.name, "live_corpus")
        self.assertNotIn("processed", data_dir.parts)
        self.assertNotIn("annotations", data_dir.parts)

        payload = demo_summary()
        self.assertFalse(payload["synthetic_only"])
        self.assertEqual(payload["provenance"]["authorization_status"], "public_anonymized_reddit")
        self.assertGreaterEqual(payload["posts"], 70)
        self.assertGreaterEqual(payload["comments"], 700)

        # Synthetic demo bundles are retired; API remains strictly locked to real live_corpus
        prev = os.environ.get("RCI_DATA_MODE")
        os.environ["RCI_DATA_MODE"] = "synthetic"
        demo_mod._data.cache_clear()
        try:
            live_payload = demo_summary()
            self.assertFalse(live_payload["synthetic_only"])
            self.assertGreaterEqual(live_payload["posts"], 300)
            self.assertGreaterEqual(live_payload["comments"], 3000)
        finally:
            if prev is None:
                os.environ.pop("RCI_DATA_MODE", None)
            else:
                os.environ["RCI_DATA_MODE"] = prev
            demo_mod._data.cache_clear()


if __name__ == "__main__":
    unittest.main()
