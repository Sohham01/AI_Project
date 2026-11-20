"""Tests for the local-only browser annotation studio (/annotate)."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.run_annotation_studio import create_annotation_studio_app


def _write_temp_candidates_csv(path: Path) -> None:
    fieldnames = [
        "document_id",
        "post_id",
        "comment_id",
        "subreddit",
        "post_title",
        "parent_comment_id",
        "parent_text",
        "parent_context_status",
        "created_at",
        "score",
        "original_text",
        "normalized_text",
        "language",
        "script",
        "is_code_mixed",
        "detected_languages",
        "sentiment",
        "emotions",
        "issues",
        "response_type",
        "is_solution",
        "review_status",
        "abstain_reason",
        "annotator_id",
        "annotator_notes",
    ]
    rows = [
        {
            "document_id": "doc_bot_1",
            "post_id": "p1",
            "comment_id": "c_bot_1",
            "subreddit": "IIMSambalpurUG",
            "post_title": "IIM Sambalpur placements",
            "parent_comment_id": "",
            "parent_text": "",
            "parent_context_status": "top_level",
            "created_at": "2026-01-10T10:00:00Z",
            "score": "1",
            "original_text": "Hi <USER_MENTION>! Welcome to r/IIMSambalpurUG. I am a bot, and this action was performed automatically.",
            "normalized_text": "hi welcome to r iimsambalpurug i am a bot and this action was performed automatically",
            "language": "en",
            "script": "latin",
            "is_code_mixed": "false",
            "detected_languages": "en",
            "sentiment": "",
            "emotions": "",
            "issues": "",
            "response_type": "",
            "is_solution": "",
            "review_status": "unreviewed",
            "abstain_reason": "",
            "annotator_id": "",
            "annotator_notes": "",
        },
        {
            "document_id": "doc_gif_1",
            "post_id": "p1",
            "comment_id": "c_gif_1",
            "subreddit": "MBAIndia",
            "post_title": "CAT percentile thread",
            "parent_comment_id": "",
            "parent_text": "",
            "parent_context_status": "top_level",
            "created_at": "2026-01-10T10:01:00Z",
            "score": "2",
            "original_text": "![gif](giphy|abc123)",
            "normalized_text": "gif giphy abc123",
            "language": "en",
            "script": "latin",
            "is_code_mixed": "false",
            "detected_languages": "en",
            "sentiment": "",
            "emotions": "",
            "issues": "",
            "response_type": "",
            "is_solution": "",
            "review_status": "unreviewed",
            "abstain_reason": "",
            "annotator_id": "",
            "annotator_notes": "",
        },
        {
            "document_id": "doc_h_1",
            "post_id": "p1",
            "comment_id": "c_h_1",
            "subreddit": "IIMSambalpurUG",
            "post_title": "IIM Sambalpur placements",
            "parent_comment_id": "",
            "parent_text": "",
            "parent_context_status": "top_level",
            "created_at": "2026-01-10T10:05:00Z",
            "score": "5",
            "original_text": "Placement transparency needs clearer median CTC reporting.",
            "normalized_text": "placement transparency needs clearer median ctc reporting",
            "language": "en",
            "script": "latin",
            "is_code_mixed": "false",
            "detected_languages": "en",
            "sentiment": "",
            "emotions": "",
            "issues": "",
            "response_type": "",
            "is_solution": "",
            "review_status": "unreviewed",
            "abstain_reason": "",
            "annotator_id": "",
            "annotator_notes": "",
        },
        {
            "document_id": "doc_h_2",
            "post_id": "p1",
            "comment_id": "c_h_2",
            "subreddit": "IIMSambalpurUG",
            "post_title": "IIM Sambalpur placements",
            "parent_comment_id": "c_h_1",
            "parent_text": "Placement transparency needs clearer median CTC reporting.",
            "parent_context_status": "available",
            "created_at": "2026-01-10T10:10:00Z",
            "score": "4",
            "original_text": "Email the placement cell with a joint batch request for audited median figures.",
            "normalized_text": "email the placement cell with a joint batch request for audited median figures",
            "language": "en",
            "script": "latin",
            "is_code_mixed": "false",
            "detected_languages": "en",
            "sentiment": "",
            "emotions": "",
            "issues": "",
            "response_type": "",
            "is_solution": "",
            "review_status": "unreviewed",
            "abstain_reason": "",
            "annotator_id": "",
            "annotator_notes": "",
        },
        {
            "document_id": "doc_h_3",
            "post_id": "p2",
            "comment_id": "c_h_3",
            "subreddit": "MBAIndia",
            "post_title": "Hostel and fees query",
            "parent_comment_id": "",
            "parent_text": "",
            "parent_context_status": "top_level",
            "created_at": "2026-01-11T09:00:00Z",
            "score": "3",
            "original_text": "SBI Scholar loan covers tuition cleanly once you upload the offer letter.",
            "normalized_text": "sbi scholar loan covers tuition cleanly once you upload the offer letter",
            "language": "en",
            "script": "latin",
            "is_code_mixed": "false",
            "detected_languages": "en",
            "sentiment": "",
            "emotions": "",
            "issues": "",
            "response_type": "",
            "is_solution": "",
            "review_status": "unreviewed",
            "abstain_reason": "",
            "annotator_id": "",
            "annotator_notes": "",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class AnnotationStudioTests(unittest.TestCase):
    def test_studio_ui_state_save_validation_abstain_noise_and_train_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "candidates.csv"
            _write_temp_candidates_csv(csv_path)

            app = create_annotation_studio_app(
                csv_path=csv_path,
                static_export_path=tmp_path / "unused.json",
                out_dir=tmp_path / "out",
            )
            client = TestClient(app)

            # 1. HTML and tokens CSS load cleanly
            r_html = client.get("/annotate")
            self.assertEqual(r_html.status_code, 200)
            self.assertIn("RCI Annotation Studio", r_html.text)
            r_tok = client.get("/annotate-tokens.css")
            self.assertEqual(r_tok.status_code, 200)
            self.assertIn("--color-paper", r_tok.text)

            # 2. State returns bot/GIF flags and thread-balanced order putting human rows first
            st = client.get("/api/annotate/state").json()
            self.assertEqual(st["total_rows"], 5)
            self.assertEqual(st["unreviewed_bot_or_gif_noise"], 2)
            by_id = {r["document_id"]: r for r in st["rows"]}
            self.assertTrue(by_id["doc_bot_1"]["is_automod"])
            self.assertTrue(by_id["doc_gif_1"]["is_gif_only"])
            self.assertLess(by_id["doc_h_1"]["balanced_rank"], by_id["doc_bot_1"]["balanced_rank"])
            self.assertLess(by_id["doc_h_3"]["balanced_rank"], by_id["doc_h_2"]["balanced_rank"])

            # 3. Save a valid labeled row and verify atomic CSV + .bak creation
            save_ok = client.post(
                "/api/annotate/row/doc_h_1",
                json={
                    "review_status": "labeled",
                    "sentiment": "negative",
                    "emotions": ["frustration"],
                    "issues": ["placements"],
                    "response_type": "opinion",
                    "is_solution": "not_solution",
                    "annotator_id": "tester",
                    "annotator_notes": "clear complaint",
                },
            )
            self.assertEqual(save_ok.status_code, 200)
            body = save_ok.json()
            self.assertEqual(body["validation"]["labeled_comments"], 1)
            self.assertEqual(body["validation"]["labeled_threads"], 1)
            self.assertTrue(csv_path.with_suffix(".csv.bak").exists())

            # 4. Reject forbidden placeholder or missing required label or missing abstain_reason
            bad_placeholder = client.post(
                "/api/annotate/row/doc_h_2",
                json={
                    "review_status": "labeled",
                    "sentiment": "unknown",
                    "response_type": "advice",
                    "is_solution": "solution",
                },
            )
            self.assertEqual(bad_placeholder.status_code, 422)

            bad_abstain = client.post(
                "/api/annotate/row/doc_h_2",
                json={"review_status": "uncertain", "abstain_reason": ""},
            )
            self.assertEqual(bad_abstain.status_code, 422)

            # 5. Uncertain / abstain clears label cells and increments adjudication queue
            unc_ok = client.post(
                "/api/annotate/row/doc_h_2",
                json={
                    "review_status": "uncertain",
                    "abstain_reason": "mixed_or_borderline",
                    "sentiment": "positive",  # must be stripped on uncertain!
                    "annotator_id": "tester",
                },
            )
            self.assertEqual(unc_ok.status_code, 200)
            unc_row = unc_ok.json()["row"]
            self.assertEqual(unc_row["review_status"], "uncertain")
            self.assertEqual(unc_row["sentiment"], "")
            self.assertEqual(unc_row["response_type"], "")
            self.assertEqual(unc_row["is_solution"], "")
            self.assertEqual(unc_ok.json()["validation"]["adjudication_queue_count"], 1)

            # 6. Bulk exclude bot/GIF noise excludes ONLY the 2 unreviewed noise rows
            excl = client.post(
                "/api/annotate/exclude-bot-noise",
                json={"annotator_id": "tester"},
            )
            self.assertEqual(excl.status_code, 200)
            self.assertEqual(excl.json()["excluded_count"], 2)
            self.assertEqual(
                set(excl.json()["excluded_ids"]), {"doc_bot_1", "doc_gif_1"}
            )
            # doc_h_3 remains unreviewed!
            after = client.get("/api/annotate/state").json()
            after_by_id = {r["document_id"]: r for r in after["rows"]}
            self.assertEqual(after_by_id["doc_h_3"]["review_status"], "unreviewed")
            self.assertEqual(after_by_id["doc_bot_1"]["review_status"], "excluded")
            self.assertEqual(after_by_id["doc_bot_1"]["abstain_reason"], "off_topic_noise")

            # 7. Training gate strictly refuses when dataset is not ready
            train_blocked = client.post(
                "/api/annotate/train", json={"include_muril": False}
            )
            self.assertEqual(train_blocked.status_code, 400)


if __name__ == "__main__":
    unittest.main()
