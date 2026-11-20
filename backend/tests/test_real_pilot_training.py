"""Tests for backend/train_real_pilot_models.py:
  1. Unsupervised NMF Topic Discovery runs cleanly on validated real/pilot static exports.
  2. Supervised 5-task training strictly refuses blank/unreviewed CSVs (no label fabrication).
  3. Supervised 5-task training runs end-to-end with zero post_id leakage when valid
     human-format labels satisfy readiness gates, while excluding uncertain/abstain items.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.train_real_pilot_models import (
    main as train_main,
    train_supervised_real_pilot,
    train_unsupervised_topic_model,
)


def _write_sample_static_export(path: Path) -> None:
    posts = [
        {
            "post_id": f"p{i}",
            "subreddit": "IIMSambalpurUG" if i % 2 == 0 else "MBAIndia",
            "title": f"IIM Sambalpur discussion thread {i}",
            "body": "Looking for honest updates on campus life and placements.",
            "created_at": f"2026-01-{(i % 20) + 1:02d}T10:00:00Z",
            "score": 5 + i,
            "num_comments": 3,
            "url": f"https://www.reddit.com/r/IIMSambalpurUG/comments/p{i}/t/",
            "comments": [
                {
                    "comment_id": f"c{i}a",
                    "body": "Placements report shows consulting and analytics roles improving this season.",
                    "created_at": f"2026-01-{(i % 20) + 1:02d}T11:00:00Z",
                    "score": 4,
                    "parent_comment_id": None,
                },
                {
                    "comment_id": f"c{i}b",
                    "body": "Hostel room maintenance and mess menu quality depend on block committee feedback.",
                    "created_at": f"2026-01-{(i % 20) + 1:02d}T11:15:00Z",
                    "score": 3,
                    "parent_comment_id": f"c{i}a",
                },
                {
                    "comment_id": f"c{i}c",
                    "body": "Tuition fees and SBI scholar loan repayment require realistic ROI planning.",
                    "created_at": f"2026-01-{(i % 20) + 1:02d}T11:30:00Z",
                    "score": 2,
                    "parent_comment_id": None,
                },
            ],
        }
        for i in range(1, 13)
    ]
    payload = {
        "format_version": 1,
        "metadata": {
            "dataset_id": "unit-test-real-pilot",
            "source": {
                "community": "IIMSambalpurUG",
                "source_reference": "test:sha256:0001",
                "source_provider": "arctic_shift",
                "source_mode": "direct",
                "source_url": "https://arctic-shift.photon-reddit.com/api",
                "collected_at": "2026-02-01T00:00:00Z",
            },
            "completeness": {"comments_complete": False},
            "authorization": {
                "authorization_status": "pending-review",
                "authorization_reference": "unit-test",
            },
        },
        "posts": posts,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_unified_csv(path: Path, *, labeled: bool) -> None:
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
    templates = [
        ("negative", "frustration|stress", "placements", "opinion", "not_solution",
         "Placement shortlist criteria felt very unclear and stressful this season."),
        ("positive", "gratitude|confidence", "hostel_facilities", "solution", "solution",
         "Submit a joint ticket on the hostel portal with room numbers to get coolers fixed fast."),
        ("neutral", "hope", "academic_workload", "advice", "solution",
         "Ask the class representative to combine quiz schedules and request a buffer."),
        ("negative", "stress", "financial_concerns", "counterargument", "not_solution",
         "Disagree that ROI is guaranteed when loan fees exceed twenty lakhs."),
        ("positive", "confidence", "curriculum_teaching", "personal_experience", "not_solution",
         "In my term finance electives used practical live corporate balance sheets."),
        ("neutral", "curiosity", "campus_infrastructure|administrative_communication", "agreement", "not_solution",
         "Totally agree about library wifi drops and late admin circulars."),
    ]

    rows = []
    idx = 0
    for thread_num in range(1, 13):  # 12 distinct threads for clean 70/15/15 post_id split
        post_id = f"thread{thread_num:02d}"
        for t_idx, (sent, emo, iss, resp, sol, text) in enumerate(templates):
            idx += 1
            rows.append(
                {
                    "document_id": f"doc_{idx:03d}",
                    "post_id": post_id,
                    "comment_id": f"c_{idx:03d}",
                    "subreddit": "IIMSambalpurUG",
                    "post_title": f"Thread {thread_num}",
                    "parent_comment_id": "",
                    "parent_text": "",
                    "parent_context_status": "top_level",
                    "created_at": "2026-01-10T10:00:00Z",
                    "score": "3",
                    "original_text": f"{text} Variant {thread_num}-{t_idx}.",
                    "normalized_text": f"{text} Variant {thread_num}-{t_idx}.",
                    "language": "en",
                    "script": "latin",
                    "is_code_mixed": "false",
                    "detected_languages": "en",
                    "sentiment": sent if labeled else "",
                    "emotions": emo if labeled else "",
                    "issues": iss if labeled else "",
                    "response_type": resp if labeled else "",
                    "is_solution": sol if labeled else "",
                    "review_status": "labeled" if labeled else "unreviewed",
                    "abstain_reason": "",
                    "annotator_id": "tester" if labeled else "",
                    "annotator_notes": "",
                }
            )
    if labeled:
        # Include an explicit uncertain row to ensure it goes to adjudication, never training
        rows.append(
            {
                "document_id": "doc_uncertain_01",
                "post_id": "thread01",
                "comment_id": "c_unc_01",
                "subreddit": "IIMSambalpurUG",
                "post_title": "Thread 1",
                "parent_comment_id": "",
                "parent_text": "",
                "parent_context_status": "top_level",
                "created_at": "2026-01-10T10:00:00Z",
                "score": "1",
                "original_text": "hmm okay idk",
                "normalized_text": "hmm okay idk",
                "language": "en",
                "script": "latin",
                "is_code_mixed": "false",
                "detected_languages": "en",
                "sentiment": "",
                "emotions": "",
                "issues": "",
                "response_type": "",
                "is_solution": "",
                "review_status": "uncertain",
                "abstain_reason": "insufficient_context",
                "annotator_id": "tester",
                "annotator_notes": "Needs parent context",
            }
        )

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RealPilotTrainingTests(unittest.TestCase):
    def test_unsupervised_nmf_trains_on_validated_static_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            export_path = tmp_path / "sanitized.json"
            out_dir = tmp_path / "artifacts"
            _write_sample_static_export(export_path)

            res = train_unsupervised_topic_model(
                export_path, out_dir, n_topics=4, n_keywords=6, stop_words="english"
            )
            self.assertEqual(res["status"], "trained")
            self.assertFalse(res["requires_labels"])
            self.assertEqual(res["threads_loaded"], 12)
            self.assertEqual(res["documents_modeled"], 36)
            self.assertEqual(res["n_topics"], 4)
            self.assertGreater(res["vocabulary_size"], 5)
            self.assertTrue((out_dir / "topic_discovery.json").exists())

    def test_supervised_refuses_blank_unlabeled_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            export_path = tmp_path / "sanitized.json"
            ann_path = tmp_path / "blank_candidates.csv"
            out_dir = tmp_path / "artifacts"
            _write_sample_static_export(export_path)
            _write_unified_csv(ann_path, labeled=False)

            sup = train_supervised_real_pilot(ann_path, out_dir)
            self.assertEqual(sup["status"], "blocked_awaiting_human_labels")
            self.assertFalse(sup["ready_for_training"])
            self.assertEqual(sup["labeled_comments"], 0)
            self.assertFalse((out_dir / "models").exists())

            # CLI in auto mode trains unsupervised and reports supervised blocked cleanly (exit 0)
            rc_auto = train_main(
                [
                    "--static-export",
                    str(export_path),
                    "--annotations",
                    str(ann_path),
                    "--out-dir",
                    str(out_dir),
                    "--mode",
                    "auto",
                    "--n-topics",
                    "4",
                ]
            )
            self.assertEqual(rc_auto, 0)
            summary = json.loads((out_dir / "training_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["unsupervised_topic_discovery"]["status"], "trained")
            self.assertEqual(
                summary["supervised_models"]["status"],
                "blocked_awaiting_human_labels",
            )

            # CLI in strict supervised mode returns exit 4 when labels are missing
            rc_sup = train_main(
                [
                    "--annotations",
                    str(ann_path),
                    "--out-dir",
                    str(out_dir),
                    "--mode",
                    "supervised",
                ]
            )
            self.assertEqual(rc_sup, 4)

    def test_supervised_trains_all_5_tasks_zero_leakage_when_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ann_path = tmp_path / "labeled_candidates.csv"
            out_dir = tmp_path / "artifacts"
            _write_unified_csv(ann_path, labeled=True)

            sup = train_supervised_real_pilot(
                ann_path,
                out_dir,
                min_comments=60,
                min_threads=10,
                min_per_class=3,
                include_muril=False,
            )
            self.assertEqual(sup["status"], "trained")
            self.assertTrue(sup["ready_for_training"])
            self.assertEqual(sup["labeled_comments"], 72)
            self.assertEqual(sup["adjudication_queue_count"], 1)
            self.assertTrue((out_dir / "validated_task_csvs" / "adjudication_queue.csv").exists())

            for task in ("sentiment", "emotion", "issue", "response", "solution"):
                t_rep = sup["tasks"][task]
                self.assertFalse(t_rep["split"]["leakage_detected"])
                self.assertEqual(t_rep["split"]["group_by"], "post_id")
                self.assertTrue(Path(t_rep["baseline_v1_tfidf"]["artifact_path"]).exists())
                self.assertTrue(
                    Path(t_rep["upgraded_v2_word_char_calibrated"]["artifact_path"]).exists()
                )
                held = t_rep["upgraded_v2_word_char_calibrated"]["held_out_test"]
                self.assertIn("macro_f1", held)
                if task in {"emotion", "issue"}:
                    self.assertIn("subset_accuracy", held)
                    self.assertIn("hamming_loss", held)
                else:
                    self.assertIn("confusion_matrix", held)


if __name__ == "__main__":
    unittest.main()
