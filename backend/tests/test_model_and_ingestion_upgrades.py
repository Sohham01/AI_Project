"""Unit tests for the model upgrades (Word+Char TF-IDF, validation threshold
calibration) and immediate real-data ingestion tools (RedditOAuthProvider,
import_reddit_export, run_decodo_pilot dry-run).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
for p in (str(ROOT), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.ingestion.providers.base import UnsupportedOperation
from app.services.ingestion.providers.reddit_oauth import RedditOAuthProvider
from app.services.ingestion.static_export import load_static_export
from app.services.nlp.issue_identification.base import IssueModelConfig
from app.services.nlp.issue_identification.models.annotation import IssueAnnotation
from app.services.nlp.issue_identification.tfidf_logistic_classifier import (
    TfidfLogisticIssueClassifier,
)
from app.services.nlp.sentiment_emotion.base import EmotionModelConfig
from app.services.nlp.sentiment_emotion.models.annotation import EmotionAnnotation
from app.services.nlp.sentiment_emotion.tfidf_logistic_emotion import (
    TfidfLogisticEmotionClassifier,
)
from backend.import_reddit_export import main as import_export_main
from backend.run_decodo_pilot import main as decodo_pilot_main


class TestClassicalModelUpgrades(unittest.TestCase):
    def test_issue_word_char_and_threshold_calibration(self):
        train_rows = [
            IssueAnnotation(document_id=f"h{i}", text=f"hostel mess water problem block {i}", labels=["hostel_facilities"])
            for i in range(6)
        ] + [
            IssueAnnotation(document_id=f"p{i}", text=f"placement package recruiter interview {i}", labels=["placements"])
            for i in range(6)
        ] + [
            IssueAnnotation(document_id=f"f{i}", text=f"fees loan scholarship tuition cost {i}", labels=["financial_concerns"])
            for i in range(6)
        ]
        val_rows = [
            IssueAnnotation(document_id="vh", text="hostl mes watr issue", labels=["hostel_facilities"]),
            IssueAnnotation(document_id="vp", text="placemnt packge intrview", labels=["placements"]),
            IssueAnnotation(document_id="vf", text="tution fee scholrship loan", labels=["financial_concerns"]),
        ]
        clf = TfidfLogisticIssueClassifier(
            config=IssueModelConfig(use_char_ngrams=True, class_weight="balanced")
        )
        report = clf.train(train_rows)
        self.assertGreater(report.n_features, 50)
        tuned = clf.calibrate_thresholds(val_rows)
        self.assertEqual(set(tuned.keys()), {"hostel_facilities", "placements", "financial_concerns"})
        res = clf.predict([row.to_issue_document() for row in val_rows])
        self.assertIn("hostel_facilities", res.predictions[0].predicted_issue_labels)
        self.assertIn("placements", res.predictions[1].predicted_issue_labels)
        self.assertIn("financial_concerns", res.predictions[2].predicted_issue_labels)

    def test_emotion_word_char_and_threshold_calibration(self):
        train_rows = [
            EmotionAnnotation(document_id=f"s{i}", text=f"feeling stressed frustrated tired {i}", emotions=["stress", "frustration"])
            for i in range(6)
        ] + [
            EmotionAnnotation(document_id=f"g{i}", text=f"grateful thankful hopeful great {i}", emotions=["gratitude", "hope"])
            for i in range(6)
        ]
        val_rows = [
            EmotionAnnotation(document_id="vs", text="stresed and frustrted", emotions=["stress", "frustration"]),
            EmotionAnnotation(document_id="vg", text="thankfl and hopefl", emotions=["gratitude", "hope"]),
        ]
        clf = TfidfLogisticEmotionClassifier(
            config=EmotionModelConfig(use_char_ngrams=True, class_weight="balanced")
        )
        report = clf.train(train_rows)
        self.assertGreater(report.n_features, 40)
        tuned = clf.calibrate_thresholds(val_rows)
        self.assertEqual(set(tuned.keys()), {"frustration", "gratitude", "hope", "stress"})
        res = clf.predict([row.to_document() for row in val_rows])
        self.assertIn("stress", res.predictions[0].emotions)
        self.assertIn("gratitude", res.predictions[1].emotions)


class TestRedditOAuthProvider(unittest.TestCase):
    def test_unconfigured_raises_unsupported(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = RedditOAuthProvider()
            self.assertFalse(provider.is_configured)
            with self.assertRaises(UnsupportedOperation):
                provider.search_posts(subreddit="IIMSambalpur", limit=5)

    def test_search_posts_and_comments_normalized(self):
        provider = RedditOAuthProvider(
            client_id="cid", client_secret="secret", user_agent="test-ua/1.0"
        )
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok123", "expires_in": 3600}

        posts_resp = MagicMock(status_code=200)
        posts_resp.json.return_value = {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "abc123",
                            "title": "IIM Sambalpur hostel review",
                            "selftext": "How is Block B wifi?",
                            "subreddit": "MBAIndia",
                            "created_utc": 1730000000,
                            "score": 14,
                            "num_comments": 2,
                            "permalink": "/r/MBAIndia/comments/abc123/iim_sambalpur_hostel_review/",
                        },
                    }
                ]
            },
        }
        comments_resp = MagicMock(status_code=200)
        comments_resp.json.return_value = [
            {"kind": "Listing", "data": {"children": []}},
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "id": "c1",
                                "body": "Wifi improved after router swap.",
                                "parent_id": "t3_abc123",
                                "created_utc": 1730000100,
                                "score": 5,
                                "replies": {
                                    "kind": "Listing",
                                    "data": {
                                        "children": [
                                            {
                                                "kind": "t1",
                                                "data": {
                                                    "id": "c2",
                                                    "body": "Same bro, much better now.",
                                                    "parent_id": "t1_c1",
                                                    "created_utc": 1730000200,
                                                    "score": 3,
                                                    "replies": "",
                                                },
                                            }
                                        ]
                                    },
                                },
                            },
                        }
                    ]
                },
            },
        ]

        with patch.object(provider._session, "post", return_value=token_resp), patch.object(
            provider._session, "get", side_effect=[posts_resp, comments_resp]
        ):
            posts = provider.search_posts(subreddit="MBAIndia", query="IIM Sambalpur", limit=5)
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0].reddit_id, "abc123")
            self.assertEqual(posts[0].subreddit, "MBAIndia")

            comments = provider.get_comments("abc123", limit=10)
            self.assertEqual(len(comments), 2)
            self.assertIsNone(comments[0].parent_reddit_id)
            self.assertEqual(comments[1].parent_reddit_id, "c1")


class TestImportRedditExportAndDecodoPilot(unittest.TestCase):
    def test_import_browser_thread_json_and_arctic_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()

            # 1. Browser-saved Reddit thread JSON
            thread_payload = [
                {
                    "kind": "Listing",
                    "data": {
                        "children": [
                            {
                                "kind": "t3",
                                "data": {
                                    "id": "p100",
                                    "subreddit": "MBAIndia",
                                    "title": "IIM Sambalpur ROI discussion",
                                    "selftext": "Is the loan manageable?",
                                    "created_utc": 1730000000,
                                    "score": 19,
                                    "num_comments": 2,
                                    "permalink": "/r/MBAIndia/comments/p100/roi/",
                                },
                            }
                        ]
                    },
                },
                {
                    "kind": "Listing",
                    "data": {
                        "children": [
                            {
                                "kind": "t1",
                                "data": {
                                    "id": "c100",
                                    "body": "Check placement reports carefully.",
                                    "parent_id": "t3_p100",
                                    "created_utc": 1730000500,
                                    "score": 7,
                                    "replies": "",
                                },
                            }
                        ]
                    },
                },
            ]
            (raw_dir / "thread_p100.json").write_text(json.dumps(thread_payload), encoding="utf-8")

            # 2. Arctic Shift JSONL posts + comments
            (raw_dir / "arctic.jsonl").write_text(
                json.dumps({
                    "id": "p200",
                    "subreddit": "IIMSambalpur",
                    "title": "Hostel mess feedback",
                    "selftext": "Menu updated this week.",
                    "created_utc": 1730010000,
                    "score": 8,
                    "num_comments": 1,
                })
                + "\n"
                + json.dumps({
                    "id": "c200",
                    "link_id": "t3_p200",
                    "parent_id": "t3_p200",
                    "body": "Email the mess committee with photos.",
                    "created_utc": 1730010300,
                    "score": 4,
                })
                + "\n",
                encoding="utf-8",
            )

            out_json = root / "validated.json"
            cand_dir = root / "candidates"
            rc = import_export_main(
                [
                    "--input", str(raw_dir),
                    "--output", str(out_json),
                    "--dataset-id", "test-real-v1",
                    "--community", "IIMSambalpur",
                    "--export-candidates", str(cand_dir),
                ]
            )
            self.assertEqual(rc, 0)
            res = load_static_export(out_json)
            self.assertTrue(res.valid)
            self.assertEqual(len(res.posts), 2)
            self.assertEqual(sum(len(p.comments) for p in res.posts), 2)
            self.assertTrue((cand_dir / "real_annotation_candidates.csv").is_file())

    def test_decodo_pilot_dry_run_makes_no_http_requests(self):
        with patch("app.services.ingestion.providers.decodo.requests.post") as mock_post:
            rc = decodo_pilot_main(["--url", "https://www.reddit.com/r/IIMSambalpur/"])
        self.assertEqual(rc, 0)
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
