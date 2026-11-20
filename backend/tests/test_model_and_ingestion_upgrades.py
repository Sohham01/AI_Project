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


try:
    import torch as _torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch optional dependency not installed")
class TestV3ModelArchitectureAndHybridEnsemble(unittest.TestCase):
    def test_pooled_multisample_head_shapes_and_llrd_groups(self):
        import types
        import torch
        from ml.models.transformer.muril_classifier import (
            _PooledMultiSampleModel,
            _build_llrd_param_groups,
        )

        class _TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(32, 8)
                self.encoder = torch.nn.Module()
                self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                x = self.embeddings(input_ids)
                for lyr in self.encoder.layer:
                    x = lyr(x)
                return types.SimpleNamespace(last_hidden_state=x)

        enc = _TinyEncoder()
        model = _PooledMultiSampleModel(enc, hidden_size=8, num_labels=6, multi_sample_dropout=5, dropout_rate=0.2)
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        logits = model(ids, mask)
        self.assertEqual(tuple(logits.shape), (2, 6))

        groups = _build_llrd_param_groups(
            model,
            base_lr=3e-5,
            head_lr_multiplier=3.0,
            llrd=0.90,
            weight_decay=0.01,
            num_hidden_layers=2,
        )
        lrs = sorted({round(g["lr"], 8) for g in groups})
        # Embeddings < layer 0 < layer 1 < head
        self.assertGreaterEqual(len(lrs), 4)
        self.assertAlmostEqual(lrs[-1], 9e-5, places=7)

    def test_validation_temperature_scaling_reduces_overconfident_nll(self):
        import numpy as np
        from ml.models.transformer.muril_classifier import _calibrate_temperature, _validation_nll

        # Deliberately overconfident logits with 1 misclassified point -> T > 1.0 lowers NLL
        logits = np.array(
            [[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [6.0, -6.0]],
            dtype=np.float32,
        )
        y_val = np.array([0, 0, 1, 1], dtype=np.int64)
        raw_nll = _validation_nll(logits, y_val, multilabel=False, temperature=1.0)
        best_t = _calibrate_temperature(logits, y_val, multilabel=False)
        cal_nll = _validation_nll(logits, y_val, multilabel=False, temperature=best_t)
        self.assertGreater(best_t, 1.0)
        self.assertLess(cal_nll, raw_nll)

    def test_dapt_mlm_masking_never_masks_special_or_pad_tokens(self):
        import torch
        from ml.models.transformer.muril_dapt import _mask_tokens_bert

        class _FakeTok:
            cls_token_id = 101
            sep_token_id = 102
            pad_token_id = 0
            unk_token_id = 100
            mask_token_id = 103

            def __len__(self):
                return 1000

        tok = _FakeTok()
        ids = torch.tensor([[101, 200, 201, 202, 203, 204, 205, 102, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
        gen = torch.Generator().manual_seed(42)
        masked_ids, labels = _mask_tokens_bert(ids, mask, tok, mlm_probability=0.5, generator=gen)
        # Special tokens & padding must NEVER be masked
        for pos in (0, 7, 8, 9):
            self.assertEqual(int(labels[0, pos].item()), -100)
            self.assertEqual(int(masked_ids[0, pos].item()), int(ids[0, pos].item()))

    def test_hybrid_blend_selects_complementary_alpha_on_validation(self):
        import numpy as np
        from backend.build_live_corpus_artifacts import _select_hybrid_blend_on_validation

        labels = ["not_solution", "solution"]
        val_targets = ["not_solution", "not_solution", "solution", "solution"]
        # Model A is strong on rows 0, 2; Model B is strong on rows 1, 3 -> blend beats either solo
        p_a = np.array([[0.95, 0.05], [0.45, 0.55], [0.05, 0.95], [0.55, 0.45]], dtype=np.float32)
        p_b = np.array([[0.45, 0.55], [0.95, 0.05], [0.55, 0.45], [0.05, 0.95]], dtype=np.float32)
        alpha, _, _, val_f1 = _select_hybrid_blend_on_validation(
            p_a, p_b, val_targets, labels, multilabel=False, is_binary_solution=True
        )
        self.assertGreater(alpha, 0.0)
        self.assertLess(alpha, 1.0)
        self.assertAlmostEqual(val_f1, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
