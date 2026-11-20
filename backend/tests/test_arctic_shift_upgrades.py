"""Unit tests for Arctic Shift live-contract upgrades and the bounded
Arctic Shift pilot runner (``run_arctic_shift_pilot.py``).
"""

from __future__ import annotations

from datetime import datetime
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

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.arctic_shift import ArcticShiftClient
from app.services.ingestion.arctic_shift_normalizer import ArcticShiftNormalizer
from app.services.ingestion.collection import _is_retryable
from app.services.ingestion.providers.arctic_shift import ArcticShiftProvider
from app.services.ingestion.providers.base import ProviderRequestError
from app.services.ingestion.static_export import load_static_export
from backend.run_arctic_shift_pilot import collect_arctic_shift_pilot, main as pilot_main


class TestArcticShiftClientUpgrades(unittest.TestCase):
    def test_http_422_timeout_error_classifies_as_retryable_timeout_without_calling_json(self):
        client = ArcticShiftClient()
        resp = MagicMock(
            status_code=422,
            text='{"data":null,"error":"Timeout. Maybe slow down a bit"}',
            headers={"x-ratelimit-reset": "12", "x-ratelimit-remaining": "0"},
        )
        with patch.object(client.session, "get", return_value=resp):
            with self.assertRaises(ProviderRequestError) as ctx:
                client._get("/api/posts/search", params={"subreddit": "CATpreparation"})
        self.assertEqual(ctx.exception.failure_category, "timeout")
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertTrue(_is_retryable(ctx.exception))
        self.assertEqual(client.last_ratelimit_reset, 12.0)
        self.assertEqual(client.last_ratelimit_remaining, 0)
        resp.json.assert_not_called()
        self.assertNotIn("slow down", str(ctx.exception))

    def test_title_and_search_comments_parameters(self):
        client = ArcticShiftClient()
        with patch.object(client, "_get", return_value={"data": []}) as mock_get:
            client.search_posts(subreddit="CATpreparation", title="IIM Sambalpur", limit=5)
        self.assertEqual(mock_get.call_args.kwargs["params"]["title"], "IIM Sambalpur")
        self.assertNotIn("query", mock_get.call_args.kwargs["params"])

        with patch.object(client, "_get", return_value={"data": []}) as mock_get:
            client.search_comments(subreddit="MBAIndia", query="IIM Sambalpur", limit=10)
        self.assertEqual(mock_get.call_args.args[0], "/api/comments/search")
        self.assertEqual(mock_get.call_args.kwargs["params"]["body"], "IIM Sambalpur")


class TestArcticNormalizerCrosspostUrls(unittest.TestCase):
    def test_relative_reddit_crosspost_url_and_empty_url_accepted(self):
        norm = ArcticShiftNormalizer()
        base = {
            "id": "abc123",
            "title": "Crossposted discussion",
            "selftext": "",
            "subreddit": "CATPrep",
            "created_utc": 1763500000,
            "score": 5,
            "num_comments": 2,
        }
        p1 = norm.normalize_post({**base, "url": "/r/MBAIndia/comments/xyz999/iim_sambalpur/"})
        self.assertEqual(p1.url, "https://www.reddit.com/r/MBAIndia/comments/xyz999/iim_sambalpur/")
        p2 = norm.normalize_post({**base, "url": ""})
        self.assertIsNone(p2.url)


class TestArcticShiftPilotRunner(unittest.TestCase):
    def test_pilot_fallback_to_title_search_and_full_sanitized_export(self):
        prov = MagicMock(spec=ArcticShiftProvider)
        prov.client = MagicMock(last_ratelimit_reset=0.0)
        prov.last_comments_truncated = False

        post = PostData(
            reddit_id="p101",
            title="IIM Sambalpur mess update u/student_demo",
            body="Reach demo@example.edu or +91 9876543210",
            subreddit="MBAIndia",
            created_at=datetime(2026, 10, 14, 17, 48, 11),
            score=42,
            num_comments=2,
            url="https://www.reddit.com/r/MBAIndia/comments/p101/mess/",
            comments=[],
        )
        comments = [
            CommentData(
                reddit_id="c101",
                body="Raise a ticket on portal; u/helper_demo confirmed.",
                created_at=datetime(2026, 10, 14, 18, 0, 0),
                score=10,
                parent_reddit_id=None,
            ),
            CommentData(
                reddit_id="c102",
                body="Same bro, worked for us!",
                created_at=datetime(2026, 10, 14, 18, 10, 0),
                score=4,
                parent_reddit_id="c101",
            ),
        ]

        # Simulate query= timing out (HTTP 422 timeout) -> automatic title= fallback succeeding!
        def fake_search(**kwargs):
            if kwargs.get("query"):
                raise ProviderRequestError("timeout", 422)
            if kwargs.get("title"):
                return [post]
            return []

        prov.search_posts.side_effect = fake_search
        prov.get_comments.return_value = comments

        posts, summary = collect_arctic_shift_pilot(
            direct_subreddits=[],
            context_subreddits=["MBAIndia"],
            query="IIM Sambalpur",
            max_posts=5,
            max_comments_per_post=10,
            request_delay=0.0,
            max_retries=0,
            provider=prov,
            sleeper=lambda _: None,
        )
        self.assertEqual(len(posts), 1)
        self.assertEqual(len(posts[0].comments), 2)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["sources"][0]["search_field"], "title")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_json = root / "sanitized.json"
            cand_dir = root / "cands"
            with patch(
                "backend.run_arctic_shift_pilot.ArcticShiftProvider",
                return_value=prov,
            ), patch("backend.run_arctic_shift_pilot.time.sleep", return_value=None):
                rc = pilot_main(
                    [
                        "--no-direct",
                        "--context-subreddit", "MBAIndia",
                        "--max-posts", "5",
                        "--max-comments", "10",
                        "--delay", "0",
                        "--max-retries", "0",
                        "--output", str(out_json),
                        "--export-candidates", str(cand_dir),
                    ]
                )
            self.assertEqual(rc, 0)
            loaded = load_static_export(out_json)
            self.assertTrue(loaded.valid)
            self.assertTrue(loaded.usable_for_pilot)
            text = out_json.read_text(encoding="utf-8")
            self.assertNotIn("u/student_demo", text)
            self.assertNotIn("u/helper_demo", text)
            self.assertNotIn("demo@example.edu", text)
            self.assertNotIn("9876543210", text)
            self.assertTrue((cand_dir / "real_annotation_candidates.csv").is_file())
            self.assertTrue((cand_dir / "review_manifest.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
