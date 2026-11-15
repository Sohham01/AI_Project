"""Offline unit tests for the Arctic Shift ingestion path.

These were previously a live-network script.  They now mock the HTTP layer so
they run deterministically without touching Reddit or the Arctic Shift API.
"""

import unittest
from unittest.mock import MagicMock, patch

from app.services.ingestion.arctic_shift import ArcticShiftClient
from app.services.ingestion.arctic_shift_normalizer import ArcticShiftNormalizer
from app.services.ingestion.providers.arctic_shift import ArcticShiftProvider
from app.services.ingestion.providers.base import ProviderRequestError
from app.services.ingestion.reddit import RedditIngestion


class TestArcticShiftClient(unittest.TestCase):
    def test_search_posts_forwards_query_and_paging(self):
        client = ArcticShiftClient()
        with patch.object(
            client, "_get", return_value={"data": []}
        ) as mock_get:
            client.search_posts(
                subreddit="CATpreparation",
                query="IIM placement",
                limit=7,
                after="10d",
                before="1d",
                sort="asc",
            )

        endpoint = mock_get.call_args.args[0]
        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(endpoint, "/api/posts/search")
        self.assertEqual(params["subreddit"], "CATpreparation")
        self.assertEqual(params["query"], "IIM placement")
        self.assertEqual(params["limit"], 7)
        self.assertEqual(params["after"], "10d")
        self.assertEqual(params["before"], "1d")
        self.assertEqual(params["sort"], "asc")

    def test_search_posts_omits_empty_query(self):
        client = ArcticShiftClient()
        with patch.object(
            client, "_get", return_value={"data": []}
        ) as mock_get:
            client.search_posts(subreddit="CATpreparation")

        params = mock_get.call_args.kwargs["params"]
        self.assertNotIn("query", params)
        self.assertIn("after", params)

    def test_get_comments_tree_normalises_link_id(self):
        client = ArcticShiftClient()
        with patch.object(
            client, "_get", return_value={"data": []}
        ) as mock_get:
            client.get_comments_tree("t3_abc123")

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["link_id"], "t3_abc123")

    def test_get_comments_tree_forwards_limit(self):
        client = ArcticShiftClient()
        with patch.object(client, "_get", return_value={"data": []}) as mock_get:
            client.get_comments_tree("p1", limit=7)
        self.assertEqual(mock_get.call_args.kwargs["params"]["limit"], 7)

    def test_http_error_does_not_expose_url_or_response_body(self):
        client = ArcticShiftClient()
        response = MagicMock(ok=False, status_code=403)
        response.url = "https://example.invalid/?token=never-print-this"
        response.text = "Bearer never-print-this"
        with patch.object(client.session, "get", return_value=response):
            with self.assertRaises(ProviderRequestError) as ctx:
                client._get("/api/posts/search", params={"query": "safe"})
        self.assertNotIn("never-print-this", str(ctx.exception))
        self.assertNotIn("example.invalid", str(ctx.exception))
        self.assertEqual(ctx.exception.failure_category, "http_status")
        self.assertEqual(ctx.exception.status_code, 403)


class TestArcticShiftNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = ArcticShiftNormalizer()

    def test_clean_id_strips_only_leading_prefix(self):
        self.assertEqual(self.normalizer._clean_id("t3_abc"), "abc")
        self.assertEqual(self.normalizer._clean_id("t1_abc"), "abc")
        self.assertEqual(self.normalizer._clean_id("abc"), "abc")
        # A prefix appearing mid-string must not be stripped.
        self.assertEqual(self.normalizer._clean_id("xt3_abc"), "xt3_abc")

    def test_normalize_post_produces_naive_utc(self):
        post = self.normalizer.normalize_post(
            {
                "id": "t3_x",
                "title": "title",
                "selftext": "body",
                "subreddit": "CATpreparation",
                "created_utc": 1_700_000_000,
                "score": 3,
                "num_comments": 2,
            }
        )
        self.assertEqual(post.reddit_id, "x")
        self.assertIsNone(post.created_at.tzinfo)


class TestArcticShiftProvider(unittest.TestCase):
    def setUp(self):
        self.provider = ArcticShiftProvider()

    def test_search_posts_forwards_query(self):
        self.provider.client = MagicMock()
        self.provider.client.search_posts.return_value = {"data": []}

        self.provider.search_posts(
            query="fees", subreddit="CATpreparation", limit=3
        )

        _, kwargs = self.provider.client.search_posts.call_args
        self.assertEqual(kwargs["query"], "fees")
        self.assertEqual(kwargs["subreddit"], "CATpreparation")
        self.assertEqual(kwargs["limit"], 3)

    def test_search_posts_rejects_non_list_data(self):
        self.provider.client = MagicMock()
        self.provider.client.search_posts.return_value = {"data": None}
        with self.assertRaises(ProviderRequestError) as ctx:
            self.provider.search_posts()
        self.assertEqual(ctx.exception.failure_category, "invalid_payload")

    def test_get_comments_walks_tree_and_links_parents(self):
        self.provider.client = MagicMock()
        self.provider.client.get_comments_tree.return_value = {
            "data": [
                {
                    "kind": "t1",
                    "data": {
                        "id": "t1_c1",
                        "body": "root",
                        "created_utc": 1_700_000_000,
                        "score": 1,
                        "parent_id": "t3_p",
                    },
                },
                {
                    "kind": "t1",
                    "data": {
                        "id": "t1_c2",
                        "body": "reply",
                        "created_utc": 1_700_000_001,
                        "score": 2,
                        "parent_id": "t1_c1",
                        "replies": {
                            "data": {
                                "children": [
                                    {
                                        "kind": "t1",
                                        "data": {
                                            "id": "t1_c3",
                                            "body": "nested",
                                            "created_utc": 1_700_000_002,
                                            "score": 1,
                                            "parent_id": "t1_c2",
                                        },
                                    }
                                ]
                            }
                        },
                    },
                },
                {"kind": "more", "data": {"id": "more_1", "children": ["c4"]}},
            ]
        }

        comments = self.provider.get_comments("p")

        self.assertEqual([c.reddit_id for c in comments], ["c1", "c2", "c3"])
        self.assertIsNone(comments[0].parent_reddit_id)
        self.assertEqual(comments[1].parent_reddit_id, "c1")
        self.assertEqual(comments[2].parent_reddit_id, "c2")

    def test_get_comments_forwards_bounded_limit(self):
        self.provider.client = MagicMock()
        self.provider.client.get_comments_tree.return_value = {"data": []}
        self.provider.get_comments("p", limit=7)
        self.provider.client.get_comments_tree.assert_called_once_with(
            "p", limit=7, timeout=None
        )

    def test_get_comments_skips_deleted_bodies(self):
        self.provider.client = MagicMock()
        self.provider.client.get_comments_tree.return_value = {
            "data": [
                {
                    "kind": "t1",
                    "data": {
                        "id": "t1_c1",
                        "body": "[deleted]",
                        "created_utc": 1_700_000_000,
                        "parent_id": "t3_p",
                    },
                }
            ]
        }
        self.assertEqual(self.provider.get_comments("p"), [])


class TestRedditIngestionAdapter(unittest.TestCase):
    def test_fetch_posts_delegates_to_provider(self):
        ingestion = RedditIngestion()
        ingestion._provider = MagicMock()
        ingestion._provider.search_posts.return_value = []

        ingestion.fetch_posts(
            query="IIM", subreddit="CATpreparation", limit=5
        )

        ingestion._provider.search_posts.assert_called_once_with(
            query="IIM", subreddit="CATpreparation", limit=5
        )

    def test_fetch_comments_delegates_to_provider(self):
        ingestion = RedditIngestion()
        ingestion._provider = MagicMock()
        ingestion._provider.get_comments.return_value = []

        ingestion.fetch_comments("p1")

        ingestion._provider.get_comments.assert_called_once_with("p1")


if __name__ == "__main__":
    unittest.main()
