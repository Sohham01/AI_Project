"""
Unit tests for HermesProvider.

These tests mock ``requests.post`` so they do not require a live Hermes server.
Each test exercises one of the five ``IngestionProvider`` operations and verifies
the contract documented in ``providers/hermes.py``: the provider raises
``UnsupportedOperation`` when Hermes is not configured, and returns properly-typed
``PostData`` / ``CommentData`` objects otherwise.

Run with::

    python -m unittest backend.test_hermes

or::

    pytest backend/test_hermes.py
"""

from __future__ import annotations

import datetime
import unittest
from unittest.mock import MagicMock, patch

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.providers.base import UnsupportedOperation
from app.services.ingestion.providers.hermes import HermesProvider


def _mk_post(
    reddit_id: str = "t3_abc123",
    title: str = "Test post title",
    body: str | None = "This is the self-text body.",
    subreddit: str = "CATpreparation",
    created_utc: str = "2024-01-15T10:30:00Z",
    score: int = 142,
    num_comments: int = 37,
    url: str = "https://www.reddit.com/r/CATpreparation/comments/abc123/title/",
    comments: list[dict] | None = None,
) -> dict:
    """Build a minimal Hermes ``/scrape`` post dict for test fixtures."""
    return {
        "reddit_id": reddit_id,
        "title": title,
        "body": body,
        "subreddit": subreddit,
        "created_utc": created_utc,
        "score": score,
        "num_comments": num_comments,
        "url": url,
        "comments": comments or [],
    }


def _mk_comment(
    reddit_id: str = "t1_xyz789",
    body: str = "Comment text",
    created_utc: str = "2024-01-15T11:00:00Z",
    score: int = 12,
    parent_reddit_id: str | None = None,
) -> dict:
    """Build a minimal Hermes comment dict for test fixtures."""
    return {
        "reddit_id": reddit_id,
        "body": body,
        "created_utc": created_utc,
        "score": score,
        "parent_reddit_id": parent_reddit_id,
    }


def _ok_response(data: Any) -> dict:
    """Wrap *data* in a successful Hermes JSON envelope."""
    return {"ok": True, "data": data}


class TestHermesProvider(unittest.TestCase):
    """Tests for the five ``IngestionProvider`` operations exposed by HermesProvider."""

    def setUp(self):
        self.provider = HermesProvider()
        # We set these on the instance so the provider thinks Hermes is configured.
        # The real env vars are not touched.
        self.provider.api_url = "http://test-hermes.local"
        self.provider.api_key = "test-key-123"

    def _post_mock(self, payload: dict) -> MagicMock:
        """Return a mock ``requests.Response`` that returns *payload* as JSON."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.text = ""
        return mock_resp

    # ------------------------------------------------------------------
    # Configuration checks
    # ------------------------------------------------------------------

    def test_unconfigured_raises_unsupported_operation(self):
        """When api_url or api_key is missing, every operation raises
        ``UnsupportedOperation("Hermes not configured")``.
        """
        broken = HermesProvider()
        broken.api_url = None
        broken.api_key = "some-key"
        with self.assertRaises(UnsupportedOperation):
            broken.scrape_url("http://example.com/post")

        broken.api_url = "http://example.com"
        broken.api_key = None
        with self.assertRaises(UnsupportedOperation):
            broken.scrape_url("http://example.com/post")

    # ------------------------------------------------------------------
    # scrape_url
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_scrape_url_returns_postdata(self, mock_post):
        """``scrape_url`` POSTs to ``/scrape`` and returns a ``PostData`` with
        comments parsed into ``CommentData`` objects.
        """
        comments = [
            _mk_comment(reddit_id="t1_a", body="First comment"),
            _mk_comment(
                reddit_id="t1_b",
                body="Reply",
                parent_reddit_id="t1_a",
            ),
        ]
        mock_post.return_value = self._post_mock(
            _ok_response(_mk_post(comments=comments))
        )

        post = self.provider.scrape_url("https://www.reddit.com/r/x/comments/abc/title/")

        self.assertIsInstance(post, PostData)
        self.assertEqual(post.reddit_id, "t3_abc123")
        self.assertEqual(post.title, "Test post title")
        self.assertEqual(post.body, "This is the self-text body.")
        self.assertEqual(post.subreddit, "CATpreparation")
        self.assertEqual(len(post.comments), 2)
        self.assertIsInstance(post.comments[0], CommentData)
        self.assertEqual(post.comments[0].reddit_id, "t1_a")
        self.assertIsNone(post.comments[0].parent_reddit_id)
        self.assertEqual(post.comments[1].parent_reddit_id, "t1_a")

    @patch("requests.post")
    def test_scrape_url_forwards_kwargs(self, mock_post):
        """Extra kwargs (``include_comments``, ``timeout_seconds``) are forwarded
        to the Hermes ``/scrape`` endpoint.
        """
        mock_post.return_value = self._post_mock(
            _ok_response(_mk_post(comments=[]))
        )
        self.provider.scrape_url(
            "https://www.reddit.com/r/x/comments/abc/title/",
            include_comments=False,
            timeout_seconds=300,
        )
        called_body = mock_post.call_args.kwargs["json"]
        self.assertFalse(called_body["include_comments"])
        self.assertEqual(called_body["timeout_seconds"], 300)

    @patch("requests.post")
    def test_scrape_url_handles_missing_body(self, mock_post):
        """A post with ``body: null`` produces ``PostData.body is None``."""
        mock_post.return_value = self._post_mock(
            _ok_response(_mk_post(body=None))
        )
        post = self.provider.scrape_url("https://www.reddit.com/r/x/comments/abc/title/")
        self.assertIsNone(post.body)

    @patch("requests.post")
    def test_scrape_url_raises_on_unsupported(
        self,
        mock_post,
    ):
        """HTTP 404 from Hermes is surfaced as ``UnsupportedOperation``."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.ok = False
        mock_resp.text = "not found"
        mock_post.return_value = mock_resp

        with self.assertRaises(UnsupportedOperation):
            self.provider.scrape_url("https://www.reddit.com/r/x/comments/abc/title/")

    @patch("requests.post")
    def test_scrape_url_raises_runtime_on_error(
        self,
        mock_post,
    ):
        """Non-200, non-404 responses raise ``RuntimeError`` with context."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.ok = False
        mock_resp.text = "internal server error"
        mock_post.return_value = mock_resp

        with self.assertRaises(RuntimeError):
            self.provider.scrape_url("https://www.reddit.com/r/x/comments/abc/title/")

    @patch("requests.post")
    def test_scrape_url_raises_runtime_on_ok_false(self, mock_post):
        """Hermes ``ok: false`` envelope raises ``RuntimeError``."""
        mock_post.return_value = self._post_mock(
            {"ok": False, "error": "browser crashed"}
        )
        with self.assertRaises(RuntimeError):
            self.provider.scrape_url("https://www.reddit.com/r/x/comments/abc/title/")

    @patch("requests.post")
    def test_provider_error_does_not_expose_url_or_response_body(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.ok = False
        mock_resp.text = "Bearer never-print-this"
        mock_post.return_value = mock_resp
        self.provider.api_url = "https://gateway.invalid/?token=never-print-this"
        with self.assertRaises(RuntimeError) as ctx:
            self.provider.scrape_url("https://www.reddit.com/r/x/comments/abc/title/")
        self.assertNotIn("never-print-this", str(ctx.exception))
        self.assertNotIn("gateway.invalid", str(ctx.exception))

    # ------------------------------------------------------------------
    # search_posts
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_search_posts_returns_list_of_postdata(self, mock_post):
        """``search_posts`` POSTs to ``/search`` and returns ``list[PostData]``."""
        posts = [
            _mk_post(reddit_id="t3_one", title="Post one"),
            _mk_post(reddit_id="t3_two", title="Post two", body=None),
        ]
        mock_post.return_value = self._post_mock(_ok_response(posts))

        results = self.provider.search_posts(query="IIM placement", subreddit="CATpreparation")

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], PostData)
        self.assertEqual(results[0].reddit_id, "t3_one")
        self.assertIsNone(results[1].body)  # link-post style

    @patch("requests.post")
    def test_search_posts_requires_query(self, mock_post):
        """Calling ``search_posts`` without a ``query`` kwarg raises
        ``UnsupportedOperation``.
        """
        with self.assertRaises(UnsupportedOperation):
            self.provider.search_posts(subreddit="CATpreparation")

    @patch("requests.post")
    def test_search_posts_forwards_kwargs(self, mock_post):
        """Kwargs like ``limit`` and ``recent_days`` are forwarded."""
        mock_post.return_value = self._post_mock(_ok_response([]))
        self.provider.search_posts(
            query="test",
            limit=25,
            recent_days=7,
        )
        called_body = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_body["limit"], 25)
        self.assertEqual(called_body["recent_days"], 7)

    @patch("requests.post")
    def test_search_posts_empty_result(self, mock_post):
        """An empty list from Hermes returns an empty Python list."""
        mock_post.return_value = self._post_mock(_ok_response([]))
        results = self.provider.search_posts(query="nomatch")
        self.assertEqual(results, [])

    @patch("requests.post")
    def test_search_posts_handles_object_envelope(self, mock_post):
        """If Hermes wraps the list in ``{"posts": [...]}``, we still extract it."""
        posts = [
            _mk_post(reddit_id="t3_wrapped", title="Wrapped post"),
        ]
        mock_post.return_value = self._post_mock(
            _ok_response({"posts": posts})
        )
        results = self.provider.search_posts(query="test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].reddit_id, "t3_wrapped")

    # ------------------------------------------------------------------
    # get_post
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_get_post_returns_postdata(self, mock_post):
        """``get_post`` POSTs to ``/scrape`` with ``include_comments: false``
        and returns a single ``PostData`` (no comments).
        """
        mock_post.return_value = self._post_mock(
            _ok_response(_mk_post(comments=[]))
        )
        post = self.provider.get_post("t3_abc123")
        self.assertIsInstance(post, PostData)
        self.assertEqual(post.reddit_id, "t3_abc123")
        self.assertEqual(len(post.comments), 0)

    @patch("requests.post")
    def test_get_post_uses_provided_url(self, mock_post):
        """When ``url`` is passed as a kwarg it is sent to Hermes."""
        mock_post.return_value = self._post_mock(
            _ok_response(_mk_post())
        )
        self.provider.get_post("t3_abc123", url="https://www.reddit.com/r/x/comments/abc/title/")
        called_body = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_body["url"], "https://www.reddit.com/r/x/comments/abc/title/")
        self.assertFalse(called_body["include_comments"])

    @patch("requests.post")
    def test_get_post_constructs_url_when_none_given(self, mock_post):
        """Without a ``url`` kwarg we construct a plausible Reddit URL from the ID."""
        mock_post.return_value = self._post_mock(
            _ok_response(_mk_post())
        )
        self.provider.get_post("t3_abc123")
        called_body = mock_post.call_args.kwargs["json"]
        self.assertIn("t3_abc123", called_body["url"])

    # ------------------------------------------------------------------
    # get_comments
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_get_comments_returns_list_of_commentdata(self, mock_post):
        """``get_comments`` POSTs to ``/comments`` and returns ``list[CommentData]``."""
        comments = [
            _mk_comment(reddit_id="t1_x", body="Top-level"),
            _mk_comment(reddit_id="t1_y", body="Reply", parent_reddit_id="t1_x"),
        ]
        mock_post.return_value = self._post_mock(_ok_response(comments))

        result = self.provider.get_comments(
            post_id="t3_abc123",
            url="https://www.reddit.com/r/x/comments/abc/title/",
        )

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], CommentData)
        self.assertEqual(result[0].reddit_id, "t1_x")
        self.assertIsNone(result[0].parent_reddit_id)
        self.assertEqual(result[1].parent_reddit_id, "t1_x")

    @patch("requests.post")
    def test_get_comments_requires_url(self, mock_post):
        """Calling ``get_comments`` without a ``url`` kwarg raises
        ``UnsupportedOperation``.
        """
        with self.assertRaises(UnsupportedOperation):
            self.provider.get_comments(post_id="t3_abc123")

    @patch("requests.post")
    def test_get_comments_forwards_limit(self, mock_post):
        """The ``limit`` kwarg is forwarded to Hermes."""
        mock_post.return_value = self._post_mock(_ok_response([]))
        self.provider.get_comments(
            post_id="t3_abc123",
            url="https://www.reddit.com/r/x/comments/abc/title/",
            limit=1000,
        )
        called_body = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_body["limit"], 1000)

    @patch("requests.post")
    def test_get_comments_handles_object_envelope(self, mock_post):
        """If Hermes wraps comments in ``{"comments": [...]}``, we extract them."""
        comments = [_mk_comment(reddit_id="t1_z", body="Wrapped comment")]
        mock_post.return_value = self._post_mock(
            _ok_response({"comments": comments})
        )
        result = self.provider.get_comments(
            post_id="t3_abc123",
            url="https://www.reddit.com/r/x/comments/abc/title/",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].reddit_id, "t1_z")

    # ------------------------------------------------------------------
    # discover_targets
    # ------------------------------------------------------------------

    @patch("requests.post")
    def test_discover_targets_returns_list_of_strings(self, mock_post):
        """``discover_targets`` POSTs to ``/discover`` and returns ``list[str]``."""
        urls = [
            "https://www.reddit.com/r/CATpreparation/comments/abc123/title/",
            "https://www.reddit.com/r/CATpreparation/comments/def456/another/",
        ]
        mock_post.return_value = self._post_mock(_ok_response(urls))

        result = self.provider.discover_targets(
            prompt="IIM Sambalpur placement PPO complaints",
            subreddit="CATpreparation",
        )

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertTrue(result[0].startswith("https://"))
        called_body = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_body["prompt"], "IIM Sambalpur placement PPO complaints")
        self.assertEqual(called_body["subreddit"], "CATpreparation")

    @patch("requests.post")
    def test_discover_targets_forwards_limit(self, mock_post):
        """The ``limit`` kwarg is forwarded."""
        mock_post.return_value = self._post_mock(_ok_response([]))
        self.provider.discover_targets(prompt="test", limit=20)
        called_body = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_body["limit"], 20)

    @patch("requests.post")
    def test_discover_targets_empty_result(self, mock_post):
        """An empty list from Hermes returns an empty Python list."""
        mock_post.return_value = self._post_mock(_ok_response([]))
        result = self.provider.discover_targets(prompt="nomatch")
        self.assertEqual(result, [])

    @patch("requests.post")
    def test_discover_targets_handles_object_envelope(self, mock_post):
        """If Hermes wraps URLs in objects like ``{"url": "..."}``, we extract the string."""
        mock_post.return_value = self._post_mock(
            _ok_response([{"url": "https://www.reddit.com/r/x/comments/abc/title/"}])
        )
        result = self.provider.discover_targets(prompt="test")
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].startswith("https://"))

    @patch("requests.post")
    def test_discover_targets_dedup_non_string_items(self, mock_post):
        """Non-string items in the response are coerced to str; we never crash."""
        mock_post.return_value = self._post_mock(
            _ok_response([123, {"url": "https://www.reddit.com/r/x/comments/abc/title/"}])
        )
        result = self.provider.discover_targets(prompt="test")
        self.assertEqual(len(result), 2)
        self.assertIn("123", result)
        self.assertTrue(result[1].startswith("https://"))

    # ------------------------------------------------------------------
    # datetime parsing
    # ------------------------------------------------------------------

    def test_parse_iso_utc_handles_z_suffix(self):
        """``created_utc`` values with a trailing ``Z`` are parsed as UTC."""
        from app.services.ingestion.providers.hermes import _parse_iso_utc

        dt = _parse_iso_utc("2024-01-15T10:30:00Z")
        self.assertEqual(dt.year, 2024)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 15)
        self.assertEqual(dt.tzinfo, datetime.timezone.utc)

    def test_parse_iso_utc_handles_missing(self):
        """Missing / None / empty ``created_utc`` falls back to now."""
        from app.services.ingestion.providers.hermes import _parse_iso_utc

        dt = _parse_iso_utc(None)
        self.assertIsNotNone(dt.tzinfo)

    def test_parse_iso_utc_handles_already_datetime(self):
        """If Hermes returns a ``datetime`` object we pass it through unchanged."""
        from app.services.ingestion.providers.hermes import _parse_iso_utc

        original = datetime.datetime(2024, 1, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)
        result = _parse_iso_utc(original)
        self.assertEqual(result, original)

    # ------------------------------------------------------------------
    # name property
    # ------------------------------------------------------------------

    def test_name(self):
        self.assertEqual(self.provider.name, "HermesProvider")


if __name__ == "__main__":
    unittest.main()
