"""Shared ingestion contracts, exercised through real adapters and the manager.

All HTTP boundaries are mocked; an additional requests guard prevents accidental
live access. Fixtures are synthetic and never touch persistence or source data.
"""

from copy import deepcopy
from datetime import datetime
import unittest
from unittest.mock import MagicMock, patch

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.manager import IngestionManager
from app.services.ingestion.providers.base import (
    ProviderOperationError, ProviderRequestError, safe_failure_metadata,
)


POST_URL = "https://www.reddit.com/r/IIMSambalpur/comments/abc123/thread/"
FEED_URL = "https://feed.invalid/recent"
SOURCE_TIME = datetime(2023, 11, 14, 22, 13, 20)
POST = {
    "id": "abc123", "title": "Source title", "selftext": "Source body",
    "subreddit": "IIMSambalpur", "created_utc": 1700000000,
    "score": 0, "num_comments": 0, "url": POST_URL,
}


def listing(children):
    return {"kind": "Listing", "data": {"children": children}}


def comment(identity, parent):
    return {"kind": "t1", "data": {
        "id": identity, "body": "Comment text", "created_utc": 1700000001,
        "score": 1, "parent_id": parent, "link_id": "t3_abc123", "replies": "",
    }}


def scrape_payload(comments=()):
    return {"results": [{"status_code": 200, "content": [
        listing([{"kind": "t3", "data": deepcopy(POST)}]), listing(list(comments)),
    ]}]}


def feed():
    return ('<rss version="2.0"><channel><item><title>Source title</title>'
            f'<link>{POST_URL}</link><guid>t3_abc123</guid>'
            '<pubDate>Tue, 14 Nov 2023 22:13:20 GMT</pubDate>'
            '<description>Source body</description></item></channel></rss>')


def response(*, payload=None, content="", status=200):
    result = MagicMock(status_code=status)
    result.json.return_value = payload
    result.content = content.encode()
    return result


class TestSharedIngestionContract(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {"DECODO_API_TOKEN": "offline-test-key",
                                              "DECODO_API_URL": "https://data.decodo.com/v1/scrape"})
        environment.start()
        self.addCleanup(environment.stop)
        guard = patch("requests.sessions.Session.request",
                      side_effect=AssertionError("unexpected live request"))
        guard.start()
        self.addCleanup(guard.stop)
        self.manager = IngestionManager()

    def dispatch(self, provider, reply, *, operation=None, **kwargs):
        if provider == "arctic_shift":
            boundary = patch.object(self.manager.get_provider(provider).client.session,
                                    "get", return_value=reply)
            operation = operation or "search_posts"
            if operation == "search_posts":
                kwargs.setdefault("subreddit", "IIMSambalpur")
        elif provider == "rss":
            boundary = patch("app.services.ingestion.providers.rss.requests.get", return_value=reply)
            operation = operation or "search_posts"
            kwargs.setdefault("url", FEED_URL)
        else:
            boundary = patch("app.services.ingestion.providers.decodo.requests.post", return_value=reply)
            operation = operation or "scrape_url"
            kwargs.setdefault("url", POST_URL)
        with boundary as request:
            try:
                return self.manager.execute_operation(operation, provider=provider, **kwargs)
            finally:
                request.assert_called_once()
                self.assertFalse(request.call_args.kwargs["allow_redirects"])
                self.assertGreater(request.call_args.kwargs["timeout"], 0)

    def test_adapters_share_post_identity_text_timestamp_and_provenance_contract(self):
        replies = {
            "arctic_shift": response(payload={"data": [deepcopy(POST)]}),
            "rss": response(content=feed()),
            "decodo": response(payload=scrape_payload()),
        }
        for provider, reply in replies.items():
            with self.subTest(provider=provider):
                result = self.dispatch(provider, reply)
                post = result if isinstance(result, PostData) else result[0]
                self.assertIsInstance(post, PostData)
                self.assertEqual(post.reddit_id, "abc123")
                self.assertEqual(post.title, "Source title")
                self.assertEqual(post.body, "Source body")
                self.assertEqual(post.subreddit, "IIMSambalpur")
                self.assertEqual(post.created_at, SOURCE_TIME)
                self.assertIsNone(post.created_at.tzinfo)
                self.assertEqual(post.url, POST_URL)
                self.assertEqual(post.comments, [])
                self.assertNotIn("source_mode", post.model_dump())

    def test_comment_adapters_share_top_level_and_reply_parent_convention(self):
        children = [comment("c1", "t3_abc123"), comment("c2", "t1_c1")]
        arctic = self.dispatch("arctic_shift", response(payload={"data": deepcopy(children)}),
                               operation="get_comments", post_id="abc123", limit=2)
        decodo = self.dispatch("decodo", response(payload=scrape_payload(children))).comments
        for comments in (arctic, decodo):
            self.assertTrue(all(isinstance(row, CommentData) for row in comments))
            self.assertEqual([row.reddit_id for row in comments], ["c1", "c2"])
            self.assertEqual([row.parent_reddit_id for row in comments], [None, "c1"])
            self.assertTrue(all(row.created_at.tzinfo is None for row in comments))

    def test_supported_empty_discovery_is_distinct_from_invalid_payload(self):
        for provider, reply in (
            ("arctic_shift", response(payload={"data": []})),
            ("rss", response(content='<rss version="2.0"><channel/></rss>')),
        ):
            with self.subTest(provider=provider):
                self.assertEqual(self.dispatch(provider, reply), [])

    def test_payload_failure_categories_survive_real_adapter_manager_boundary(self):
        for provider, reply, category in (
            ("arctic_shift", response(payload={}), "invalid_payload"),
            ("rss", response(content="<rss>"), "invalid_xml"),
            ("rss", response(content="<html/>"), "invalid_payload"),
            ("decodo", response(payload={"results": []}), "invalid_payload"),
        ):
            with self.subTest(provider=provider, category=category):
                with self.assertRaises(ProviderOperationError) as caught:
                    self.dispatch(provider, reply)
                self.assertEqual(caught.exception.failure_category, category)
                self.assertIsNone(caught.exception.status_code)
                self.assertIsNone(caught.exception.__cause__)
                self.assertTrue(caught.exception.__suppress_context__)

    def test_http_statuses_survive_without_retry_redirect_or_body_parsing(self):
        for provider in ("arctic_shift", "rss", "decodo"):
            for status in (301, 422, 429, 503):
                with self.subTest(provider=provider, status=status):
                    reply = response(payload={"secret": "private-response"},
                                     content="private-response", status=status)
                    with self.assertRaises(ProviderOperationError) as caught:
                        self.dispatch(provider, reply)
                    self.assertEqual(caught.exception.failure_category, "http_status")
                    self.assertEqual(caught.exception.status_code, status)
                    self.assertNotIn("private-response", str(caught.exception))
                    reply.json.assert_not_called()

    def test_registry_order_capabilities_and_disabled_hermes_remain_unchanged(self):
        self.assertEqual(list(self.manager.providers), ["arctic_shift", "rss", "hermes", "decodo"])
        expected = {
            "arctic_shift": {"search_posts", "get_post", "get_comments"},
            "rss": {"search_posts"}, "hermes": set(), "decodo": {"scrape_url", "scrape_source"},
        }
        operations = ("search_posts", "get_post", "get_comments", "scrape_url", "scrape_source", "discover_targets")
        for name, supported in expected.items():
            self.assertEqual({op for op in operations if self.manager.get_provider(name).supports(op)},
                             supported)


class TestSafeFailureContract(unittest.TestCase):
    def test_shape_categories_are_preserved_without_payload_details(self):
        for category in ("invalid_json", "invalid_xml", "invalid_payload"):
            with self.subTest(category=category):
                error = ProviderRequestError(category)
                self.assertEqual(safe_failure_metadata(error), (category, None))
                self.assertEqual(str(error), f"provider request failed ({category})")

    def test_untrusted_category_and_invalid_status_are_not_echoed(self):
        for status in (True, False, "503", 99, 600, None):
            with self.subTest(status=status):
                error = ProviderRequestError("Bearer private-input", status)
                self.assertEqual(str(error), "provider request failed (provider_error)")
                self.assertEqual(safe_failure_metadata(error), ("provider_request_error", None))

    def test_hostile_metadata_properties_and_error_text_are_not_required(self):
        class HostileError(Exception):
            @property
            def failure_category(self):
                raise ValueError("private-input")

            @property
            def status_code(self):
                raise ValueError("private-input")

            @property
            def response(self):
                raise ValueError("private-input")

            def __str__(self):
                raise AssertionError("error text must not be inspected")

        self.assertEqual(safe_failure_metadata(HostileError()), ("hostile_error", None))


if __name__ == "__main__":
    unittest.main()
