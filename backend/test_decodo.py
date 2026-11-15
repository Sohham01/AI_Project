from copy import deepcopy
import json
import unittest
from unittest.mock import MagicMock, patch

import requests

from app.services.ingestion.providers.base import ProviderRequestError, UnsupportedOperation
from app.services.ingestion.providers.decodo import DecodoProvider


POST_URL = "https://www.reddit.com/r/IIMSambalpur/comments/abc123/sample_post/"


def _listing_child(kind, data):
    return {"kind": kind, "data": data}


def _fixture_payload(comment_count=3):
    post = {
        "id": "abc123",
        "name": "t3_abc123",
        "title": "IIM Sambalpur discussion",
        "selftext": "Post body",
        "subreddit": "IIMSambalpur",
        "created_utc": 1710000000,
        "score": 8,
        "num_comments": comment_count,
    }
    comments = [
        {
            "id": "cmt001",
            "name": "t1_cmt001",
            "body": "Top-level comment",
            "created_utc": 1710000010,
            "score": 2,
            "parent_id": "t3_abc123",
            "link_id": "t3_abc123",
            "subreddit": "IIMSambalpur",
            "replies": "",
        },
        {
            "id": "cmt002",
            "name": "t1_cmt002",
            "body": "Reply",
            "created_utc": 1710000020,
            "score": 1,
            "parent_id": "t1_cmt001",
            "link_id": "t3_abc123",
            "subreddit": "IIMSambalpur",
            "replies": "",
        },
        {
            "id": "cmt003",
            "body": "Another top-level comment",
            "created_utc": 1710000030,
            "score": 0,
            "parent_id": "t3_abc123",
            "link_id": "t3_abc123",
            "subreddit": "IIMSambalpur",
            "replies": "",
        },
    ][:comment_count]
    return [
        {"kind": "Listing", "data": {"children": [_listing_child("t3", post)]}},
        {"kind": "Listing", "data": {"children": [_listing_child("t1", c) for c in comments]}},
    ]


def _response(payload, *, ok=True, status_code=200):
    response = MagicMock()
    response.ok = ok
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _envelope(content=None, **metadata):
    return {
        "results": [
            {"content": _fixture_payload() if content is None else content, "status_code": 200, **metadata}
        ]
    }


def _nested_payload():
    payload = _fixture_payload()
    top, reply, deep = payload[1]["data"]["children"]
    deep["data"]["parent_id"] = "t1_cmt002"
    reply["data"]["replies"] = {"kind": "Listing", "data": {"children": [deep]}}
    top["data"]["replies"] = {"kind": "Listing", "data": {"children": [reply]}}
    payload[1]["data"]["children"] = [top]
    return payload


class TestDecodoProvider(unittest.TestCase):
    def setUp(self):
        # Synthetic configuration only: never inspect local credentials.
        config = {"DECODO_API_TOKEN": "test-token"}
        self.getenv = patch(
            "app.services.ingestion.providers.decodo.os.getenv",
            side_effect=lambda name, default=None: config.get(name, default),
        )
        self.getenv.start()
        self.addCleanup(self.getenv.stop)
        self.provider = DecodoProvider()
        self.provider.api_token = "test-token"

    def test_realtime_envelope_accepts_listing_content_and_json_content(self):
        for content in (_fixture_payload(), json.dumps(_fixture_payload())):
            with self.subTest(encoded=isinstance(content, str)):
                with patch("requests.post", return_value=_response(_envelope(content))):
                    post = self.provider.scrape_url(POST_URL)
                self.assertEqual(post.reddit_id, "abc123")
                self.assertEqual(len(post.comments), 3)

    def test_posts_universal_configuration_and_normalizes_listing_with_thread(self):
        with patch("requests.post", return_value=_response(_fixture_payload())) as request:
            post = self.provider.scrape_url(POST_URL)

        self.assertEqual(request.call_args.args[0], "https://data.decodo.com/v1/scrape")
        self.assertEqual(
            request.call_args.kwargs["json"],
            {"url": POST_URL, "proxy_pool": "premium", "headless": "html"},
        )
        self.assertEqual(request.call_args.kwargs["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(request.call_args.kwargs["timeout"], 10)
        self.assertFalse(request.call_args.kwargs["allow_redirects"])
        request.assert_called_once()
        self.assertNotIn("test-token", json.dumps(request.call_args.kwargs["json"]))
        self.assertEqual(post.reddit_id, "abc123")
        self.assertEqual(post.subreddit, "IIMSambalpur")
        self.assertEqual(post.created_at.isoformat(), "2024-03-09T16:00:00")
        self.assertEqual(post.url, POST_URL)
        self.assertEqual([comment.reddit_id for comment in post.comments], ["cmt001", "cmt002", "cmt003"])
        self.assertIsNone(post.comments[0].parent_reddit_id)
        self.assertEqual(post.comments[1].parent_reddit_id, "cmt001")

    def test_enforces_25_comment_output_cap(self):
        payload = _fixture_payload()
        payload[1]["data"]["children"] *= 10
        with patch("requests.post", return_value=_response(payload)):
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                post = self.provider.scrape_url(POST_URL, comment_limit=2)
        self.assertEqual(len(post.comments), 2)

    def test_rejects_limit_above_project_cap_before_request(self):
        with patch("requests.post") as request:
            with self.assertRaises(ValueError):
                self.provider.scrape_url(POST_URL, comment_limit=26)
        request.assert_not_called()

    def test_rejects_unapproved_subreddit_before_request(self):
        with patch("requests.post") as request:
            with self.assertRaisesRegex(ValueError, "approved collection scope"):
                self.provider.scrape_url("https://www.reddit.com/r/all/comments/abc123/title/")
        request.assert_not_called()

    def test_rejects_non_reddit_or_out_of_scope_url_before_request(self):
        for url in (
            "http://www.reddit.com/r/IIMSambalpur/comments/abc123/title/",
            "https://example.com/r/IIMSambalpur/comments/abc123/title/",
            "https://www.reddit.com/r/all/",
            "https://www.reddit.com/",
        ):
            with self.subTest(url=url), patch("requests.post") as request:
                with self.assertRaises(ValueError):
                    self.provider.scrape_url(url)
                request.assert_not_called()

    def test_rejects_payload_with_wrong_post_identity(self):
        payload = _fixture_payload()
        payload[0]["data"]["children"][0]["data"]["id"] = "other"
        with patch("requests.post", return_value=_response(payload)):
            with self.assertRaises(ProviderRequestError) as ctx:
                self.provider.scrape_url(POST_URL)
        self.assertEqual(ctx.exception.failure_category, "invalid_payload")

    def test_missing_timestamp_is_rejected_not_replaced_with_now(self):
        payload = _fixture_payload()
        del payload[0]["data"]["children"][0]["data"]["created_utc"]
        with patch("requests.post", return_value=_response(payload)):
            with self.assertRaises(ProviderRequestError) as ctx:
                self.provider.scrape_url(POST_URL)
        self.assertEqual(ctx.exception.failure_category, "invalid_payload")

    def test_unconfigured_provider_makes_no_request(self):
        self.provider.api_token = None
        with patch("requests.post") as request:
            with self.assertRaises(UnsupportedOperation):
                self.provider.scrape_url(POST_URL)
        request.assert_not_called()

    def test_http_error_is_sanitized(self):
        self.provider.api_token = "sensitive-token"
        response = _response({"error": "sensitive-token"}, ok=False, status_code=403)
        response.reason = "sensitive-token"
        with patch("requests.post", return_value=response):
            with self.assertRaises(ProviderRequestError) as ctx:
                self.provider.scrape_url(POST_URL)
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertNotIn("sensitive-token", str(ctx.exception))
        response.json.assert_not_called()

    def test_manager_dispatches_scrape_url_to_decodo(self):
        from app.services.ingestion.manager import IngestionManager

        manager = IngestionManager()
        manager.get_provider("decodo").api_token = "test-token"
        with patch("requests.post", return_value=_response(_fixture_payload())):
            result = manager.execute_operation(
                "scrape_url", provider="decodo", url=POST_URL
            )
        self.assertEqual(result.reddit_id, "abc123")
        self.assertEqual(result.subreddit, "IIMSambalpur")

    def test_api_url_encodes_only_supported_fields(self):
        with patch("requests.post", return_value=_response(_fixture_payload())) as request:
            self.provider.scrape_url(POST_URL, comment_limit=25)
        self.assertEqual(
            set(request.call_args.kwargs["json"]),
            {"url", "proxy_pool", "headless"},
        )

    def test_subreddit_source_url_is_supported_by_universal_workflow(self):
        source_url = "https://www.reddit.com/r/IIMSambalpur/"
        with patch("requests.post", return_value=_response(_fixture_payload())) as request:
            post = self.provider.scrape_url(source_url)
        self.assertEqual(post.reddit_id, "abc123")
        self.assertEqual(request.call_args.kwargs["json"]["url"], source_url)

    def test_nested_replies_preserve_all_parent_relationships(self):
        with patch("requests.post", return_value=_response(_envelope(_nested_payload()))):
            post = self.provider.scrape_url(POST_URL)
        self.assertEqual([comment.reddit_id for comment in post.comments], ["cmt001", "cmt002", "cmt003"])
        self.assertEqual([comment.parent_reddit_id for comment in post.comments], [None, "cmt001", "cmt002"])
        self.assertEqual(self.provider.last_diagnostics, ())

    def test_parsed_result_wrapper_accepts_only_success(self):
        content = {"results": _nested_payload(), "parse_status_code": 12000, "errors": []}
        with patch("requests.post", return_value=_response(_envelope(content, parse_status_code=12000))):
            self.assertEqual(len(self.provider.scrape_url(POST_URL).comments), 3)

    def test_duplicate_comments_do_not_consume_output_cap(self):
        payload = _fixture_payload()
        children = payload[1]["data"]["children"]
        children.insert(1, deepcopy(children[0]))
        with patch("requests.post", return_value=_response(_envelope(payload))):
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                post = self.provider.scrape_url(POST_URL, comment_limit=3)
        self.assertEqual([comment.reddit_id for comment in post.comments], ["cmt001", "cmt002", "cmt003"])
        self.assertIn("duplicate_comments_removed", self.provider.last_diagnostics)

    def test_conflicting_duplicate_comments_are_rejected(self):
        payload = _fixture_payload()
        children = payload[1]["data"]["children"]
        duplicate = deepcopy(children[0])
        duplicate["data"]["body"] = "sensitive-token"
        children.append(duplicate)
        self.assert_invalid_payload(_envelope(payload))

    def test_comment_output_is_bounded_to_25_unique_rows(self):
        payload = _fixture_payload()
        prototype = payload[1]["data"]["children"][0]
        children = []
        for index in range(40):
            row = deepcopy(prototype)
            row["data"]["id"] = f"c{index}"
            row["data"]["name"] = f"t1_c{index}"
            children.append(row)
        payload[1]["data"]["children"] = children
        payload[0]["data"]["children"][0]["data"]["num_comments"] = 40
        with patch("requests.post", return_value=_response(_envelope(payload))):
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                post = self.provider.scrape_url(POST_URL)
        self.assertEqual(len(post.comments), 25)
        self.assertEqual(len({comment.reddit_id for comment in post.comments}), 25)
        self.assertIn("comments_truncated_by_limit", self.provider.last_diagnostics)
        self.assertIn("fewer_comments_than_reported", self.provider.last_diagnostics)

    def test_zero_comment_limit_returns_no_fabricated_comments(self):
        with patch("requests.post", return_value=_response(_envelope())) as request:
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                post = self.provider.scrape_url(POST_URL, comment_limit=0)
        self.assertEqual(post.comments, [])
        self.assertIn("comments_truncated_by_limit", self.provider.last_diagnostics)
        request.assert_called_once()

    def test_more_placeholder_is_diagnosed_never_a_comment_or_extra_request(self):
        payload = _nested_payload()
        payload[1]["data"]["after"] = "sensitive-token"
        payload[1]["data"]["children"].append(
            _listing_child("more", {"parent_id": "t3_abc123", "children": ["notfetched"], "count": 4})
        )
        with patch("requests.post", return_value=_response(_envelope(payload))) as request:
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING") as logs:
                post = self.provider.scrape_url(POST_URL)
        self.assertEqual(len(post.comments), 3)
        self.assertIn("comments_more_available", self.provider.last_diagnostics)
        self.assertIn("listing_cursor_not_followed", self.provider.last_diagnostics)
        self.assertNotIn("sensitive-token", " ".join(logs.output))
        request.assert_called_once()

    def test_deleted_and_removed_bodies_are_preserved_with_safe_diagnostics(self):
        payload = _fixture_payload()
        children = payload[1]["data"]["children"]
        children[0]["data"]["body"] = "[deleted]"
        children[1]["data"]["body"] = "[removed]"
        children[1]["data"]["removal_reason"] = "sensitive-token"
        with patch("requests.post", return_value=_response(_envelope(payload))):
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING") as logs:
                post = self.provider.scrape_url(POST_URL)
        self.assertEqual([comment.body for comment in post.comments[:2]], ["[deleted]", "[removed]"])
        self.assertIn("comment_content_unavailable", self.provider.last_diagnostics)
        self.assertNotIn("sensitive-token", " ".join(logs.output))
        self.assertEqual(set(post.comments[0].model_dump()), {"reddit_id", "body", "created_at", "score", "parent_reddit_id"})

    def test_missing_comment_listing_is_diagnosed_without_synthetic_rows(self):
        payload = _fixture_payload()[:1]
        with patch("requests.post", return_value=_response(_envelope(payload))):
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                post = self.provider.scrape_url(POST_URL)
        self.assertEqual(post.comments, [])
        self.assertIn("comments_listing_missing", self.provider.last_diagnostics)
        self.assertIn("fewer_comments_than_reported", self.provider.last_diagnostics)

    def test_missing_link_id_and_unreturned_parent_are_diagnosed_not_invented(self):
        payload = _fixture_payload(1)
        comment = payload[1]["data"]["children"][0]["data"]
        del comment["link_id"]
        del comment["replies"]
        comment["parent_id"] = "t1_missing"
        with patch("requests.post", return_value=_response(_envelope(payload))):
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                post = self.provider.scrape_url(POST_URL)
        self.assertEqual(len(post.comments), 1)
        self.assertEqual(post.comments[0].parent_reddit_id, "missing")
        self.assertIn("comment_link_id_missing", self.provider.last_diagnostics)
        self.assertIn("comment_replies_missing", self.provider.last_diagnostics)
        self.assertIn("comment_parent_not_returned", self.provider.last_diagnostics)

    def test_empty_comment_listing_is_valid_for_reported_zero_comments(self):
        payload = _fixture_payload(0)
        with patch("requests.post", return_value=_response(_envelope(payload))):
            post = self.provider.scrape_url(POST_URL)
        self.assertEqual(post.comments, [])
        self.assertEqual(post.num_comments, 0)
        self.assertEqual(self.provider.last_diagnostics, ())

    def test_missing_comment_metadata_fails_instead_of_fabricating(self):
        for field in ("id", "created_utc", "body", "parent_id", "score"):
            with self.subTest(field=field):
                payload = _fixture_payload(1)
                comment = payload[1]["data"]["children"][0]["data"]
                del comment[field]
                if field == "id":
                    del comment["name"]
                self.assert_invalid_payload(_envelope(payload))

    def test_comment_identity_and_scope_mismatches_are_rejected(self):
        mutations = (
            ("id", "t3_wrong"), ("name", "t1_other"), ("id", "bad_id"),
            ("link_id", "t3_other"), ("link_id", "t1_abc123"),
            ("subreddit", "other"), ("parent_id", "t3_other"),
            ("parent_id", "t1_cmt001"), ("parent_id", "unknown"),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                payload = _fixture_payload(1)
                payload[1]["data"]["children"][0]["data"][field] = value
                self.assert_invalid_payload(_envelope(payload))

    def test_nested_parent_mismatch_is_rejected(self):
        payload = _nested_payload()
        reply = payload[1]["data"]["children"][0]["data"]["replies"]["data"]["children"][0]["data"]
        reply["parent_id"] = "t3_abc123"
        self.assert_invalid_payload(_envelope(payload))

    def test_flat_comment_parent_cycle_is_rejected(self):
        payload = _fixture_payload()
        payload[1]["data"]["children"][0]["data"]["parent_id"] = "t1_cmt002"
        self.assert_invalid_payload(_envelope(payload))

    def test_bad_comment_shapes_are_rejected(self):
        for malformed in (None, {}, {"kind": "t1", "data": None}, {"kind": "post", "data": {}}):
            with self.subTest(malformed=malformed):
                payload = _fixture_payload()
                payload[1]["data"]["children"].insert(0, malformed)
                self.assert_invalid_payload(_envelope(payload))
        payload = _fixture_payload()
        payload[1]["data"]["children"][0]["data"]["replies"] = []
        self.assert_invalid_payload(_envelope(payload))

    def test_post_identity_metadata_and_types_are_validated(self):
        mutations = (
            ("id", "other"), ("name", "t3_other"), ("id", "bad_id"),
            ("subreddit", "other"), ("subreddit", None),
            ("title", ""), ("title", "  "), ("title", None), ("title", 42),
            ("selftext", {}), ("created_utc", None), ("created_utc", True),
            ("created_utc", float("nan")), ("created_utc", float("inf")),
            ("created_utc", "invalid sensitive-token"), ("created_utc", "2024-01-01"),
            ("score", True), ("score", None), ("num_comments", -1), ("num_comments", "3"),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                payload = _fixture_payload()
                payload[0]["data"]["children"][0]["data"][field] = value
                self.assert_invalid_payload(_envelope(payload))

    def test_no_post_or_multiple_posts_or_empty_payload_is_explicit_failure(self):
        for payload in (None, {}, [], "", [None], [{"kind": "Listing", "data": {"children": []}}]):
            with self.subTest(payload=payload):
                self.assert_invalid_payload(payload)
        payload = _fixture_payload()
        payload[0]["data"]["children"] *= 2
        self.assert_invalid_payload(_envelope(payload))

    def test_bad_envelope_shapes_do_not_silently_return_empty_success(self):
        for payload in (
            {"results": []}, {"results": {}}, {"results": [None]},
            {"results": [{}, {}]}, {"results": [{}]},
            {"results": [{"content": _fixture_payload()}]},
            _envelope("<html>sensitive-token</html>"), _envelope("{}"), _envelope("[]"),
            _envelope({"post": {"id": "abc123"}}),
        ):
            with self.subTest(payload=payload):
                self.assert_invalid_payload(payload)

    def test_api_and_parser_failure_envelopes_are_sanitized(self):
        for metadata in (
            {"parse_status_code": 12002}, {"parse_status_code": 12004},
            {"parse_status_code": 12005}, {"parse_status_code": 12007},
            {"parse_status_code": "sensitive-token"}, {"status_code": "sensitive-token"},
            {"status_code": 613}, {"status_code": True}, {"errors": ["sensitive-token"]},
            {"error": {"authorization": "sensitive-token"}},
        ):
            with self.subTest(metadata=metadata):
                self.assert_invalid_payload(_envelope(**metadata))
        for status in (401, 403, 429, 500, 524):
            with self.subTest(status=status):
                with patch("requests.post", return_value=_response(_envelope(status_code=status))):
                    with self.assertRaises(ProviderRequestError) as ctx:
                        self.provider.scrape_url(POST_URL)
                self.assertEqual(ctx.exception.status_code, status)
                self.assertEqual(ctx.exception.failure_category, "http_status")
        payload = _envelope()
        payload["status_code"] = "sensitive-token"
        self.assert_invalid_payload(payload)
        self.assert_invalid_payload(_envelope({"results": _fixture_payload(), "parse_status_code": 12004}))

    def test_http_failures_redirects_and_rate_limits_are_single_attempt(self):
        for status in (202, 204, 301, 302, 307, 308, 400, 401, 403, 404, 422, 429, 500, 524, 613):
            with self.subTest(status=status):
                response = _response({"error": "sensitive-token"}, ok=status < 400, status_code=status)
                response.headers = {"Location": "https://evil.invalid/?sensitive-token", "Retry-After": "0"}
                with patch("requests.post", return_value=response) as request:
                    with self.assertRaises(ProviderRequestError) as ctx:
                        self.provider.scrape_url(POST_URL)
                self.assertEqual(ctx.exception.failure_category, "http_status")
                self.assertNotIn("sensitive-token", str(ctx.exception))
                request.assert_called_once()
                self.assertFalse(request.call_args.kwargs["allow_redirects"])
                response.json.assert_not_called()

    def test_transport_failures_never_expose_transport_messages_or_retry(self):
        for error, category in (
            (requests.Timeout("sensitive-token https://evil.invalid"), "timeout"),
            (requests.ConnectionError("sensitive-token"), "connection"),
            (requests.RequestException("sensitive-token"), "transport"),
            (requests.exceptions.InvalidHeader("sensitive-token"), "transport"),
        ):
            with self.subTest(category=category):
                with patch("requests.post", side_effect=error) as request:
                    with self.assertRaises(ProviderRequestError) as ctx:
                        self.provider.scrape_url(POST_URL, timeout=2.5)
                self.assertEqual(ctx.exception.failure_category, category)
                self.assertNotIn("sensitive-token", str(ctx.exception))
                self.assertIsNone(ctx.exception.__cause__)
                self.assertTrue(ctx.exception.__suppress_context__)
                request.assert_called_once()

    def test_outer_json_error_is_sanitized(self):
        response = _response(None)
        response.json.side_effect = ValueError("sensitive-token")
        with patch("requests.post", return_value=response):
            with self.assertRaises(ProviderRequestError) as ctx:
                self.provider.scrape_url(POST_URL)
        self.assertEqual(ctx.exception.failure_category, "invalid_json")
        self.assertNotIn("sensitive-token", str(ctx.exception))

    def test_timeout_is_bounded_and_separate_from_json(self):
        with patch("requests.post", return_value=_response(_envelope())) as request:
            self.provider.scrape_url(POST_URL, timeout=2.5)
        self.assertEqual(request.call_args.kwargs["timeout"], 2.5)
        self.assertNotIn("timeout", request.call_args.kwargs["json"])
        for timeout in (None, False, 0, -1, "2", [], float("nan"), float("inf"), 31, 10 ** 1000):
            with self.subTest(timeout=timeout), patch("requests.post") as request:
                with self.assertRaises(ValueError):
                    self.provider.scrape_url(POST_URL, timeout=timeout)
                request.assert_not_called()

    def test_invalid_comment_limits_and_unsupported_options_make_no_request(self):
        for limit in (None, True, -1, 26, 1.5, "2", []):
            with self.subTest(limit=limit), patch("requests.post") as request:
                with self.assertRaises(ValueError):
                    self.provider.scrape_url(POST_URL, comment_limit=limit)
                request.assert_not_called()
        for options in ({"after": "cursor"}, {"limit": 2}, {"parse": False}, {"api_token": "sensitive-token"}, {"sort": "new"}):
            with self.subTest(options=options), patch("requests.post") as request:
                with self.assertRaises(ValueError) as ctx:
                    self.provider.scrape_url(POST_URL, **options)
                self.assertNotIn("sensitive-token", str(ctx.exception))
                request.assert_not_called()

    def test_unsafe_endpoint_is_rejected_without_exposing_value_or_sending_key(self):
        for endpoint in (
            "http://data.decodo.com/v1/scrape", "https://evil.invalid/v1/scrape",
            "https://data.decodo.com.evil.invalid/v1/scrape", "https://data.decodo.com:444/v1/scrape",
            "https://sensitive-token@data.decodo.com/v1/scrape", "https://data.decodo.com/v1/scrape?token=sensitive-token",
            "https://data.decodo.com/v1/scrape#sensitive-token", "https://data.decodo.com/v1/scrape/",
            "https://data.decodo.com:sensitive-token/v1/scrape", "https://[sensitive-token/v1/scrape",
            "https://data.decodo.com/v1/scrape\nsensitive-token", None,
        ):
            with self.subTest(endpoint=endpoint), patch("requests.post") as request:
                self.provider.endpoint = endpoint
                with self.assertRaises(ValueError) as ctx:
                    self.provider.scrape_url(POST_URL)
                self.assertNotIn("sensitive-token", str(ctx.exception))
                request.assert_not_called()

    def test_unsafe_tokens_are_rejected_before_request(self):
        for token in ("Bearer sensitive-token", "Basic sensitive-token", "sensitive-token\r\nX-Evil: yes", " sensitive-token", "sensitive-token\t", "sensitive-token\x00", "sécret", 123, "x" * 4097):
            with self.subTest(token=token), patch("requests.post") as request:
                self.provider.api_token = token
                with self.assertRaises(ValueError) as ctx:
                    self.provider.scrape_url(POST_URL)
                self.assertNotIn("sensitive-token", str(ctx.exception))
                request.assert_not_called()

    def test_unsafe_target_url_values_are_rejected_without_echo(self):
        for url in (
            POST_URL + "?api_key=sensitive-token", POST_URL + "#sensitive-token",
            POST_URL.replace("www.reddit.com", "sensitive-token@www.reddit.com"),
            POST_URL.replace("www.reddit.com", "www.reddit.com:sensitive-token"),
            POST_URL.replace("www.reddit.com", "www.reddit.com:444"),
            POST_URL + "\nsensitive-token", "https://[sensitive-token", None,
        ):
            with self.subTest(url=url), patch("requests.post") as request:
                with self.assertRaises(ValueError) as ctx:
                    self.provider.scrape_url(url)
                self.assertNotIn("sensitive-token", str(ctx.exception))
                request.assert_not_called()

    def test_target_dot_segments_are_rejected_before_request(self):
        for slug in (
            ".", "..", "%2e", "%2E", "%2e%2e", "%2E%2E", ".%2e", "%2e.",
            "../sensitive-target", "sensitive-target/..", "%2e%2e/sensitive-target",
            "..;sensitive-target",
        ):
            with self.subTest(slug=slug):
                self.assert_invalid_target_url(POST_URL.replace("sample_post", slug))

    def test_target_separators_and_ambiguous_escapes_are_rejected_before_request(self):
        for slug in (
            "title%2fcomment", "title%2Fcomment", "title%5ccomment", "title%5Ccomment",
            "title\\comment", "title/comment", "title%3fquery", "title%23fragment",
            "title%3bparams", "title;params", "%252e%252e", "title%252fcomment",
            "title%255ccomment", "title%25252fcomment", "title%25", "title%",
            "title%2", "title%GG", "title%2g", "%u002e%u002e", "/", "title;",
        ):
            with self.subTest(slug=slug):
                self.assert_invalid_target_url(POST_URL.replace("sample_post", slug))

        # urlparse removes even an empty final params delimiter from its path.
        self.assert_invalid_target_url(POST_URL.rstrip("/") + ";")

    def test_target_controls_and_invalid_utf8_are_rejected_before_request(self):
        for suffix in (
            "%00", "%09", "%0a", "%0D", "%1f", "%20", "%7f", "%C2%80",
            "%C2%9F", "\x00", "\x1f", "\x7f", "\x85", "%FF", "%C0%AF",
            "%E0%80%AE", "%ED%A0%80", "%E2%82", "\ud800",
        ):
            with self.subTest(suffix=ascii(suffix)):
                self.assert_invalid_target_url(POST_URL.replace("sample_post", "sensitive-target" + suffix))

    def test_safe_standard_target_forms_preserve_original_url(self):
        for url in (
            POST_URL, POST_URL.rstrip("/"), POST_URL.replace("sample_post/", ""),
            POST_URL.replace("www.reddit.com", "reddit.com"),
            POST_URL.replace("www.reddit.com", "www.reddit.com:443"),
            POST_URL.replace("IIMSambalpur/comments/abc123", "iimsambalpur/comments/ABC123"),
            POST_URL.replace("sample_post", "a.normal-slug_123~"),
            POST_URL.replace("sample_post", "%73ample%5Fpost"),
            POST_URL.replace("sample_post", "caf%C3%A9"),
            POST_URL.replace("sample_post", "%e0%a4%a8%e0%a4%ae%e0%a4%b8%e0%a5%8d%e0%a4%a4%e0%a5%87"),
            POST_URL.replace("sample_post", "%F0%9F%8E%93"),
            POST_URL.replace("sample_post", "caf\u00e9"),
        ):
            with self.subTest(url=url), patch("requests.post", return_value=_response(_fixture_payload())) as request:
                post = self.provider.scrape_url(url)
                request.assert_called_once()
                self.assertEqual(request.call_args.args[0], self.provider.DEFAULT_ENDPOINT)
                self.assertEqual(
                    request.call_args.kwargs["json"],
                    {"url": url, "proxy_pool": "premium", "headless": "html"},
                )
                self.assertEqual(request.call_args.kwargs["headers"]["Authorization"], "Bearer test-token")
                self.assertEqual(request.call_args.kwargs["timeout"], 10)
                self.assertFalse(request.call_args.kwargs["allow_redirects"])
                self.assertEqual(post.url, url)
                self.assertEqual(post.reddit_id, "abc123")
                self.assertEqual(len(post.comments), 3)
                self.assertEqual(self.provider.last_diagnostics, ())

    def test_comment_traversal_has_a_work_bound(self):
        payload = _fixture_payload()
        payload[1]["data"]["children"] *= 10
        with patch.object(DecodoProvider, "MAX_COMMENT_NODES", 4):
            with patch("requests.post", return_value=_response(_envelope(payload))):
                with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                    post = self.provider.scrape_url(POST_URL)
        self.assertEqual(len(post.comments), 3)
        self.assertIn("comment_traversal_limit", self.provider.last_diagnostics)

    def test_comment_depth_has_a_bound_and_reports_truncation(self):
        with patch.object(DecodoProvider, "MAX_COMMENT_DEPTH", 1):
            with patch("requests.post", return_value=_response(_envelope(_nested_payload()))):
                with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                    post = self.provider.scrape_url(POST_URL)
        self.assertEqual([comment.reddit_id for comment in post.comments], ["cmt001", "cmt002"])
        self.assertIn("comment_depth_limit", self.provider.last_diagnostics)

    def test_diagnostics_are_reset_between_calls(self):
        with patch("requests.post", return_value=_response(_envelope())):
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                self.provider.scrape_url(POST_URL, comment_limit=0)
        self.assertTrue(self.provider.last_diagnostics)
        with patch("requests.post", return_value=_response(_envelope(_nested_payload()))):
            self.provider.scrape_url(POST_URL)
        self.assertEqual(self.provider.last_diagnostics, ())

    def test_timestamp_normalization_is_naive_utc(self):
        payload = _fixture_payload(0)
        payload[0]["data"]["children"][0]["data"]["created_utc"] = "2024-03-09T18:00:00+02:00"
        with patch("requests.post", return_value=_response(_envelope(payload))):
            post = self.provider.scrape_url(POST_URL)
        self.assertEqual(post.created_at.isoformat(), "2024-03-09T16:00:00")
        self.assertIsNone(post.created_at.tzinfo)

    def assert_invalid_target_url(self, url):
        self.provider.api_token = "sensitive-token"
        with patch("app.services.ingestion.providers.decodo.logger.warning") as warning:
            with patch("requests.post", return_value=_response(_fixture_payload())) as request:
                with self.assertRaises(ValueError) as ctx:
                    self.provider.scrape_url(url)
                request.assert_not_called()
        warning.assert_not_called()
        self.assertNotIn(url, str(ctx.exception))
        self.assertNotIn("sensitive-target", str(ctx.exception))
        self.assertNotIn("sensitive-token", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)

    def assert_invalid_payload(self, payload):
        # Production logs may contain only static diagnostics, never payloads.
        with patch("app.services.ingestion.providers.decodo.logger.warning") as warning:
            with patch("requests.post", return_value=_response(payload)):
                with self.assertRaises(ProviderRequestError) as ctx:
                    self.provider.scrape_url(POST_URL)
        self.assertEqual(ctx.exception.failure_category, "invalid_payload")
        self.assertNotIn("sensitive-token", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)
        self.assertNotIn("sensitive-token", str(warning.call_args_list))


if __name__ == "__main__":
    unittest.main()
