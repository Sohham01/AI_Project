"""Offline Arctic Shift contract regressions; every HTTP operation is mocked."""

from copy import deepcopy
from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock, patch

import requests

from app.services.ingestion.arctic_shift import ArcticShiftClient
from app.services.ingestion.arctic_shift_normalizer import ArcticShiftNormalizer
from app.services.ingestion.providers.arctic_shift import ArcticShiftProvider
from app.services.ingestion.providers.base import ProviderRequestError


def post(identity="p1", **overrides):
    row = {
        "id": identity, "title": "A title", "selftext": "A body",
        "subreddit": "CATpreparation", "created_utc": 1700000000,
        "score": 3, "num_comments": 2,
    }
    row.update(overrides)
    return row


def comment(identity="c1", parent="t3_p1", **overrides):
    row = {
        "id": identity, "parent_id": parent, "body": "A comment",
        "created_utc": 1700000001, "score": 1,
    }
    row.update(overrides)
    return {"kind": "t1", "data": row}


def with_replies(node, *children):
    node["data"]["replies"] = {"kind": "Listing", "data": {"children": list(children)}}
    return node


class OfflineCase(unittest.TestCase):
    def setUp(self):
        self.network_guard = patch(
            "requests.sessions.Session.request", side_effect=AssertionError("unexpected network")
        )
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)

    def assertInvalidPayload(self, call):
        with self.assertRaises(ProviderRequestError) as caught:
            call()
        self.assertEqual(caught.exception.failure_category, "invalid_payload")
        self.assertNotIn("secret-payload", str(caught.exception))


class TestArcticHttpContract(OfflineCase):
    def setUp(self):
        super().setUp()
        self.client = ArcticShiftClient()

    def response(self, payload, status=200):
        response = MagicMock(status_code=status)
        response.json.return_value = payload
        response.url = "https://secret-payload.invalid/?key=secret-payload"
        response.text = "secret-payload"
        return response

    def test_valid_and_empty_data_are_preserved(self):
        for payload in ({"data": []}, {"data": [post()]}):
            with self.subTest(payload=payload), patch.object(
                self.client.session, "get", return_value=self.response(payload)
            ) as get:
                self.assertEqual(self.client._get("/api/posts/search"), payload)
                get.assert_called_once()

    def test_every_snapshot_get_disables_redirects(self):
        for operation, args in (
            (self.client.search_posts, ()),
            (self.client.get_post, ("p1",)),
            (self.client.get_comments_tree, ("p1",)),
        ):
            with self.subTest(operation=operation.__name__), patch.object(
                self.client.session, "get", return_value=self.response({"data": []})
            ) as get:
                operation(*args)
                get.assert_called_once()
                self.assertIs(get.call_args.kwargs.get("allow_redirects"), False)

    def test_redirects_fail_sanitized_without_parsing_or_follow_up(self):
        for status in range(300, 400):
            response = self.response({"data": []}, status)
            response.headers = {"Location": "https://secret-payload.invalid/?key=secret-payload"}
            with self.subTest(status=status), patch.object(
                self.client.session, "get", return_value=response
            ) as get:
                with self.assertRaises(ProviderRequestError) as caught:
                    self.client._get("/api/posts/search")
                self.assertEqual(caught.exception.failure_category, "http_status")
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(
                    str(caught.exception), f"provider request failed (http_status) HTTP {status}"
                )
                get.assert_called_once_with(
                    f"{self.client.BASE_URL}/api/posts/search",
                    params=None, timeout=self.client.timeout, allow_redirects=False,
                )
                response.json.assert_not_called()

    def test_json_object_and_list_of_objects_are_required(self):
        for payload in (None, [], "secret-payload", {}, {"data": None},
                        {"data": {}}, {"data": "secret-payload"}, {"data": [None]}):
            with self.subTest(payload=payload), patch.object(
                self.client.session, "get", return_value=self.response(payload)
            ):
                self.assertInvalidPayload(lambda: self.client._get("/api/posts/search"))

    def test_invalid_json_is_sanitized(self):
        response = self.response(None)
        response.json.side_effect = ValueError("secret-payload")
        with patch.object(self.client.session, "get", return_value=response):
            with self.assertRaises(ProviderRequestError) as caught:
                self.client._get("/api/posts/search")
        self.assertEqual(caught.exception.failure_category, "invalid_json")
        self.assertNotIn("secret-payload", str(caught.exception))

    def test_http_errors_include_status_without_payload_or_retries(self):
        for status in (403, 422, 429, 500):
            response = self.response({"data": []}, status)
            with self.subTest(status=status), patch.object(
                self.client.session, "get", return_value=response
            ) as get:
                with self.assertRaises(ProviderRequestError) as caught:
                    self.client._get("/api/posts/search")
                self.assertEqual(caught.exception.failure_category, "http_status")
                self.assertEqual(caught.exception.status_code, status)
                self.assertNotIn("secret-payload", str(caught.exception))
                get.assert_called_once()
                response.json.assert_not_called()

    def test_transport_failures_are_sanitized(self):
        for error, category in (
            (requests.Timeout("secret-payload"), "timeout"),
            (requests.ConnectionError("secret-payload"), "connection"),
            (requests.RequestException("secret-payload"), "transport"),
        ):
            with self.subTest(category=category), patch.object(
                self.client.session, "get", side_effect=error
            ):
                with self.assertRaises(ProviderRequestError) as caught:
                    self.client._get("/api/posts/search")
                self.assertEqual(caught.exception.failure_category, category)
                self.assertNotIn("secret-payload", str(caught.exception))

    def test_timeout_override_is_forwarded(self):
        with patch.object(self.client.session, "get", return_value=self.response({"data": []})) as get:
            self.client.search_posts(timeout=2.5)
        self.assertEqual(get.call_args.kwargs["timeout"], 2.5)

    def test_invalid_timeouts_fail_before_network(self):
        for timeout in (True, False, 0, -1, "10", float("nan"), float("inf"), 10 ** 1000):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    ArcticShiftClient(timeout=timeout)
                with self.assertRaises(ValueError):
                    self.client.search_posts(timeout=timeout)
                with self.assertRaises(ValueError):
                    self.client.get_comments_tree("p1", timeout=timeout)


class TestArcticParameters(OfflineCase):
    def setUp(self):
        super().setUp()
        self.client = ArcticShiftClient()

    def search_params(self, **kwargs):
        with patch.object(self.client, "_get", return_value={"data": []}) as get:
            self.client.search_posts(**kwargs)
        self.assertEqual(get.call_args.args[0], "/api/posts/search")
        get.assert_called_once()
        return get.call_args.kwargs["params"]

    def test_documented_query_and_fields_are_forwarded(self):
        params = self.search_params(subreddit="r/CATpreparation", query="fees", limit=7)
        self.assertEqual(params["query"], "fees")
        self.assertEqual(params["subreddit"], "CATpreparation")
        self.assertEqual(params["limit"], 7)
        self.assertEqual(params["after"], "30d")
        self.assertEqual(set(params["fields"].split(",")), {
            "id", "title", "selftext", "subreddit", "created_utc", "score", "num_comments", "url",
        })
        self.assertNotIn("title", params)
        self.assertNotIn("selftext", params)

    def test_unscoped_keyword_query_fails_before_request(self):
        with self.assertRaisesRegex(ValueError, "requires a subreddit"):
            self.client.search_posts(query="fees")

    def test_optional_bounds_and_empty_query_are_omitted(self):
        params = self.search_params(query="  ", after=None, before=None)
        for key in ("query", "after", "before", "subreddit"):
            self.assertNotIn(key, params)

    def test_post_limits_are_integer_and_bounded(self):
        for limit in (True, False, 0, -1, 101, 1.0, "auto", None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.client.search_posts(limit=limit)
        self.assertEqual(self.search_params(limit=100)["limit"], 100)

    def test_comment_limits_match_documented_hard_maximum(self):
        for limit in (True, False, 0, -1, 25001, 1.0, "auto", None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.client.get_comments_tree("p1", limit=limit)
        with patch.object(self.client, "_get", return_value={"data": []}) as get:
            self.client.get_comments_tree("t3_p1", limit=25000)
        self.assertEqual(get.call_args.args[0], "/api/comments/tree")
        self.assertEqual(get.call_args.kwargs["params"], {"link_id": "t3_p1", "limit": 25000})

    def test_ids_are_not_coerced_or_prefix_confused(self):
        for identity in (None, True, 123, "", "t1_p1", "t3_", "p 1", "t3_t3_p1"):
            with self.subTest(identity=identity):
                with self.assertRaises(ValueError):
                    self.client.get_post(identity)
                with self.assertRaises(ValueError):
                    self.client.get_comments_tree(identity)
        with patch.object(self.client, "_get", return_value={"data": []}) as get:
            self.client.get_post("t3_p1")
        self.assertEqual(get.call_args.args[0], "/api/posts/ids")
        self.assertEqual(get.call_args.kwargs["params"], {"ids": "p1"})

    def test_malformed_search_fields_fail_before_request(self):
        for kwargs in ({"sort": True}, {"sort": "score"}, {"query": {}},
                       {"subreddit": True}, {"subreddit": ""}, {"subreddit": "a/b"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.client.search_posts(**kwargs)

    def test_absolute_bounds_normalize_timezone(self):
        params = self.search_params(
            after="2024-01-01T03:00:00+05:30", before="2024-01-01T00:00:00Z"
        )
        self.assertEqual(params["after"], "2023-12-31T21:30:00Z")
        self.assertEqual(params["before"], "2024-01-01T00:00:00Z")

    def test_epoch_seconds_milliseconds_and_zero(self):
        seconds = self.search_params(after=1700000000, before=None)["after"]
        millis = self.search_params(after=1700000000000, before=None)["after"]
        self.assertEqual(seconds, millis)
        self.assertEqual(self.search_params(after=0)["after"], "1970-01-01T00:00:00Z")

    def test_partial_iso_and_naive_dates_use_utc(self):
        self.assertEqual(self.search_params(after="2024")["after"], "2024-01-01T00:00:00Z")
        self.assertEqual(self.search_params(after="2024-02")["after"], "2024-02-01T00:00:00Z")
        self.assertEqual(self.search_params(after="2024-02-03")["after"], "2024-02-03T00:00:00Z")

    def test_reversed_equal_and_mixed_date_bounds_fail(self):
        for after, before in (
            ("2024-02-01", "2024-01-01"),
            ("2024-01-01T01:00:00+01:00", "2024-01-01T00:00:00Z"),
            ("1d", "30d"), ("3000-01-01", "1d"),
        ):
            with self.subTest(after=after, before=before), self.assertRaises(ValueError):
                self.client.search_posts(after=after, before=before)

    def test_relative_dates_are_preserved_and_calendar_aware(self):
        self.assertEqual(self.search_params(after="10d", before="1d")["before"], "1d")
        now = datetime(2024, 3, 31, tzinfo=timezone.utc)
        self.assertEqual(ArcticShiftClient._date("1m", now)[1], datetime(2024, 2, 29, tzinfo=timezone.utc))
        for value in ("1year", "3m", "2d", "1hour", "5min", "10s"):
            with self.subTest(value=value):
                self.assertEqual(self.search_params(after=value)["after"], value)

    def test_invalid_date_bounds_fail_explicitly(self):
        for value in (True, False, "", "secret-payload", "2024-02-30", "1fortnight", {},
                      float("nan"), float("inf"), 10 ** 100):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                self.client.search_posts(after=value)
            self.assertNotIn("secret-payload", str(caught.exception))


class TestArcticNormalizedRecords(OfflineCase):
    def setUp(self):
        super().setUp()
        self.normalizer = ArcticShiftNormalizer()

    def test_valid_post_has_no_fabricated_url_and_naive_utc(self):
        normalized = self.normalizer.normalize_post(post(id="t3_p1"))
        self.assertEqual(normalized.reddit_id, "p1")
        self.assertEqual(normalized.created_at, datetime(2023, 11, 14, 22, 13, 20))
        self.assertIsNone(normalized.created_at.tzinfo)
        self.assertIsNone(normalized.url)

    def test_timestamp_timezones_and_naive_utc_are_preserved(self):
        for value in ("2024-01-01T03:30:00+05:30", datetime(2023, 12, 31, 22)):
            with self.subTest(value=value):
                normalized = self.normalizer.normalize_post(post(created_utc=value))
                self.assertEqual(normalized.created_at, datetime(2023, 12, 31, 22))
                self.assertIsNone(normalized.created_at.tzinfo)

    def test_required_post_metadata_is_never_synthesized(self):
        for field in ("id", "title", "subreddit", "created_utc"):
            raw = post()
            del raw[field]
            with self.subTest(field=field):
                self.assertInvalidPayload(lambda: self.normalizer.normalize_post(raw))

    def test_malformed_post_fields_fail_without_coercion(self):
        for overrides in (
            {"id": None}, {"id": "t1_p1"}, {"title": {}}, {"title": ""},
            {"subreddit": None}, {"subreddit": ""}, {"subreddit": "r/example"},
            {"created_utc": True}, {"created_utc": float("nan")}, {"created_utc": float("inf")},
            {"created_utc": "secret-payload"}, {"created_utc": 10 ** 100},
            {"score": True}, {"score": "1"}, {"num_comments": -1}, {"num_comments": False},
            {"selftext": {}}, {"url": {}}, {"url": "secret-payload"},
        ):
            with self.subTest(overrides=overrides):
                self.assertInvalidPayload(lambda: self.normalizer.normalize_post(post(**overrides)))

    def test_removed_post_body_is_none(self):
        for body in (None, "", "[deleted]", "[removed]"):
            with self.subTest(body=body):
                self.assertIsNone(self.normalizer.normalize_post(post(selftext=body)).body)

    def test_valid_comment_preserves_t1_parent_and_t3_root_convention(self):
        root = self.normalizer.normalize_comment(comment()["data"])
        reply = self.normalizer.normalize_comment(comment("c2", "t1_c1")["data"])
        self.assertIsNone(root.parent_reddit_id)
        self.assertEqual(reply.parent_reddit_id, "c1")

    def test_missing_and_malformed_comment_metadata_fail(self):
        for field in ("id", "body", "created_utc", "parent_id"):
            row = comment()["data"]
            del row[field]
            with self.subTest(field=field):
                self.assertInvalidPayload(lambda: self.normalizer.normalize_comment(row))
        for overrides in ({"id": "t3_c1"}, {"body": None}, {"body": {}},
                          {"parent_id": "c2"}, {"parent_id": "t1_c1"},
                          {"created_utc": False}, {"score": True}):
            with self.subTest(overrides=overrides):
                self.assertInvalidPayload(lambda: self.normalizer.normalize_comment(comment(**overrides)["data"]))


class TestArcticProviderSnapshots(OfflineCase):
    def setUp(self):
        super().setUp()
        self.provider = ArcticShiftProvider()
        self.provider.client = MagicMock()
        self.provider.client.search_posts.return_value = {"data": []}
        self.provider.client.get_post.return_value = {"data": []}
        self.provider.client.get_comments_tree.return_value = {"data": []}

    def comments(self, nodes, **kwargs):
        self.provider.client.get_comments_tree.return_value = {"data": nodes}
        return self.provider.get_comments("p1", **kwargs)

    def test_empty_results_are_valid_but_get_post_is_not_found(self):
        self.assertEqual(self.provider.search_posts(), [])
        self.assertEqual(self.provider.get_comments("p1"), [])
        with self.assertRaisesRegex(ValueError, "post not found"):
            self.provider.get_post("p1")

    def test_provider_rejects_malformed_envelopes_for_each_operation(self):
        for payload in (None, [], {}, {"data": None}, {"data": {}}, {"data": [None]}):
            with self.subTest(payload=payload):
                self.provider.client.search_posts.return_value = payload
                self.provider.client.get_post.return_value = payload
                self.provider.client.get_comments_tree.return_value = payload
                self.assertInvalidPayload(lambda: self.provider.search_posts())
                self.assertInvalidPayload(lambda: self.provider.get_post("p1"))
                self.assertInvalidPayload(lambda: self.provider.get_comments("p1"))

    def test_search_is_single_page_deduplicated_and_capped(self):
        self.provider.client.search_posts.return_value = {"data": [post(), post(), post("p2"), post("p3")]}
        with self.assertLogs("app.services.ingestion.providers.arctic_shift", "WARNING"):
            posts = self.provider.search_posts(limit=2)
        self.assertEqual([p.reddit_id for p in posts], ["p1", "p2"])
        self.provider.client.search_posts.assert_called_once()
        self.assertFalse(self.provider.last_search_complete)

    def test_search_conflicting_duplicates_and_wrong_subreddit_fail(self):
        for rows, kwargs in (
            ([post(), post(title="different")], {}),
            ([post(subreddit="other")], {"subreddit": "CATpreparation"}),
        ):
            self.provider.client.search_posts.return_value = {"data": rows}
            with self.subTest(rows=rows):
                self.assertInvalidPayload(lambda: self.provider.search_posts(**kwargs))

    def test_provider_does_not_supply_synthetic_subreddit_scope(self):
        self.provider.search_posts()
        self.assertIsNone(self.provider.client.search_posts.call_args.kwargs["subreddit"])

    def test_provider_rejects_unknown_filters_and_invalid_limits(self):
        with self.assertRaises(ValueError):
            self.provider.search_posts(author="someone")
        with self.assertRaises(ValueError):
            self.provider.search_posts(limit=True)
        with self.assertRaises(ValueError):
            self.provider.get_comments("p1", limit=True)
        self.provider.client.search_posts.assert_not_called()
        self.provider.client.get_comments_tree.assert_not_called()

    def test_get_post_checks_requested_identity_and_exact_data_shape(self):
        self.provider.client.get_post.return_value = {"data": [post()]}
        self.assertEqual(self.provider.get_post("t3_p1").reddit_id, "p1")
        for payload in ({"data": [post("p2")]}, {"data": [post(), post()]}):
            self.provider.client.get_post.return_value = payload
            with self.subTest(payload=payload):
                self.assertInvalidPayload(lambda: self.provider.get_post("p1"))

    def test_nested_tree_and_flat_parent_chain_preserve_parent_ids(self):
        nodes = [with_replies(comment(), comment("c2", "t1_c1")), comment("c3", "t1_c2")]
        comments = self.comments(nodes)
        self.assertEqual([c.reddit_id for c in comments], ["c1", "c2", "c3"])
        self.assertEqual([c.parent_reddit_id for c in comments], [None, "c1", "c2"])

    def test_deleted_parent_does_not_hide_valid_replies(self):
        nodes = [with_replies(comment(body="[deleted]"), comment("c2", "t1_c1"))]
        comments = self.comments(nodes)
        self.assertEqual([c.reddit_id for c in comments], ["c2"])
        self.assertEqual(comments[0].parent_reddit_id, "c1")

    def test_wrong_root_or_link_provenance_fails_at_every_depth(self):
        for nodes in (
            [comment(parent="t3_p2")],
            [comment(link_id="t3_p2")],
            [with_replies(comment(), comment("c2", "t1_c1", link_id="t3_p2"))],
            [with_replies(comment(), comment("c2", "t3_p2"))],
        ):
            with self.subTest(nodes=nodes):
                self.assertInvalidPayload(lambda: self.comments(nodes))

    def test_nested_parent_mismatch_fails_even_with_correct_link_id(self):
        nodes = [with_replies(comment(), comment("c2", "t1_other", link_id="t3_p1"))]
        self.assertInvalidPayload(lambda: self.comments(nodes))

    def test_orphan_parent_requires_explicit_matching_link_id(self):
        self.assertInvalidPayload(lambda: self.comments([comment(parent="t1_missing")]))
        result = self.comments([comment(parent="t1_missing", link_id="t3_p1")])
        self.assertEqual(result[0].parent_reddit_id, "missing")

    def test_flattened_parent_order_does_not_require_parent_first(self):
        result = self.comments([comment("c2", "t1_c1"), comment()])
        self.assertEqual([c.reddit_id for c in result], ["c2", "c1"])

    def test_duplicate_comments_are_not_counted_twice(self):
        result = self.comments([comment(), deepcopy(comment()), comment("c2")], limit=2)
        self.assertEqual([c.reddit_id for c in result], ["c1", "c2"])
        self.assertFalse(self.provider.last_comments_truncated)

    def test_conflicting_duplicate_comments_fail_explicitly(self):
        self.assertInvalidPayload(lambda: self.comments([comment(), comment(body="different")]))

    def test_collapsed_more_is_not_fetched_and_reports_incomplete(self):
        nodes = [comment(), {"kind": "more", "data": {"children": ["c2", "c3"]}}]
        with self.assertLogs("app.services.ingestion.providers.arctic_shift", "WARNING"):
            result = self.comments(nodes)
        self.assertEqual([c.reddit_id for c in result], ["c1"])
        self.assertTrue(self.provider.last_comments_truncated)
        self.assertFalse(self.provider.last_comments_complete)
        self.provider.client.get_comments_tree.assert_called_once()

    def test_output_cap_is_enforced_and_reset_between_calls(self):
        with self.assertLogs("app.services.ingestion.providers.arctic_shift", "WARNING"):
            result = self.comments([comment(), comment("c2"), comment("c3")], limit=2)
        self.assertEqual([c.reddit_id for c in result], ["c1", "c2"])
        self.assertTrue(self.provider.last_comments_truncated)
        self.assertEqual(self.comments([]), [])
        self.assertFalse(self.provider.last_comments_truncated)

    def test_malformed_nodes_and_replies_are_not_silently_ignored(self):
        for nodes in (
            [{}], [None], [{"kind": {}, "data": {}}], [{"kind": "t1", "data": None}],
            [comment(replies={})], [comment(replies={"data": {"children": None}})],
            [comment(replies={"data": {"children": [None]}})],
            [{"kind": "more", "data": {}}],
            [{"kind": "more", "data": {"children": [None]}}],
        ):
            with self.subTest(nodes=nodes):
                self.assertInvalidPayload(lambda: self.comments(nodes))

    def test_malformed_node_beyond_cap_still_fails(self):
        self.assertInvalidPayload(lambda: self.comments([comment(), {}], limit=1))

    def test_cap_applies_across_nested_replies(self):
        nodes = [with_replies(comment(), with_replies(
            comment("c2", "t1_c1"), comment("c3", "t1_c2")
        ))]
        with self.assertLogs("app.services.ingestion.providers.arctic_shift", "WARNING"):
            result = self.comments(nodes, limit=2)
        self.assertEqual([c.reddit_id for c in result], ["c1", "c2"])
        self.assertTrue(self.provider.last_comments_truncated)

    def test_parent_cycles_and_object_cycles_fail_explicitly(self):
        self.assertInvalidPayload(lambda: self.comments([
            comment("c1", "t1_c2", link_id="t3_p1"),
            comment("c2", "t1_c1", link_id="t3_p1"),
        ]))
        node = comment()
        with_replies(node, node)
        self.assertInvalidPayload(lambda: self.comments([node]))


if __name__ == "__main__":
    unittest.main()
