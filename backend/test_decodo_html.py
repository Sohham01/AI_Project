"""Offline HTML-parser contracts for Decodo Universal snapshots."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.services.ingestion.providers.decodo_html import parse_reddit_html
from app.services.ingestion.providers.decodo import DecodoProvider


HTML_SOURCE_URL = "https://www.reddit.com/r/IIMSambalpur/"


def _html(*, include_optional: bool = True, duplicate: bool = False) -> str:
    optional = (
        ' post-body="Body" post-created="2024-03-09T16:00:00Z" score="8" '
        'comment-count="2" permalink="https://www.reddit.com/r/IIMSambalpur/comments/abc123/thread/"'
        if include_optional
        else ""
    )
    comment = (
        '<shreddit-comment id="t1_cmt001" parent-id="t3_abc123" '
        'created-at="2024-03-09T16:00:10Z" score="2">'
        '<div slot="comment">Top-level comment</div></shreddit-comment>'
        '<shreddit-comment id="t1_cmt002" parent-id="t1_cmt001" '
        'created-at="2024-03-09T16:00:20Z" score="0">'
        '<div slot="comment">Reply</div></shreddit-comment>'
    )
    extra = (
        '<shreddit-post id="t3_abc123" subreddit-prefixed-name="IIMSambalpur" '
        'post-title="IIM Sambalpur discussion"' + optional + '>' + comment + '</shreddit-post>'
        if duplicate
        else ""
    )
    return (
        '<html><body><shreddit-post id="t3_abc123" '
        'subreddit-prefixed-name="IIMSambalpur" post-title="IIM Sambalpur discussion"'
        + optional
        + ">"
        + comment
        + '<div data-testid="load-more-comments" data-more-comments="true"></div>'
        + "</shreddit-post>"
        + extra
        + "</body></html>"
    )


class DecodoHtmlParserTests(unittest.TestCase):
    def test_parses_post_comments_parent_ids_timestamps_scores_and_provenance(self) -> None:
        parsed = parse_reddit_html(_html())
        self.assertEqual(len(parsed), 1)
        post = parsed[0]
        self.assertEqual(post.reddit_id, "abc123")
        self.assertEqual(post.subreddit, "IIMSambalpur")
        self.assertEqual(post.title, "IIM Sambalpur discussion")
        self.assertEqual(post.url, "https://www.reddit.com/r/IIMSambalpur/comments/abc123/thread/")
        self.assertEqual(post.created_at.isoformat(), "2024-03-09T16:00:00")
        self.assertEqual(post.score, 8)
        self.assertEqual([comment.reddit_id for comment in post.comments], ["cmt001", "cmt002"])
        self.assertEqual([comment.parent_reddit_id for comment in post.comments], [None, "cmt001"])
        self.assertEqual(post.comments[1].score, 0)
        self.assertIn("comments_more_available", parsed.diagnostics)

    def test_missing_optional_fields_are_not_fabricated(self) -> None:
        html = _html().replace(
            ' post-body="Body" post-created="2024-03-09T16:00:00Z" score="8" '
            'comment-count="2" '
            'permalink="https://www.reddit.com/r/IIMSambalpur/comments/abc123/thread/"',
            ' post-created="2024-03-09T16:00:00Z"',
        )
        parsed = parse_reddit_html(html)
        post = parsed[0]
        self.assertIsNone(post.body)
        self.assertIsNone(post.url)
        self.assertEqual(post.score, 0)
        self.assertEqual(post.num_comments, 0)
        self.assertIn("post_score_missing", parsed.diagnostics)
        self.assertIn("post_comment_count_missing", parsed.diagnostics)

    def test_missing_parent_is_preserved_as_unresolved_diagnostic(self) -> None:
        html = _html().replace('parent-id="t1_cmt001"', "")
        parsed = parse_reddit_html(html)
        self.assertIsNone(parsed[0].comments[1].parent_reddit_id)
        self.assertIn("comment_parent_missing", parsed.diagnostics)

    def test_duplicate_identical_posts_are_deduplicated(self) -> None:
        parsed = parse_reddit_html(_html(duplicate=True))
        self.assertEqual(len(parsed), 1)
        self.assertIn("duplicate_posts_removed", parsed.diagnostics)

    def test_conflicting_duplicate_posts_fail_closed(self) -> None:
        html = _html(duplicate=True).replace(
            'post-title="IIM Sambalpur discussion"',
            'post-title="Different post"',
            1,
        )
        with self.assertRaises(ValueError):
            parse_reddit_html(html)

    def test_malformed_empty_and_unexpected_html_fail_without_network(self) -> None:
        for html in ("", "<html><body></body></html>", "<html><shreddit-post"):
            with self.subTest(html=html):
                with self.assertRaises(ValueError):
                    parse_reddit_html(html)

    def test_comment_cap_is_explicit_and_partial(self) -> None:
        parsed = parse_reddit_html(_html(), comment_limit=1)
        self.assertEqual(len(parsed[0].comments), 1)
        self.assertIn("comments_truncated_by_limit", parsed.diagnostics)
        self.assertIn("fewer_comments_than_reported", parsed.diagnostics)

    def test_invalid_required_post_metadata_is_rejected(self) -> None:
        html = _html().replace('post-title="IIM Sambalpur discussion"', 'post-title=""')
        with self.assertRaises(ValueError):
            parse_reddit_html(html)

    def test_provider_maps_universal_html_response_to_existing_schema(self) -> None:
        response = MagicMock(status_code=200)
        response.json.side_effect = ValueError("synthetic raw HTML response")
        response.text = _html()
        provider = DecodoProvider()
        provider.api_token = "synthetic-token"
        with patch("requests.post", return_value=response) as request:
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                post = provider.scrape_url(HTML_SOURCE_URL)
        self.assertEqual(post.reddit_id, "abc123")
        self.assertEqual(post.comments[1].parent_reddit_id, "cmt001")
        self.assertEqual(request.call_args.kwargs["json"], {
            "url": HTML_SOURCE_URL,
            "proxy_pool": "premium",
            "headless": "html",
        })
        self.assertIn("comments_more_available", provider.last_diagnostics)

    def test_provider_source_operation_returns_a_bounded_post_list(self) -> None:
        second = (
            '<shreddit-post id="t3_pilot2" subreddit-prefixed-name="IIMSambalpur" '
            'post-title="Second thread" post-created="2024-03-09T17:00:00Z" '
            'permalink="https://www.reddit.com/r/IIMSambalpur/comments/pilot2/thread/">'
            "</shreddit-post>"
        )
        response = MagicMock(status_code=200)
        response.json.side_effect = ValueError("synthetic raw HTML response")
        response.text = _html() + second
        provider = DecodoProvider()
        provider.api_token = "synthetic-token"
        with patch("requests.post", return_value=response):
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                posts = provider.scrape_source(HTML_SOURCE_URL, max_posts=2, comment_limit=1)
        self.assertEqual([post.reddit_id for post in posts], ["abc123", "pilot2"])
        self.assertTrue(provider.last_diagnostics)

    def test_proxy_pool_and_rendering_mode_are_explicit_overrides(self) -> None:
        response = MagicMock(status_code=200)
        response.json.side_effect = ValueError("synthetic raw HTML response")
        response.text = _html()
        provider = DecodoProvider()
        provider.api_token = "synthetic-token"
        with patch("requests.post", return_value=response) as request:
            with self.assertLogs("app.services.ingestion.providers.decodo", level="WARNING"):
                provider.scrape_url(
                    HTML_SOURCE_URL,
                    proxy_pool="standard",
                    headless="html",
                )
        self.assertEqual(request.call_args.kwargs["json"]["proxy_pool"], "standard")
        self.assertEqual(request.call_args.kwargs["json"]["headless"], "html")


if __name__ == "__main__":
    unittest.main()
