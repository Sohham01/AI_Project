from datetime import datetime
import unittest
from unittest.mock import MagicMock, patch

import requests

from app.services.ingestion.providers.base import ProviderRequestError
from app.services.ingestion.providers.rss import RSSProvider


POST_URL = "https://www.reddit.com/r/IIMSambalpur/comments/abc123/sample_post/"
FEED_URL = "https://feed.invalid/recent?token=never-print-this"
RSS_DATE = "Mon, 15 Jan 2024 10:30:00 GMT"
ATOM_DATE = "2024-01-15T10:30:00Z"


def _response(content, *, status_code=200):
    response = MagicMock()
    response.content = content.encode() if isinstance(content, str) else content
    response.status_code = status_code
    response.ok = 200 <= status_code < 400
    response.text = "password=never-print-this"
    return response


def _rss_item(*, post_id="abc123", link=None, title="Test Post", date=RSS_DATE,
              guid=None, body="<description><![CDATA[<p>Test body</p>]]></description>"):
    if link is None:
        link = f"https://www.reddit.com/r/IIMSambalpur/comments/{post_id}/sample_post/"
    title_xml = "" if title is None else f"<title>{title}</title>"
    date_xml = "" if date is None else f"<pubDate>{date}</pubDate>"
    guid_xml = "" if guid is None else f'<guid isPermaLink="false">{guid}</guid>'
    return f"<item>{title_xml}<link>{link}</link>{guid_xml}{date_xml}{body}</item>"


def _rss(*items, extra=""):
    return '<rss version="2.0"><channel>' + "".join(items) + extra + "</channel></rss>"


def _atom_entry(*, post_id="abc123", link=POST_URL, title="Atom Post", date=ATOM_DATE,
                identifier=None, category="IIMSambalpur", body=""):
    identifier = f"t3_{post_id}" if identifier is None else identifier
    date_xml = "" if date is None else f"<updated>{date}</updated>"
    title_xml = "" if title is None else f"<title>{title}</title>"
    category_xml = "" if category is None else f'<category term="{category}" label="r/{category}"/>'
    return (f"<entry><id>{identifier}</id>{title_xml}"
            f'<link href="{link}"/>{category_xml}{date_xml}{body}</entry>')


def _atom(*entries, extra=""):
    return '<feed xmlns="http://www.w3.org/2005/Atom">' + "".join(entries) + extra + "</feed>"


class TestRSSProvider(unittest.TestCase):
    def setUp(self):
        self.provider = RSSProvider()
        self.request_patch = patch("app.services.ingestion.providers.rss.requests.get")
        self.request = self.request_patch.start()
        self.addCleanup(self.request_patch.stop)

    def search(self, content, **kwargs):
        self.request.return_value = _response(content)
        return self.provider.search_posts(url=FEED_URL, **kwargs)

    def assert_invalid_payload(self, content):
        with self.assertRaises(ProviderRequestError) as ctx:
            self.search(content)
        self.assertEqual(ctx.exception.failure_category, "invalid_payload")
        self.assertIsNone(ctx.exception.status_code)
        self.assert_sanitized(ctx.exception)

    def assert_sanitized(self, exc):
        self.assertNotIn("never-print-this", str(exc))
        self.assertNotIn("feed.invalid", str(exc))
        self.assertNotIn("password=", str(exc))

    def test_rss_uses_actual_returned_permalink_for_bare_id_subreddit_and_url(self):
        # The requested feed and caller hints cannot supply entry identity.
        post = self.search(_rss(_rss_item()), subreddit="MBAIndia", query="other")[0]
        self.assertEqual(post.reddit_id, "abc123")
        self.assertEqual(post.subreddit, "IIMSambalpur")
        self.assertEqual(post.url, POST_URL)
        self.assertEqual(post.title, "Test Post")
        self.assertEqual(post.body, "<p>Test body</p>")
        self.assertEqual(post.created_at, datetime(2024, 1, 15, 10, 30))
        self.assertEqual(post.comments, [])
        self.assertEqual(post.score, 0)
        self.assertEqual(post.num_comments, 0)
        self.assertFalse(hasattr(post, "source_mode"))

    def test_rss_accepts_matching_fullname_guid(self):
        post = self.search(_rss(_rss_item(guid="t3_abc123")))[0]
        self.assertEqual(post.reddit_id, "abc123")

    def test_rss_accepts_matching_canonical_permalink_guid(self):
        post = self.search(_rss(_rss_item(guid=POST_URL)))[0]
        self.assertEqual(post.reddit_id, "abc123")

    def test_rss_rejects_guid_identity_conflicts_and_non_post_ids(self):
        for guid in ("t3_def456", "t1_abc123", "not-a-reddit-id", "https://example.com/abc123"):
            with self.subTest(guid=guid):
                self.assert_invalid_payload(_rss(_rss_item(guid=guid)))

    def test_rss_requires_returned_canonical_link_even_with_explicit_guid(self):
        item = _rss_item(guid="t3_abc123").replace(f"<link>{POST_URL}</link>", "")
        self.assert_invalid_payload(_rss(item))

    def test_generic_feed_cannot_borrow_identity_from_request_or_title(self):
        self.request.return_value = _response(_rss(_rss_item(link="https://example.com/story")))
        with self.assertRaises(ProviderRequestError) as ctx:
            self.provider.search_posts(url=POST_URL, subreddit="IIMSambalpur")
        self.assertEqual(ctx.exception.failure_category, "invalid_payload")

    def test_canonical_permalink_validation_rejects_non_posts_and_unsafe_hosts(self):
        invalid_links = (
            "http://www.reddit.com/r/IIMSambalpur/comments/abc123/title/",
            "https://reddit.com.example/r/IIMSambalpur/comments/abc123/title/",
            "https://www.reddit.com@evil.invalid/r/IIMSambalpur/comments/abc123/title/",
            "https://user:never-print-this@www.reddit.com/r/IIMSambalpur/comments/abc123/title/",
            "https://www.reddit.com:444/r/IIMSambalpur/comments/abc123/title/",
            "https://www.reddit.com/r/IIMSambalpur/",
            "https://www.reddit.com/comments/abc123/",
            "https://www.reddit.com/r/IIMSambalpur/comments/t3_abc123/title/",
            "https://www.reddit.com/r/IIMSambalpur/comments/abc-123/title/",
            "https://www.reddit.com/r/IIMSambalpur/comments/abc123/title/comment123/",
            "https://www.reddit.com/r/IIMSambalpur/comments/abc123/title/?token=never-print-this",
            "https://www.reddit.com/r/IIMSambalpur/comments/abc123/title/#comment",
            "https://www.reddit.com/r/IIMSambalpur/comments/abc123/..",
            "https://www.reddit.com/r/IIMSambalpur/comments/abc123/%2e%2e/",
            "https://www.reddit.com/r/IIMSambalpur/comments/abc123/title%2fcomment/",
            "https://www.reddit.com/r/IIMSambalpur/comments/abc123/title%5ccomment/",
            "https://www.reddit.com/r/IIMSambalpur/comments/abc123/%ZZ/",
            "/r/IIMSambalpur/comments/abc123/title/",
        )
        for link in invalid_links:
            with self.subTest(link=link):
                self.assert_invalid_payload(_rss(_rss_item(link=link)))

    def test_missing_and_blank_titles_are_rejected(self):
        for title in (None, "", "   "):
            with self.subTest(title=title):
                self.assert_invalid_payload(_rss(_rss_item(title=title)))
                self.assert_invalid_payload(_atom(_atom_entry(title=title)))

    def test_rss_missing_malformed_or_timezone_unknown_dates_are_rejected(self):
        for date in (None, "", "yesterday", "31 Feb 2024 10:30:00 GMT",
                     "15 Jan 2024 10:30:00", "15 Jan 2024 10:30:00 -0000",
                     "15 Jan 2024 10:30:00 +0199", "15 Jan 2024 10:30:00 GMT junk"):
            with self.subTest(date=date):
                self.assert_invalid_payload(_rss(_rss_item(date=date)))

    def test_rss_date_offset_is_converted_to_naive_utc(self):
        for date, expected in (
            ("Mon, 15 Jan 2024 10:30:00 +0530", datetime(2024, 1, 15, 5, 0)),
            ("Mon, 15 Jan 2024 22:30:00 -0500", datetime(2024, 1, 16, 3, 30)),
            ("Mon, 15 Jan 2024 10:30:00 EST", datetime(2024, 1, 15, 15, 30)),
        ):
            with self.subTest(date=date):
                post = self.search(_rss(_rss_item(date=date)))[0]
                self.assertEqual(post.created_at, expected)
                self.assertIsNone(post.created_at.tzinfo)

    def test_rss_optional_description_may_be_absent(self):
        self.assertIsNone(self.search(_rss(_rss_item(body="")))[0].body)

    def test_rss_encoded_content_preserves_html(self):
        body = ('<content:encoded xmlns:content="http://purl.org/rss/1.0/modules/content/">'
                '<![CDATA[<div><a href="https://example.com">Link</a> & body</div>]]>'
                '</content:encoded><description>Short summary</description>')
        post = self.search(_rss(_rss_item(body=body)))[0]
        self.assertEqual(post.body, '<div><a href="https://example.com">Link</a> & body</div>')

    def test_atom_normalizes_real_fullname_category_link_and_timestamp(self):
        post = self.search(_atom(_atom_entry()))[0]
        self.assertEqual(post.reddit_id, "abc123")
        self.assertEqual(post.subreddit, "IIMSambalpur")
        self.assertEqual(post.url, POST_URL)
        self.assertEqual(post.title, "Atom Post")
        self.assertEqual(post.created_at, datetime(2024, 1, 15, 10, 30))
        self.assertIsNone(post.body)

    def test_atom_uses_actual_permalink_when_id_and_category_are_absent(self):
        entry = _atom_entry(category=None).replace("<id>t3_abc123</id>", "")
        post = self.search(_atom(entry))[0]
        self.assertEqual((post.reddit_id, post.subreddit), ("abc123", "IIMSambalpur"))

    def test_atom_id_and_category_conflicts_fail_closed(self):
        for entry in (_atom_entry(identifier="t3_def456"),
                      _atom_entry(identifier="t1_abc123"),
                      _atom_entry(identifier="urn:uuid:never-print-this"),
                      _atom_entry(category="MBAIndia")):
            with self.subTest(entry=entry):
                self.assert_invalid_payload(_atom(entry))

    def test_atom_prefers_published_over_later_updated_timestamp(self):
        entry = _atom_entry(date="2024-02-15T10:30:00Z")
        entry = entry.replace("</entry>", "<published>2024-01-15T10:30:00Z</published></entry>")
        post = self.search(_atom(entry))[0]
        self.assertEqual(post.created_at, datetime(2024, 1, 15, 10, 30))

    def test_atom_malformed_published_is_not_replaced_by_valid_updated(self):
        entry = _atom_entry().replace("</entry>", "<published>not-a-date</published></entry>")
        self.assert_invalid_payload(_atom(entry))

    def test_atom_date_offsets_fractional_seconds_and_utc_day_rollover(self):
        post = self.search(_atom(_atom_entry(date="2024-01-15T00:30:00.123456+05:30")))[0]
        self.assertEqual(post.created_at, datetime(2024, 1, 14, 19, 0, 0, 123456))
        self.assertIsNone(post.created_at.tzinfo)

    def test_atom_missing_or_malformed_required_dates_fail(self):
        for date in (None, "", "2024-01-15", "2024-01-15T10:30:00",
                     "2024-02-31T10:30:00Z", "2024-01-15T10:30:00+01:99"):
            with self.subTest(date=date):
                self.assert_invalid_payload(_atom(_atom_entry(date=date)))

    def test_atom_escaped_html_content_is_preserved(self):
        body = '<content type="html">&lt;p&gt;Hello &amp;amp; world&lt;/p&gt;</content>'
        post = self.search(_atom(_atom_entry(body=body)))[0]
        self.assertEqual(post.body, "<p>Hello &amp; world</p>")

    def test_atom_xhtml_body_is_preserved_as_markup(self):
        body = '<content type="xhtml"><div xmlns="http://www.w3.org/1999/xhtml"><p>Hello</p></div></content>'
        post = self.search(_atom(_atom_entry(body=body)))[0]
        self.assertIn("Hello</", post.body)
        self.assertIn("http://www.w3.org/1999/xhtml", post.body)
        self.assertIn("<", post.body)

    def test_atom_summary_is_used_only_when_content_is_absent(self):
        post = self.search(_atom(_atom_entry(body="<summary>Summary</summary>")))[0]
        self.assertEqual(post.body, "Summary")
        post = self.search(_atom(_atom_entry(body="<summary>Summary</summary><content>Full</content>")))[0]
        self.assertEqual(post.body, "Full")

    def test_atom_ignores_enclosures_when_selecting_alternate_post_link(self):
        entry = _atom_entry().replace(f'<link href="{POST_URL}"/>',
                                     '<link rel="enclosure" href="https://media.invalid/image"/>'
                                     f'<link rel="alternate" type="text/html" href="{POST_URL}"/>')
        self.assertEqual(self.search(_atom(entry))[0].url, POST_URL)

    def test_atom_multiple_conflicting_alternate_links_fail(self):
        entry = _atom_entry().replace("</entry>", '<link rel="alternate" '
                                     'href="https://www.reddit.com/r/MBAIndia/comments/def456/title/"/></entry>')
        self.assert_invalid_payload(_atom(entry))

    def test_atom_requires_actual_alternate_permalink(self):
        for replacement in ('<link rel="self" href="https://feed.invalid/entry"/>',
                            '<link rel="alternate"/>', ""):
            with self.subTest(replacement=replacement):
                entry = _atom_entry().replace(f'<link href="{POST_URL}"/>', replacement)
                self.assert_invalid_payload(_atom(entry))

    def test_ambiguous_duplicate_required_entry_metadata_fails(self):
        for extra in ("<title>Conflicting title</title>", f"<pubDate>{RSS_DATE}</pubDate>",
                      f"<link>{POST_URL}</link>"):
            with self.subTest(extra=extra):
                self.assert_invalid_payload(_rss(_rss_item().replace("</item>", f"{extra}</item>")))
        self.assert_invalid_payload(_atom(_atom_entry().replace("</entry>", "<id>t3_abc123</id></entry>")))

    def test_rss_explicit_subreddit_category_conflict_is_rejected(self):
        self.assert_invalid_payload(_rss(_rss_item(body="<category>r/MBAIndia</category>")))
        post = self.search(_rss(_rss_item(body="<category>Discussion</category>")))[0]
        self.assertEqual(post.subreddit, "IIMSambalpur")

    def test_atom_does_not_follow_external_content_or_next_links(self):
        entry = _atom_entry(body='<content src="https://content.invalid/body"/>')
        posts = self.search(_atom(entry, extra='<link rel="next" href="https://feed.invalid/page2"/>'))
        self.assertEqual(len(posts), 1)
        self.assertIsNone(posts[0].body)
        self.request.assert_called_once_with(FEED_URL, timeout=30, allow_redirects=False)

    def test_supported_empty_feeds_return_empty_lists(self):
        for content in (_rss(), _atom()):
            with self.subTest(content=content):
                self.assertEqual(self.search(content), [])

    def test_valid_but_unsupported_or_invalid_feed_shapes_are_not_empty_success(self):
        for content in ("<html><body>never-print-this</body></html>",
                        "<rss/>", '<rss version="1.0"><channel/></rss>',
                        '<rss version="2.0"/>', '<rss version="2.0"><channel/><channel/></rss>',
                        "<feed><entry/></feed>"):
            with self.subTest(content=content):
                self.assert_invalid_payload(content)

    def test_malformed_xml_has_a_distinct_sanitized_category(self):
        for content in (b"", b"<rss>never-print-this", b"not XML: password=never-print-this"):
            with self.subTest(content=content):
                with self.assertRaises(ProviderRequestError) as ctx:
                    self.search(content)
                self.assertEqual(ctx.exception.failure_category, "invalid_xml")
                self.assert_sanitized(ctx.exception)

    def test_dedup_happens_before_limit_and_keeps_first_post(self):
        content = _rss(_rss_item(), _rss_item(title="Duplicate"),
                       _rss_item(post_id="def456"), _rss_item(post_id="ghi789"))
        posts = self.search(content, limit=2)
        self.assertEqual([post.reddit_id for post in posts], ["abc123", "def456"])
        self.assertEqual(posts[0].title, "Test Post")

    def test_atom_dedup_and_limit(self):
        second_link = "https://www.reddit.com/r/IIMSambalpur/comments/def456/another/"
        content = _atom(_atom_entry(), _atom_entry(),
                        _atom_entry(post_id="def456", link=second_link))
        self.assertEqual([post.reddit_id for post in self.search(content, limit=2)], ["abc123", "def456"])

    def test_default_and_hard_output_limits_are_bounded(self):
        content = _rss(*(_rss_item(post_id=f"a{i}") for i in range(110)))
        self.assertEqual(len(self.search(content)), 25)
        self.assertEqual(len(self.search(content, limit=100)), 100)

    def test_zero_limit_performs_no_request(self):
        self.assertEqual(self.provider.search_posts(url=FEED_URL, limit=0), [])
        self.request.assert_not_called()

    def test_invalid_limits_fail_before_request(self):
        for limit in (-1, 101, True, 1.5, "3", None):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    self.provider.search_posts(url=FEED_URL, limit=limit)
        self.request.assert_not_called()

    def test_limit_stops_before_unneeded_bad_entries(self):
        content = _rss(_rss_item(), _rss_item(post_id="def456", date="bad"))
        self.assertEqual(len(self.search(content, limit=1)), 1)

    def test_duplicate_identity_cannot_claim_a_different_subreddit(self):
        alternate = POST_URL.replace("IIMSambalpur", "MBAIndia")
        self.assert_invalid_payload(_rss(_rss_item(), _rss_item(link=alternate)))

    def test_missing_or_invalid_input_url_is_sanitized_and_makes_no_request(self):
        for url in (None, "", "  ", 1, "file:///never-print-this", "not-a-url"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    self.provider.search_posts(url=url)
                self.assert_sanitized(ctx.exception)
        self.request.assert_not_called()

    def test_http_status_errors_are_typed_sanitized_and_not_retried(self):
        for status in (301, 403, 429, 500):
            with self.subTest(status=status):
                self.request.reset_mock()
                self.request.return_value = _response(b"never-print-this", status_code=status)
                with self.assertRaises(ProviderRequestError) as ctx:
                    self.provider.search_posts(url=FEED_URL)
                self.assertEqual(ctx.exception.failure_category, "http_status")
                self.assertEqual(ctx.exception.status_code, status)
                self.assert_sanitized(ctx.exception)
                self.request.assert_called_once()

    def test_request_failures_have_typed_sanitized_categories_and_no_retry(self):
        for error, category in ((requests.Timeout, "timeout"),
                                (requests.ConnectionError, "connection"),
                                (requests.RequestException, "transport")):
            with self.subTest(error=error):
                self.request.reset_mock()
                self.request.side_effect = error(f"{FEED_URL} password=never-print-this")
                with self.assertRaises(ProviderRequestError) as ctx:
                    self.provider.search_posts(url=FEED_URL)
                self.assertEqual(ctx.exception.failure_category, category)
                self.assertIsNone(ctx.exception.status_code)
                self.assert_sanitized(ctx.exception)
                self.request.assert_called_once()

    def test_http_exception_retains_only_safe_status(self):
        response = _response(b"never-print-this", status_code=403)
        self.request.side_effect = requests.HTTPError(FEED_URL, response=response)
        with self.assertRaises(ProviderRequestError) as ctx:
            self.provider.search_posts(url=FEED_URL)
        self.assertEqual(ctx.exception.failure_category, "http_status")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assert_sanitized(ctx.exception)
        self.request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
