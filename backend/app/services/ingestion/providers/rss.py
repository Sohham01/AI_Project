"""Bounded Reddit post discovery from a single returned RSS2 or Atom document.

Entry identity comes from an actual HTTPS Reddit post permalink in the returned
``item/link`` or Atom alternate ``entry/link``: its ``/r/<sub>/comments/<id>/``
path supplies the subreddit and bare base36 Reddit post ID. An explicit guid/id
must agree when present. Neither the requested feed URL, caller hints, title nor
body supplies identity. Generic feeds without that permalink are unsupported.

RSS pubDate is required. Atom uses published when present, otherwise the actual
updated timestamp (which need not represent original creation). Missing titles,
dates, unknown timezones and conflicting identity fail closed. No pagination,
content-link fetches, redirects, retries, or source_mode inference are performed.
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Any
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import requests

from app.schemas.ingestion import PostData
from app.services.ingestion.providers.base import IngestionProvider, ProviderRequestError


_ATOM = "{http://www.w3.org/2005/Atom}"
_ENCODED_CONTENT = "{http://purl.org/rss/1.0/modules/content/}encoded"
_POST_PATH = re.compile(
    r"/r/(?P<subreddit>[A-Za-z0-9_]{1,21})/comments/"
    r"(?P<post_id>[A-Za-z0-9]+)(?:/[^/\s?#]+)?/?"
)
_POST_ID = re.compile(r"(?:t3_)?([A-Za-z0-9]+)")
_RSS_DATE = re.compile(
    r"(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+)?"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(?:\d{4}|\d{2})\s+\d{2}:\d{2}(?::\d{2})?\s+"
    r"(?:[+-](?:[01]\d|2[0-3])[0-5]\d|UT|GMT|[ECMP][DS]T)",
    re.IGNORECASE,
)
_ATOM_DATE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)"
)


def _optional_element(parent: ET.Element, tag: str) -> ET.Element | None:
    elements = parent.findall(tag)
    if len(elements) > 1:
        raise ValueError("Ambiguous entry metadata")
    return elements[0] if elements else None


def _text(element: ET.Element | None) -> str:
    if element is None or len(element) or not element.text or not element.text.strip():
        raise ValueError("Required entry metadata is missing")
    return element.text.strip()


def _post_identity(value: str) -> tuple[str, str]:
    """Validate a returned canonical thread URL, then parse its source identity."""
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Invalid Reddit permalink")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc.casefold() not in {"reddit.com", "www.reddit.com"}
        or parsed.query
        or parsed.fragment
        or re.search(r"%(?![0-9A-Fa-f]{2})", parsed.path)
    ):
        raise ValueError("A canonical Reddit thread permalink is required")
    decoded_path = unquote(parsed.path)
    if (
        "\\" in decoded_path
        or decoded_path.count("/") != parsed.path.count("/")
        or any(segment in {".", ".."} for segment in decoded_path.split("/"))
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in decoded_path)
    ):
        raise ValueError("Invalid Reddit permalink path")
    match = _POST_PATH.fullmatch(parsed.path)
    if not match:
        raise ValueError("A canonical Reddit thread permalink is required")
    return match.group("post_id").lower(), match.group("subreddit")


def _validate_identifier(element: ET.Element | None, post_id: str, subreddit: str) -> None:
    if element is None:
        return
    identifier = _text(element)
    match = _POST_ID.fullmatch(identifier)
    if match:
        actual_id, actual_subreddit = match.group(1).lower(), subreddit
    else:
        actual_id, actual_subreddit = _post_identity(identifier)
    if actual_id != post_id or actual_subreddit.casefold() != subreddit.casefold():
        raise ValueError("Conflicting Reddit entry identity")


def _parse_pub_date(value: str | None) -> datetime:
    """Require a real RSS RFC822 date with a known zone; return naive UTC."""
    if not value or not _RSS_DATE.fullmatch(value):
        raise ValueError("An RFC822 publication date with a known zone is required")
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        raise ValueError("A publication timezone is required")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_atom_date(value: str) -> datetime:
    if not _ATOM_DATE.fullmatch(value):
        raise ValueError("An RFC3339 timestamp is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _body(element: ET.Element | None) -> str | None:
    if element is None or element.get("src") is not None:
        return None
    # Escaped HTML/CDATA remains HTML. Preserve inline XHTML as serialized markup
    # instead of silently dropping its child elements or following a content src.
    return (element.text or "") + "".join(ET.tostring(child, encoding="unicode") for child in element)


def _atom_link(entry: ET.Element) -> str:
    links = [
        element for element in entry.findall(f"{_ATOM}link")
        if element.get("rel", "alternate") == "alternate"
        and element.get("type", "text/html") in {"text/html", "application/xhtml+xml"}
    ]
    values = {element.get("href", "").strip() for element in links}
    if len(values) != 1 or not next(iter(values)):
        raise ValueError("One unambiguous alternate post permalink is required")
    return next(iter(values))


def _validate_categories(entry: ET.Element, subreddit: str, atom: bool) -> None:
    for category in entry.findall(f"{_ATOM}category" if atom else "category"):
        # Reddit Atom categories explicitly identify r/<subreddit>. RSS2's
        # generic category tags do not, unless they actually carry an r/ prefix.
        term = category.get("term", "").strip() if atom else _text(category)
        if not atom and not term.startswith("r/"):
            continue
        actual = term.removeprefix("r/")
        if actual.casefold() != subreddit.casefold():
            raise ValueError("Conflicting entry subreddit")
        label = category.get("label", "")
        if atom and label.startswith("r/") and label[2:].casefold() != subreddit.casefold():
            raise ValueError("Conflicting entry subreddit")


def _normalize_entry(entry: ET.Element, atom: bool) -> PostData:
    namespace = _ATOM if atom else ""
    link = _atom_link(entry) if atom else _text(_optional_element(entry, "link"))
    post_id, subreddit = _post_identity(link)
    identifier = _optional_element(entry, f"{_ATOM}id" if atom else "guid")
    _validate_identifier(identifier, post_id, subreddit)
    _validate_categories(entry, subreddit, atom)
    title = _text(_optional_element(entry, f"{namespace}title"))
    if atom:
        date_element = _optional_element(entry, f"{_ATOM}published")
        if date_element is None:
            date_element = _optional_element(entry, f"{_ATOM}updated")
        created_at = _parse_atom_date(_text(date_element))
        body_element = _optional_element(entry, f"{_ATOM}content")
        if body_element is None:
            body_element = _optional_element(entry, f"{_ATOM}summary")
    else:
        created_at = _parse_pub_date(_text(_optional_element(entry, "pubDate")))
        body_element = _optional_element(entry, _ENCODED_CONTENT)
        if body_element is None:
            body_element = _optional_element(entry, "description")
    return PostData(
        reddit_id=post_id,
        title=title,
        body=_body(body_element),
        subreddit=subreddit,
        created_at=created_at,
        url=link,
        comments=[],
    )


class RSSProvider(IngestionProvider):
    DEFAULT_LIMIT = 25
    MAX_POSTS = 100

    def search_posts(self, **kwargs: Any) -> list[PostData]:
        """Read at most limit unique posts (0..100) from one feed response.

        Invalid entries encountered before the output cap reject the response;
        invalid/unsupported payloads are not confused with a valid empty feed.
        Only standard RSS2 channels and namespaced Atom feeds are supported.
        """
        url = kwargs.get("url")
        if not isinstance(url, str) or not url.strip() or any(char.isspace() for char in url):
            raise ValueError("An HTTP(S) RSS URL is required")
        try:
            parsed_url = urlsplit(url)
            valid_url = parsed_url.scheme in {"http", "https"} and bool(parsed_url.hostname)
            parsed_url.port  # Reject invalid port syntax without disclosing the URL.
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError("An HTTP(S) RSS URL is required")
        limit = kwargs.get("limit", self.DEFAULT_LIMIT)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= self.MAX_POSTS:
            raise ValueError("limit must be an integer between 0 and 100")
        if limit == 0:
            return []

        try:
            response = requests.get(url, timeout=30, allow_redirects=False)
        except requests.Timeout:
            raise ProviderRequestError("timeout") from None
        except requests.ConnectionError:
            raise ProviderRequestError("connection") from None
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            raise ProviderRequestError("http_status", status) from None
        except requests.RequestException:
            raise ProviderRequestError("transport") from None
        if not 200 <= response.status_code < 300:
            raise ProviderRequestError("http_status", response.status_code)
        try:
            root = ET.fromstring(response.content)
        except (ET.ParseError, TypeError, ValueError):
            raise ProviderRequestError("invalid_xml") from None

        try:
            if root.tag == "rss" and root.get("version") == "2.0":
                channel = _optional_element(root, "channel")
                if channel is None:
                    raise ValueError("An RSS2 channel is required")
                entries, atom = channel.findall("item"), False
            elif root.tag == f"{_ATOM}feed":
                entries, atom = root.findall(f"{_ATOM}entry"), True
            else:
                raise ValueError("Unsupported feed format")

            posts: list[PostData] = []
            seen: dict[str, str] = {}
            for entry in entries:
                post = _normalize_entry(entry, atom)
                previous_subreddit = seen.get(post.reddit_id)
                if previous_subreddit is not None:
                    if previous_subreddit != post.subreddit.casefold():
                        raise ValueError("Conflicting duplicate post identity")
                    continue
                seen[post.reddit_id] = post.subreddit.casefold()
                posts.append(post)
                if len(posts) >= limit:
                    break
            return posts
        except (TypeError, ValueError, OverflowError):
            raise ProviderRequestError("invalid_payload") from None
