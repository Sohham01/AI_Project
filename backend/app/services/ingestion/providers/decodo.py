"""Decodo Universal Web Data API adapter for bounded Reddit HTML snapshots.

The request uses the Universal scraper configuration (explicit source URL,
proxy pool, and rendering mode). Reddit-specific targets such as
``reddit_post`` are intentionally not sent: Decodo returns raw HTML and the
application parses it into the existing ``PostData``/``CommentData`` models.

For backwards compatibility with already-tested offline fixtures, structured
Reddit Listings are still accepted when supplied in a mocked response. That
fallback does not change the live request contract and is not used to infer
pagination or completeness.

Contract: https://help.decodo.com/docs/web-scraping-api-reddit-post
Envelope: https://help.decodo.com/docs/web-data-api-scrape-real-time-requests
Auth: https://help.decodo.com/docs/web-data-api-migration-guide
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
import re
from typing import Any
from urllib.parse import unquote_to_bytes, urlparse

import requests

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.providers.base import (
    IngestionProvider,
    ProviderRequestError,
    UnsupportedOperation,
)
from app.services.ingestion.providers.decodo_html import parse_reddit_html


_REDDIT_POST_URL = re.compile(
    r"^/r/(?P<subreddit>[A-Za-z0-9_]+)/comments/(?P<post_id>[A-Za-z0-9]+)/(?P<slug>[^/]*)/?$",
    re.IGNORECASE,
)
_REDDIT_SUBREDDIT_URL = re.compile(
    r"^/r/(?P<subreddit>[A-Za-z0-9_]+)/?$",
    re.IGNORECASE,
)
_ALLOWED_SUBREDDITS = frozenset(
    {"iimsambalpur", "iimsambalpurug", "mbaindia", "catprep", "catpreparation"}
)
_ID = re.compile(r"[A-Za-z0-9]+")
_BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]+")
logger = logging.getLogger(__name__)


def _validated_reddit_url(url: str) -> tuple[str, str | None]:
    """Accept a canonical Reddit subreddit or post source URL in project scope."""
    if not isinstance(url, str) or any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise ValueError("A Reddit subreddit or post URL is required")
    parsed = None
    try:
        parsed = urlparse(url)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname in {"reddit.com", "www.reddit.com"}
            and parsed.username is None
            and parsed.password is None
            and parsed.port in (None, 443)
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid or parsed is None:
        raise ValueError("URL must be an HTTPS Reddit subreddit or post URL")
    path_error = "URL must identify an unambiguous Reddit subreddit or post"
    if ";" in url or "//" in parsed.path or re.search(r"%(?![A-Fa-f0-9]{2})", parsed.path):
        raise ValueError(path_error)
    try:
        segments = [unquote_to_bytes(segment).decode("utf-8") for segment in parsed.path.split("/")]
    except UnicodeError:
        raise ValueError(path_error) from None
    # Decode once for validation only; submit the original URL unchanged.
    # A remaining percent sign could acquire a different meaning if decoded again.
    if any(
        segment in {".", ".."}
        or any(ord(char) <= 32 or 127 <= ord(char) <= 159 or char in "/\\%?#;" for char in segment)
        for segment in segments
    ):
        raise ValueError(path_error)
    match = _REDDIT_POST_URL.fullmatch(parsed.path)
    if match:
        subreddit = match.group("subreddit")
        post_id = match.group("post_id").lower()
    else:
        subreddit_match = _REDDIT_SUBREDDIT_URL.fullmatch(parsed.path)
        if not subreddit_match:
            raise ValueError(path_error)
        subreddit = subreddit_match.group("subreddit")
        post_id = None
    if subreddit.casefold() not in _ALLOWED_SUBREDDITS:
        raise ValueError("Subreddit is outside the approved collection scope")
    return subreddit, post_id


def _validated_post_url(url: str) -> tuple[str, str]:
    """Compatibility helper for callers that still require a post URL."""

    subreddit, post_id = _validated_reddit_url(url)
    if post_id is None:
        raise ValueError("URL must identify a Reddit post thread")
    return subreddit, post_id


def _validated_option(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or not re.fullmatch(r"[A-Za-z0-9._-]+", value)
    ):
        raise ValueError(f"{field} must be a safe nonblank token")
    return value


def _validated_endpoint(endpoint: Any) -> str:
    """Never send a key to an unrelated host, URL credential, or redirect."""
    message = "DECODO_API_URL must be the HTTPS Web Data API scrape endpoint"
    if not isinstance(endpoint, str) or any(ord(char) <= 32 or ord(char) == 127 for char in endpoint):
        raise ValueError(message)
    try:
        parsed = urlparse(endpoint)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname == "data.decodo.com"
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
            and parsed.path == "/v1/scrape"
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(message)
    return endpoint


def _utc_datetime(value: Any) -> datetime:
    """Parse an actual Reddit epoch or ISO timestamp; never synthesize a date."""
    if isinstance(value, bool):
        raise ValueError("Invalid timestamp")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Invalid timestamp")
        parsed = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str) and len(value) >= 19:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Invalid timestamp") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        raise ValueError("Missing timestamp")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _check_envelope_status(value: dict[str, Any]) -> None:
    """Codes and error bodies are untrusted; never include them in messages."""
    if "status_code" in value:
        code = value["status_code"]
        if type(code) is not int or code not in (200, 12000):
            if type(code) is int and 400 <= code <= 599:
                raise ProviderRequestError("http_status", code)
            raise ProviderRequestError("invalid_payload")
    if "parse_status_code" in value:
        code = value["parse_status_code"]
        # Partial/defaulted parser output cannot establish complete metadata.
        if type(code) is not int or code != 12000:
            raise ProviderRequestError("invalid_payload")
    for key in ("errors", "error"):
        if value.get(key) not in (None, "", [], {}):
            raise ProviderRequestError("invalid_payload")


def _thread_content(payload: Any) -> list[Any]:
    """Accept the documented template output or one real-time API result."""
    if isinstance(payload, list):
        content = payload
    else:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected API response")
        _check_envelope_status(payload)
        results = payload.get("results")
        # A scrape_url call submits one task, never a batch or a paginated job.
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            raise ValueError("Expected one scrape result")
        result = results[0]
        if "status_code" not in result:
            raise ValueError("Missing scrape result status")
        _check_envelope_status(result)
        content = result.get("content")
        if isinstance(content, str):
            content = json.loads(content)
        # Parsed templates may wrap their parsed result in a results object.
        if isinstance(content, dict):
            _check_envelope_status(content)
            content = content.get("results")
    if not isinstance(content, list) or not 1 <= len(content) <= 2:
        raise ValueError("Expected Reddit thread Listings")
    return content


def _result_content(payload: Any) -> Any:
    """Extract one Universal-scraper result without interpreting its body."""

    # Existing offline adapter fixtures may provide the result body directly;
    # the live Universal API uses the envelope below.
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Unexpected API response")
    _check_envelope_status(payload)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("Expected one scrape result")
    result = results[0]
    if "status_code" not in result:
        raise ValueError("Missing scrape result status")
    _check_envelope_status(result)
    if "content" not in result:
        raise ValueError("Missing scrape result content")
    content = result["content"]
    if isinstance(content, dict) and "results" in content:
        _check_envelope_status(content)
        content = content["results"]
    return content


def _listing_children(value: Any, diagnostics: set[str] | None = None) -> list[dict[str, Any]]:
    """Unwrap exactly one Listing without silently accepting missing children."""
    if not isinstance(value, dict) or value.get("kind") != "Listing":
        raise ValueError("Unexpected Reddit listing")
    data = value.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("children"), list):
        raise ValueError("Unexpected Reddit listing children")
    if diagnostics is not None and (data.get("after") is not None or data.get("before") is not None):
        diagnostics.add("listing_cursor_not_followed")
    return data["children"]


def _bare_id(value: Any, kind: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Missing Reddit identifier")
    if value.startswith(f"{kind}_"):
        value = value[3:]
    if not _ID.fullmatch(value):
        raise ValueError("Invalid Reddit identifier")
    return value.lower()


def _row_id(row: dict[str, Any], kind: str) -> str:
    supplied = [_bare_id(row[key], kind) for key in ("id", "name") if key in row]
    if not supplied or any(value != supplied[0] for value in supplied):
        raise ValueError("Missing or inconsistent Reddit identifier")
    return supplied[0]


def _integer(value: Any, *, nonnegative: bool = False) -> int:
    if type(value) is not int or (nonnegative and value < 0):
        raise ValueError("Invalid Reddit count")
    return value


def _parent(value: Any, post_id: str) -> str | None:
    if not isinstance(value, str):
        raise ValueError("Missing comment parent")
    if value.startswith("t3_"):
        if _bare_id(value, "t3") != post_id:
            raise ValueError("Comment belongs to another post")
        return None
    if value.startswith("t1_"):
        return _bare_id(value, "t1")
    raise ValueError("Invalid comment parent")


def _validate_comment_scope(row: dict[str, Any], post_id: str, subreddit: str) -> None:
    if "link_id" in row:
        if not isinstance(row["link_id"], str) or not row["link_id"].startswith("t3_"):
            raise ValueError("Invalid comment post reference")
        if _bare_id(row["link_id"], "t3") != post_id:
            raise ValueError("Comment belongs to another post")
    if "subreddit" in row:
        if not isinstance(row["subreddit"], str) or row["subreddit"].casefold() != subreddit.casefold():
            raise ValueError("Comment belongs to another subreddit")


def _validate_parent_graph(parents: dict[str, str | None]) -> None:
    complete: set[str] = set()
    for comment_id in parents:
        path: set[str] = set()
        current = comment_id
        while current in parents and current not in complete:
            if current in path:
                raise ValueError("Cyclic Reddit comment parents")
            path.add(current)
            current = parents[current]
        complete.update(path)


class DecodoProvider(IngestionProvider):
    """Fetch one bounded Reddit HTML snapshot using Decodo Universal."""

    DEFAULT_ENDPOINT = "https://data.decodo.com/v1/scrape"
    DEFAULT_PROXY_POOL = "premium"
    DEFAULT_HEADLESS = "html"
    MAX_COMMENTS_PER_POST = 25
    DEFAULT_TIMEOUT = 10
    MAX_TIMEOUT = 30
    MAX_COMMENT_NODES = 1000
    MAX_COMMENT_DEPTH = 100

    def __init__(self) -> None:
        self.api_token = os.getenv("DECODO_API_TOKEN")
        self.endpoint = os.getenv("DECODO_API_URL", self.DEFAULT_ENDPOINT)
        self.proxy_pool = os.getenv("DECODO_PROXY_POOL", self.DEFAULT_PROXY_POOL)
        self.headless = os.getenv("DECODO_HEADLESS", self.DEFAULT_HEADLESS)
        # The canonical schemas have no flags. Expose safe, fixed diagnostics
        # instead of inventing fields, missing comments, or deleted content.
        self.last_diagnostics: tuple[str, ...] = ()

    def _request_universal(
        self,
        url: str,
        *,
        proxy_pool: str,
        headless: str,
        timeout: float,
    ) -> Any:
        try:
            response = requests.post(
                _validated_endpoint(self.endpoint),
                json={
                    "url": url,
                    "proxy_pool": proxy_pool,
                    "headless": headless,
                },
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
                allow_redirects=False,
            )
        except requests.Timeout:
            raise ProviderRequestError("timeout") from None
        except requests.ConnectionError:
            raise ProviderRequestError("connection") from None
        except requests.RequestException:
            raise ProviderRequestError("transport") from None

        if type(response.status_code) is not int:
            raise ProviderRequestError("invalid_payload")
        if response.status_code != 200:
            raise ProviderRequestError("http_status", response.status_code)
        try:
            return response.json()
        except (ValueError, RecursionError):
            raw_html = getattr(response, "text", None)
            if not isinstance(raw_html, str) or not raw_html.strip():
                raw_bytes = getattr(response, "content", None)
                if isinstance(raw_bytes, bytes):
                    try:
                        raw_html = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        raw_html = None
            if not isinstance(raw_html, str) or not raw_html.strip():
                raise ProviderRequestError("invalid_json") from None
            return raw_html

    @classmethod
    def _normalize_universal_content(
        cls,
        payload: Any,
        *,
        url: str,
        requested_subreddit: str,
        requested_id: str | None,
        max_posts: int,
        comment_limit: int,
        diagnostics: set[str],
    ) -> list[PostData]:
        content = payload if isinstance(payload, str) else _result_content(payload)
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    structured = json.loads(stripped)
                except (TypeError, ValueError, RecursionError):
                    structured = None
                if structured is not None:
                    content = structured
            if isinstance(content, str):
                parsed = parse_reddit_html(
                    content,
                    max_posts=max_posts,
                    comment_limit=comment_limit,
                )
                diagnostics.update(parsed.diagnostics)
                posts = list(parsed)
            else:
                posts = [
                    cls._normalize_post(
                        content,
                        url,
                        requested_subreddit,
                        requested_id,
                        comment_limit,
                        diagnostics,
                    )
                ]
        else:
            posts = [
                cls._normalize_post(
                    content,
                    url,
                    requested_subreddit,
                    requested_id,
                    comment_limit,
                    diagnostics,
                )
            ]

        for post in posts:
            if post.subreddit.casefold() != requested_subreddit.casefold():
                raise ValueError("Response subreddit does not match requested URL")
            if requested_id is not None and post.reddit_id != requested_id:
                raise ValueError("Response post ID does not match requested URL")
        return posts[:max_posts]

    def scrape_url(self, url: str, **kwargs: Any) -> PostData:
        self.last_diagnostics = ()
        if not self.api_token:
            raise UnsupportedOperation("Decodo not configured")
        if (
            not isinstance(self.api_token, str)
            or len(self.api_token) > 4096
            or not _BEARER_TOKEN.fullmatch(self.api_token)
        ):
            raise ValueError("DECODO_API_TOKEN must contain only a Web Data API key")
        _validated_endpoint(self.endpoint)
        subreddit, requested_id = _validated_reddit_url(url)
        proxy_pool = _validated_option(
            kwargs.pop("proxy_pool", self.proxy_pool), field="DECODO_PROXY_POOL"
        )
        headless = _validated_option(
            kwargs.pop("headless", self.headless), field="DECODO_HEADLESS"
        )
        comment_limit = kwargs.pop("comment_limit", self.MAX_COMMENTS_PER_POST)
        if isinstance(comment_limit, bool) or not isinstance(comment_limit, int):
            raise ValueError("comment_limit must be an integer")
        if not 0 <= comment_limit <= self.MAX_COMMENTS_PER_POST:
            raise ValueError(f"comment_limit must be between 0 and {self.MAX_COMMENTS_PER_POST}")
        timeout = kwargs.pop("timeout", self.DEFAULT_TIMEOUT)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= self.MAX_TIMEOUT
            or not math.isfinite(timeout)
        ):
            raise ValueError(f"timeout must be positive and at most {self.MAX_TIMEOUT} seconds")
        if kwargs:
            raise ValueError("Unsupported scrape options")

        diagnostics: set[str] = set()
        try:
            posts = self._normalize_universal_content(
                self._request_universal(
                    url,
                    proxy_pool=proxy_pool,
                    headless=headless,
                    timeout=timeout,
                ),
                url=url,
                requested_subreddit=subreddit,
                requested_id=requested_id,
                max_posts=1,
                comment_limit=comment_limit,
                diagnostics=diagnostics,
            )
            if len(posts) != 1:
                raise ValueError("Expected one Reddit post in response")
            return posts[0]
        except (KeyError, TypeError, ValueError, OverflowError, OSError, RecursionError):
            raise ProviderRequestError("invalid_payload") from None
        finally:
            self.last_diagnostics = tuple(sorted(diagnostics))
            if self.last_diagnostics:
                logger.warning("Decodo normalization diagnostics: %s", ",".join(self.last_diagnostics))

    def scrape_source(self, url: str, **kwargs: Any) -> list[PostData]:
        """Fetch a bounded Universal subreddit snapshot as multiple posts."""

        self.last_diagnostics = ()
        if not self.api_token:
            raise UnsupportedOperation("Decodo not configured")
        if (
            not isinstance(self.api_token, str)
            or len(self.api_token) > 4096
            or not _BEARER_TOKEN.fullmatch(self.api_token)
        ):
            raise ValueError("DECODO_API_TOKEN must contain only a Web Data API key")
        subreddit, requested_id = _validated_reddit_url(url)
        proxy_pool = _validated_option(
            kwargs.pop("proxy_pool", self.proxy_pool), field="DECODO_PROXY_POOL"
        )
        headless = _validated_option(
            kwargs.pop("headless", self.headless), field="DECODO_HEADLESS"
        )
        max_posts = kwargs.pop("max_posts", 10)
        comment_limit = kwargs.pop("comment_limit", self.MAX_COMMENTS_PER_POST)
        timeout = kwargs.pop("timeout", self.DEFAULT_TIMEOUT)
        if isinstance(max_posts, bool) or not isinstance(max_posts, int) or not 1 <= max_posts <= 200:
            raise ValueError("max_posts must be between 1 and 200")
        if isinstance(comment_limit, bool) or not isinstance(comment_limit, int) or not 0 <= comment_limit <= self.MAX_COMMENTS_PER_POST:
            raise ValueError(f"comment_limit must be between 0 and {self.MAX_COMMENTS_PER_POST}")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= self.MAX_TIMEOUT
            or not math.isfinite(timeout)
        ):
            raise ValueError(f"timeout must be positive and at most {self.MAX_TIMEOUT} seconds")
        if kwargs:
            raise ValueError("Unsupported scrape options")
        diagnostics: set[str] = set()
        try:
            posts = self._normalize_universal_content(
                self._request_universal(
                    url,
                    proxy_pool=proxy_pool,
                    headless=headless,
                    timeout=timeout,
                ),
                url=url,
                requested_subreddit=subreddit,
                requested_id=requested_id,
                max_posts=max_posts,
                comment_limit=comment_limit,
                diagnostics=diagnostics,
            )
            return posts
        except (KeyError, TypeError, ValueError, OverflowError, OSError, RecursionError):
            raise ProviderRequestError("invalid_payload") from None
        finally:
            self.last_diagnostics = tuple(sorted(diagnostics))
            if self.last_diagnostics:
                logger.warning("Decodo normalization diagnostics: %s", ",".join(self.last_diagnostics))

    @classmethod
    def _normalize_post(
        cls,
        payload: Any,
        url: str,
        requested_subreddit: str,
        requested_id: str | None,
        comment_limit: int,
        diagnostics: set[str] | None = None,
    ) -> PostData:
        diagnostics = diagnostics if diagnostics is not None else set()
        listings = _thread_content(payload)
        post_rows = _listing_children(listings[0], diagnostics)
        if len(post_rows) != 1 or not isinstance(post_rows[0], dict) or post_rows[0].get("kind") != "t3":
            raise ValueError("Expected one post in Reddit thread response")
        raw_post = post_rows[0].get("data")
        if not isinstance(raw_post, dict):
            raise ValueError("Missing post data")
        post_id = _row_id(raw_post, "t3")
        if requested_id is not None and post_id != requested_id.lower():
            raise ValueError("Response post ID does not match requested URL")
        actual_subreddit = raw_post.get("subreddit")
        if not isinstance(actual_subreddit, str) or actual_subreddit.casefold() != requested_subreddit.casefold():
            raise ValueError("Response subreddit does not match requested URL")

        title = raw_post.get("title")
        body = raw_post.get("selftext")
        if not isinstance(title, str) or not title.strip() or (body is not None and not isinstance(body, str)):
            raise ValueError("Invalid post text")
        post = PostData(
            reddit_id=post_id,
            title=title,
            body=body,
            subreddit=actual_subreddit,
            created_at=_utc_datetime(raw_post.get("created_utc", raw_post.get("created"))),
            score=_integer(raw_post.get("score")),
            num_comments=_integer(raw_post.get("num_comments"), nonnegative=True),
            url=url,
        )
        if body in ("[deleted]", "[removed]") or raw_post.get("removed_by_category"):
            diagnostics.add("post_content_unavailable")

        if len(listings) == 1:
            diagnostics.add("comments_listing_missing")
            children = []
        else:
            children = _listing_children(listings[1], diagnostics)

        # Iterative depth-first traversal avoids unbounded Python recursion.
        # Frames hold iterators so a wide Listing does not grow the work stack.
        stack: list[tuple[Any, str | None, int]] = [(iter(children), None, 0)]
        seen: dict[str, CommentData] = {}
        parents: dict[str, str | None] = {}
        visited = 0
        end = object()
        while stack:
            iterator, expected_parent, depth = stack[-1]
            child = next(iterator, end)
            if child is end:
                stack.pop()
                continue
            if visited >= cls.MAX_COMMENT_NODES:
                diagnostics.add("comment_traversal_limit")
                break
            visited += 1
            if not isinstance(child, dict) or not isinstance(child.get("data"), dict):
                raise ValueError("Invalid comment data")
            kind, row = child.get("kind"), child["data"]
            if kind == "more":
                diagnostics.add("comments_more_available")
                _validate_comment_scope(row, post_id, actual_subreddit)
                if "parent_id" in row:
                    parent_id = _parent(row["parent_id"], post_id)
                    if expected_parent is not None and parent_id != expected_parent:
                        raise ValueError("Inconsistent nested comment parent")
                continue
            if kind != "t1":
                raise ValueError("Unexpected comment kind")
            _validate_comment_scope(row, post_id, actual_subreddit)
            if "link_id" not in row:
                diagnostics.add("comment_link_id_missing")
            comment_id = _row_id(row, "t1")
            parent_id = _parent(row.get("parent_id"), post_id)
            if parent_id == comment_id or (expected_parent is not None and parent_id != expected_parent):
                raise ValueError("Inconsistent nested comment parent")
            body = row.get("body")
            if not isinstance(body, str) or not body.strip():
                diagnostics.add("comment_required_metadata_missing")
                raise ValueError("Missing comment body")
            comment = CommentData(
                reddit_id=comment_id,
                body=body,
                created_at=_utc_datetime(row.get("created_utc", row.get("created"))),
                score=_integer(row.get("score")),
                parent_reddit_id=parent_id,
            )
            if body in ("[deleted]", "[removed]") or row.get("removal_reason"):
                diagnostics.add("comment_content_unavailable")
            if comment_id in seen:
                if seen[comment_id] != comment:
                    raise ValueError("Conflicting duplicate comments")
                diagnostics.add("duplicate_comments_removed")
            else:
                seen[comment_id] = comment
                parents[comment_id] = parent_id
                if len(post.comments) < comment_limit:
                    post.comments.append(comment)
                else:
                    diagnostics.add("comments_truncated_by_limit")
            replies = row.get("replies")
            if replies is None:
                diagnostics.add("comment_replies_missing")
            if replies not in (None, ""):
                nested = _listing_children(replies, diagnostics)
                if nested and depth >= cls.MAX_COMMENT_DEPTH:
                    diagnostics.add("comment_depth_limit")
                elif nested:
                    stack.append((iter(nested), comment_id, depth + 1))

        _validate_parent_graph(parents)
        returned_ids = {comment.reddit_id for comment in post.comments}
        if any(comment.parent_reddit_id and comment.parent_reddit_id not in returned_ids for comment in post.comments):
            diagnostics.add("comment_parent_not_returned")
        if post.num_comments > len(post.comments):
            diagnostics.add("fewer_comments_than_reported")
        return post
