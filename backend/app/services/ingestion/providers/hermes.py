"""
Hermes-based ingestion provider for the IIM Sambalpur Community Intelligence pipeline.

## Unverified draft routes (not the Hermes gateway API contract)

This module contains a draft adapter sketch for custom scraping routes. The
local Hermes gateway is documented as OpenAI-compatible; it does not establish
that these Reddit-specific routes or browser tools are available. The provider
is disabled in manager capability dispatch until that contract exists.
"""

import os
from datetime import datetime, timezone
from typing import Any

import requests

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.providers.base import IngestionProvider, UnsupportedOperation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso_utc(value: Any) -> datetime:
    """Parse a Hermes ``created_utc`` field (ISO-8601 string or None) into a UTC datetime.

    Hermes is expected to return ISO-8601 strings like ``"2024-01-15T10:30:00Z"``.
    If the field is missing or unparseable we fall back to now — this is defensive,
    not correct, and should be treated as a signal that the Hermes response shape
    diverged from the contract above.
    """
    if not value:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Handle trailing 'Z' and common ISO variants without importing dateutil.
        cleaned = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _strip_default_field(value: Any, default: Any) -> Any:
    """Return *value* unless it equals its schema default, in which case return None.

    Hermes sometimes returns the literal default (e.g. score=0, num_comments=0) even
    when the field is genuinely absent. We treat a value equal to the schema default as
    "not provided" so the persistence layer can decide whether to trust it. This is a
    heuristic — tune it if Hermes starts returning meaningful zeros.
    """
    if value == default:
        return None
    return value


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class HermesProvider(IngestionProvider):
    """Disabled until the gateway exposes a verified Reddit scraping API."""

    def supports(self, operation: str) -> bool:
        # Current Hermes gateway contract is OpenAI-compatible only; custom
        # /search, /scrape, /comments, and /discover routes are unverified.
        return False



    def __init__(self) -> None:
        # Read from env on construction so a process-wide .env load (done in
        # app/core/database.py and expected at entry points) is picked up once.
        self.api_url: str | None = os.getenv("HERMES_API_URL")
        self.api_key: str | None = os.getenv("HERMES_API_KEY")

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, body: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
        """Send a POST to a Hermes endpoint and return the parsed JSON body.

        Raises:
            UnsupportedOperation: if Hermes is not configured or the response
                indicates an error (``ok: false`` or non-200 status).
        """
        if not self.api_url or not self.api_key:
            raise UnsupportedOperation("Hermes not configured")

        url = f"{self.api_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            resp = requests.post(url, json=body, headers=self._headers(), timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Hermes transport failed ({type(exc).__name__})"
            ) from None

        if resp.status_code == 404:
            raise UnsupportedOperation("Hermes endpoint unavailable (HTTP 404)")

        if not resp.ok:
            raise RuntimeError(f"Hermes request failed (HTTP {resp.status_code})")

        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError("Hermes returned invalid JSON") from None
        if not isinstance(payload, dict):
            raise RuntimeError("Hermes returned a non-object response")

        if not payload.get("ok"):
            raise RuntimeError("Hermes reported an application error")

        data = payload.get("data")
        if data is None:
            # ``ok: true`` but no ``data`` key — return an empty container of the
            # expected shape rather than None so callers can iterate safely.
            return {}
        return data

    # ------------------------------------------------------------------
    # IngestionProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "HermesProvider"

    # -- search_posts ---------------------------------------------------

    def search_posts(self, **kwargs: Any) -> list[PostData]:
        """Search Reddit for posts matching a natural-language query via Hermes.

        Accepted kwargs (all optional — pass only what you need):

        * ``query``             — search string (required for anything useful)
        * ``subreddit``         — restrict to a subreddit
        * ``limit``             — max posts to return (default 10, max 50)
        * ``recent_days``       — restrict to posts from the last N days (default 30)
        """
        query = kwargs.get("query")
        if not query:
            raise UnsupportedOperation(
                "Hermes search_posts requires a 'query' kwarg"
            )

        body: dict[str, Any] = {
            "query": str(query),
            "subreddit": kwargs.get("subreddit"),
            "limit": int(kwargs.get("limit", 10)),
            "recent_days": int(kwargs.get("recent_days", 30)),
        }

        data = self._post("search", body)
        raw_posts = data if isinstance(data, list) else data.get("posts", [])

        if not raw_posts:
            return []

        return [self._normalize_post_dict(p) for p in raw_posts]

    # -- get_post --------------------------------------------------------

    def get_post(self, reddit_id: str, **kwargs: Any) -> PostData:
        """Fetch a single post by its Reddit ID.

        Hermes needs the post URL to navigate, so you may pass it as a kwarg.
        If omitted, we construct the standard Reddit URL from the ID — this works
        for public posts but may fail for private/gated communities.
        """
        url = kwargs.get("url")
        if not url:
            url = f"https://www.reddit.com/r/unknown/comments/{reddit_id}/"

        body: dict[str, Any] = {
            "url": url,
            "include_comments": False,
        }

        data = self._post("scrape", body)
        return self._normalize_post_dict(data)

    # -- get_comments ---------------------------------------------------

    def get_comments(self, post_id: str, **kwargs: Any) -> list[CommentData]:
        """Fetch the comment tree for a post.

        Requires the post URL (passed as ``url`` kwarg) so Hermes can navigate to the
        comments section. The ``post_id`` is the Reddit post ID (e.g. ``t3_abc123``).
        """
        url = kwargs.get("url")
        if not url:
            raise UnsupportedOperation(
                "Hermes get_comments requires a 'url' kwarg for the post"
            )

        body: dict[str, Any] = {
            "reddit_id": post_id,
            "url": url,
            "limit": int(kwargs.get("limit", 500)),
        }

        data = self._post("comments", body)
        raw_comments = data if isinstance(data, list) else data.get("comments", [])

        return [self._normalize_comment_dict(c) for c in raw_comments]

    # -- scrape_url -----------------------------------------------------

    def scrape_url(self, url: str, **kwargs: Any) -> PostData:
        """Scrape a single Reddit post URL and return the post plus its top-level comments.

        This is the most commonly used Hermes operation — give it any Reddit post URL and
        it returns a fully-populated ``PostData`` with comments.

        Additional kwargs are forwarded to the Hermes ``/scrape`` endpoint:

        * ``include_comments`` — default True
        * ``timeout_seconds``  — default 60
        """
        include_comments = kwargs.get("include_comments", True)
        timeout_seconds = int(kwargs.get("timeout_seconds", 60))

        body: dict[str, Any] = {
            "url": url,
            "include_comments": bool(include_comments),
            "timeout_seconds": timeout_seconds,
        }

        data = self._post("scrape", body)
        return self._normalize_post_dict(data)

    # -- discover_targets -----------------------------------------------

    def discover_targets(self, prompt: str, **kwargs: Any) -> list[str]:
        """Ask Hermes to find Reddit post URLs matching a natural-language prompt.

        This is the most agentic operation — Hermes uses its browser + LLM to search
        Reddit and return URLs that match the prompt. Results are URLs only; scrape
        each one with ``scrape_url`` to get the actual post data.

        Additional kwargs:

        * ``subreddit`` — restrict to a subreddit
        * ``limit``     — max URLs to return (default 10, max 50)
        """
        body: dict[str, Any] = {
            "prompt": prompt,
            "subreddit": kwargs.get("subreddit"),
            "limit": int(kwargs.get("limit", 10)),
        }

        data = self._post("discover", body)
        raw_urls = data if isinstance(data, list) else data.get("urls", [])

        if not raw_urls:
            return []

        # Ensure every returned item is a string URL.
        urls: list[str] = []
        for item in raw_urls:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict):
                # Some Hermes response shapes may wrap URLs in objects.
                urls.append(str(item.get("url") or item.get("link") or item.get("post_url") or ""))
            else:
                urls.append(str(item))
        return urls

    # ------------------------------------------------------------------
    # Normalization — Hermes response dict -> PostData / CommentData
    # ------------------------------------------------------------------

    def _normalize_post_dict(self, data: dict[str, Any]) -> PostData:
        """Map a Hermes ``/scrape`` or ``/search`` post dict to ``PostData``.

        The Hermes contract (documented above) returns post objects with these fields.
        Any missing field is replaced with a sensible default so the rest of the pipeline
        never sees a KeyError.
        """
        raw_comments = data.get("comments", [])
        if not isinstance(raw_comments, list):
            raw_comments = []

        try:
            return PostData(
                reddit_id=str(data.get("reddit_id") or data.get("id") or ""),
                title=str(data.get("title") or ""),
                body=data.get("body"),  # may be None for link posts
                subreddit=str(data.get("subreddit") or "unknown"),
                created_at=_parse_iso_utc(data.get("created_utc")),
                score=int(data.get("score", 0)),
                num_comments=int(data.get("num_comments", 0)),
                url=str(data.get("url") or ""),
                comments=[self._normalize_comment_dict(c) for c in raw_comments],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Hermes post normalization failed ({type(exc).__name__})"
            ) from None

    def _normalize_comment_dict(self, data: dict[str, Any]) -> CommentData:
        """Map a Hermes comment dict to ``CommentData``.

        Hermes returns a flat comment list with ``parent_reddit_id`` links. We preserve
        that shape so the persistence layer can reconstruct the tree.
        """
        try:
            return CommentData(
                reddit_id=str(data.get("reddit_id") or data.get("id") or ""),
                body=str(data.get("body") or ""),
                created_at=_parse_iso_utc(data.get("created_utc")),
                score=int(data.get("score", 0)),
                parent_reddit_id=data.get("parent_reddit_id"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Hermes comment normalization failed ({type(exc).__name__})"
            ) from None
