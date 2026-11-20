"""Official Reddit Data API (OAuth2) provider for bounded research ingestion.

Uses Reddit's official OAuth2 application-only flow (``client_credentials``)
against ``https://oauth.reddit.com`` so requests authenticate cleanly rather
than hitting unauthenticated ``403`` blocks on ``www.reddit.com/*.json``.

Configuration (via environment variables — never hardcoded or logged):
  - ``REDDIT_CLIENT_ID``     (required)
  - ``REDDIT_CLIENT_SECRET`` (required)
  - ``REDDIT_USER_AGENT``    (optional; defaults to a descriptive academic UA)

When credentials are not configured, ``supports()`` remains ``True`` for the
implemented operations while calls raise ``UnsupportedOperation`` cleanly,
matching the unconfigured-provider contract of ``IngestionManager``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import re
import time
from typing import Any, Optional

import requests

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.providers.base import (
    IngestionProvider,
    ProviderRequestError,
    UnsupportedOperation,
)

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_API_BASE = "https://oauth.reddit.com"
_SUBREDDIT_RE = re.compile(r"^[A-Za-z0-9_]{1,21}$")
_ID_RE = re.compile(r"^[a-z0-9]+$")


class RedditOAuthProvider(IngestionProvider):
    """Official Reddit OAuth2 API provider implementing ``search_posts``, ``get_post``, and ``get_comments``."""

    MAX_POSTS = 100
    MAX_COMMENTS = 500
    DEFAULT_TIMEOUT = 20.0

    def __init__(
        self,
        *,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self.timeout = float(timeout)
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._session = requests.Session()
        self.last_comments_truncated = False

    def _credentials(self) -> tuple[str, str, str]:
        cid = (
            self._client_id
            if self._client_id is not None
            else os.getenv("REDDIT_CLIENT_ID", "")
        ).strip()
        secret = (
            self._client_secret
            if self._client_secret is not None
            else os.getenv("REDDIT_CLIENT_SECRET", "")
        ).strip()
        ua = (
            self._user_agent
            if self._user_agent is not None
            else os.getenv(
                "REDDIT_USER_AGENT",
                "python:rci-academic-research:v1.0 (IIM Sambalpur NLP research)",
            )
        ).strip()
        if not cid or not secret:
            raise UnsupportedOperation(
                "RedditOAuthProvider is not configured (set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)"
            )
        return cid, secret, ua

    @property
    def is_configured(self) -> bool:
        try:
            self._credentials()
            return True
        except UnsupportedOperation:
            return False

    def _ensure_token(self, timeout: float) -> tuple[str, str]:
        cid, secret, ua = self._credentials()
        now = time.monotonic()
        if self._access_token and now < self._token_expires_at - 30.0:
            return self._access_token, ua

        try:
            response = self._session.post(
                _TOKEN_URL,
                auth=(cid, secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": ua},
                timeout=timeout,
                allow_redirects=False,
            )
        except requests.Timeout:
            raise ProviderRequestError("timeout") from None
        except requests.ConnectionError:
            raise ProviderRequestError("connection") from None
        except requests.RequestException:
            raise ProviderRequestError("transport") from None

        if not 200 <= response.status_code < 300:
            raise ProviderRequestError("http_status", response.status_code)
        try:
            payload = response.json()
        except ValueError:
            raise ProviderRequestError("invalid_json") from None

        token = payload.get("access_token") if isinstance(payload, dict) else None
        expires_in = payload.get("expires_in", 3600) if isinstance(payload, dict) else 3600
        if not isinstance(token, str) or not token.strip():
            raise ProviderRequestError("invalid_payload")
        self._access_token = token.strip()
        self._token_expires_at = now + max(float(expires_in), 60.0)
        return self._access_token, ua

    def _api_get(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        req_timeout = self.timeout if timeout is None else float(timeout)
        token, ua = self._ensure_token(req_timeout)
        try:
            response = self._session.get(
                f"{_API_BASE}{path}",
                params=params,
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": ua,
                    "Accept": "application/json",
                },
                timeout=req_timeout,
                allow_redirects=False,
            )
        except requests.Timeout:
            raise ProviderRequestError("timeout") from None
        except requests.ConnectionError:
            raise ProviderRequestError("connection") from None
        except requests.RequestException:
            raise ProviderRequestError("transport") from None

        if not 200 <= response.status_code < 300:
            raise ProviderRequestError("http_status", response.status_code)
        try:
            return response.json()
        except ValueError:
            raise ProviderRequestError("invalid_json") from None

    @staticmethod
    def _clean_subreddit(subreddit: Any) -> str:
        if not isinstance(subreddit, str):
            raise ValueError("subreddit must be a string")
        clean = subreddit.strip().removeprefix("r/").removeprefix("/r/").strip("/")
        if not _SUBREDDIT_RE.fullmatch(clean):
            raise ValueError("invalid subreddit name")
        return clean

    @staticmethod
    def _clean_id(value: Any, prefix: str) -> str:
        if not isinstance(value, str):
            raise ValueError("invalid Reddit identity")
        clean = value.strip().removeprefix(f"{prefix}_").lower()
        if not _ID_RE.fullmatch(clean):
            raise ValueError("invalid Reddit identity")
        return clean

    @staticmethod
    def _utc_from_epoch(value: Any) -> datetime:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProviderRequestError("invalid_payload")
        return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)

    def _normalize_post(self, raw: dict[str, Any]) -> PostData:
        if not isinstance(raw, dict):
            raise ProviderRequestError("invalid_payload")
        post_id = self._clean_id(raw.get("id", ""), "t3")
        title = str(raw.get("title") or "").strip()
        subreddit = str(raw.get("subreddit") or "").strip()
        if not title or not subreddit:
            raise ProviderRequestError("invalid_payload")
        created_at = self._utc_from_epoch(raw.get("created_utc"))
        body = raw.get("selftext")
        body_clean = body.strip() if isinstance(body, str) and body.strip() else None
        permalink = raw.get("permalink")
        url = (
            f"https://www.reddit.com{permalink}"
            if isinstance(permalink, str) and permalink.startswith("/")
            else (raw.get("url") if isinstance(raw.get("url"), str) else None)
        )
        return PostData(
            reddit_id=post_id,
            title=title,
            body=body_clean,
            subreddit=subreddit,
            created_at=created_at,
            score=int(raw.get("score") or 0),
            num_comments=int(raw.get("num_comments") or 0),
            url=url,
            comments=[],
        )

    def search_posts(self, **kwargs: Any) -> list[PostData]:
        """Search a subreddit (with optional ``query``) or list newest posts via OAuth2."""
        subreddit = kwargs.get("subreddit")
        query = kwargs.get("query")
        limit = int(kwargs.get("limit", 25))
        timeout = kwargs.get("timeout")
        sort = str(kwargs.get("sort", "new"))
        if sort == "desc":
            sort = "new"
        elif sort not in {"new", "relevance", "top", "hot", "comments"}:
            sort = "new"
        if not 1 <= limit <= self.MAX_POSTS:
            raise ValueError(f"limit must be between 1 and {self.MAX_POSTS}")

        clean_sub = self._clean_subreddit(subreddit) if subreddit is not None else None
        clean_query = query.strip() if isinstance(query, str) and query.strip() else None

        if clean_query:
            path = f"/r/{clean_sub}/search" if clean_sub else "/search"
            params: dict[str, Any] = {
                "q": clean_query,
                "sort": sort,
                "limit": limit,
                "raw_json": 1,
            }
            if clean_sub:
                params["restrict_sr"] = "on"
        else:
            if not clean_sub:
                raise ValueError("subreddit is required when query is omitted")
            path = f"/r/{clean_sub}/new"
            params = {"limit": limit, "raw_json": 1}

        payload = self._api_get(path, params=params, timeout=timeout)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise ProviderRequestError("invalid_payload")
        children = payload["data"].get("children")
        if not isinstance(children, list):
            raise ProviderRequestError("invalid_payload")

        posts: dict[str, PostData] = {}
        for child in children:
            if not isinstance(child, dict) or child.get("kind") != "t3":
                continue
            raw = child.get("data")
            if not isinstance(raw, dict):
                continue
            post = self._normalize_post(raw)
            posts.setdefault(post.reddit_id, post)
        return list(posts.values())[:limit]

    def get_post(self, reddit_id: str) -> PostData:
        clean_id = self._clean_id(reddit_id, "t3")
        payload = self._api_get("/api/info", params={"id": f"t3_{clean_id}", "raw_json": 1})
        children = (
            payload.get("data", {}).get("children", [])
            if isinstance(payload, dict)
            else []
        )
        if not children or not isinstance(children[0], dict):
            raise ValueError("Reddit post not found")
        return self._normalize_post(children[0].get("data", {}))

    def get_comments(
        self,
        post_id: str,
        timeout: float | None = None,
        limit: int = 100,
    ) -> list[CommentData]:
        """Fetch and flatten the comment tree for a single post via OAuth2."""
        clean_id = self._clean_id(post_id, "t3")
        if not 1 <= limit <= self.MAX_COMMENTS:
            raise ValueError(f"limit must be between 1 and {self.MAX_COMMENTS}")
        self.last_comments_truncated = False

        payload = self._api_get(
            f"/comments/{clean_id}",
            params={"limit": limit, "depth": 8, "sort": "confidence", "raw_json": 1},
            timeout=timeout,
        )
        if not isinstance(payload, list) or len(payload) < 2:
            raise ProviderRequestError("invalid_payload")
        comment_listing = payload[1]
        if not isinstance(comment_listing, dict) or not isinstance(
            comment_listing.get("data", {}).get("children"), list
        ):
            raise ProviderRequestError("invalid_payload")

        comments: list[CommentData] = []
        seen: set[str] = set()
        stack = list(reversed(comment_listing["data"]["children"]))
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            kind = node.get("kind")
            data = node.get("data")
            if kind == "more":
                self.last_comments_truncated = True
                continue
            if kind != "t1" or not isinstance(data, dict):
                continue
            cid_raw = data.get("id")
            if not isinstance(cid_raw, str) or not cid_raw.strip():
                continue
            cid = self._clean_id(cid_raw, "t1")
            if cid in seen:
                continue
            seen.add(cid)
            body = str(data.get("body") or "").strip()
            parent_raw = str(data.get("parent_id") or "")
            parent_id = (
                self._clean_id(parent_raw, "t1")
                if parent_raw.startswith("t1_")
                else None
            )
            if body and body not in {"[deleted]", "[removed]"}:
                created_at = self._utc_from_epoch(data.get("created_utc"))
                comments.append(
                    CommentData(
                        reddit_id=cid,
                        body=body,
                        created_at=created_at,
                        score=int(data.get("score") or 0),
                        parent_reddit_id=parent_id,
                    )
                )
            replies = data.get("replies")
            if isinstance(replies, dict):
                children = replies.get("data", {}).get("children", [])
                if isinstance(children, list):
                    stack.extend(reversed(children))

        if len(comments) > limit:
            self.last_comments_truncated = True
        return comments[:limit]
