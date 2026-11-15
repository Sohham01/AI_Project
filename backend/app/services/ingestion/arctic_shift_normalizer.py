from datetime import datetime, timezone
import math
from typing import Any
from urllib.parse import urlsplit

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.arctic_shift import ArcticShiftClient
from app.services.ingestion.providers.base import ProviderRequestError


class ArcticShiftNormalizer:
    @staticmethod
    def _utc_datetime(value: Any) -> datetime:
        try:
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, datetime):
                dt = value
            elif isinstance(value, (int, float)) and math.isfinite(value):
                dt = datetime.fromtimestamp(value, tz=timezone.utc)
            elif isinstance(value, str):
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            else:
                raise ValueError
            # Canonical ingestion timestamps are naive UTC.
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except (ValueError, OverflowError, OSError):
            raise ProviderRequestError("invalid_payload") from None

    @staticmethod
    def _clean_id(value: str) -> str:
        # Keep the compatibility helper prefix-only; validate identities below.
        for prefix in ("t3_", "t1_"):
            if value.startswith(prefix):
                return value[len(prefix):]
        return value

    @staticmethod
    def _identity(value: Any, kind: str) -> str:
        try:
            return ArcticShiftClient._id(value, kind)
        except ValueError:
            raise ProviderRequestError("invalid_payload") from None

    @staticmethod
    def _integer(value: Any, *, nonnegative: bool = False) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or (nonnegative and value < 0):
            raise ProviderRequestError("invalid_payload")
        return value

    @staticmethod
    def _text(value: Any, *, required: bool = False) -> str:
        if not isinstance(value, str) or (required and not value.strip()):
            raise ProviderRequestError("invalid_payload")
        return value

    @staticmethod
    def _url(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or any(char.isspace() for char in value):
            raise ProviderRequestError("invalid_payload")
        try:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError
            _ = parsed.port
        except ValueError:
            raise ProviderRequestError("invalid_payload") from None
        return value

    def normalize_post(self, raw_post: dict[str, Any]) -> PostData:
        if not isinstance(raw_post, dict):
            raise ProviderRequestError("invalid_payload")
        reddit_id = self._identity(raw_post.get("id"), "t3")
        body = raw_post.get("selftext")
        if body in ("", "[removed]", "[deleted]"):
            body = None
        if body is not None:
            body = self._text(body)
        subreddit = self._text(raw_post.get("subreddit"), required=True)
        if any(char.isspace() for char in subreddit) or "/" in subreddit:
            raise ProviderRequestError("invalid_payload")
        return PostData(
            reddit_id=reddit_id,
            title=self._text(raw_post.get("title"), required=True),
            body=body,
            subreddit=subreddit,
            created_at=self._utc_datetime(raw_post.get("created_utc")),
            score=self._integer(raw_post.get("score", 0)),
            num_comments=self._integer(raw_post.get("num_comments", 0), nonnegative=True),
            url=self._url(raw_post.get("url")),
            comments=[],
        )

    def normalize_comment(self, raw_comment: dict[str, Any]) -> CommentData:
        if not isinstance(raw_comment, dict):
            raise ProviderRequestError("invalid_payload")
        reddit_id = self._identity(raw_comment.get("id"), "t1")
        parent_id = raw_comment.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id.startswith(("t1_", "t3_")):
            raise ProviderRequestError("invalid_payload")
        parent = self._identity(parent_id, parent_id[:2])
        parent_reddit_id = parent if parent_id.startswith("t1_") else None
        if parent_reddit_id == reddit_id:
            raise ProviderRequestError("invalid_payload")
        return CommentData(
            reddit_id=reddit_id,
            body=self._text(raw_comment.get("body")),
            created_at=self._utc_datetime(raw_comment.get("created_utc")),
            score=self._integer(raw_comment.get("score", 0)),
            parent_reddit_id=parent_reddit_id,
        )
