from datetime import datetime, timezone
from typing import Any

from app.schemas.ingestion import CommentData, PostData


class ArcticShiftNormalizer:

    @staticmethod
    def _utc_datetime(value: Any) -> datetime:

        if isinstance(value, datetime):
            dt = value

        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(
                value,
                tz=timezone.utc,
            )

        elif isinstance(value, str):
            dt = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )

        else:
            raise ValueError(
                f"Unsupported datetime value: {value!r}"
            )

        # Our PostgreSQL columns currently use timestamp
        # rather than timestamp with time zone.
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(
                tzinfo=None
            )

        return dt

    @staticmethod
    def _clean_id(value: str) -> str:
        return value.replace("t3_", "").replace("t1_", "")

    def normalize_post(
        self,
        raw_post: dict[str, Any],
    ) -> PostData:

        reddit_id = self._clean_id(
            str(raw_post["id"])
        )

        body = raw_post.get("selftext")

        if body in ("", "[removed]", "[deleted]"):
            body = None

        return PostData(
            reddit_id=reddit_id,
            title=str(raw_post.get("title", "")),
            body=body,
            subreddit=str(
                raw_post.get("subreddit", "")
            ),
            created_at=self._utc_datetime(
                raw_post["created_utc"]
            ),
            score=int(
                raw_post.get("score", 0)
            ),
            num_comments=int(
                raw_post.get("num_comments", 0)
            ),
            url=raw_post.get("url"),
            comments=[],
        )

    def normalize_comment(
            self,
            raw_comment: dict[str, Any],
    ) -> CommentData:

        parent_id = raw_comment.get("parent_id")

        parent_reddit_id = None

        if parent_id and str(parent_id).startswith("t1_"):
            parent_reddit_id = self._clean_id(
                str(parent_id)
            )

        return CommentData(
            reddit_id=self._clean_id(
                str(raw_comment["id"])
            ),
            body=str(
                raw_comment.get("body", "")
            ),
            created_at=self._utc_datetime(
                raw_comment["created_utc"]
            ),
            score=int(
                raw_comment.get("score", 0)
            ),
            parent_reddit_id=parent_reddit_id,
        )