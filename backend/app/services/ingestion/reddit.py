"""Legacy Reddit ingestion entry point.

Historically this module contained its own Arctic Shift client usage plus a
duplicated comment-tree traversal.  The canonical implementation now lives in
:class:`app.services.ingestion.providers.arctic_shift.ArcticShiftProvider`; this
class is a thin compatibility adapter so existing callers keep working while
there is exactly one traversal/filtering implementation.
"""

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.base import IngestionSource
from app.services.ingestion.providers.arctic_shift import ArcticShiftProvider


class RedditIngestion(IngestionSource):

    def __init__(self) -> None:
        self._provider = ArcticShiftProvider()

    def fetch_posts(
            self,
            query: str,
            subreddit: str,
            limit: int = 25,
    ) -> list[PostData]:
        # ``query`` is forwarded to the Arctic Shift API (server-side search)
        # rather than downloading an unfiltered page and filtering locally.
        return self._provider.search_posts(
            query=query,
            subreddit=subreddit,
            limit=limit,
        )

    def fetch_comments(
            self,
            post_id: str,
    ) -> list[CommentData]:
        return self._provider.get_comments(post_id)
