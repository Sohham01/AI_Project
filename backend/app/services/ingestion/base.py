from abc import ABC, abstractmethod

from app.schemas.ingestion import PostData


class IngestionSource(ABC):

    @abstractmethod
    def fetch_posts(self, **kwargs) -> list[PostData]:
        """Fetch and normalize posts from the source."""
        raise NotImplementedError

    @abstractmethod
    def fetch_comments(self, post_id: str) -> list[dict]:
        """Fetch comments belonging to a post."""
        raise NotImplementedError