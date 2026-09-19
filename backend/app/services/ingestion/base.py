from abc import ABC, abstractmethod


class IngestionSource(ABC):

    @abstractmethod
    def fetch_posts(self):
        """Fetch posts from the source."""
        pass

    @abstractmethod
    def fetch_comments(self, post_id: str):
        """Fetch comments for a post."""
        pass