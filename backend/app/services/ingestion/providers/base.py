from abc import ABC
from typing import Any, Optional

from app.schemas.ingestion import CommentData, PostData

class UnsupportedOperation(Exception):
    pass

class IngestionProvider(ABC):
    
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def search_posts(self, **kwargs: Any) -> list[PostData]:
        raise UnsupportedOperation(f"{self.name} does not support search_posts")

    def get_post(self, reddit_id: str) -> PostData:
        raise UnsupportedOperation(f"{self.name} does not support get_post")

    def get_comments(self, post_id: str) -> list[CommentData]:
        raise UnsupportedOperation(f"{self.name} does not support get_comments")

    def scrape_url(self, url: str) -> PostData:
        raise UnsupportedOperation(f"{self.name} does not support scrape_url")

    def discover_targets(self, prompt: str) -> list[str]:
        raise UnsupportedOperation(f"{self.name} does not support discover_targets")
