from datetime import datetime

from pydantic import BaseModel, Field


class CommentData(BaseModel):
    reddit_id: str
    body: str
    created_at: datetime
    score: int = 0
    parent_reddit_id: str | None = None


class PostData(BaseModel):
    reddit_id: str
    title: str
    body: str | None = None
    subreddit: str
    created_at: datetime
    score: int = 0
    num_comments: int = 0
    url: str | None = None
    comments: list[CommentData] = Field(default_factory=list)