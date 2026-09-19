from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class RedditPost(Base):
    __tablename__ = "reddit_posts"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    reddit_id: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    body: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    subreddit: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False
    )

    score: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    num_comments: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    collected_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    source_id: Mapped[int] = mapped_column(
        ForeignKey("data_sources.id"),
        nullable=False,
        index=True
    )