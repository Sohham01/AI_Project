from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class RedditComment(Base):
    __tablename__ = "reddit_comments"

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

    post_id: Mapped[int] = mapped_column(
        ForeignKey("reddit_posts.id"),
        nullable=False,
        index=True
    )

    parent_comment_id: Mapped[int | None] = mapped_column(
        ForeignKey("reddit_comments.id"),
        nullable=True,
        index=True
    )

    body: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False
    )

    score: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    collected_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )