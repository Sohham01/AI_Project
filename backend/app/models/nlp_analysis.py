from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Float, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class NLPPostAnalysis(Base):
    __tablename__ = "nlp_analysis_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    post_id: Mapped[int] = mapped_column(
        ForeignKey("reddit_posts.id"), nullable=False, index=True, unique=True
    )

    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    language_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    script: Mapped[str | None] = mapped_column(String(50), nullable=True)
    script_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_code_mixed: Mapped[bool] = mapped_column(Boolean, default=False)
    detected_languages: Mapped[str | None] = mapped_column(Text, nullable=True)
    detector_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    detector_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Preprocessing fields
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    preprocessing_version: Mapped[str | None] = mapped_column(String(50), nullable=True)


class NLPCommentAnalysis(Base):
    __tablename__ = "nlp_analysis_comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    comment_id: Mapped[int] = mapped_column(
        ForeignKey("reddit_comments.id"), nullable=False, index=True, unique=True
    )

    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    language_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    script: Mapped[str | None] = mapped_column(String(50), nullable=True)
    script_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_code_mixed: Mapped[bool] = mapped_column(Boolean, default=False)
    detected_languages: Mapped[str | None] = mapped_column(Text, nullable=True)
    detector_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    detector_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Preprocessing fields
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    preprocessing_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
