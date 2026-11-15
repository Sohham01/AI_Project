"""Create the pre-source_id application tables for new databases.

Revision ID: 0c4a1b2d3e5f
Revises:

This is the original schema baseline *before* e9fa8685c983 adds
reddit_posts.source_id. Existing databases at e9fa8685c983 or later do not run
this migration when upgrading; their recorded revision already descends from it.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy import inspect


revision: str = "0c4a1b2d3e5f"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_BASE_TABLES = ("data_sources", "reddit_posts", "reddit_comments")


def _refuse_existing_unversioned_schema(bind) -> None:
    """Stop rather than guessing how to adopt tables without Alembic history."""
    present = sorted(name for name in _BASE_TABLES if inspect(bind).has_table(name))
    if present:
        raise RuntimeError(
            "Cannot apply the initial schema baseline: existing base table(s) "
            f"found without a recognized Alembic revision: {', '.join(present)}. "
            "Do not stamp automatically. Compare the live schema with the "
            "revision history and follow docs/database_migrations.md."
        )


def upgrade() -> None:
    """Create only the schema that predates the existing source-id revision."""
    if not context.is_offline_mode():
        _refuse_existing_unversioned_schema(op.get_bind())

    op.create_table(
        "data_sources",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("last_synced_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_data_sources_id", "data_sources", ["id"], unique=False)

    op.create_table(
        "reddit_posts",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("reddit_id", sa.String(length=50), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("subreddit", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("num_comments", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("collected_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_reddit_posts_id", "reddit_posts", ["id"], unique=False)
    op.create_index(
        "ix_reddit_posts_reddit_id", "reddit_posts", ["reddit_id"], unique=True
    )
    op.create_index(
        "ix_reddit_posts_subreddit", "reddit_posts", ["subreddit"], unique=False
    )

    op.create_table(
        "reddit_comments",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("reddit_id", sa.String(length=50), nullable=False),
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("parent_comment_id", sa.Integer(), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("collected_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["post_id"], ["reddit_posts.id"]),
        sa.ForeignKeyConstraint(["parent_comment_id"], ["reddit_comments.id"]),
    )
    op.create_index("ix_reddit_comments_id", "reddit_comments", ["id"], unique=False)
    op.create_index(
        "ix_reddit_comments_reddit_id",
        "reddit_comments",
        ["reddit_id"],
        unique=True,
    )
    op.create_index(
        "ix_reddit_comments_post_id",
        "reddit_comments",
        ["post_id"],
        unique=False,
    )
    op.create_index(
        "ix_reddit_comments_parent_comment_id",
        "reddit_comments",
        ["parent_comment_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop baseline tables in foreign-key-safe order."""
    op.drop_table("reddit_comments")
    op.drop_table("reddit_posts")
    op.drop_table("data_sources")
