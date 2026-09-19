"""add source id to reddit posts

Revision ID: e9fa8685c983
Revises:
Create Date: 2026-09-19 11:11:00.713345

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e9fa8685c983"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # Add the column first as nullable
    op.add_column(
        "reddit_posts",
        sa.Column("source_id", sa.Integer(), nullable=True)
    )

    # Add the foreign key
    op.create_foreign_key(
        "reddit_posts_source_id_fkey",
        "reddit_posts",
        "data_sources",
        ["source_id"],
        ["id"]
    )

    # Add an index for faster lookups
    op.create_index(
        "ix_reddit_posts_source_id",
        "reddit_posts",
        ["source_id"],
        unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(
        "ix_reddit_posts_source_id",
        table_name="reddit_posts"
    )

    op.drop_constraint(
        "reddit_posts_source_id_fkey",
        "reddit_posts",
        type_="foreignkey"
    )

    op.drop_column(
        "reddit_posts",
        "source_id"
    )