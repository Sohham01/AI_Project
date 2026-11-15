from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text


BACKEND = Path(__file__).resolve().parent
BASELINE_FILE = BACKEND / "alembic" / "versions" / "0c4a1b2d3e5f_create_base_tables.py"
BASELINE_ID = "0c4a1b2d3e5f"
FIRST_EXISTING_ID = "e9fa8685c983"


def _alembic_config() -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("prepend_sys_path", str(BACKEND))
    return config


def _render_postgresql_upgrade(revision: str) -> str:
    # Offline mode renders PostgreSQL DDL and never opens a database connection.
    output = io.StringIO()
    with patch.dict(
        os.environ,
        {"DATABASE_URL": "postgresql://migration_test:offline@127.0.0.1/migration_test"},
    ), contextlib.redirect_stdout(output):
        command.upgrade(_alembic_config(), revision, sql=True)
    return output.getvalue()


def test_migration_graph_has_one_baseline_and_preserves_existing_revision_ids():
    script = ScriptDirectory.from_config(_alembic_config())
    revisions = {revision.revision: revision for revision in script.walk_revisions()}

    assert script.get_heads() == ["8f1300a2a8f0"]
    assert revisions[BASELINE_ID].down_revision is None
    assert revisions[FIRST_EXISTING_ID].down_revision == BASELINE_ID
    assert revisions["966e480ef159"].down_revision == FIRST_EXISTING_ID
    assert revisions["17cb1444081b"].down_revision == "966e480ef159"
    assert revisions["8f1300a2a8f0"].down_revision == "17cb1444081b"
    assert set(revisions) == {
        BASELINE_ID,
        FIRST_EXISTING_ID,
        "966e480ef159",
        "17cb1444081b",
        "8f1300a2a8f0",
    }


def test_fresh_postgresql_upgrade_sql_creates_base_schema_before_alterations():
    sql = _render_postgresql_upgrade("head")
    sources = sql.index("CREATE TABLE data_sources")
    posts = sql.index("CREATE TABLE reddit_posts")
    comments = sql.index("CREATE TABLE reddit_comments")
    source_column = sql.index("ADD COLUMN source_id")
    nlp_posts = sql.index("CREATE TABLE nlp_analysis_posts")
    nlp_comments = sql.index("CREATE TABLE nlp_analysis_comments")

    assert sources < posts < comments < source_column < nlp_posts < nlp_comments
    assert "PRIMARY KEY (id)" in sql
    expected_columns = (
        "name VARCHAR(100) NOT NULL",
        "type VARCHAR(50) NOT NULL",
        "status VARCHAR(30) NOT NULL",
        "last_synced_at TIMESTAMP WITHOUT TIME ZONE",
        "reddit_id VARCHAR(50) NOT NULL",
        "title TEXT NOT NULL",
        "body TEXT",
        "subreddit VARCHAR(100) NOT NULL",
        "num_comments INTEGER NOT NULL",
        "parent_comment_id INTEGER",
        "language_confidence FLOAT",
        "script_confidence FLOAT",
        "is_code_mixed BOOLEAN NOT NULL",
        "analyzed_at TIMESTAMP WITHOUT TIME ZONE NOT NULL",
        "detected_languages TEXT",
        "normalized_text TEXT",
        "preprocessing_version VARCHAR(50)",
    )
    assert all(column in sql for column in expected_columns)
    expected_indexes = (
        "ix_data_sources_id",
        "ix_reddit_posts_id",
        "CREATE UNIQUE INDEX ix_reddit_posts_reddit_id",
        "ix_reddit_posts_subreddit",
        "ix_reddit_posts_source_id",
        "ix_reddit_comments_id",
        "CREATE UNIQUE INDEX ix_reddit_comments_reddit_id",
        "ix_reddit_comments_post_id",
        "ix_reddit_comments_parent_comment_id",
        "CREATE UNIQUE INDEX ix_nlp_analysis_posts_post_id",
        "CREATE UNIQUE INDEX ix_nlp_analysis_comments_comment_id",
    )
    assert all(index in sql for index in expected_indexes)
    assert "FOREIGN KEY(post_id) REFERENCES reddit_posts (id)" in sql
    assert "FOREIGN KEY(parent_comment_id) REFERENCES reddit_comments (id)" in sql
    assert "FOREIGN KEY(source_id) REFERENCES data_sources (id)" in sql
    assert "FOREIGN KEY(post_id) REFERENCES reddit_posts (id)" in sql
    assert "FOREIGN KEY(comment_id) REFERENCES reddit_comments (id)" in sql
    assert "reddit_posts_source_id_fkey" in sql
    assert "ALTER COLUMN source_id SET NOT NULL" in sql
    assert "detected_languages" in sql
    assert "normalized_text" in sql
    assert "preprocessing_version" in sql
    assert "DEFAULT " not in sql


def test_valid_existing_revision_upgrade_sql_does_not_recreate_base_tables():
    sql = _render_postgresql_upgrade(f"{FIRST_EXISTING_ID}:head")

    assert "CREATE TABLE data_sources" not in sql
    assert "CREATE TABLE reddit_posts" not in sql
    assert "CREATE TABLE reddit_comments" not in sql
    assert "ADD COLUMN source_id" not in sql
    assert "CREATE TABLE nlp_analysis_posts" in sql
    assert "CREATE TABLE nlp_analysis_comments" in sql
    assert "normalized_text" in sql


def test_baseline_refuses_to_adopt_existing_tables_without_revision():
    spec = importlib.util.spec_from_file_location("migration_baseline_test", BASELINE_FILE)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = migration
    spec.loader.exec_module(migration)

    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE reddit_posts (id INTEGER PRIMARY KEY)"))
            with pytest.raises(RuntimeError, match="without a recognized Alembic revision"):
                migration._refuse_existing_unversioned_schema(connection)
            assert connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='reddit_posts'")
            ).scalar_one() == "reddit_posts"
            assert connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='data_sources'")
            ).scalar_one_or_none() is None
    finally:
        engine.dispose()
