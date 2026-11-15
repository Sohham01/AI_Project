"""Guarded integration tests for the PostgreSQL Alembic history.

These tests require TEST_DATABASE_URL in backend/.env. They never fall back to
DATABASE_URL, and every schema reset verifies both configured and live identity.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import ipaddress
import os
from pathlib import Path
import shutil
from typing import Iterator

import pytest
from alembic import command
from alembic.config import Config
from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.sql.schema import UniqueConstraint
from sqlalchemy.sql.sqltypes import DateTime, Integer, String
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parent
ENV_FILE = BACKEND / ".env"
BASELINE_REVISION = "0c4a1b2d3e5f"
EXISTING_REVISION = "e9fa8685c983"
HEAD_REVISION = "8f1300a2a8f0"
EXPECTED_DATABASE = "ai_project_migration_test"
EXPECTED_ROLE = "ai_project_test"
EXPECTED_PORT = 5432


def _url_target(url: URL) -> tuple[str, int, str | None, str | None]:
    host = (url.host or "").lower().rstrip(".")
    if host in {"localhost", "127.0.0.1", "::1"}:
        host = "loopback"
    return host, url.port or 5432, url.database, url.username


def _load_test_url() -> tuple[URL, tuple[str, int, str | None, str | None]]:
    # Explicitly read only TEST_DATABASE_URL from backend/.env; no fallback.
    values = dotenv_values(ENV_FILE)
    raw_test_url = values.get("TEST_DATABASE_URL")
    if not raw_test_url:
        pytest.skip("TEST_DATABASE_URL is not configured in backend/.env")

    try:
        test_url = make_url(raw_test_url).set(drivername="postgresql+psycopg")
    except Exception as exc:
        pytest.fail(f"TEST_DATABASE_URL could not be parsed ({type(exc).__name__})")

    working_raw = os.environ.get("DATABASE_URL") or values.get("DATABASE_URL")
    if working_raw:
        try:
            working_url = make_url(working_raw)
        except Exception as exc:
            pytest.fail(f"Configured working database target could not be parsed ({type(exc).__name__})")
        if _url_target(test_url) == _url_target(working_url):
            pytest.fail("Safety refusal: TEST_DATABASE_URL matches the configured working database")

    target = _url_target(test_url)
    if target != ("loopback", EXPECTED_PORT, EXPECTED_DATABASE, EXPECTED_ROLE):
        pytest.fail("Safety refusal: TEST_DATABASE_URL is not the approved local test target")
    return test_url, target


def _assert_live_identity(connection, target: tuple[str, int, str | None, str | None]) -> None:
    database, role, server_address, server_port = connection.execute(
        text(
            "SELECT current_database(), current_user, "
            "inet_server_addr()::text, inet_server_port()"
        )
    ).one()
    if (database, role, server_port) != (EXPECTED_DATABASE, EXPECTED_ROLE, EXPECTED_PORT):
        raise RuntimeError("Safety refusal: connected PostgreSQL identity does not match the approved test target")
    if not server_address or not ipaddress.ip_interface(server_address).ip.is_loopback:
        raise RuntimeError("Safety refusal: PostgreSQL server address is not loopback")
    if target != ("loopback", EXPECTED_PORT, EXPECTED_DATABASE, EXPECTED_ROLE):
        raise RuntimeError("Safety refusal: configured test target changed during preflight")


@contextmanager
def _safe_connection() -> Iterator[object]:
    test_url, target = _load_test_url()
    engine = create_engine(test_url, hide_parameters=True, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            _assert_live_identity(connection, target)
            yield connection
    finally:
        engine.dispose()


def _reset_public_schema() -> None:
    # Re-parse TEST_DATABASE_URL and query database identity immediately before reset.
    with _safe_connection() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public AUTHORIZATION ai_project_test"))


def _alembic_config(script_location: Path | None = None) -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("prepend_sys_path", str(BACKEND))
    if script_location is not None:
        config.set_main_option("script_location", str(script_location))
    return config


def _run_upgrade(revision: str, *, script_location: Path | None = None) -> None:
    test_url, target = _load_test_url()
    with _safe_connection() as connection:
        _assert_live_identity(connection, target)
    # env.py uses DATABASE_URL; set it only to the explicit TEST_DATABASE_URL
    # for this command so it cannot load the configured working target.
    url_for_alembic = test_url.render_as_string(hide_password=False)
    with patch.dict(os.environ, {"DATABASE_URL": url_for_alembic}):
        command.upgrade(_alembic_config(script_location), revision)


def _create_historical_pre_source_schema() -> None:
    """Create the pre-e9fa tables as an existing deployment would have had them."""
    statements = (
        """CREATE TABLE public.data_sources (
            id SERIAL NOT NULL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            type VARCHAR(50) NOT NULL,
            status VARCHAR(30) NOT NULL,
            last_synced_at TIMESTAMP WITHOUT TIME ZONE NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
        )""",
        "CREATE INDEX ix_data_sources_id ON public.data_sources (id)",
        """CREATE TABLE public.reddit_posts (
            id SERIAL NOT NULL PRIMARY KEY,
            reddit_id VARCHAR(50) NOT NULL,
            title TEXT NOT NULL,
            body TEXT NULL,
            subreddit VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            score INTEGER NOT NULL,
            num_comments INTEGER NOT NULL,
            url TEXT NULL,
            collected_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
        )""",
        "CREATE INDEX ix_reddit_posts_id ON public.reddit_posts (id)",
        "CREATE UNIQUE INDEX ix_reddit_posts_reddit_id ON public.reddit_posts (reddit_id)",
        "CREATE INDEX ix_reddit_posts_subreddit ON public.reddit_posts (subreddit)",
        """CREATE TABLE public.reddit_comments (
            id SERIAL NOT NULL PRIMARY KEY,
            reddit_id VARCHAR(50) NOT NULL,
            post_id INTEGER NOT NULL REFERENCES public.reddit_posts(id),
            parent_comment_id INTEGER NULL REFERENCES public.reddit_comments(id),
            body TEXT NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            score INTEGER NOT NULL,
            collected_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
        )""",
        "CREATE INDEX ix_reddit_comments_id ON public.reddit_comments (id)",
        "CREATE UNIQUE INDEX ix_reddit_comments_reddit_id ON public.reddit_comments (reddit_id)",
        "CREATE INDEX ix_reddit_comments_post_id ON public.reddit_comments (post_id)",
        "CREATE INDEX ix_reddit_comments_parent_comment_id ON public.reddit_comments (parent_comment_id)",
    )
    with _safe_connection() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _historical_alembic_dir(tmp_path: Path) -> Path:
    """Materialize the pre-baseline graph without changing repository files."""
    script_dir = tmp_path / "historical_alembic"
    versions_dir = script_dir / "versions"
    versions_dir.mkdir(parents=True)
    shutil.copy2(BACKEND / "alembic" / "env.py", script_dir / "env.py")
    migration_names = (
        "e9fa8685c983_add_source_id_to_reddit_posts.py",
        "966e480ef159_add_nlp_analysis_tables.py",
        "17cb1444081b_add_detected_languages_column.py",
        "8f1300a2a8f0_add_preprocessing_columns.py",
    )
    for name in migration_names:
        source = BACKEND / "alembic" / "versions" / name
        content = source.read_text(encoding="utf-8")
        if name.startswith("e9fa"):
            current_parent = 'down_revision: Union[str, Sequence[str], None] = "0c4a1b2d3e5f"'
            if content.count(current_parent) != 1:
                raise RuntimeError("Could not derive the historical pre-baseline root revision")
            content = content.replace(current_parent, "down_revision: Union[str, Sequence[str], None] = None", 1)
        (versions_dir / name).write_text(content, encoding="utf-8")
    return script_dir


def _assert_final_model_schema(connection) -> None:
    # Import all five model tables only after ensuring backend is importable.
    import sys
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from app.core.database import Base
    import app.models  # noqa: F401

    inspector = inspect(connection)
    expected_table_names = set(Base.metadata.tables)
    actual_table_names = set(inspector.get_table_names(schema="public")) - {"alembic_version"}
    assert actual_table_names == expected_table_names

    for table_name, model_table in Base.metadata.tables.items():
        actual_columns = {column["name"]: column for column in inspector.get_columns(table_name, schema="public")}
        expected_columns = {column.name: column for column in model_table.columns}
        assert set(actual_columns) == set(expected_columns), table_name
        for column_name, expected in expected_columns.items():
            actual = actual_columns[column_name]
            assert actual["nullable"] == expected.nullable, (table_name, column_name)
            assert actual["type"]._type_affinity == expected.type._type_affinity, (table_name, column_name)
            if isinstance(expected.type, String) and expected.type.length is not None:
                assert actual["type"].length == expected.type.length, (table_name, column_name)
            if isinstance(expected.type, DateTime):
                assert actual["type"].timezone == expected.type.timezone, (table_name, column_name)
            if expected.server_default is None:
                if expected.primary_key and isinstance(expected.type, Integer) and expected.autoincrement is not False:
                    assert actual["default"] and "nextval(" in actual["default"], (table_name, column_name)
                else:
                    assert actual["default"] is None, (table_name, column_name)

        actual_pk = tuple((inspector.get_pk_constraint(table_name, schema="public").get("constrained_columns") or []))
        expected_pk = tuple(column.name for column in model_table.primary_key.columns)
        assert actual_pk == expected_pk, table_name

        expected_fks = {
            (tuple(fk.parent.name for fk in constraint.elements), constraint.referred_table.name,
             tuple(fk.column.name for fk in constraint.elements))
            for constraint in model_table.foreign_key_constraints
        }
        actual_fks = {
            (tuple(fk["constrained_columns"]), fk["referred_table"], tuple(fk["referred_columns"]))
            for fk in inspector.get_foreign_keys(table_name, schema="public")
        }
        assert actual_fks == expected_fks, table_name

        expected_indexes = {
            index.name: (tuple(column.name for column in index.columns), bool(index.unique))
            for index in model_table.indexes
        }
        actual_indexes = {
            index["name"]: (tuple(index["column_names"]), bool(index["unique"]))
            for index in inspector.get_indexes(table_name, schema="public")
        }
        assert actual_indexes == expected_indexes, table_name

        expected_unique_constraints = {
            tuple(column.name for column in constraint.columns)
            for constraint in model_table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        actual_unique_constraints = {
            tuple(constraint["column_names"])
            for constraint in inspector.get_unique_constraints(table_name, schema="public")
        }
        assert actual_unique_constraints == expected_unique_constraints, table_name


def test_existing_e9fa_database_upgrades_without_replaying_baseline(tmp_path):
    _reset_public_schema()
    _create_historical_pre_source_schema()

    # Execute the existing root migration against a manually constructed
    # pre-baseline schema using its historical root graph, then upgrade that
    # resulting e9fa database using the current graph.
    historical_dir = _historical_alembic_dir(tmp_path)
    _run_upgrade(EXISTING_REVISION, script_location=historical_dir)

    with _safe_connection() as connection:
        initial_revision = connection.execute(text("SELECT version_num FROM public.alembic_version")).scalar_one()
        assert initial_revision == EXISTING_REVISION
        source = connection.execute(
            text("INSERT INTO public.data_sources (name, type, status, created_at) "
                 "VALUES ('synthetic-migration-test', 'fixture', 'active', :created_at) RETURNING id"),
            {"created_at": datetime(2026, 1, 2, tzinfo=timezone.utc).replace(tzinfo=None)},
        ).scalar_one()
        marker = "synthetic-existing-row-phase-i2"
        post_id = connection.execute(
            text("INSERT INTO public.reddit_posts "
                 "(reddit_id, title, body, subreddit, created_at, score, num_comments, url, collected_at, source_id) "
                 "VALUES (:reddit_id, 'synthetic marker', 'synthetic body', 'migration_test', :created_at, "
                 "0, 0, 'https://example.invalid/synthetic', :created_at, :source_id) RETURNING id"),
            {"reddit_id": marker, "created_at": datetime(2026, 1, 2), "source_id": source},
        ).scalar_one()
        base_oids = dict(connection.execute(
            text("SELECT relname, oid FROM pg_class WHERE relnamespace='public'::regnamespace "
                 "AND relname IN ('data_sources','reddit_posts','reddit_comments')")
        ).all())
        assert set(base_oids) == {"data_sources", "reddit_posts", "reddit_comments"}

    _run_upgrade("head")

    with _safe_connection() as connection:
        revision = connection.execute(text("SELECT version_num FROM public.alembic_version")).scalar_one()
        assert revision == HEAD_REVISION
        preserved = connection.execute(
            text("SELECT id, reddit_id, title, source_id FROM public.reddit_posts WHERE id=:id"),
            {"id": post_id},
        ).one()
        assert tuple(preserved) == (post_id, marker, "synthetic marker", source)
        final_oids = dict(connection.execute(
            text("SELECT relname, oid FROM pg_class WHERE relnamespace='public'::regnamespace "
                 "AND relname IN ('data_sources','reddit_posts','reddit_comments')")
        ).all())
        assert final_oids == base_oids
        _assert_final_model_schema(connection)


def test_unversioned_existing_schema_is_refused_without_changes():
    _reset_public_schema()
    with _safe_connection() as connection:
        connection.execute(text(
            "CREATE TABLE public.data_sources (id INTEGER PRIMARY KEY, marker TEXT NOT NULL)"
        ))
        connection.execute(text(
            "INSERT INTO public.data_sources (id, marker) VALUES (31415, 'synthetic-unversioned-marker')"
        ))
        before_oid = connection.execute(text(
            "SELECT oid FROM pg_class WHERE oid='public.data_sources'::regclass"
        )).scalar_one()
        before_columns = tuple(connection.execute(text(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='data_sources' ORDER BY ordinal_position"
        )).all())
        before_row = tuple(connection.execute(text(
            "SELECT id, marker FROM public.data_sources WHERE id=31415"
        )).one())
        assert connection.execute(text(
            "SELECT to_regclass('public.alembic_version') IS NULL"
        )).scalar_one()

    with pytest.raises(RuntimeError) as failure:
        _run_upgrade("head")
    message = str(failure.value)
    assert "existing base table(s)" in message
    assert "Do not stamp automatically" in message
    assert "docs/database_migrations.md" in message

    with _safe_connection() as connection:
        after_oid = connection.execute(text(
            "SELECT oid FROM pg_class WHERE oid='public.data_sources'::regclass"
        )).scalar_one()
        after_columns = tuple(connection.execute(text(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='data_sources' ORDER BY ordinal_position"
        )).all())
        after_row = tuple(connection.execute(text(
            "SELECT id, marker FROM public.data_sources WHERE id=31415"
        )).one())
        revisions = connection.execute(
            text("SELECT version_num FROM public.alembic_version ORDER BY version_num")
        ).scalars().all() if inspect(connection).has_table("alembic_version", schema="public") else []
        assert after_oid == before_oid
        assert after_columns == before_columns
        assert after_row == before_row
        assert revisions == []
