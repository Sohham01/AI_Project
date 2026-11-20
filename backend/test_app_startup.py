"""Startup / schema-authority / health-endpoint tests.

These avoid ``fastapi.testclient`` (which needs ``httpx``); the route handlers
are plain functions and are called directly.
"""

import os
import pathlib
import re
import unittest
from unittest.mock import MagicMock, patch

from fastapi.responses import JSONResponse

from app.core.database import Base, ConfigurationError, get_database_url
import app.main as main


VERSIONS_DIR = pathlib.Path(__file__).parent / "alembic" / "versions"
REV_RE = re.compile(r"^revision(?::\s*str)?\s*=\s*['\"]([^'\"]+)['\"]", re.M)
DOWN_RE = re.compile(
    r"^down_revision[^=]*=\s*(None|['\"]([^'\"]+)['\"])", re.M
)


EXPECTED_TABLES = {
    "reddit_posts",
    "reddit_comments",
    "data_sources",
    "nlp_analysis_posts",
    "nlp_analysis_comments",
}


class TestSchemaMetadata(unittest.TestCase):
    def test_all_model_tables_registered(self):
        # Regression guard: nlp_analysis used to be missing from app.main's
        # imports, so its tables were absent from Base.metadata.
        self.assertTrue(EXPECTED_TABLES <= set(Base.metadata.tables))

    def test_auto_create_is_off_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTO_CREATE_SCHEMA", None)
            self.assertFalse(main._auto_create_enabled())

    def test_auto_create_flag_respected(self):
        with patch.dict(os.environ, {"AUTO_CREATE_SCHEMA": "true"}):
            self.assertTrue(main._auto_create_enabled())


class TestHealthEndpoints(unittest.TestCase):
    def test_root(self):
        self.assertIn("message", main.root())

    def test_health(self):
        self.assertEqual(main.health(), {"status": "ok"})

    def test_db_health_not_configured_returns_503(self):
        with patch.object(
            main, "get_engine",
            side_effect=ConfigurationError("password=never-print-this"),
        ):
            with self.assertLogs(main.logger, level="ERROR") as captured:
                response = main.database_health()
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(b"never-print-this", response.body)
        self.assertNotIn("never-print-this", " ".join(captured.output))

    def test_db_health_connected(self):
        result = MagicMock()
        result.scalar.return_value = 1
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.__exit__.return_value = False
        connection.execute.return_value = result
        engine = MagicMock()
        engine.connect.return_value = connection

        with patch.object(main, "get_engine", return_value=engine):
            response = main.database_health()

        self.assertEqual(response, {"database": "connected", "result": 1})

    def test_db_health_error_returns_503(self):
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("connection refused")
        with patch.object(main, "get_engine", return_value=engine):
            response = main.database_health()
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 503)

    def test_db_health_does_not_expose_exception_details(self):
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError(
            "password=private-secret host=internal-db.local"
        )
        with patch.object(main, "get_engine", return_value=engine):
            with self.assertLogs(main.logger, level="ERROR") as captured:
                response = main.database_health()

        self.assertEqual(response.status_code, 503)
        self.assertNotIn(b"private-secret", response.body)
        self.assertNotIn(b"internal-db.local", response.body)
        self.assertNotIn("private-secret", " ".join(captured.output))
        self.assertNotIn("internal-db.local", " ".join(captured.output))


class TestLazyDatabase(unittest.TestCase):
    def test_get_database_url_raises_without_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATABASE_URL", None)
            with self.assertRaises(ConfigurationError):
                get_database_url()

    def test_lazy_proxies_allow_mock_patch_without_database_url(self):
        import app.core.database as database_mod

        fake_session = MagicMock()
        fake_engine = MagicMock()
        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(database_mod, "_engine", None),
            patch.object(database_mod, "_session_factory", None),
        ):
            os.environ.pop("DATABASE_URL", None)
            with (
                patch("app.core.database.SessionLocal", return_value=fake_session),
                patch("app.core.database.engine", return_value=fake_engine),
            ):
                self.assertIs(database_mod.SessionLocal(), fake_session)
                self.assertIs(database_mod.engine(), fake_engine)


class TestSecurityHeaders(unittest.TestCase):
    def test_app_emits_defensive_security_headers(self):
        from fastapi.testclient import TestClient
        from app.core.security import BASE_SECURITY_HEADERS

        client = TestClient(main.app)
        for path in ("/health", "/app/", "/api/demo/summary"):
            res = client.get(path)
            self.assertEqual(res.status_code, 200, path)
            for key, expected in BASE_SECURITY_HEADERS.items():
                self.assertEqual(res.headers.get(key.lower()), expected, f"{path}: {key}")
            csp = res.headers.get("content-security-policy", "")
            self.assertIn("default-src 'self'", csp)
            self.assertIn("frame-ancestors 'none'", csp)


class TestMigrationGraph(unittest.TestCase):
    """The Alembic history must stay a single, linear, connected chain."""

    def _revisions(self):
        files = sorted(VERSIONS_DIR.glob("*.py"))
        revisions = {}
        downs = {}
        for path in files:
            text = path.read_text(encoding="utf-8")
            rev = REV_RE.search(text)
            down = DOWN_RE.search(text)
            self.assertIsNotNone(rev, f"no revision in {path.name}")
            self.assertIsNotNone(down, f"no down_revision in {path.name}")
            revisions[rev.group(1)] = path
            downs[rev.group(1)] = down.group(2)  # None or the parent id
        return revisions, downs

    def test_single_head_linear_chain(self):
        revisions, downs = self._revisions()
        parents = {d for d in downs.values() if d is not None}
        # Every parent must exist as a revision.
        self.assertTrue(parents <= set(revisions))
        heads = set(revisions) - parents
        self.assertEqual(len(heads), 1, f"expected one head, got {heads}")

    def test_nlp_tables_are_created_by_a_migration(self):
        texts = [
            p.read_text(encoding="utf-8")
            for p in VERSIONS_DIR.glob("*.py")
        ]
        blob = "\n".join(texts)
        for table in ("nlp_analysis_posts", "nlp_analysis_comments"):
            self.assertIn(f"create_table('{table}'", blob)


if __name__ == "__main__":
    unittest.main()
