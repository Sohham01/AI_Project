import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.ingestion.manager import IngestionManager
from app.services.ingestion.providers.base import (
    IngestionProvider,
    ProviderOperationError,
    ProviderRequestError,
    UnsupportedOperation,
)


class _StubProvider(IngestionProvider):
    def __init__(self, *, supports, result=None, error=None):
        self._supports = supports
        self._result = result
        self._error = error

    def supports(self, operation: str) -> bool:
        return operation in self._supports

    def search_posts(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._result


class TestIngestionManager(unittest.TestCase):
    def setUp(self):
        # These tests must not depend on (or inspect) local provider credentials.
        # Individual configured cases set synthetic values explicitly below.
        configuration = patch(
            "app.services.ingestion.providers.decodo.os.getenv",
            side_effect=lambda name, default=None: default,
        )
        configuration.start()
        self.addCleanup(configuration.stop)
        self.manager = IngestionManager()

    def test_arctic_shift_client_can_be_imported_first_in_fresh_process(self):
        backend_dir = Path(__file__).resolve().parent
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from app.services.ingestion.arctic_shift import ArcticShiftClient; "
                "from app.services.ingestion.providers import (ArcticShiftProvider, "
                "RSSProvider, HermesProvider, DecodoProvider); "
                "assert all((ArcticShiftClient, ArcticShiftProvider, RSSProvider, "
                "HermesProvider, DecodoProvider))",
            ],
            cwd=backend_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_get_provider(self):
        provider = self.manager.get_provider("arctic_shift")
        self.assertIsNotNone(provider)

    def test_get_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            self.manager.get_provider("nope")

    def test_hermes_is_not_advertised_as_a_supported_provider(self):
        hermes = self.manager.get_provider("hermes")
        self.assertTrue(all(not hermes.supports(op) for op in (
            "search_posts", "get_post", "get_comments", "scrape_url", "scrape_source"
        )))
        self.assertEqual(next(iter(self.manager.providers)), "arctic_shift")
        self.assertTrue(self.manager.get_provider("arctic_shift").supports("search_posts"))
        self.assertTrue(self.manager.get_provider("arctic_shift").supports("get_comments"))
        with self.assertRaises(UnsupportedOperation):
            self.manager.execute_operation(
                "search_posts", provider="hermes", query="IIM Sambalpur"
            )

    def test_execute_operation_unsupported(self):
        with self.assertRaises(UnsupportedOperation):
            self.manager.execute_operation("scrape_url", url="http://invalid.url")

    def test_configured_decodo_invalid_url_is_safe_and_makes_no_request(self):
        self.manager.get_provider("decodo").api_token = "offline-test-key"
        with patch("app.services.ingestion.providers.decodo.requests.post") as request:
            with self.assertRaises(ProviderOperationError) as caught:
                self.manager.execute_operation("scrape_url", url="http://invalid.url")
        request.assert_not_called()
        self.assertEqual(caught.exception.failure_category, "value_error")
        self.assertEqual(caught.exception.provider, "decodo")
        self.assertNotIn("invalid.url", str(caught.exception))
        self.assertNotIn("offline-test-key", str(caught.exception))

    def test_unknown_operation_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.manager.execute_operation("launch_rocket")

    def test_capability_dispatch_uses_supporting_provider(self):
        arctic = MagicMock()
        arctic.supports.return_value = True
        arctic.search_posts.return_value = ["post"]
        # A provider that does NOT support the operation must never be called.
        noise = MagicMock()
        noise.supports.return_value = False
        self.manager.providers = {"noise": noise, "arctic_shift": arctic}

        result = self.manager.execute_operation("search_posts", query="x")

        self.assertEqual(result, ["post"])
        arctic.search_posts.assert_called_once_with(query="x")
        noise.search_posts.assert_not_called()

    def test_provider_errors_are_surfaced(self):
        failing = _StubProvider(
            supports={"search_posts"}, error=ValueError("boom")
        )
        self.manager.providers = {"failing": failing}
        with self.assertRaises(RuntimeError) as ctx:
            self.manager.execute_operation("search_posts")
        self.assertIn("value_error", str(ctx.exception))
        self.assertNotIn("boom", str(ctx.exception))

    def test_safe_http_metadata_survives_manager_wrapping(self):
        failing = _StubProvider(
            supports={"search_posts"},
            error=ProviderRequestError("http_status", 503),
        )
        self.manager.providers = {"arctic_shift": failing}
        with self.assertRaises(RuntimeError) as ctx:
            self.manager.execute_operation("search_posts", provider="arctic_shift")
        self.assertEqual(ctx.exception.failure_category, "http_status")
        self.assertEqual(ctx.exception.status_code, 503)

    def test_provider_error_details_are_suppressed(self):
        failing = _StubProvider(
            supports={"search_posts"},
            error=ValueError("Authorization: Bearer never-print-this"),
        )
        self.manager.providers = {"failing": failing}
        with self.assertRaises(RuntimeError) as ctx:
            self.manager.execute_operation("search_posts", provider="failing")
        self.assertNotIn("never-print-this", str(ctx.exception))
        self.assertEqual(ctx.exception.__cause__, None)
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_explicit_provider_selection(self):
        ok = MagicMock()
        ok.supports.return_value = True
        ok.search_posts.return_value = ["ok"]
        self.manager.providers = {"arctic_shift": ok}

        result = self.manager.execute_operation(
            "search_posts", provider="arctic_shift", query="q"
        )

        self.assertEqual(result, ["ok"])
        ok.search_posts.assert_called_once_with(query="q")

    def test_explicit_provider_that_does_not_support_raises(self):
        provider = MagicMock()
        provider.supports.return_value = False
        self.manager.providers = {"arctic_shift": provider}

        with self.assertRaises(UnsupportedOperation):
            self.manager.execute_operation(
                "search_posts", provider="arctic_shift"
            )

    def test_unconfigured_provider_falls_through(self):
        fallback = _StubProvider(supports={"search_posts"}, result=["fallback"])
        unconfigured = _StubProvider(
            supports={"search_posts"}, error=UnsupportedOperation("no key")
        )
        self.manager.providers = {
            "unconfigured": unconfigured,
            "fallback": fallback,
        }
        self.assertEqual(
            self.manager.execute_operation("search_posts"), ["fallback"]
        )


if __name__ == "__main__":
    unittest.main()
