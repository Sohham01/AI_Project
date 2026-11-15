from abc import ABC
import re
from typing import Any, Optional

from app.schemas.ingestion import CommentData, PostData

class UnsupportedOperation(Exception):
    pass


_SAFE_FAILURE_CATEGORIES = {
    "http_status",
    "timeout",
    "connection",
    "transport",
    "invalid_json",
    "invalid_payload",
    "invalid_xml",
}


class ProviderRequestError(RuntimeError):
    """Sanitized provider request failure with safe retry metadata."""

    def __init__(self, failure_category: str, status_code: Optional[int] = None):
        if (
            not isinstance(failure_category, str)
            or failure_category not in _SAFE_FAILURE_CATEGORIES
        ):
            failure_category = "provider_error"
        if (
            isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or not 100 <= status_code <= 599
        ):
            status_code = None
        self.failure_category = failure_category
        self.status_code = status_code
        status = f" HTTP {status_code}" if status_code is not None else ""
        super().__init__(f"provider request failed ({failure_category}){status}")


def safe_failure_metadata(exc: BaseException) -> tuple[str, Optional[int]]:
    """Return bounded category/status metadata without inspecting error text."""
    try:
        category = getattr(exc, "failure_category", None)
    except Exception:  # noqa: BLE001 - metadata is optional and untrusted
        category = None
    if not isinstance(category, str) or category not in _SAFE_FAILURE_CATEGORIES:
        category = re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).lower()
        category = re.sub(r"[^a-z0-9_]+", "_", category)[:48].strip("_") or "provider_error"

    try:
        status = getattr(exc, "status_code", None)
    except Exception:  # noqa: BLE001 - metadata is optional and untrusted
        status = None
    if status is None:
        try:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        except Exception:  # noqa: BLE001 - metadata is optional and untrusted
            status = None
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        status = None
    return category, status


class ProviderOperationError(RuntimeError):
    """Safe manager-level failure preserving only category and HTTP status."""

    def __init__(
        self,
        provider: str,
        operation: str,
        failure_category: str,
        status_code: Optional[int] = None,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.failure_category = failure_category
        self.status_code = status_code
        status = f" HTTP {status_code}" if status_code is not None else ""
        super().__init__(
            f"Provider {provider!r} failed {operation!r} "
            f"({failure_category}{status})"
        )

#: Operations an :class:`IngestionProvider` may implement.
KNOWN_OPERATIONS = (
    "search_posts",
    "get_post",
    "get_comments",
    "scrape_url",
    "scrape_source",
    "discover_targets",
)


class IngestionProvider(ABC):

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def supports(self, operation: str) -> bool:
        """Return True when this provider actually implements *operation*.

        The base class defines every operation as raising
        :class:`UnsupportedOperation`; a provider "supports" an operation only
        when it overrides it.  This lets the manager dispatch to the right
        provider instead of calling every provider and swallowing errors.
        """
        own = getattr(type(self), operation, None)
        base = getattr(IngestionProvider, operation, None)
        return callable(own) and own is not base

    def search_posts(self, **kwargs: Any) -> list[PostData]:
        raise UnsupportedOperation(f"{self.name} does not support search_posts")

    def get_post(self, reddit_id: str) -> PostData:
        raise UnsupportedOperation(f"{self.name} does not support get_post")

    def get_comments(self, post_id: str) -> list[CommentData]:
        raise UnsupportedOperation(f"{self.name} does not support get_comments")

    def scrape_url(self, url: str) -> PostData:
        raise UnsupportedOperation(f"{self.name} does not support scrape_url")

    def scrape_source(self, url: str, **kwargs: Any) -> list[PostData]:
        raise UnsupportedOperation(f"{self.name} does not support scrape_source")

    def discover_targets(self, prompt: str) -> list[str]:
        raise UnsupportedOperation(f"{self.name} does not support discover_targets")
