from typing import Any, Optional
from app.services.ingestion.providers.base import (
    IngestionProvider,
    KNOWN_OPERATIONS,
    ProviderOperationError,
    UnsupportedOperation,
    safe_failure_metadata,
)
from app.services.ingestion.providers.arctic_shift import ArcticShiftProvider
from app.services.ingestion.providers.rss import RSSProvider
from app.services.ingestion.providers.hermes import HermesProvider
from app.services.ingestion.providers.decodo import DecodoProvider


class IngestionManager:
    def __init__(self) -> None:
        self.providers = {
            "arctic_shift": ArcticShiftProvider(),
            "rss": RSSProvider(),
            "hermes": HermesProvider(),
            "decodo": DecodoProvider(),
        }

    def get_provider(self, name: str) -> IngestionProvider:
        if name not in self.providers:
            raise ValueError(f"Provider {name} not found")
        return self.providers[name]

    def execute_operation(
        self,
        operation: str,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute *operation* against a supporting provider.

        Selection is capability-based: only providers that actually implement
        *operation* are considered. Provider failures surface as a safe operation
        error with a sanitized category and optional HTTP status only;
        unconfigured providers may be skipped in favor of another capable
        provider. Pass ``provider=`` to force a specific one.
        """
        if operation not in KNOWN_OPERATIONS:
            raise ValueError(
                f"Unknown ingestion operation {operation!r}. Known: "
                f"{', '.join(KNOWN_OPERATIONS)}"
            )

        if provider is not None:
            chosen = self.get_provider(provider)
            if not chosen.supports(operation):
                raise UnsupportedOperation(
                    f"{provider} does not support {operation}"
                )
            try:
                return getattr(chosen, operation)(**kwargs)
            except Exception as exc:  # noqa: BLE001 - hide provider-supplied details
                category, status = safe_failure_metadata(exc)
                raise ProviderOperationError(
                    provider, operation, category, status
                ) from None

        errors: list[tuple[str, str, Optional[int]]] = []
        for name, candidate in self.providers.items():
            if not candidate.supports(operation):
                continue
            try:
                return getattr(candidate, operation)(**kwargs)
            except UnsupportedOperation:
                # Provider advertises support but is unconfigured/unavailable;
                # try the next capable provider.
                continue
            except Exception as exc:  # noqa: BLE001 - surfaced below
                category, status = safe_failure_metadata(exc)
                errors.append((name, category, status))

        if errors:
            name, category, status = errors[0]
            raise ProviderOperationError(name, operation, category, status) from None

        raise UnsupportedOperation(
            f"No provider supports operation {operation!r}"
        )
