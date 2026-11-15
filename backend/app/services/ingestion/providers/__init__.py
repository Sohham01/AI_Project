"""Ingestion providers, imported lazily to avoid client/provider cycles."""

from importlib import import_module
from typing import Any

_PROVIDER_EXPORTS = {
    "ArcticShiftProvider": ("arctic_shift", "ArcticShiftProvider"),
    "RSSProvider": ("rss", "RSSProvider"),
    "HermesProvider": ("hermes", "HermesProvider"),
    "DecodoProvider": ("decodo", "DecodoProvider"),
}

__all__ = list(_PROVIDER_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a provider export only when requested by the package consumer."""
    try:
        module_name, attribute = _PROVIDER_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
