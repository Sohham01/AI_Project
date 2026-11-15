"""Compatibility shim for the Hermes ingestion provider.

This module was historically an empty placeholder.  The canonical Hermes
implementation lives in :mod:`app.services.ingestion.providers.hermes`; it is
re-exported here so any older ``from app.services.ingestion.hermes import
HermesProvider`` import keeps working.
"""

from app.services.ingestion.providers.hermes import HermesProvider

__all__ = ["HermesProvider"]
