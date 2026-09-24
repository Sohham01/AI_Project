from typing import Any, Optional
from app.services.ingestion.providers.base import IngestionProvider, UnsupportedOperation
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

    def execute_operation(self, operation: str, **kwargs: Any) -> Any:
        # Simple capability-based selection
        # Could be more complex with fallbacks
        for name, provider in self.providers.items():
            try:
                method = getattr(provider, operation)
                return method(**kwargs)
            except (UnsupportedOperation, Exception):
                continue
        raise Exception(f"No provider found for operation {operation}")
