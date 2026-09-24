from datetime import datetime, timezone
import requests
import os
from app.services.ingestion.providers.base import IngestionProvider, UnsupportedOperation
from app.schemas.ingestion import PostData

class HermesProvider(IngestionProvider):
    def __init__(self) -> None:
        self.api_url = os.getenv("HERMES_API_URL")
        self.api_key = os.getenv("HERMES_API_KEY")

    def scrape_url(self, url: str) -> PostData:
        if not self.api_url or not self.api_key:
            raise UnsupportedOperation("Hermes not configured")
            
        response = requests.post(
            f"{self.api_url}/scrape",
            json={"url": url},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        data = response.json()
        
        # Assume Hermes returns structured output that we need to normalize
        return PostData(
            reddit_id=data.get("id"),
            title=data.get("title"),
            body=data.get("body"),
            subreddit=data.get("subreddit", "unknown"),
            created_at=data.get("created_at") or datetime.now(timezone.utc), # Need parsing
            comments=[]
        )
