from datetime import datetime, timezone
import requests
import xml.etree.ElementTree as ET
from app.services.ingestion.providers.base import IngestionProvider
from app.schemas.ingestion import PostData

class RSSProvider(IngestionProvider):
    def search_posts(self, **kwargs: Any) -> list[PostData]:
        url = kwargs.get("url")
        if not url:
            raise ValueError("RSS URL required")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        posts = []
        
        # RSS/Atom parsing (simplified)
        for item in root.findall(".//item"):
            title = item.findtext("title")
            link = item.findtext("link")
            description = item.findtext("description")
            pub_date = item.findtext("pubDate")
            
            posts.append(PostData(
                reddit_id=link or title, # Fallback
                title=title or "Unknown",
                body=description,
                subreddit="unknown", # RSS feeds often don't explicitly have this in the item
                created_at=datetime.now(timezone.utc), # Parse pub_date if needed
                comments=[]
            ))
            
        return posts
