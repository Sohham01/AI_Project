from app.services.ingestion.base import IngestionSource


class RedditIngestion(IngestionSource):

    def fetch_posts(self):
        raise NotImplementedError(
            "Reddit post collection has not been implemented yet."
        )

    def fetch_comments(self, post_id: str):
        raise NotImplementedError(
            "Reddit comment collection has not been implemented yet."
        )