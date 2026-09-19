from app.services.ingestion.base import IngestionSource


class HermesIngestion(IngestionSource):
    """
    Experimental browser-based ingestion source.

    This is not the production Reddit collector.
    It must only be used for permitted data collection.
    """

    def fetch_posts(self):
        raise NotImplementedError(
            "Hermes post collection has not been implemented yet."
        )

    def fetch_comments(self, post_id: str):
        raise NotImplementedError(
            "Hermes comment collection has not been implemented yet."
        )