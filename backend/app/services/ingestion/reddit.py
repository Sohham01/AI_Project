from app.schemas.ingestion import PostData
from app.services.ingestion.arctic_shift import ArcticShiftClient
from app.services.ingestion.arctic_shift_normalizer import (
    ArcticShiftNormalizer,
)
from app.services.ingestion.base import IngestionSource


class RedditIngestion(IngestionSource):

    def __init__(self) -> None:
        self.client = ArcticShiftClient()
        self.normalizer = ArcticShiftNormalizer()

    def fetch_posts(
            self,
            query: str,
            subreddit: str,
            limit: int = 25,
    ) -> list[PostData]:

        raw_response = self.client.search_posts(
            subreddit=subreddit,
            limit=limit,
        )

        if not raw_response or "data" not in raw_response:
            return []

        raw_posts = raw_response["data"]

        if not isinstance(raw_posts, list):
            return []

        # Local filtering logic
        query_terms = query.lower().split()
        filtered_posts = []

        for post in raw_posts:
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            combined_text = f"{title} {selftext}".lower()

            if all(term in combined_text for term in query_terms):
                filtered_posts.append(
                    self.normalizer.normalize_post(post)
                )

        return filtered_posts

    def fetch_comments(
            self,
            post_id: str,
    ) -> list:

        raw_response = self.client.get_comments_tree(
            post_id
        )

        raw_comments = raw_response.get(
            "data",
            []
        )

        comments = []

        def walk(nodes):
            for node in nodes:

                # "more" nodes represent collapsed comments.
                if node.get("kind") == "more":
                    continue

                data = node.get("data")

                if not isinstance(data, dict):
                    continue

                # Only process actual comments.
                if node.get("kind") != "t1":
                    continue

                body = data.get("body")

                if body in ("[deleted]", "[removed]", ""):
                    continue

                comments.append(
                    self.normalizer.normalize_comment(data)
                )

                # Reddit-style replies may be nested in `replies`.
                replies = data.get("replies")

                if isinstance(replies, dict):
                    reply_children = (
                        replies
                        .get("data", {})
                        .get("children", [])
                    )

                    walk(reply_children)

        walk(raw_comments)

        return comments