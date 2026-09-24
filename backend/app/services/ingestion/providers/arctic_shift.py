from typing import Any
from app.services.ingestion.providers.base import IngestionProvider
from app.services.ingestion.arctic_shift import ArcticShiftClient
from app.services.ingestion.arctic_shift_normalizer import ArcticShiftNormalizer
from app.schemas.ingestion import CommentData, PostData

class ArcticShiftProvider(IngestionProvider):
    def __init__(self) -> None:
        self.client = ArcticShiftClient()
        self.normalizer = ArcticShiftNormalizer()

    def search_posts(self, **kwargs: Any) -> list[PostData]:
        # ArcticShiftClient.search_posts takes subreddit, limit, etc.
        # We need to map kwargs to these.
        subreddit = kwargs.get("subreddit", "CATpreparation")
        limit = kwargs.get("limit", 25)
        after = kwargs.get("after", "30d")
        
        raw_response = self.client.search_posts(
            subreddit=subreddit,
            limit=limit,
            after=after
        )
        
        raw_posts = raw_response.get("data", [])
        return [self.normalizer.normalize_post(post) for post in raw_posts]

    def get_post(self, reddit_id: str) -> PostData:
        raw_response = self.client.get_post(reddit_id)
        # Assuming raw_response['data'] contains the post
        raw_post = raw_response.get("data", [])
        if not raw_post:
            raise ValueError(f"Post {reddit_id} not found")
        # Arctic Shift returns a list of posts usually, or dict. 
        # Inspecting arctic_shift.py: it returns a dict. Let's see get_post in arctic_shift.py.
        # It calls _get("/api/posts/ids", params={"ids": reddit_id})
        # Let's assume it returns one post.
        return self.normalizer.normalize_post(raw_post[0] if isinstance(raw_post, list) else raw_post)

    def get_comments(self, post_id: str) -> list[CommentData]:
        raw_response = self.client.get_comments_tree(post_id)
        raw_comments = raw_response.get("data", [])
        
        comments = []
        
        # We need the traversal logic here, currently in RedditIngestion.
        # Let's move the traversal logic to normalizer or here?
        # The requirement says: "Tree traversal should be handled by the ingestion/parser layer."
        # Arctic Shift provider should probably handle it or delegate.
        
        def walk(nodes):
            for node in nodes:
                if node.get("kind") == "more":
                    continue
                data = node.get("data")
                if not isinstance(data, dict):
                    continue
                if node.get("kind") != "t1":
                    continue
                body = data.get("body")
                if body in ("[deleted]", "[removed]", ""):
                    continue
                
                comments.append(self.normalizer.normalize_comment(data))
                
                replies = data.get("replies")
                if isinstance(replies, dict):
                    reply_children = replies.get("data", {}).get("children", [])
                    walk(reply_children)
                    
        walk(raw_comments)
        return comments
