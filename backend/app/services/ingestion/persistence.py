from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.data_source import DataSource
from app.models.reddit_comment import RedditComment
from app.models.reddit_post import RedditPost
from app.schemas.ingestion import CommentData, PostData
from app.core.database import SessionLocal

class PersistenceService:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_reddit_source(self) -> DataSource:
        stmt = select(DataSource).where(
            DataSource.name == "Reddit",
            DataSource.type == "reddit"
        )
        source = self.db.execute(stmt).scalar_one_or_none()
        if not source:
            source = DataSource(name="Reddit", type="reddit", status="active")
            self.db.add(source)
            self.db.commit()
            self.db.refresh(source)
        return source

    def save_post_and_comments(self, post_data: PostData):
        source = self.get_or_create_reddit_source()

        # Deduplicate post
        stmt = select(RedditPost).where(RedditPost.reddit_id == post_data.reddit_id)
        post = self.db.execute(stmt).scalar_one_or_none()

        if not post:
            post = RedditPost(
                reddit_id=post_data.reddit_id,
                title=post_data.title,
                body=post_data.body,
                subreddit=post_data.subreddit,
                created_at=post_data.created_at,
                score=post_data.score,
                num_comments=post_data.num_comments,
                url=post_data.url,
                source_id=source.id
            )
            self.db.add(post)
            self.db.flush()
        else:
            # Update mutable metadata
            post.score = post_data.score
            post.num_comments = post_data.num_comments
            post.body = post_data.body
            self.db.flush()

        # Deduplicate comments and handle hierarchy
        # Using a map to track DB IDs for already processed comments in this batch
        # or already in DB
        
        # To handle parent-child, we might need a two-pass approach or a robust dictionary
        # First, ensure all comments exist in DB.
        # Since we need parent_comment_id (int), we need the ID of the parent comment record.
        
        processed_comments = {} # reddit_id -> RedditComment object

        # Sort comments: root comments first, then replies.
        # However, a reply's parent might not be in the batch if it's already in DB.
        
        for comment_data in post_data.comments:
            stmt = select(RedditComment).where(RedditComment.reddit_id == comment_data.reddit_id)
            comment = self.db.execute(stmt).scalar_one_or_none()
            
            parent_comment_id = None
            if comment_data.parent_reddit_id:
                # Find parent comment DB ID
                parent_stmt = select(RedditComment.id).where(RedditComment.reddit_id == comment_data.parent_reddit_id)
                parent_comment_id = self.db.execute(parent_stmt).scalar_one_or_none()
            
            if not comment:
                comment = RedditComment(
                    reddit_id=comment_data.reddit_id,
                    post_id=post.id,
                    parent_comment_id=parent_comment_id,
                    body=comment_data.body,
                    created_at=comment_data.created_at,
                    score=comment_data.score
                )
                self.db.add(comment)
                self.db.flush()
            else:
                # Update mutable metadata
                comment.score = comment_data.score
                comment.body = comment_data.body
                comment.parent_comment_id = parent_comment_id # Ensure it's correct
                self.db.flush()
        
        self.db.commit()
