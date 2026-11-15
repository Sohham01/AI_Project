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

        # Comment hierarchy is resolved in two passes so the result does not
        # depend on the order comments appear in ``post_data.comments`` (a reply
        # may arrive before its parent, or the parent may already be in the DB).

        # Pass 1: ensure every comment row exists and refresh mutable fields.
        by_reddit_id: dict[str, RedditComment] = {}
        for comment_data in post_data.comments:
            stmt = select(RedditComment).where(
                RedditComment.reddit_id == comment_data.reddit_id
            )
            comment = self.db.execute(stmt).scalar_one_or_none()

            if not comment:
                comment = RedditComment(
                    reddit_id=comment_data.reddit_id,
                    post_id=post.id,
                    body=comment_data.body,
                    created_at=comment_data.created_at,
                    score=comment_data.score,
                )
                self.db.add(comment)
            else:
                comment.post_id = post.id
                comment.body = comment_data.body
                comment.score = comment_data.score

            self.db.flush()  # assigns integer ids for new rows
            by_reddit_id[comment_data.reddit_id] = comment

        # Pass 2: resolve parent links now that every comment has a DB id.
        for comment_data in post_data.comments:
            parent_id = None
            if comment_data.parent_reddit_id:
                parent = by_reddit_id.get(comment_data.parent_reddit_id)
                if parent is not None:
                    parent_id = parent.id
                else:
                    parent_stmt = select(RedditComment.id).where(
                        RedditComment.reddit_id == comment_data.parent_reddit_id
                    )
                    parent_id = self.db.execute(parent_stmt).scalar_one_or_none()

            by_reddit_id[comment_data.reddit_id].parent_comment_id = parent_id

        self.db.commit()
