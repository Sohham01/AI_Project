"""Offline unit tests for ingestion persistence.

These previously ran against the live Reddit API and the real database.  They
now use a temporary in-memory SQLite database and hand-built payloads.
"""

import unittest
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base

# Importing the models registers their tables on ``Base.metadata``.
from app.models.data_source import DataSource  # noqa: F401
from app.models.reddit_comment import RedditComment
from app.models.reddit_post import RedditPost
from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.persistence import PersistenceService


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


class TestPersistenceService(unittest.TestCase):
    def setUp(self):
        self.db = _make_session()
        self.service = PersistenceService(self.db)

    def tearDown(self):
        self.db.close()

    def _post(self, reddit_id="p1", comments=None):
        return PostData(
            reddit_id=reddit_id,
            title="A title",
            subreddit="CATpreparation",
            created_at=datetime(2024, 1, 1),
            comments=comments or [],
        )

    def test_saves_post_and_comments_with_hierarchy(self):
        comments = [
            CommentData(
                reddit_id="c1",
                body="root",
                created_at=datetime(2024, 1, 1),
            ),
            CommentData(
                reddit_id="c2",
                body="reply",
                created_at=datetime(2024, 1, 2),
                parent_reddit_id="c1",
            ),
        ]
        self.service.save_post_and_comments(self._post(comments=comments))

        self.assertEqual(self.db.query(RedditPost).count(), 1)
        saved = self.db.query(RedditComment).order_by(RedditComment.id).all()
        self.assertEqual(len(saved), 2)
        self.assertIsNone(saved[0].parent_comment_id)
        self.assertEqual(saved[1].parent_comment_id, saved[0].id)

    def test_reply_listed_before_parent_is_still_linked(self):
        comments = [
            CommentData(
                reddit_id="c2",
                body="reply",
                created_at=datetime(2024, 1, 2),
                parent_reddit_id="c1",
            ),
            CommentData(
                reddit_id="c1",
                body="root",
                created_at=datetime(2024, 1, 1),
            ),
        ]
        self.service.save_post_and_comments(self._post(comments=comments))

        by_id = {
            c.reddit_id: c for c in self.db.query(RedditComment).all()
        }
        self.assertEqual(by_id["c2"].parent_comment_id, by_id["c1"].id)

    def test_second_save_is_idempotent(self):
        comments = [
            CommentData(
                reddit_id="c1",
                body="root",
                created_at=datetime(2024, 1, 1),
            )
        ]
        self.service.save_post_and_comments(self._post(comments=comments))
        self.service.save_post_and_comments(self._post(comments=comments))

        self.assertEqual(self.db.query(RedditPost).count(), 1)
        self.assertEqual(self.db.query(RedditComment).count(), 1)
        self.assertEqual(self.db.query(DataSource).count(), 1)

    def test_missing_parent_is_left_unlinked(self):
        comments = [
            CommentData(
                reddit_id="c1",
                body="orphan",
                created_at=datetime(2024, 1, 1),
                parent_reddit_id="does_not_exist",
            )
        ]
        self.service.save_post_and_comments(self._post(comments=comments))
        saved = self.db.query(RedditComment).one()
        self.assertIsNone(saved.parent_comment_id)

    def test_comment_becoming_top_level_clears_old_parent(self):
        root = CommentData(
            reddit_id="c1",
            body="root",
            created_at=datetime(2024, 1, 1),
        )
        reply = CommentData(
            reddit_id="c2",
            body="reply",
            created_at=datetime(2024, 1, 2),
            parent_reddit_id="c1",
        )
        self.service.save_post_and_comments(self._post(comments=[root, reply]))

        updated = CommentData(
            reddit_id="c2",
            body="now top-level",
            created_at=datetime(2024, 1, 2),
        )
        self.service.save_post_and_comments(self._post(comments=[root, updated]))

        saved = self.db.query(RedditComment).filter_by(reddit_id="c2").one()
        self.assertIsNone(saved.parent_comment_id)


if __name__ == "__main__":
    unittest.main()
