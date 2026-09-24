from app.services.ingestion.reddit import RedditIngestion
from app.services.ingestion.persistence import PersistenceService
from app.core.database import SessionLocal
from app.models.reddit_post import RedditPost
from app.models.reddit_comment import RedditComment
from sqlalchemy import select

def run_test(run_name: str):
    print(f"\n--- {run_name} ---")
    db = SessionLocal()
    ingestion = RedditIngestion()
    persistence = PersistenceService(db)

    # 1. Fetch posts
    posts = ingestion.fetch_posts(
        query="IIM",
        subreddit="CATpreparation",
        limit=10,
    )
    
    print(f"Posts fetched: {len(posts)}")

    inserted_posts = 0
    inserted_comments = 0

    for post in posts:
        # 2. Fetch comments for the post
        post.comments = ingestion.fetch_comments(post.reddit_id)
        
        # 3. Save to DB
        persistence.save_post_and_comments(post)
        
        # In a real scenario, we'd track counts from the persistence layer.
        # Here we just verify DB counts.

    print("Saved posts and comments.")
    
    # 4. Print stats
    print(f"Total posts in DB: {db.query(RedditPost).count()}")
    print(f"Total comments in DB: {db.query(RedditComment).count()}")

    # 5. Print a sample
    sample_comments = db.query(RedditComment).limit(5).all()
    for c in sample_comments:
        print(f"COMMENT DB_ID: {c.id}, Reddit_ID: {c.reddit_id}, Post_ID: {c.post_id}, Parent_ID: {c.parent_comment_id}, Body: {c.body[:50]}...")

    db.close()

if __name__ == "__main__":
    # First run
    run_test("First Run")
    
    # Second run (idempotency check)
    run_test("Second Run")
