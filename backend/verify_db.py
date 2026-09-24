from app.core.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()

print("\n--- SQL Verification ---")
print("COUNT(*):", db.execute(text("SELECT COUNT(*) FROM reddit_posts")).scalar())
print("COUNT(*):", db.execute(text("SELECT COUNT(*) FROM reddit_comments")).scalar())

print("\n--- Joined Sample ---")
# Adjusting to SQLite/Postgres compatibility. The model files suggest generic SQL.
# Assuming PostgreSQL as stated in the issue.
query = """
SELECT
    p.reddit_id AS post_reddit_id,
    c.reddit_id AS comment_reddit_id,
    c.parent_comment_id,
    c.body
FROM reddit_posts p
JOIN reddit_comments c
    ON c.post_id = p.id
LIMIT 5;
"""
results = db.execute(text(query)).fetchall()
for r in results:
    print(r)

print("\n--- Source Linkage ---")
query_source = """
SELECT
    p.id,
    p.reddit_id,
    p.source_id,
    s.name,
    s.type
FROM reddit_posts p
JOIN data_sources s
    ON s.id = p.source_id
LIMIT 5;
"""
results_source = db.execute(text(query_source)).fetchall()
for r in results_source:
    print(r)

db.close()
