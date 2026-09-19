from fastapi import FastAPI
from sqlalchemy import text
from app.core.database import Base, engine

from app.models.reddit_post import RedditPost
from app.models.reddit_comment import RedditComment
from app.models.data_source import DataSource

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="IIM Sambalpur Community Intelligence",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "IIM Sambalpur Community Intelligence API is running"
    }


@app.get("/health/db")
def database_health():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        return {
            "database": "connected",
            "result": result.scalar()
        }