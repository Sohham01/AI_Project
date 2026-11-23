import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.core.database import Base, ConfigurationError, get_engine
from app.core.security import SecurityHeadersMiddleware
from app.api.demo import router as demo_router

# Importing every model module registers its table with ``Base.metadata`` so
# that migrations/autogenerate and (opt-in) schema creation see the full schema.
from app.models.data_source import DataSource
from app.models.reddit_post import RedditPost
from app.models.reddit_comment import RedditComment
from app.models.nlp_analysis import NLPPostAnalysis, NLPCommentAnalysis

logger = logging.getLogger(__name__)

_MODELS = (
    DataSource,
    RedditPost,
    RedditComment,
    NLPPostAnalysis,
    NLPCommentAnalysis,
)


def _auto_create_enabled() -> bool:
    """Schema creation is opt-in; Alembic is the source of truth by default."""
    return os.getenv("AUTO_CREATE_SCHEMA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _auto_create_enabled():
        Base.metadata.create_all(bind=get_engine())
    yield


app = FastAPI(
    title="Reddit Community Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(SecurityHeadersMiddleware)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.include_router(demo_router, prefix="/api/demo")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="demo-assets")


@app.get("/")
def root(request: Request = None):  # type: ignore[assignment]
    if request is not None:
        accept = (request.headers.get("accept") or "").lower()
        if "text/html" in accept and "application/json" not in accept.split(",")[0]:
            return RedirectResponse(url="/app", status_code=307)
    return {
        "message": "IIM Sambalpur Community Intelligence API is running"
    }


@app.get("/app", include_in_schema=False)
@app.get("/app/", include_in_schema=False)
def demo_app():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Liveness probe that never depends on the database."""
    return {"status": "ok"}


@app.get("/health/db")
def database_health():
    """Read-only database connectivity check with actionable failure output."""
    try:
        engine = get_engine()
    except ConfigurationError as exc:
        logger.error("Database health check configuration failed (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "database": "not_configured",
                "detail": "Database configuration is missing or invalid.",
            },
        )

    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            value = result.scalar()
    except Exception as exc:  # noqa: BLE001 - reported without leaking secrets
        logger.error("Database health check failed (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "database": "unavailable",
                "detail": "Database connectivity check failed.",
            },
        )

    return {"database": "connected", "result": value}
