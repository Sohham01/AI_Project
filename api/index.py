"""Vercel FastAPI entrypoint for Reddit Community Intelligence (RCI)."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
STATIC_DIR = BACKEND_DIR / "app" / "static"

for candidate in (str(BACKEND_DIR), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from app.api.demo import router as demo_router  # noqa: E402
from app.core.security import SecurityHeadersMiddleware  # noqa: E402

app = FastAPI(
    title="Reddit Community Intelligence",
    version="1.0.0",
)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(demo_router, prefix="/api/demo")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="demo-assets")


@app.get("/")
def root():
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
    """Read-only database status in serverless mode."""
    return JSONResponse(
        status_code=503,
        content={
            "database": "not_configured",
            "detail": "Database configuration is missing or invalid.",
        },
    )
