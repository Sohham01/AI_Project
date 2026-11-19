"""Vercel serverless ASGI entrypoint for Reddit Community Intelligence (RCI).

Exposes ``app`` for ``@vercel/python``. When running in the full stack
environment (with SQLAlchemy installed), re-exports ``app.main:app``. When
running in the slim Vercel serverless bundle (FastAPI + Pydantic only), mounts
the read-only synthetic demo router (``/api/demo/*``), static studio assets
(``/assets/*``), the SPA shell (``/app`` and ``/app/``), and health probes.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"

for candidate in (str(BACKEND_DIR), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def _create_serverless_app():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from app.api.demo import router as demo_router

    serverless_app = FastAPI(
        title="Reddit Community Intelligence",
        version="1.0.0",
    )

    static_dir = BACKEND_DIR / "app" / "static"
    serverless_app.include_router(demo_router, prefix="/api/demo")
    serverless_app.mount(
        "/assets",
        StaticFiles(directory=static_dir),
        name="demo-assets",
    )

    @serverless_app.get("/")
    def root():
        return {
            "message": "IIM Sambalpur Community Intelligence API is running"
        }

    @serverless_app.get("/app", include_in_schema=False)
    @serverless_app.get("/app/", include_in_schema=False)
    def demo_app():
        return FileResponse(static_dir / "index.html")

    @serverless_app.get("/health")
    def health():
        return {"status": "ok"}

    @serverless_app.get("/health/db")
    def database_health():
        return JSONResponse(
            status_code=503,
            content={
                "database": "not_configured",
                "detail": "Database configuration is missing or invalid.",
            },
        )

    return serverless_app


try:
    from app.main import app  # type: ignore[assignment]
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("sqlalchemy"):
        app = _create_serverless_app()
    else:
        raise
