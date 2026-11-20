import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing or invalid."""


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


_engine: "Engine | None" = None
_session_factory: "sessionmaker | None" = None


def get_database_url() -> str:
    """Return ``DATABASE_URL`` or raise an actionable configuration error.

    The engine is intentionally *not* created at import time: the test suite and
    tooling must be able to import the app without a configured database.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ConfigurationError(
            "DATABASE_URL is not set. Copy backend/.env.example to backend/.env "
            "and set DATABASE_URL (e.g. "
            "'postgresql+psycopg://user:password@localhost:5432/dbname')."
        )
    return url


def get_engine() -> Engine:
    """Lazily create (and cache) the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(get_database_url(), pool_pre_ping=True)
    return _engine


def get_sessionmaker() -> sessionmaker:
    """Lazily build (and cache) the session factory bound to the engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _session_factory


_INTROSPECTION_ATTRS = frozenset({
    "_is_coroutine_marker",
    "_is_coroutine",
    "_mock_methods",
    "_mock_unsafe",
})


def _is_introspection_attr(name: str) -> bool:
    return (name.startswith("__") and name.endswith("__")) or name in _INTROSPECTION_ATTRS


class _LazyEngine:
    """Backward-compatible proxy that resolves the engine on first use.

    Code that did ``from app.core.database import engine`` keeps working, but
    merely importing the module no longer requires ``DATABASE_URL`` to be set.
    """

    def __getattr__(self, name: str):
        if _is_introspection_attr(name):
            raise AttributeError(name)
        return getattr(get_engine(), name)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "<lazy SQLAlchemy engine>"


class _LazySessionLocal:
    """Backward-compatible, callable proxy for the session factory."""

    def __call__(self, *args, **kwargs):
        return get_sessionmaker()(*args, **kwargs)

    def __getattr__(self, name: str):
        if _is_introspection_attr(name):
            raise AttributeError(name)
        return getattr(get_sessionmaker(), name)


# Module-level names retained for existing imports.  Accessing them triggers
# lazy engine creation, so a missing DATABASE_URL surfaces at first use.
engine = _LazyEngine()
SessionLocal = _LazySessionLocal()
