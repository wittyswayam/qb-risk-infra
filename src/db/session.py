"""SQLAlchemy engine and session factory setup."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


def build_engine(database_url: str | None = None):
    """Create and return a SQLAlchemy engine.

    Args:
        database_url: Override the URL from settings.

    Returns:
        SQLAlchemy Engine.
    """
    url = database_url or settings.database.url
    engine = create_engine(
        url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_pre_ping=True,  # Recycle stale connections
        echo=settings.debug,
    )

    @event.listens_for(engine, "connect")
    def set_search_path(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("SET search_path TO public")
        cursor.close()

    logger.info("Database engine created: host=%s db=%s", settings.database.host, settings.database.name)
    return engine


_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionFactory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager providing a database session with automatic cleanup.

    Usage::

        with get_db_session() as session:
            repo = BacktestRepository(session)
            run_id = repo.save(result, config)
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_tables() -> None:
    """Create all ORM-mapped tables if they do not exist."""
    from src.db.models import Base
    Base.metadata.create_all(bind=get_engine())
    logger.info("Database tables created/verified.")
