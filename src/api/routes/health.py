"""Health check endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])

VERSION = "1.0.0"


@router.get("/", response_model=HealthResponse, summary="System health check")
async def health_check() -> HealthResponse:
    """Returns the current service status and connectivity to external stores.

    The endpoint is designed to be polled by container orchestrators and
    load balancers. A 200 response indicates the API is reachable; consumers
    should additionally check `db_connected` and `redis_connected`.
    """
    db_ok = False
    redis_ok = False

    try:
        from src.db.session import get_engine
        with get_engine().connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    try:
        import redis as redis_lib
        from src.core.config import settings
        client = redis_lib.Redis.from_url(settings.redis.url, socket_connect_timeout=2)
        client.ping()
        redis_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version=VERSION,
        timestamp=datetime.now(timezone.utc),
        db_connected=db_ok,
        redis_connected=redis_ok,
    )


@router.get("/ping", summary="Liveness probe")
async def ping() -> dict:
    """Minimal liveness endpoint for container health checks."""
    return {"pong": True}
