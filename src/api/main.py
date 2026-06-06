"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.logging import configure_logging
from src.api.routes import health, backtest, simulation, analytics

configure_logging(
    level=settings.logging.level,
    fmt=settings.logging.format,
    log_file=settings.logging.file,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown event handling."""
    logger.info("Starting qb-risk-infra API (env=%s)", settings.env)
    try:
        from src.db.session import create_all_tables
        create_all_tables()
        logger.info("Database schema verified.")
    except Exception as exc:
        logger.warning("DB init skipped (likely no DB in dev mode): %s", exc)
    yield
    logger.info("API shutdown.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Quantitative Backtesting & Risk Analytics API",
        description=(
            "REST API for strategy backtesting, walk-forward validation, "
            "Monte Carlo stress simulation, and portfolio risk analytics."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )

    app.include_router(health.router)
    app.include_router(backtest.router, prefix="/api/v1")
    app.include_router(simulation.router, prefix="/api/v1")
    app.include_router(analytics.router, prefix="/api/v1")

    return app


app = create_app()
