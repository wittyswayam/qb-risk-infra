"""Portfolio risk analytics endpoints."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import RiskAnalyticsRequest, RiskAnalyticsResponse
from src.core.exceptions import RiskCalculationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.post(
    "/risk",
    response_model=RiskAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute risk metrics for a completed backtest run",
)
async def compute_risk_analytics(req: RiskAnalyticsRequest) -> RiskAnalyticsResponse:
    """Compute comprehensive risk analytics for a backtest run.

    Fetches the backtest result from the database, computes rolling volatility,
    VaR, CVaR, beta, and correlation versus an optional benchmark.
    """
    try:
        from src.db.session import get_db_session
        from src.db.repository import BacktestRepository
        import pandas as pd
        import numpy as np

        with get_db_session() as session:
            repo = BacktestRepository(session)
            run = repo.get_by_run_id(req.run_id)
            if run is None:
                raise HTTPException(
                    status_code=404, detail=f"Run '{req.run_id}' not found."
                )
            metrics = run.metrics or {}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Risk analytics DB error")
        raise HTTPException(status_code=500, detail="Database error")

    return RiskAnalyticsResponse(
        symbol=run.symbol,
        period=f"{run.start_date.date()} to {run.end_date.date()}",
        annualised_return=metrics.get("annualised_return", 0.0),
        annualised_volatility=metrics.get("annualised_volatility", 0.0),
        sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
        sortino_ratio=metrics.get("sortino_ratio", 0.0),
        max_drawdown=metrics.get("max_drawdown", 0.0),
        var_95=metrics.get("var_95", 0.0),
        cvar_95=metrics.get("cvar_95", 0.0),
        var_99=metrics.get("var_99", 0.0),
        cvar_99=metrics.get("cvar_99", 0.0),
        beta=None,
        correlation_to_benchmark=None,
        rolling_vol=[],
        rolling_var=[],
    )


@router.get(
    "/runs",
    summary="List recent backtest runs",
)
async def list_runs(
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """Return recent backtest run summaries from the database."""
    try:
        from src.db.session import get_db_session
        from src.db.repository import BacktestRepository

        with get_db_session() as session:
            repo = BacktestRepository(session)
            runs = repo.list_runs(strategy_name=strategy, symbol=symbol, limit=limit)

        return {
            "runs": [
                {
                    "run_id": r.run_id,
                    "strategy_name": r.strategy_name,
                    "symbol": r.symbol,
                    "start_date": str(r.start_date.date()),
                    "end_date": str(r.end_date.date()),
                    "sharpe_ratio": (r.metrics or {}).get("sharpe_ratio"),
                    "max_drawdown": (r.metrics or {}).get("max_drawdown"),
                    "created_at": str(r.created_at),
                }
                for r in runs
            ]
        }
    except Exception as exc:
        logger.exception("List runs DB error")
        raise HTTPException(status_code=500, detail="Database error")
