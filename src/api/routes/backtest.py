"""Backtest execution endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import BacktestRequest, BacktestResponse, FillResponse, MetricsResponse
from src.backtesting.engine import BacktestEngine
from src.backtesting.order_router import OrderRouter
from src.backtesting.portfolio import Portfolio
from src.core.config import settings
from src.core.exceptions import (
    BacktestError,
    InsufficientDataError,
    IngestionError,
    StrategyError,
)
from src.ingestion.csv_adapter import CSVIngestionAdapter
from src.strategies.registry import get_strategy

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/backtest", tags=["Backtest"])


@router.post(
    "/run",
    response_model=BacktestResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute a strategy backtest",
)
async def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Execute a full backtest for a given strategy and symbol.

    The endpoint loads OHLCV data from the configured data source, runs the
    backtesting engine bar-by-bar, and returns metrics and the equity curve.

    Returns HTTP 422 on invalid parameters and 400 on domain logic errors
    (e.g., insufficient data, unknown strategy).
    """
    logger.info(
        "Backtest request: strategy=%s symbol=%s %s->%s",
        req.strategy, req.symbol, req.start_date, req.end_date,
    )

    # 1. Load data
    try:
        adapter = CSVIngestionAdapter(
            data_dir=str(settings.ingestion.data_dir),
            missing_strategy=settings.ingestion.missing_value_strategy,
        )
        start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
        data = adapter.fetch(
            symbol=req.symbol,
            start=start_dt,
            end=end_dt,
            interval=req.interval,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=f"Data ingestion error: {exc}")
    except Exception as exc:
        logger.exception("Unexpected ingestion failure")
        raise HTTPException(status_code=500, detail="Data loading failed")

    # 2. Initialise strategy
    try:
        strategy = get_strategy(req.strategy, req.strategy_params)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Strategy init error: {exc}")

    # 3. Run backtest
    try:
        router_obj = OrderRouter(
            commission_bps=req.commission_bps,
            slippage_bps=req.slippage_bps,
        )
        portfolio = Portfolio(
            initial_capital=req.initial_capital,
            max_position_size=settings.backtest.max_position_size,
        )
        engine = BacktestEngine(
            strategy=strategy,
            order_router=router_obj,
            portfolio=portfolio,
            risk_free_rate=settings.backtest.risk_free_rate,
            trading_days=settings.backtest.trading_days_per_year,
        )
        result = engine.run(data=data, symbol=req.symbol)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except BacktestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Backtest engine error")
        raise HTTPException(status_code=500, detail="Backtest execution failed")

    # 4. Optionally persist
    run_id: Optional[str] = None
    if req.persist:
        try:
            from src.db.session import get_db_session
            from src.db.repository import BacktestRepository
            with get_db_session() as session:
                repo = BacktestRepository(session)
                run_id = repo.save(
                    result,
                    config={
                        "interval": req.interval,
                        "initial_capital": req.initial_capital,
                        "commission_bps": req.commission_bps,
                        "slippage_bps": req.slippage_bps,
                        "strategy_params": req.strategy_params,
                    },
                )
        except Exception as exc:
            logger.warning("Could not persist backtest result: %s", exc)

    # 5. Build response
    equity_curve_data = [
        {"timestamp": str(ts), "equity": float(eq)}
        for ts, eq in result.equity_curve.items()
    ]

    fills_data = [
        FillResponse(
            symbol=f.order.symbol,
            side=f.order.side.value,
            quantity=f.fill_quantity,
            fill_price=f.fill_price,
            commission=f.commission,
            slippage=f.slippage,
            timestamp=f.timestamp,
        )
        for f in result.fills
    ]

    m = result.metrics
    return BacktestResponse(
        run_id=run_id,
        strategy_name=result.strategy_name,
        symbol=result.symbol,
        start_date=req.start_date,
        end_date=req.end_date,
        metrics=MetricsResponse(
            annualised_return=m.get("annualised_return", 0.0),
            annualised_volatility=m.get("annualised_volatility", 0.0),
            sharpe_ratio=m.get("sharpe_ratio", 0.0),
            sortino_ratio=m.get("sortino_ratio", 0.0),
            max_drawdown=m.get("max_drawdown", 0.0),
            calmar_ratio=m.get("calmar_ratio", 0.0),
            hit_rate=m.get("hit_rate", 0.0),
            profit_factor=m.get("profit_factor", 0.0),
            skewness=m.get("skewness", 0.0),
            kurtosis=m.get("kurtosis", 0.0),
            var_95=m.get("var_95", 0.0),
            cvar_95=m.get("cvar_95", 0.0),
            var_99=m.get("var_99", 0.0),
            cvar_99=m.get("cvar_99", 0.0),
            n_periods=m.get("n_periods", 0.0),
        ),
        n_fills=len(result.fills),
        equity_curve=equity_curve_data,
        fills=fills_data,
    )


@router.get(
    "/strategies",
    summary="List available strategies",
)
async def list_strategies() -> dict:
    """Return the names of all registered trading strategies."""
    from src.strategies.registry import list_strategies
    return {"strategies": list_strategies()}
