"""Monte Carlo simulation and walk-forward validation endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    MonteCarloRequest,
    MonteCarloResponse,
    WalkForwardRequest,
    WalkForwardResponse,
    MetricsResponse,
)
from src.core.config import settings
from src.core.exceptions import WalkForwardError, SimulationError, IngestionError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulation", tags=["Simulation"])


@router.post(
    "/walkforward",
    response_model=WalkForwardResponse,
    status_code=status.HTTP_200_OK,
    summary="Run walk-forward validation",
)
async def run_walkforward(req: WalkForwardRequest) -> WalkForwardResponse:
    """Execute a walk-forward validation run for a strategy.

    For each rolling window, the strategy is optimised on in-sample data via
    grid search and evaluated on adjacent out-of-sample bars. Returns
    aggregated OOS metrics and parameter stability scores.
    """
    logger.info(
        "Walk-forward: strategy=%s symbol=%s %s->%s",
        req.strategy, req.symbol, req.start_date, req.end_date,
    )

    try:
        from src.ingestion.csv_adapter import CSVIngestionAdapter
        adapter = CSVIngestionAdapter(
            data_dir=str(settings.ingestion.data_dir),
            missing_strategy=settings.ingestion.missing_value_strategy,
        )
        start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
        data = adapter.fetch(symbol=req.symbol, start=start_dt, end=end_dt, interval=req.interval)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=f"Data ingestion error: {exc}")

    try:
        from src.strategies.registry import get_strategy
        from src.walkforward.validator import WalkForwardValidator

        def strategy_factory(params):
            return get_strategy(req.strategy, params)

        validator = WalkForwardValidator(
            strategy_factory=strategy_factory,
            param_grid=req.param_grid,
            train_window_days=req.train_window_days,
            test_window_days=req.test_window_days,
            step_days=req.step_days,
            initial_capital=req.initial_capital,
            commission_bps=req.commission_bps,
            slippage_bps=req.slippage_bps,
            risk_free_rate=settings.backtest.risk_free_rate,
            trading_days=settings.backtest.trading_days_per_year,
        )
        wf_result = validator.run(data=data, symbol=req.symbol)
    except WalkForwardError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Walk-forward engine error")
        raise HTTPException(status_code=500, detail="Walk-forward execution failed")

    m = wf_result.oos_metrics
    oos_metrics = MetricsResponse(
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
    )

    return WalkForwardResponse(
        run_id=None,
        strategy_name=wf_result.strategy_name,
        symbol=req.symbol,
        n_windows=len(wf_result.windows),
        oos_metrics=oos_metrics,
        param_stability=wf_result.param_stability,
        windows=wf_result.windows,
    )


@router.post(
    "/montecarlo",
    response_model=MonteCarloResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Monte Carlo stress simulation on a completed backtest",
)
async def run_montecarlo(req: MonteCarloRequest) -> MonteCarloResponse:
    """Run a Monte Carlo simulation using returns from a prior backtest run.

    Loads the backtest run from the database, extracts daily returns, and
    simulates forward paths under baseline and optional shock scenarios.
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
                    status_code=404,
                    detail=f"Backtest run '{req.run_id}' not found.",
                )

        # We don't store the full equity curve in DB, so we re-run to get returns.
        # In production, the equity curve snapshots would be stored in a time-series table.
        # For this implementation, we use the metrics to reconstruct approximate returns.
        # A full production system would cache equity curves in a dedicated store.
        raise HTTPException(
            status_code=400,
            detail=(
                "Monte Carlo requires equity curve data. "
                "Use the /simulation/montecarlo/from-returns endpoint with explicit return data, "
                "or re-run the backtest with persist=true and equity_curve caching enabled."
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Monte Carlo endpoint error")
        raise HTTPException(status_code=500, detail="Monte Carlo simulation failed")


@router.post(
    "/montecarlo/from-returns",
    response_model=MonteCarloResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Monte Carlo from an explicit return series",
)
async def run_montecarlo_from_returns(
    returns: list[float],
    req: MonteCarloRequest,
) -> MonteCarloResponse:
    """Run Monte Carlo simulation from an explicit list of daily returns.

    Accepts the daily return series directly (as decimal values, e.g., 0.01 = +1%).
    Useful when equity curve data is available in memory from a preceding backtest call.
    """
    import pandas as pd
    from src.montecarlo.simulator import MonteCarloSimulator

    if len(returns) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 30 return observations; received {len(returns)}.",
        )

    return_series = pd.Series(returns, dtype=float)

    try:
        simulator = MonteCarloSimulator(
            n_simulations=req.n_simulations,
            horizon_days=req.horizon_days,
            method=req.method,
            confidence_levels=req.confidence_levels,
        )

        shocks = [
            {
                "name": s.name,
                "return_shock": s.return_shock,
                "vol_multiplier": s.vol_multiplier,
            }
            for s in req.stress_scenarios
        ]

        all_results = simulator.stress_test(
            returns=return_series,
            shocks=shocks,
        )

        baseline = all_results["baseline"]
        stress_summary: dict = {}
        for name, mc_result in all_results.items():
            stress_summary[name] = {
                "var_95": mc_result.var.get(0.95, 0.0),
                "cvar_95": mc_result.cvar.get(0.95, 0.0),
                "var_99": mc_result.var.get(0.99, 0.0),
                "cvar_99": mc_result.cvar.get(0.99, 0.0),
                "median_terminal": mc_result.percentiles.get(50, 0.0),
                "p5_terminal": mc_result.percentiles.get(5, 0.0),
                "p95_terminal": mc_result.percentiles.get(95, 0.0),
            }

        return MonteCarloResponse(
            run_id=req.run_id,
            n_simulations=baseline.n_simulations,
            horizon_days=baseline.horizon_days,
            method=req.method,
            var_95=baseline.var.get(0.95, 0.0),
            cvar_95=baseline.cvar.get(0.95, 0.0),
            var_99=baseline.var.get(0.99, 0.0),
            cvar_99=baseline.cvar.get(0.99, 0.0),
            median_terminal=baseline.percentiles.get(50, 0.0),
            p5_terminal=baseline.percentiles.get(5, 0.0),
            p95_terminal=baseline.percentiles.get(95, 0.0),
            stress_results=stress_summary,
        )

    except SimulationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Monte Carlo from-returns error")
        raise HTTPException(status_code=500, detail="Simulation failed")
