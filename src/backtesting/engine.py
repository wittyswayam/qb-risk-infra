"""Event-driven backtesting engine.

The engine iterates bar-by-bar through historical data, calling the strategy
on each bar, routing the resulting signal to the order router, and updating
the portfolio. Metrics are computed at the end of the run.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.core.types import BacktestResult, Order, OrderSide
from src.core.exceptions import BacktestError, InsufficientDataError
from src.strategies.base import BaseStrategy
from src.backtesting.order_router import OrderRouter
from src.backtesting.portfolio import Portfolio
from src.backtesting.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Bar-by-bar event-driven backtesting engine.

    The engine operates on a single symbol. Multi-symbol backtests
    can be constructed by running multiple single-symbol engines and
    combining their results in the portfolio layer.

    Args:
        strategy: Initialised strategy instance.
        order_router: Order execution simulator.
        portfolio: Portfolio state tracker.
        risk_free_rate: Annualised risk-free rate for Sharpe computation.
        trading_days: Trading days per year for annualisation.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        order_router: OrderRouter,
        portfolio: Portfolio,
        risk_free_rate: float = 0.04,
        trading_days: int = 252,
    ) -> None:
        self._strategy = strategy
        self._router = order_router
        self._portfolio = portfolio
        self._risk_free_rate = risk_free_rate
        self._trading_days = trading_days

    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
    ) -> BacktestResult:
        """Execute a full backtest over *data* for *symbol*.

        Args:
            data: OHLCV DataFrame with DatetimeIndex. Must have enough bars
                  to satisfy strategy.required_history.
            symbol: Ticker string for labelling outputs.

        Returns:
            BacktestResult with equity curve, fills, and performance metrics.

        Raises:
            InsufficientDataError: If data has fewer bars than required_history.
            BacktestError: On unexpected engine failure.
        """
        required = self._strategy.required_history
        if len(data) < required:
            raise InsufficientDataError(
                f"Strategy '{self._strategy.name}' requires {required} bars; "
                f"got {len(data)} for {symbol}."
            )

        self._portfolio.reset()
        self._strategy.reset()

        logger.info(
            "Starting backtest: strategy=%s symbol=%s bars=%d",
            self._strategy.name,
            symbol,
            len(data),
        )

        for i in range(required, len(data)):
            history = data.iloc[: i + 1]
            current_bar = data.iloc[i]
            timestamp = data.index[i]

            # 1. Generate signal from strategy
            try:
                signal = self._strategy.on_bar(
                    symbol=symbol,
                    history=history,
                    current_bar=current_bar,
                )
            except Exception as exc:
                raise BacktestError(
                    f"Strategy error at bar {timestamp}: {exc}"
                ) from exc

            # 2. Determine desired position change
            current_prices = {symbol: float(current_bar["close"])}
            current_equity = self._portfolio.equity(current_prices)

            order = self._portfolio.compute_order_quantity(
                symbol=symbol,
                direction=signal.direction,
                current_price=float(current_bar["close"]),
                current_equity=current_equity,
            )

            # 3. Execute order if any (fills on next bar open; here we use
            #    current bar open for simplicity with same-bar data)
            if order is not None:
                try:
                    fill = self._router.execute(
                        order=order,
                        bar=current_bar,
                        timestamp=timestamp,
                    )
                    self._portfolio.apply_fill(fill)
                except Exception as exc:
                    logger.warning("Order execution failed at %s: %s", timestamp, exc)

            # 4. Snapshot the portfolio at close
            close_prices = {symbol: float(current_bar["close"])}
            self._portfolio.record_snapshot(timestamp, close_prices)

        # 5. Compute performance metrics
        equity_curve = self._portfolio.equity_curve()
        if equity_curve.empty:
            raise BacktestError("Backtest produced an empty equity curve.")

        returns = equity_curve.pct_change().dropna()
        metrics = compute_all_metrics(
            returns=returns,
            equity_curve=equity_curve,
            risk_free_rate=self._risk_free_rate,
            trading_days=self._trading_days,
        )

        logger.info(
            "Backtest complete: sharpe=%.3f max_dd=%.3f ann_ret=%.3f",
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown", 0),
            metrics.get("annualised_return", 0),
        )

        return BacktestResult(
            strategy_name=self._strategy.name,
            symbol=symbol,
            start_date=data.index[0].to_pydatetime(),
            end_date=data.index[-1].to_pydatetime(),
            equity_curve=equity_curve,
            returns=returns,
            fills=self._portfolio.fills,
            snapshots=self._portfolio.snapshots,
            metrics=metrics,
        )
