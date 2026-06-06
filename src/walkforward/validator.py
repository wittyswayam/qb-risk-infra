"""Walk-forward validation engine for out-of-sample performance assessment.

Walk-forward validation prevents look-ahead bias by training strategy
parameters exclusively on past data, then evaluating on unseen future bars.
This mirrors how a strategy would be operated live: the model is periodically
retrained on available history, and the next out-of-sample window is traded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.types import WalkForwardResult
from src.core.exceptions import WalkForwardError
from src.strategies.base import BaseStrategy
from src.backtesting.engine import BacktestEngine
from src.backtesting.order_router import OrderRouter
from src.backtesting.portfolio import Portfolio
from src.backtesting.metrics import compute_all_metrics, sharpe_ratio

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """A single train/test split used in walk-forward validation."""

    window_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: pd.DataFrame
    test_data: pd.DataFrame


class WalkForwardValidator:
    """Anchored or rolling walk-forward validation.

    Generates non-overlapping out-of-sample windows. For each window:
    1. Train on the in-sample period (parameter selection via grid search).
    2. Evaluate on the adjacent out-of-sample period.
    3. Step forward and repeat.

    The concatenated out-of-sample results form an unbiased performance
    estimate across the full data range.

    Args:
        strategy_factory: Callable(params) -> BaseStrategy. Used to
            instantiate fresh strategy copies with each parameter set.
        param_grid: List of parameter dicts to evaluate during training.
        train_window_days: Length of each training window in trading days.
        test_window_days: Length of each test window in trading days.
        step_days: How many days to advance the window each iteration.
        initial_capital: Portfolio start capital.
        commission_bps: Transaction cost in basis points.
        slippage_bps: Slippage in basis points.
        risk_free_rate: Annualised risk-free rate for Sharpe scoring.
        trading_days: Trading days per year.
    """

    def __init__(
        self,
        strategy_factory,
        param_grid: list[dict[str, Any]],
        train_window_days: int = 252,
        test_window_days: int = 63,
        step_days: int = 21,
        initial_capital: float = 100_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        risk_free_rate: float = 0.04,
        trading_days: int = 252,
    ) -> None:
        self._factory = strategy_factory
        self._param_grid = param_grid
        self._train_days = train_window_days
        self._test_days = test_window_days
        self._step_days = step_days
        self._initial_capital = initial_capital
        self._commission_bps = commission_bps
        self._slippage_bps = slippage_bps
        self._rfr = risk_free_rate
        self._trading_days = trading_days

    def run(self, data: pd.DataFrame, symbol: str) -> WalkForwardResult:
        """Execute walk-forward validation on *data* for *symbol*.

        Args:
            data: Full OHLCV DataFrame with DatetimeIndex.
            symbol: Ticker label for reporting.

        Returns:
            WalkForwardResult with concatenated OOS returns and metrics.
        """
        windows = self._build_windows(data)
        if not windows:
            raise WalkForwardError(
                f"Insufficient data for walk-forward validation. "
                f"Need at least {self._train_days + self._test_days} bars, "
                f"got {len(data)}."
            )

        logger.info(
            "Walk-forward: %d windows, train=%d days, test=%d days",
            len(windows),
            self._train_days,
            self._test_days,
        )

        window_results: list[dict[str, Any]] = []
        oos_returns_list: list[pd.Series] = []
        best_params_per_window: list[dict[str, Any]] = []

        for win in windows:
            logger.debug(
                "Window %d: train [%s, %s] test [%s, %s]",
                win.window_idx,
                win.train_start.date(),
                win.train_end.date(),
                win.test_start.date(),
                win.test_end.date(),
            )

            # In-sample: find best parameters by Sharpe ratio
            best_params, best_sharpe = self._select_params(win.train_data, symbol)

            # Out-of-sample: evaluate with selected parameters
            oos_result = self._evaluate(
                data=win.test_data,
                symbol=symbol,
                params=best_params,
            )

            oos_returns = oos_result.returns if oos_result is not None else pd.Series(dtype=float)
            oos_returns_list.append(oos_returns)
            best_params_per_window.append(best_params)

            window_results.append({
                "window": win.window_idx,
                "train_start": str(win.train_start.date()),
                "train_end": str(win.train_end.date()),
                "test_start": str(win.test_start.date()),
                "test_end": str(win.test_end.date()),
                "best_params": best_params,
                "is_sharpe": round(best_sharpe, 4),
                "oos_sharpe": round(
                    sharpe_ratio(oos_returns, self._rfr, self._trading_days), 4
                ) if not oos_returns.empty else None,
                "oos_n_bars": len(oos_returns),
            })

        combined_returns = pd.concat(oos_returns_list).sort_index() if oos_returns_list else pd.Series(dtype=float)

        oos_metrics: dict[str, float] = {}
        if not combined_returns.empty:
            equity = (1 + combined_returns).cumprod() * self._initial_capital
            oos_metrics = compute_all_metrics(
                returns=combined_returns,
                equity_curve=equity,
                risk_free_rate=self._rfr,
                trading_days=self._trading_days,
            )

        param_stability = self._compute_param_stability(best_params_per_window)

        return WalkForwardResult(
            strategy_name=self._factory({}).name,
            windows=window_results,
            oos_returns=combined_returns,
            oos_metrics=oos_metrics,
            param_stability=param_stability,
        )

    def _build_windows(self, data: pd.DataFrame) -> list[WalkForwardWindow]:
        n = len(data)
        windows: list[WalkForwardWindow] = []
        start_idx = 0
        win_idx = 0

        while start_idx + self._train_days + self._test_days <= n:
            train_end_idx = start_idx + self._train_days
            test_end_idx = train_end_idx + self._test_days

            train_slice = data.iloc[start_idx:train_end_idx]
            test_slice = data.iloc[train_end_idx:test_end_idx]

            if len(train_slice) >= self._train_days and len(test_slice) >= 1:
                windows.append(WalkForwardWindow(
                    window_idx=win_idx,
                    train_start=data.index[start_idx].to_pydatetime(),
                    train_end=data.index[train_end_idx - 1].to_pydatetime(),
                    test_start=data.index[train_end_idx].to_pydatetime(),
                    test_end=data.index[test_end_idx - 1].to_pydatetime(),
                    train_data=train_slice,
                    test_data=test_slice,
                ))
                win_idx += 1

            start_idx += self._step_days

        return windows

    def _select_params(
        self,
        train_data: pd.DataFrame,
        symbol: str,
    ) -> tuple[dict[str, Any], float]:
        """Grid-search parameters on in-sample data, scoring by Sharpe ratio."""
        best_params: dict[str, Any] = self._param_grid[0] if self._param_grid else {}
        best_sharpe = float("-inf")

        for params in self._param_grid:
            result = self._evaluate(train_data, symbol, params)
            if result is None or result.returns.empty:
                continue
            sr = sharpe_ratio(result.returns, self._rfr, self._trading_days)
            if sr > best_sharpe:
                best_sharpe = sr
                best_params = params

        return best_params, best_sharpe

    def _evaluate(self, data: pd.DataFrame, symbol: str, params: dict[str, Any]):
        try:
            strategy = self._factory(params)
            router = OrderRouter(
                commission_bps=self._commission_bps,
                slippage_bps=self._slippage_bps,
            )
            portfolio = Portfolio(initial_capital=self._initial_capital)
            engine = BacktestEngine(
                strategy=strategy,
                order_router=router,
                portfolio=portfolio,
                risk_free_rate=self._rfr,
                trading_days=self._trading_days,
            )
            return engine.run(data=data, symbol=symbol)
        except Exception as exc:
            logger.warning("Evaluation failed for params %s: %s", params, exc)
            return None

    def _compute_param_stability(
        self, params_per_window: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Compute coefficient of variation for each numeric parameter.

        Lower CV => more stable parameter selection across windows.
        CV = std / mean; values > 0.3 suggest high instability.
        """
        if not params_per_window:
            return {}

        param_names = [
            k for k in params_per_window[0] if isinstance(params_per_window[0][k], (int, float))
        ]
        stability: dict[str, float] = {}
        for name in param_names:
            values = np.array([
                float(p[name]) for p in params_per_window if name in p
            ])
            if len(values) < 2 or values.mean() == 0:
                stability[name] = 0.0
            else:
                stability[name] = round(float(values.std() / abs(values.mean())), 4)

        return stability
