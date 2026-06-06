"""Unit tests for the backtesting engine and order router."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine
from src.backtesting.order_router import OrderRouter
from src.backtesting.portfolio import Portfolio
from src.core.exceptions import InsufficientDataError, OrderExecutionError
from src.core.types import Order, OrderSide, OrderType
from src.strategies.moving_average_crossover import MovingAverageCrossover
from src.strategies.momentum import MomentumStrategy


def make_data(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + abs(rng.normal(0, 0.003, n)))
    lows = np.minimum(opens, closes) * (1 - abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1e6 * np.ones(n)},
        index=dates,
    )


@pytest.fixture
def data():
    return make_data(300)


@pytest.fixture
def router():
    return OrderRouter(commission_bps=5.0, slippage_bps=2.0)


@pytest.fixture
def portfolio():
    return Portfolio(initial_capital=100_000.0, max_position_size=0.20)


class TestOrderRouter:
    def test_buy_fill_price_above_open(self, router, data):
        order = Order(symbol="TEST", side=OrderSide.BUY, quantity=100)
        bar = data.iloc[50]
        ts = data.index[50].to_pydatetime()
        fill = router.execute(order, bar, ts)
        assert fill.fill_price > bar["open"]
        assert fill.fill_quantity == 100
        assert fill.commission > 0
        assert fill.slippage > 0

    def test_sell_fill_price_below_open(self, router, data):
        order = Order(symbol="TEST", side=OrderSide.SELL, quantity=100)
        bar = data.iloc[50]
        ts = data.index[50].to_pydatetime()
        fill = router.execute(order, bar, ts)
        assert fill.fill_price < bar["open"]

    def test_zero_quantity_raises(self, router, data):
        order = Order(symbol="TEST", side=OrderSide.BUY, quantity=0)
        with pytest.raises(OrderExecutionError):
            router.execute(order, data.iloc[0], data.index[0].to_pydatetime())

    def test_negative_quantity_raises(self, router, data):
        order = Order(symbol="TEST", side=OrderSide.BUY, quantity=-10)
        with pytest.raises(OrderExecutionError):
            router.execute(order, data.iloc[0], data.index[0].to_pydatetime())

    def test_gross_value_property(self, router, data):
        order = Order(symbol="TEST", side=OrderSide.BUY, quantity=100)
        fill = router.execute(order, data.iloc[0], data.index[0].to_pydatetime())
        assert abs(fill.gross_value - fill.fill_price * fill.fill_quantity) < 1e-8


class TestPortfolio:
    def test_initial_equity_equals_capital(self, portfolio):
        assert portfolio.equity({}) == 100_000.0

    def test_apply_buy_fill_reduces_cash(self, router, portfolio, data):
        order = Order(symbol="TEST", side=OrderSide.BUY, quantity=100)
        fill = router.execute(order, data.iloc[0], data.index[0].to_pydatetime())
        portfolio.apply_fill(fill)
        assert portfolio.cash < 100_000.0

    def test_buy_then_sell_restores_cash_approximately(self, router, portfolio, data):
        # Buy then immediately sell at the same price
        buy_order = Order(symbol="TEST", side=OrderSide.BUY, quantity=100)
        buy_fill = router.execute(buy_order, data.iloc[0], data.index[0].to_pydatetime())
        portfolio.apply_fill(buy_fill)

        sell_order = Order(symbol="TEST", side=OrderSide.SELL, quantity=100)
        sell_fill = router.execute(sell_order, data.iloc[0], data.index[0].to_pydatetime())
        portfolio.apply_fill(sell_fill)

        # After round-trip, cash should be reduced by total transaction costs
        total_cost = buy_fill.commission + buy_fill.slippage + sell_fill.commission + sell_fill.slippage
        expected = 100_000.0 - total_cost
        assert portfolio.cash == pytest.approx(expected, rel=0.01)

    def test_position_created_after_buy(self, router, portfolio, data):
        order = Order(symbol="TEST", side=OrderSide.BUY, quantity=50)
        fill = router.execute(order, data.iloc[0], data.index[0].to_pydatetime())
        portfolio.apply_fill(fill)
        assert "TEST" in portfolio.positions
        assert portfolio.positions["TEST"].quantity == pytest.approx(50)

    def test_reset_clears_state(self, router, portfolio, data):
        order = Order(symbol="TEST", side=OrderSide.BUY, quantity=10)
        fill = router.execute(order, data.iloc[0], data.index[0].to_pydatetime())
        portfolio.apply_fill(fill)
        portfolio.reset()
        assert portfolio.cash == 100_000.0
        assert portfolio.positions == {}
        assert portfolio.fills == []

    def test_snapshot_records_equity(self, router, portfolio, data):
        portfolio.record_snapshot(data.index[0].to_pydatetime(), {})
        assert len(portfolio.snapshots) == 1
        assert portfolio.snapshots[0].equity == 100_000.0

    def test_compute_order_quantity_long(self, portfolio):
        order = portfolio.compute_order_quantity(
            symbol="TEST", direction=1,
            current_price=100.0, current_equity=100_000.0
        )
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.quantity > 0

    def test_compute_order_quantity_flat_returns_none(self, portfolio):
        order = portfolio.compute_order_quantity(
            symbol="TEST", direction=0,
            current_price=100.0, current_equity=100_000.0
        )
        assert order is None  # No existing position to close


class TestBacktestEngine:
    def test_run_produces_result(self, data, router, portfolio):
        strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
        engine = BacktestEngine(strategy=strategy, order_router=router, portfolio=portfolio)
        result = engine.run(data=data, symbol="TEST")
        assert result.strategy_name == "MovingAverageCrossover"
        assert result.symbol == "TEST"
        assert not result.equity_curve.empty
        assert len(result.metrics) > 0

    def test_insufficient_data_raises(self, router, portfolio):
        strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
        engine = BacktestEngine(strategy=strategy, order_router=router, portfolio=portfolio)
        tiny_data = make_data(10)  # Only 10 bars, need 51
        with pytest.raises(InsufficientDataError):
            engine.run(data=tiny_data, symbol="TEST")

    def test_equity_curve_length(self, data, router, portfolio):
        strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
        engine = BacktestEngine(strategy=strategy, order_router=router, portfolio=portfolio)
        result = engine.run(data=data, symbol="TEST")
        # Should have one snapshot per bar after required_history
        expected_bars = len(data) - strategy.required_history
        assert len(result.equity_curve) == expected_bars

    def test_metrics_keys_present(self, data, router, portfolio):
        strategy = MomentumStrategy(lookback=30, vol_window=15)
        engine = BacktestEngine(strategy=strategy, order_router=router, portfolio=portfolio)
        result = engine.run(data=data, symbol="TEST")
        for key in ["sharpe_ratio", "max_drawdown", "annualised_return", "var_95"]:
            assert key in result.metrics

    def test_returns_match_equity_changes(self, data, router, portfolio):
        strategy = MovingAverageCrossover(fast_period=5, slow_period=20)
        engine = BacktestEngine(strategy=strategy, order_router=router, portfolio=portfolio)
        result = engine.run(data=data, symbol="TEST")
        # Computed returns from equity curve should match stored returns
        computed_returns = result.equity_curve.pct_change().dropna()
        stored_returns = result.returns
        pd.testing.assert_series_equal(
            computed_returns.round(10),
            stored_returns.round(10),
            check_names=False,
        )
