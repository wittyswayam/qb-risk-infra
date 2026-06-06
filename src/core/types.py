"""Shared domain types and data containers used across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class OHLCV:
    """A single OHLCV bar for one symbol."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"OHLCV integrity failure for {self.symbol} @ {self.timestamp}: "
                f"open={self.open} outside [{self.low}, {self.high}]"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"OHLCV integrity failure for {self.symbol} @ {self.timestamp}: "
                f"close={self.close} outside [{self.low}, {self.high}]"
            )
        if self.volume < 0:
            raise ValueError(f"Negative volume for {self.symbol} @ {self.timestamp}")


@dataclass
class Order:
    """Instruction to buy or sell a quantity of a symbol."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class Fill:
    """Confirmed execution of an order with cost details."""

    order: Order
    fill_price: float
    fill_quantity: float
    commission: float
    slippage: float
    timestamp: datetime

    @property
    def gross_value(self) -> float:
        return self.fill_price * self.fill_quantity

    @property
    def net_value(self) -> float:
        """Value after commissions and slippage."""
        return self.gross_value + self.commission + self.slippage


@dataclass
class Position:
    """Current holding in a single symbol."""

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0

    @property
    def side(self) -> PositionSide:
        if self.quantity > 0:
            return PositionSide.LONG
        if self.quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealised_pnl(self, price: float) -> float:
        return self.quantity * (price - self.avg_cost)


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state captured at the end of each bar."""

    timestamp: datetime
    cash: float
    positions: dict[str, Position]
    prices: dict[str, float]

    @property
    def equity(self) -> float:
        mv = sum(p.market_value(self.prices.get(sym, 0.0)) for sym, p in self.positions.items())
        return self.cash + mv

    @property
    def gross_exposure(self) -> float:
        return sum(
            abs(p.market_value(self.prices.get(sym, 0.0)))
            for sym, p in self.positions.items()
        )

    @property
    def net_exposure(self) -> float:
        return sum(
            p.market_value(self.prices.get(sym, 0.0))
            for sym, p in self.positions.items()
        )


@dataclass
class BacktestResult:
    """Aggregated output from a single backtest run."""

    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    equity_curve: pd.Series  # indexed by datetime
    returns: pd.Series
    fills: list[Fill] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, float | str]:
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "start": str(self.start_date.date()),
            "end": str(self.end_date.date()),
            **self.metrics,
        }


@dataclass
class WalkForwardResult:
    """Aggregated output from a walk-forward validation run."""

    strategy_name: str
    windows: list[dict[str, object]]
    oos_returns: pd.Series
    oos_metrics: dict[str, float] = field(default_factory=dict)
    param_stability: dict[str, float] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """Output from a Monte Carlo simulation run."""

    n_simulations: int
    horizon_days: int
    paths: np.ndarray  # shape (n_simulations, horizon_days)
    terminal_values: np.ndarray  # shape (n_simulations,)
    var: dict[float, float]  # confidence_level -> VaR
    cvar: dict[float, float]  # confidence_level -> CVaR
    percentiles: dict[int, float]  # percentile -> terminal value


@dataclass
class RiskMetrics:
    """Risk analytics for a return series."""

    symbol: str
    period: str
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    beta: Optional[float] = None
    correlation_to_benchmark: Optional[float] = None
