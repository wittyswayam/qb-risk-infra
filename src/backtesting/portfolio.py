"""Portfolio state tracker for the event-driven backtesting engine.

Tracks cash, positions, fills, and computes equity at each bar. Designed to be
called once per bar by the BacktestEngine after order execution.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.core.types import Fill, Order, OrderSide, Position, PortfolioSnapshot
from src.core.exceptions import OrderExecutionError

logger = logging.getLogger(__name__)


class Portfolio:
    """Stateful portfolio that processes fills and tracks P&L.

    Args:
        initial_capital: Starting cash in account currency.
        max_position_size: Max fraction of equity in any single position.
        allow_short: Whether short positions are permitted.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        max_position_size: float = 0.25,
        allow_short: bool = False,
    ) -> None:
        self._initial_capital = initial_capital
        self._max_position_size = max_position_size
        self._allow_short = allow_short
        self._cash: float = initial_capital
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._snapshots: list[PortfolioSnapshot] = []

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def snapshots(self) -> list[PortfolioSnapshot]:
        return list(self._snapshots)

    def equity(self, prices: dict[str, float]) -> float:
        """Current total equity given a map of symbol -> last price."""
        mv = sum(
            pos.market_value(prices.get(sym, pos.avg_cost))
            for sym, pos in self._positions.items()
        )
        return self._cash + mv

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(
            abs(pos.market_value(prices.get(sym, pos.avg_cost)))
            for sym, pos in self._positions.items()
        )

    def compute_order_quantity(
        self,
        symbol: str,
        direction: int,
        current_price: float,
        current_equity: float,
    ) -> Optional[Order]:
        """Compute an order to move *symbol* to the target direction.

        Args:
            symbol: Ticker.
            direction: +1, -1, or 0 from the strategy signal.
            current_price: Last known price for the symbol.
            current_equity: Current portfolio equity for position sizing.

        Returns:
            Order to submit, or None if no action required.
        """
        if current_price <= 0:
            return None

        current_pos = self._positions.get(symbol, Position(symbol=symbol))
        current_qty = current_pos.quantity

        max_notional = current_equity * self._max_position_size
        target_qty = (max_notional / current_price) * direction

        if not self._allow_short:
            target_qty = max(target_qty, 0.0)

        delta_qty = target_qty - current_qty

        if abs(delta_qty) < 1e-6:
            return None

        side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
        qty = abs(delta_qty)

        # Cash check for buys
        if side == OrderSide.BUY:
            required_cash = qty * current_price * 1.001  # slight buffer for costs
            if required_cash > self._cash:
                qty = (self._cash * 0.99) / current_price
                if qty < 1e-6:
                    return None

        from src.core.types import OrderType
        return Order(symbol=symbol, side=side, quantity=qty, order_type=OrderType.MARKET)

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and positions based on a confirmed fill.

        Args:
            fill: Executed fill to apply.
        """
        self._fills.append(fill)
        symbol = fill.order.symbol
        qty = fill.fill_quantity
        price = fill.fill_price
        total_cost = fill.commission + fill.slippage

        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)

        pos = self._positions[symbol]

        if fill.order.side == OrderSide.BUY:
            new_qty = pos.quantity + qty
            if new_qty != 0:
                pos.avg_cost = (pos.quantity * pos.avg_cost + qty * price) / new_qty
            pos.quantity = new_qty
            self._cash -= qty * price + total_cost
        else:  # SELL
            pos.quantity -= qty
            self._cash += qty * price - total_cost

        if abs(pos.quantity) < 1e-8:
            del self._positions[symbol]

        logger.debug(
            "Applied fill: %s %s %.4f @ %.4f. Cash=%.2f",
            fill.order.side.value,
            symbol,
            qty,
            price,
            self._cash,
        )

    def record_snapshot(
        self, timestamp: datetime, prices: dict[str, float]
    ) -> PortfolioSnapshot:
        """Capture and store a point-in-time portfolio snapshot."""
        snap = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self._cash,
            positions=dict(self._positions),
            prices=dict(prices),
        )
        self._snapshots.append(snap)
        return snap

    def reset(self) -> None:
        """Reset portfolio to initial state for a new backtest run."""
        self._cash = self._initial_capital
        self._positions.clear()
        self._fills.clear()
        self._snapshots.clear()

    def equity_curve(self) -> pd.Series:
        """Build and return the equity curve from recorded snapshots."""
        if not self._snapshots:
            return pd.Series(dtype=float)
        data = {
            snap.timestamp: snap.equity
            for snap in self._snapshots
        }
        return pd.Series(data, name="equity")
