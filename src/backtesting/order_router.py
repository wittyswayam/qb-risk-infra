"""Simulated order execution with slippage and commission modelling.

The order router translates strategy signals into Fill objects, applying
realistic transaction cost assumptions. The model uses a proportional
slippage model (slippage scales with price) and a flat commission per trade.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from src.core.types import Fill, Order, OrderSide, Position
from src.core.exceptions import OrderExecutionError

logger = logging.getLogger(__name__)


class OrderRouter:
    """Simulates order fills against bar data.

    Transaction cost model:
        - Commission: ``commission_bps`` basis points on gross notional.
        - Slippage: ``slippage_bps`` basis points of adverse price movement
          applied to the fill price (buys fill slightly higher, sells lower).

    Args:
        commission_bps: Commission in basis points (1 bp = 0.01%).
        slippage_bps: Slippage in basis points.
    """

    def __init__(
        self,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
    ) -> None:
        self._commission_bps = commission_bps / 10_000
        self._slippage_bps = slippage_bps / 10_000

    def execute(
        self,
        order: Order,
        bar: pd.Series,
        timestamp: datetime,
    ) -> Fill:
        """Execute *order* against *bar* prices and return a Fill.

        Fill price is the bar's open (next-bar execution convention), adjusted
        for slippage. Commission is charged as a flat percentage of notional.

        Args:
            order: The order to execute.
            bar: OHLCV Series for the execution bar.
            timestamp: Timestamp of the execution bar.

        Returns:
            Fill object with complete cost details.

        Raises:
            OrderExecutionError: If order quantity is zero or fill price invalid.
        """
        if order.quantity <= 0:
            raise OrderExecutionError(
                f"Order quantity must be positive, got {order.quantity}"
            )

        # Use the bar's open as the base execution price (next-open convention)
        base_price = float(bar["open"])
        if base_price <= 0:
            raise OrderExecutionError(
                f"Invalid base price {base_price} for {order.symbol} @ {timestamp}"
            )

        # Apply directional slippage: buys fill above open, sells below
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + self._slippage_bps)
            slippage_cost = base_price * self._slippage_bps * order.quantity
        else:
            fill_price = base_price * (1 - self._slippage_bps)
            slippage_cost = base_price * self._slippage_bps * order.quantity

        gross_notional = fill_price * order.quantity
        commission_cost = gross_notional * self._commission_bps

        logger.debug(
            "Fill: %s %s %.2f @ %.4f (slip=%.4f, comm=%.4f)",
            order.side.value,
            order.symbol,
            order.quantity,
            fill_price,
            slippage_cost,
            commission_cost,
        )

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission_cost,
            slippage=slippage_cost,
            timestamp=timestamp,
        )
