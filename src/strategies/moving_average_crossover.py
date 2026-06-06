"""Moving Average Crossover strategy.

Generates a long signal when the fast EMA crosses above the slow EMA and
exits (goes flat) when the fast EMA crosses below the slow EMA.  The strategy
does not go short by default; set ``allow_short=True`` to invert on cross-down.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import BaseStrategy, Signal


class MovingAverageCrossover(BaseStrategy):
    """Dual EMA crossover signal generator.

    Args:
        fast_period: Lookback for the fast exponential moving average.
        slow_period: Lookback for the slow exponential moving average.
        allow_short: If True, emit -1 signals on bearish crosses.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        allow_short: bool = False,
    ) -> None:
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.allow_short = allow_short
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None
        self._position: int = 0  # current position: +1, -1, 0

    @property
    def name(self) -> str:
        return "MovingAverageCrossover"

    @property
    def required_history(self) -> int:
        return self.slow_period + 1

    def on_bar(
        self,
        symbol: str,
        history: pd.DataFrame,
        current_bar: pd.Series,
    ) -> Signal:
        close = history["close"]

        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        curr_fast = fast_ema.iloc[-1]
        curr_slow = slow_ema.iloc[-1]
        prev_fast = fast_ema.iloc[-2]
        prev_slow = slow_ema.iloc[-2]

        bullish_cross = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
        bearish_cross = (prev_fast >= prev_slow) and (curr_fast < curr_slow)

        if bullish_cross:
            self._position = 1
        elif bearish_cross:
            self._position = -1 if self.allow_short else 0

        spread = (curr_fast - curr_slow) / curr_slow if curr_slow != 0 else 0.0

        return Signal(
            symbol=symbol,
            direction=self._position,
            strength=float(spread),
            metadata={
                "fast_ema": round(curr_fast, 4),
                "slow_ema": round(curr_slow, 4),
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
            },
        )

    def reset(self) -> None:
        self._prev_fast = None
        self._prev_slow = None
        self._position = 0

    def get_params(self) -> dict:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "allow_short": self.allow_short,
        }
