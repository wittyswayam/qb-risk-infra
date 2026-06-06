"""Volatility Breakout strategy based on Bollinger Bands and ATR.

Enters a long position when price closes above the upper Bollinger Band
(a volatility expansion signal) and exits when price reverts inside the
bands. Uses Average True Range to size exposure relative to current
volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, Signal


class VolatilityBreakoutStrategy(BaseStrategy):
    """Bollinger Band breakout with ATR-based confirmation.

    A breakout is identified when the closing price crosses outside the
    Bollinger Bands. The signal strength is the normalised distance of price
    from the band in units of ATR.

    Args:
        bb_period: Lookback for Bollinger Band mean and std.
        bb_std: Number of standard deviations for the bands.
        atr_period: Lookback for Average True Range computation.
        allow_short: If True, emit short signals on lower-band breakouts.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        allow_short: bool = False,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.allow_short = allow_short
        self._position: int = 0

    @property
    def name(self) -> str:
        return "VolatilityBreakout"

    @property
    def required_history(self) -> int:
        return max(self.bb_period, self.atr_period) + 1

    def on_bar(
        self,
        symbol: str,
        history: pd.DataFrame,
        current_bar: pd.Series,
    ) -> Signal:
        close = history["close"]
        high = history["high"]
        low = history["low"]

        roll_mean = close.rolling(self.bb_period).mean()
        roll_std = close.rolling(self.bb_period).std(ddof=1)
        upper_band = roll_mean + self.bb_std * roll_std
        lower_band = roll_mean - self.bb_std * roll_std

        atr = self._compute_atr(high, low, close)

        curr_close = close.iloc[-1]
        curr_upper = upper_band.iloc[-1]
        curr_lower = lower_band.iloc[-1]
        curr_atr = atr.iloc[-1]
        curr_mean = roll_mean.iloc[-1]

        above_upper = curr_close > curr_upper
        below_lower = curr_close < curr_lower
        inside_bands = curr_lower <= curr_close <= curr_upper

        if above_upper:
            self._position = 1
        elif below_lower:
            self._position = -1 if self.allow_short else 0
        elif inside_bands:
            self._position = 0

        if curr_atr > 0:
            distance = (curr_close - curr_mean) / curr_atr
        else:
            distance = 0.0

        return Signal(
            symbol=symbol,
            direction=self._position,
            strength=float(np.clip(distance, -5.0, 5.0)),
            metadata={
                "upper_band": round(curr_upper, 4),
                "lower_band": round(curr_lower, 4),
                "bb_mean": round(curr_mean, 4),
                "atr": round(curr_atr, 4),
                "above_upper": above_upper,
                "below_lower": below_lower,
            },
        )

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute Wilder's Average True Range."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()

    def reset(self) -> None:
        self._position = 0

    def get_params(self) -> dict:
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "atr_period": self.atr_period,
            "allow_short": self.allow_short,
        }
