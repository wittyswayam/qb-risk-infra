"""Mean-reversion strategy using Z-score of price relative to a rolling mean.

The strategy assumes that short-term price deviations from a rolling mean
are mean-reverting. Positions are entered when the Z-score exceeds a threshold
and exited when it returns toward zero.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """Z-score based mean reversion.

    Entry logic:
        Z-score < -entry_z  -> go LONG (price cheap relative to mean)
        Z-score >  entry_z  -> go SHORT (price rich relative to mean)

    Exit logic:
        |Z-score| < exit_z  -> close position (price near fair value)

    Args:
        lookback: Rolling window for mean and standard deviation.
        entry_z: Z-score magnitude required to open a position.
        exit_z: Z-score magnitude below which position is closed.
        allow_short: If False, short signals are replaced with flat.
    """

    def __init__(
        self,
        lookback: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        allow_short: bool = True,
    ) -> None:
        if entry_z <= exit_z:
            raise ValueError("entry_z must be greater than exit_z")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.allow_short = allow_short
        self._position: int = 0

    @property
    def name(self) -> str:
        return "MeanReversion"

    @property
    def required_history(self) -> int:
        return self.lookback + 1

    def on_bar(
        self,
        symbol: str,
        history: pd.DataFrame,
        current_bar: pd.Series,
    ) -> Signal:
        close = history["close"]
        roll_mean = close.rolling(self.lookback).mean()
        roll_std = close.rolling(self.lookback).std(ddof=1)

        mean_val = roll_mean.iloc[-1]
        std_val = roll_std.iloc[-1]

        if std_val == 0 or pd.isna(std_val):
            return Signal(symbol=symbol, direction=0, strength=0.0)

        z_score = (close.iloc[-1] - mean_val) / std_val

        if self._position == 0:
            if z_score < -self.entry_z:
                self._position = 1
            elif z_score > self.entry_z:
                self._position = -1 if self.allow_short else 0
        elif self._position == 1 and z_score > self.exit_z:
            self._position = 0
        elif self._position == -1 and z_score < -self.exit_z:
            self._position = 0

        # Re-check direction flip
        if self._position == 1 and z_score > self.entry_z:
            self._position = -1 if self.allow_short else 0
        elif self._position == -1 and z_score < -self.entry_z:
            self._position = 1

        return Signal(
            symbol=symbol,
            direction=self._position,
            strength=float(-z_score / self.entry_z),  # normalised; positive = bullish
            metadata={
                "z_score": round(z_score, 4),
                "roll_mean": round(mean_val, 4),
                "roll_std": round(std_val, 4),
            },
        )

    def reset(self) -> None:
        self._position = 0

    def get_params(self) -> dict:
        return {
            "lookback": self.lookback,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "allow_short": self.allow_short,
        }
