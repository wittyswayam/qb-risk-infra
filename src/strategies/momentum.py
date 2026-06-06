"""Time-series momentum strategy based on trailing return.

A momentum signal is computed as the total return over a lookback window,
with position sizing proportional to signal strength when ``volatility_scaled``
is enabled (cf. Moskowitz, Ooi & Pedersen 2012).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """Time-series momentum (trend-following) signal.

    Signal is the sign of the past ``lookback``-day total return.  When
    ``volatility_scaled`` is True the signal strength is the return divided
    by the realised volatility, matching the TSMOM approach.

    Args:
        lookback: Lookback window in bars for return computation.
        vol_window: Lookback window for volatility estimation.
        volatility_scaled: Scale signal by inverse realised volatility.
        allow_short: If False, negative momentum produces a flat signal.
    """

    def __init__(
        self,
        lookback: int = 63,
        vol_window: int = 21,
        volatility_scaled: bool = True,
        allow_short: bool = True,
    ) -> None:
        self.lookback = lookback
        self.vol_window = vol_window
        self.volatility_scaled = volatility_scaled
        self.allow_short = allow_short

    @property
    def name(self) -> str:
        return "Momentum"

    @property
    def required_history(self) -> int:
        return max(self.lookback, self.vol_window) + 1

    def on_bar(
        self,
        symbol: str,
        history: pd.DataFrame,
        current_bar: pd.Series,
    ) -> Signal:
        close = history["close"]

        if len(close) < self.required_history:
            return Signal(symbol=symbol, direction=0, strength=0.0)

        # Total return over lookback period (excluding current bar)
        past_return = (close.iloc[-1] / close.iloc[-(self.lookback + 1)]) - 1.0

        # Realised volatility: annualised std of daily log returns
        log_rets = np.log(close / close.shift(1)).dropna()
        realized_vol = float(log_rets.iloc[-self.vol_window:].std() * np.sqrt(252))

        if self.volatility_scaled and realized_vol > 0:
            strength = past_return / realized_vol
        else:
            strength = float(np.sign(past_return))

        direction = int(np.sign(past_return))
        if not self.allow_short and direction == -1:
            direction = 0

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(np.clip(strength, -3.0, 3.0)),
            metadata={
                "past_return": round(past_return, 6),
                "realized_vol": round(realized_vol, 6),
                "vol_scaled_signal": self.volatility_scaled,
            },
        )

    def reset(self) -> None:
        pass  # stateless between bars

    def get_params(self) -> dict:
        return {
            "lookback": self.lookback,
            "vol_window": self.vol_window,
            "volatility_scaled": self.volatility_scaled,
            "allow_short": self.allow_short,
        }
