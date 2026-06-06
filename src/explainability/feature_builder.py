"""Feature engineering for strategy signal explainability.

Builds a feature matrix from raw OHLCV data for use with SHAP-based
interpretability analysis. Features are chosen to be interpretable and
financially meaningful rather than purely ML-driven.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Constructs interpretable technical features from OHLCV bars.

    All features are computed in-place from raw price and volume data.
    Features are scaled and named for use with tree-based models and SHAP.

    Args:
        short_window: Short-term lookback (e.g., 10 days).
        medium_window: Medium-term lookback (e.g., 21 days).
        long_window: Long-term lookback (e.g., 63 days).
        vol_window: Window for realised volatility.
    """

    def __init__(
        self,
        short_window: int = 10,
        medium_window: int = 21,
        long_window: int = 63,
        vol_window: int = 21,
    ) -> None:
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.vol_window = vol_window

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Construct the feature matrix from an OHLCV DataFrame.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.

        Returns:
            Feature DataFrame with the same index, NaN rows dropped.
        """
        df = data.copy()
        features: dict[str, pd.Series] = {}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", pd.Series(np.nan, index=df.index))

        # --- Trend features ---
        log_ret = np.log(close / close.shift(1))

        features["ret_1d"] = log_ret
        features["ret_5d"] = np.log(close / close.shift(5))
        features["ret_21d"] = np.log(close / close.shift(self.medium_window))
        features["ret_63d"] = np.log(close / close.shift(self.long_window))

        # EMAs and spread
        ema_short = close.ewm(span=self.short_window, adjust=False).mean()
        ema_medium = close.ewm(span=self.medium_window, adjust=False).mean()
        ema_long = close.ewm(span=self.long_window, adjust=False).mean()

        features["ema_spread_sm"] = (ema_short - ema_medium) / ema_medium
        features["ema_spread_ml"] = (ema_medium - ema_long) / ema_long
        features["price_vs_ema_long"] = (close - ema_long) / ema_long

        # --- Volatility features ---
        roll_std_short = log_ret.rolling(self.short_window).std(ddof=1)
        roll_std_long = log_ret.rolling(self.long_window).std(ddof=1)

        features["realised_vol_21d"] = log_ret.rolling(self.vol_window).std(ddof=1) * np.sqrt(252)
        features["vol_ratio"] = roll_std_short / roll_std_long.replace(0, np.nan)

        # Parkinson's high-low volatility estimator
        features["parkinson_vol"] = (
            np.log(high / low) ** 2 / (4 * np.log(2))
        ).rolling(self.vol_window).mean().apply(lambda x: np.sqrt(max(x, 0)) * np.sqrt(252))

        # --- Mean reversion features ---
        roll_mean = close.rolling(self.medium_window).mean()
        roll_std = close.rolling(self.medium_window).std(ddof=1)
        features["z_score_21d"] = (close - roll_mean) / roll_std.replace(0, np.nan)

        # Relative Strength Index (RSI)
        features["rsi_14"] = self._compute_rsi(close, period=14)

        # --- Volume features ---
        vol_ma = volume.rolling(self.medium_window).mean()
        features["volume_ratio"] = volume / vol_ma.replace(0, np.nan)
        features["log_volume"] = np.log1p(volume)

        # --- Drawdown from recent high ---
        roll_high = close.rolling(self.long_window).max()
        features["drawdown_from_high"] = (close - roll_high) / roll_high

        # --- ATR-normalised range ---
        atr = self._compute_atr(high, low, close, period=14)
        features["atr_pct"] = atr / close

        feature_df = pd.DataFrame(features, index=df.index)
        n_before = len(feature_df)
        feature_df = feature_df.dropna()
        logger.debug(
            "FeatureBuilder: %d -> %d rows after dropping NaN (%.1f%% kept)",
            n_before,
            len(feature_df),
            100 * len(feature_df) / n_before if n_before > 0 else 0,
        )
        return feature_df

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's RSI."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
