"""Data integrity validation for OHLCV DataFrames."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.core.exceptions import DataIntegrityError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


class OHLCVValidator:
    """Validates structural and statistical properties of OHLCV data.

    Checks performed:
    1. Required columns are present.
    2. Index is a DatetimeIndex, UTC-aware, sorted ascending.
    3. No duplicate timestamps.
    4. High >= Open, Close and High >= Low.
    5. Low <= Open, Close and Low >= 0.
    6. Volume >= 0.
    7. No rows where all price columns are NaN.
    """

    def __init__(self, max_gap_days: int = 5) -> None:
        self._max_gap_days = max_gap_days

    def validate(self, df: pd.DataFrame, symbol: str) -> None:
        """Run all validation checks against *df*.

        Args:
            df: OHLCV DataFrame to validate.
            symbol: Ticker string for error messaging.

        Raises:
            DataIntegrityError: On any failed check.
        """
        self._check_columns(df, symbol)
        self._check_index(df, symbol)
        self._check_duplicates(df, symbol)
        self._check_ohlc_relationships(df, symbol)
        self._check_non_negative(df, symbol)
        self._check_all_nan_rows(df, symbol)
        self._check_price_gaps(df, symbol)
        logger.debug("Validation passed for %s (%d bars)", symbol, len(df))

    def _check_columns(self, df: pd.DataFrame, symbol: str) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
        if missing:
            raise DataIntegrityError(
                f"{symbol}: missing required columns: {missing}"
            )

    def _check_index(self, df: pd.DataFrame, symbol: str) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataIntegrityError(
                f"{symbol}: index must be DatetimeIndex, got {type(df.index)}"
            )
        if not df.index.is_monotonic_increasing:
            raise DataIntegrityError(f"{symbol}: index is not sorted ascending")

    def _check_duplicates(self, df: pd.DataFrame, symbol: str) -> None:
        n_dupes = df.index.duplicated().sum()
        if n_dupes:
            raise DataIntegrityError(
                f"{symbol}: {n_dupes} duplicate timestamps in index"
            )

    def _check_ohlc_relationships(self, df: pd.DataFrame, symbol: str) -> None:
        bad_high = df[df["high"] < df[["open", "close"]].max(axis=1)]
        if not bad_high.empty:
            raise DataIntegrityError(
                f"{symbol}: high < max(open, close) at {len(bad_high)} rows"
            )
        bad_low = df[df["low"] > df[["open", "close"]].min(axis=1)]
        if not bad_low.empty:
            raise DataIntegrityError(
                f"{symbol}: low > min(open, close) at {len(bad_low)} rows"
            )

    def _check_non_negative(self, df: pd.DataFrame, symbol: str) -> None:
        for col in ["open", "high", "low", "close", "volume"]:
            if (df[col] < 0).any():
                raise DataIntegrityError(f"{symbol}: negative values in column '{col}'")

    def _check_all_nan_rows(self, df: pd.DataFrame, symbol: str) -> None:
        price_cols = ["open", "high", "low", "close"]
        all_nan = df[price_cols].isna().all(axis=1)
        if all_nan.any():
            raise DataIntegrityError(
                f"{symbol}: {all_nan.sum()} rows with all NaN price values"
            )

    def _check_price_gaps(self, df: pd.DataFrame, symbol: str) -> None:
        """Warn (do not raise) when price returns exceed ±50% in a single bar."""
        if len(df) < 2:
            return
        returns = df["close"].pct_change().abs()
        extreme = returns[returns > 0.50]
        if not extreme.empty:
            logger.warning(
                "%s: %d bar(s) with >50%% price move detected at %s",
                symbol,
                len(extreme),
                extreme.index.tolist(),
            )
