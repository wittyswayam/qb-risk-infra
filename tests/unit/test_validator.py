"""Unit tests for OHLCV data validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import DataIntegrityError
from src.ingestion.validator import OHLCVValidator


def make_valid_df(n: int = 50) -> pd.DataFrame:
    """Generate a structurally valid OHLCV DataFrame."""
    dates = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    closes = 100 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.ones(n) * 1_000_000,
        },
        index=dates,
    )


@pytest.fixture
def validator() -> OHLCVValidator:
    return OHLCVValidator()


class TestColumnValidation:
    def test_valid_df_passes(self, validator):
        validator.validate(make_valid_df(), "TEST")

    def test_missing_close_raises(self, validator):
        df = make_valid_df().drop(columns=["close"])
        with pytest.raises(DataIntegrityError, match="missing required columns"):
            validator.validate(df, "TEST")

    def test_missing_volume_raises(self, validator):
        df = make_valid_df().drop(columns=["volume"])
        with pytest.raises(DataIntegrityError, match="missing required columns"):
            validator.validate(df, "TEST")


class TestIndexValidation:
    def test_non_datetime_index_raises(self, validator):
        df = make_valid_df().reset_index()
        with pytest.raises(DataIntegrityError, match="DatetimeIndex"):
            validator.validate(df, "TEST")

    def test_unsorted_index_raises(self, validator):
        df = make_valid_df().iloc[::-1]  # reverse order
        with pytest.raises(DataIntegrityError, match="sorted ascending"):
            validator.validate(df, "TEST")


class TestDuplicateTimestamps:
    def test_duplicate_timestamps_raise(self, validator):
        df = make_valid_df()
        df = pd.concat([df, df.iloc[:1]])
        with pytest.raises(DataIntegrityError, match="duplicate timestamps"):
            validator.validate(df, "TEST")


class TestOHLCRelationships:
    def test_high_below_open_raises(self, validator):
        df = make_valid_df()
        df.loc[df.index[5], "high"] = df.loc[df.index[5], "open"] - 1  # high < open
        with pytest.raises(DataIntegrityError, match="high < max"):
            validator.validate(df, "TEST")

    def test_low_above_close_raises(self, validator):
        df = make_valid_df()
        df.loc[df.index[3], "low"] = df.loc[df.index[3], "close"] + 2  # low > close
        with pytest.raises(DataIntegrityError, match="low > min"):
            validator.validate(df, "TEST")


class TestNonNegativeValues:
    def test_negative_volume_raises(self, validator):
        df = make_valid_df()
        df.loc[df.index[0], "volume"] = -1.0
        with pytest.raises(DataIntegrityError, match="negative values"):
            validator.validate(df, "TEST")

    def test_negative_price_raises(self, validator):
        df = make_valid_df()
        df.loc[df.index[0], "open"] = -0.1
        df.loc[df.index[0], "low"] = -0.2
        with pytest.raises(DataIntegrityError, match="negative values"):
            validator.validate(df, "TEST")
