"""Shared pytest fixtures available to all tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_ohlcv() -> pd.DataFrame:
    """Session-scoped synthetic OHLCV data (500 trading days).

    Uses a deterministic GBM-like process so tests are reproducible.
    """
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    closes = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.013, n))
    opens = closes * np.exp(rng.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volumes = rng.uniform(5e5, 5e6, n)

    return pd.DataFrame(
        {
            "open": opens.round(4),
            "high": highs.round(4),
            "low": lows.round(4),
            "close": closes.round(4),
            "volume": volumes.astype(int),
        },
        index=dates,
    )


@pytest.fixture(scope="session")
def sample_returns(sample_ohlcv) -> pd.Series:
    """Daily log-return series derived from the session OHLCV fixture."""
    return sample_ohlcv["close"].pct_change().dropna()


@pytest.fixture(scope="session")
def benchmark_returns() -> pd.Series:
    """Synthetic benchmark return series for beta/correlation tests."""
    rng = np.random.default_rng(7)
    n = 499
    return pd.Series(rng.normal(0.0002, 0.010, n), name="benchmark")
