"""Generate synthetic OHLCV CSV data for local development and testing.

This script creates realistic-looking price series using a geometric Brownian
motion model with configurable drift and volatility. It is not a market data
substitute but provides self-contained data so the system can be run without
external API keys.

Usage::

    python scripts/generate_sample_data.py \
        --symbols AAPL MSFT SPY QQQ \
        --start 2015-01-01 \
        --end 2023-12-31 \
        --output data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def generate_gbm_ohlcv(
    symbol: str,
    start: str,
    end: str,
    initial_price: float = 100.0,
    mu: float = 0.08,       # annualised drift
    sigma: float = 0.20,    # annualised volatility
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate daily OHLCV data via Geometric Brownian Motion.

    The close price is modelled as:
        S(t) = S(0) * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z)
    where Z ~ N(0, 1) and dt = 1/252.

    Open, high, and low are derived from close with noise to maintain
    structural validity (high >= max(open, close), low <= min(open, close)).

    Args:
        symbol: Ticker label used in output filename.
        start: Start date string YYYY-MM-DD.
        end: End date string YYYY-MM-DD.
        initial_price: Starting price.
        mu: Annualised drift (log-return mean).
        sigma: Annualised volatility (log-return std).
        seed: Random seed for reproducibility.

    Returns:
        OHLCV DataFrame with UTC DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    trading_dates = pd.bdate_range(start=start, end=end, tz="UTC")
    n = len(trading_dates)

    dt = 1.0 / 252.0
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)

    prices = initial_price * np.exp(np.cumsum(log_ret))
    # Shift so prices[0] == initial_price
    prices = np.concatenate([[initial_price], prices[:-1]])

    # Intraday noise (open vs close spread)
    intraday_vol = sigma * np.sqrt(dt) * 0.5
    opens = prices * np.exp(rng.normal(0, intraday_vol * 0.5, n))
    highs = np.maximum(opens, prices) * (1 + abs(rng.normal(0, intraday_vol, n)))
    lows = np.minimum(opens, prices) * (1 - abs(rng.normal(0, intraday_vol, n)))

    # Ensure structural validity
    highs = np.maximum(highs, np.maximum(opens, prices))
    lows = np.minimum(lows, np.minimum(opens, prices))
    lows = np.maximum(lows, 0.01)  # floor at 1 cent

    # Volume: log-normal with mean ~1M, occasional spikes
    base_volume = np.exp(rng.normal(13.8, 0.5, n))  # mean ~1M
    volume = base_volume * (1 + rng.exponential(0.5, n) * (rng.random(n) > 0.95))

    df = pd.DataFrame(
        {
            "open": opens.round(2),
            "high": highs.round(2),
            "low": lows.round(2),
            "close": prices.round(2),
            "volume": volume.astype(int),
        },
        index=trading_dates,
    )
    df.index.name = "timestamp"
    return df


# Realistic-ish parameters per symbol
SYMBOL_PARAMS: dict[str, dict] = {
    "AAPL": {"initial_price": 30.0,  "mu": 0.22, "sigma": 0.28, "seed": 1},
    "MSFT": {"initial_price": 45.0,  "mu": 0.20, "sigma": 0.25, "seed": 2},
    "GOOGL": {"initial_price": 550.0, "mu": 0.18, "sigma": 0.27, "seed": 3},
    "AMZN": {"initial_price": 300.0, "mu": 0.24, "sigma": 0.32, "seed": 4},
    "SPY":  {"initial_price": 200.0, "mu": 0.10, "sigma": 0.15, "seed": 5},
    "QQQ":  {"initial_price": 90.0,  "mu": 0.14, "sigma": 0.20, "seed": 6},
    "GLD":  {"initial_price": 115.0, "mu": 0.04, "sigma": 0.14, "seed": 7},
    "TLT":  {"initial_price": 120.0, "mu": 0.02, "sigma": 0.12, "seed": 8},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic OHLCV CSV data for development."
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=list(SYMBOL_PARAMS.keys()),
        help="Space-separated list of ticker symbols to generate.",
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default="2023-12-31", help="End date YYYY-MM-DD.")
    parser.add_argument("--output", default="data/raw", help="Output directory.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating data from {args.start} to {args.end} -> {output_dir}")

    for symbol in args.symbols:
        params = SYMBOL_PARAMS.get(symbol, {"initial_price": 100.0, "mu": 0.10, "sigma": 0.20})
        df = generate_gbm_ohlcv(
            symbol=symbol,
            start=args.start,
            end=args.end,
            **params,
        )
        path = output_dir / f"{symbol}.csv"
        df.to_csv(path)
        print(f"  {symbol}: {len(df)} bars -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
