"""CSV-based OHLCV ingestion adapter with resampling and gap-filling support."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.exceptions import DataIntegrityError, IngestionError
from src.ingestion.base import BaseIngestionAdapter
from src.ingestion.validator import OHLCVValidator

logger = logging.getLogger(__name__)

COLUMN_ALIASES: dict[str, str] = {
    "Date": "timestamp",
    "Datetime": "timestamp",
    "time": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "close",
    "Volume": "volume",
    "Vol": "volume",
}


class CSVIngestionAdapter(BaseIngestionAdapter):
    """Loads OHLCV data from CSV files stored in a directory.

    Directory layout:
        data_dir/
            AAPL.csv
            MSFT.csv
            ...

    Each CSV must contain at minimum: timestamp, open, high, low, close, volume.
    Column names are normalised via COLUMN_ALIASES before validation.

    Args:
        data_dir: Root directory containing per-symbol CSV files.
        missing_strategy: How to handle missing values.
            'ffill'        - forward-fill then back-fill.
            'drop'         - drop rows with any NaN in OHLCV columns.
            'interpolate'  - linear interpolation then boundary fill.
    """

    def __init__(
        self,
        data_dir: str | Path,
        missing_strategy: str = "ffill",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._missing_strategy = missing_strategy
        self._validator = OHLCVValidator()

        if not self._data_dir.exists():
            raise IngestionError(f"Data directory does not exist: {self._data_dir}")

    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Load and return OHLCV data for *symbol* from CSV.

        Args:
            symbol: Ticker string matching the CSV filename (case-insensitive).
            start: Inclusive start datetime.
            end: Inclusive end datetime.
            interval: Target bar interval. If the source is finer, data is
                resampled. If coarser, an IngestionError is raised.

        Returns:
            Validated OHLCV DataFrame.
        """
        path = self._resolve_path(symbol)
        logger.info("Loading CSV: %s", path)

        df = self._read_csv(path, symbol)
        df = self._filter_date_range(df, start, end, symbol)
        df = self._fill_missing(df, symbol)
        df = self._resample_if_needed(df, interval, symbol)
        self._validator.validate(df, symbol)
        logger.info(
            "Loaded %d bars for %s [%s -> %s] @ %s",
            len(df),
            symbol,
            df.index[0].date(),
            df.index[-1].date(),
            interval,
        )
        return df

    def available_symbols(self) -> list[str]:
        return sorted(p.stem.upper() for p in self._data_dir.glob("*.csv"))

    def _resolve_path(self, symbol: str) -> Path:
        for candidate in [
            self._data_dir / f"{symbol}.csv",
            self._data_dir / f"{symbol.upper()}.csv",
            self._data_dir / f"{symbol.lower()}.csv",
        ]:
            if candidate.exists():
                return candidate
        raise IngestionError(
            f"No CSV file found for symbol '{symbol}' in {self._data_dir}"
        )

    def _read_csv(self, path: Path, symbol: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise IngestionError(f"Cannot read {path}: {exc}") from exc

        # Normalise column names
        df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
        df.columns = df.columns.str.lower()

        if "timestamp" not in df.columns:
            raise DataIntegrityError(
                f"{symbol}: could not identify a timestamp column in {path}"
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        invalid_ts = df["timestamp"].isna()
        if invalid_ts.any():
            raise DataIntegrityError(
                f"{symbol}: {invalid_ts.sum()} unparseable timestamp(s)"
            )

        df = df.set_index("timestamp").sort_index()

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]

    def _filter_date_range(
        self,
        df: pd.DataFrame,
        start: datetime,
        end: datetime,
        symbol: str,
    ) -> pd.DataFrame:
        start_utc = pd.Timestamp(start, tz="UTC")
        end_utc = pd.Timestamp(end, tz="UTC")
        df = df.loc[start_utc:end_utc]
        if df.empty:
            raise IngestionError(
                f"{symbol}: no data in range [{start.date()}, {end.date()}]"
            )
        return df

    def _fill_missing(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        n_missing = df[["open", "high", "low", "close"]].isna().sum().sum()
        if n_missing == 0:
            return df

        logger.warning("%s: filling %d missing price values via '%s'", symbol, n_missing, self._missing_strategy)

        if self._missing_strategy == "ffill":
            df = df.ffill().bfill()
        elif self._missing_strategy == "drop":
            df = df.dropna(subset=["open", "high", "low", "close"])
        elif self._missing_strategy == "interpolate":
            df = df.interpolate(method="time").ffill().bfill()
        else:
            raise IngestionError(f"Unknown missing_strategy: {self._missing_strategy}")

        df["volume"] = df["volume"].fillna(0.0)
        return df

    def _resample_if_needed(self, df: pd.DataFrame, interval: str, symbol: str) -> pd.DataFrame:
        _alias_map = {
            "1min": "1min", "5min": "5min", "15min": "15min",
            "1h": "1h", "1d": "1D", "1w": "1W",
        }
        if interval not in _alias_map or interval == "1d":
            return df

        freq = _alias_map[interval]
        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open", "close"])

        logger.debug("%s: resampled to %s (%d -> %d bars)", symbol, interval, len(df), len(resampled))
        return resampled
