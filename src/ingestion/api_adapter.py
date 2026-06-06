"""HTTP API ingestion adapter targeting the Alpha Vantage REST API.

This adapter is intentionally thin: it fetches raw JSON, normalises it
into the shared OHLCV schema, and delegates validation to OHLCVValidator.
Swapping providers requires only a new adapter implementing BaseIngestionAdapter.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import requests

from src.core.exceptions import IngestionError
from src.ingestion.base import BaseIngestionAdapter
from src.ingestion.validator import OHLCVValidator

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
_INTERVAL_MAP: dict[str, str] = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1h": "60min",
    "1d": "daily",
}


class AlphaVantageAdapter(BaseIngestionAdapter):
    """Ingests OHLCV data from Alpha Vantage.

    Args:
        api_key: Alpha Vantage API key.
        timeout: HTTP request timeout in seconds.
        max_retries: Number of retry attempts on transient failures.
        retry_backoff: Base wait time (seconds) between retries (exponential).
    """

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        if not api_key:
            raise IngestionError("AlphaVantage API key must be non-empty.")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._validator = OHLCVValidator()
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "qb-risk-infra/1.0"

    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV from Alpha Vantage and return a validated DataFrame."""
        if interval not in _INTERVAL_MAP:
            raise IngestionError(
                f"Unsupported interval '{interval}'. Supported: {list(_INTERVAL_MAP)}"
            )

        av_interval = _INTERVAL_MAP[interval]
        if av_interval == "daily":
            raw = self._request_daily(symbol)
        else:
            raw = self._request_intraday(symbol, av_interval)

        df = self._parse_response(raw, symbol)
        df = df.loc[pd.Timestamp(start, tz="UTC"): pd.Timestamp(end, tz="UTC")]

        if df.empty:
            raise IngestionError(
                f"{symbol}: no data returned for [{start.date()}, {end.date()}]"
            )

        self._validator.validate(df, symbol)
        logger.info(
            "AlphaVantage: fetched %d bars for %s @ %s", len(df), symbol, interval
        )
        return df

    def available_symbols(self) -> list[str]:
        # Alpha Vantage does not expose a symbol list endpoint in its free tier.
        return []

    def _request_daily(self, symbol: str) -> dict[str, Any]:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self._api_key,
        }
        return self._get(params)

    def _request_intraday(self, symbol: str, interval: str) -> dict[str, Any]:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "full",
            "adjusted": "true",
            "apikey": self._api_key,
        }
        return self._get(params)

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.get(
                    ALPHA_VANTAGE_BASE, params=params, timeout=self._timeout
                )
                response.raise_for_status()
                data = response.json()
                if "Error Message" in data:
                    raise IngestionError(f"API error: {data['Error Message']}")
                if "Note" in data:
                    raise IngestionError("Alpha Vantage rate limit hit.")
                return data
            except IngestionError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = self._retry_backoff ** attempt
                logger.warning(
                    "Request failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise IngestionError(
            f"All {self._max_retries} attempts failed. Last error: {last_exc}"
        )

    @staticmethod
    def _parse_response(data: dict[str, Any], symbol: str) -> pd.DataFrame:
        ts_key = next(
            (k for k in data if "Time Series" in k or "time series" in k.lower()),
            None,
        )
        if ts_key is None:
            raise IngestionError(
                f"{symbol}: unexpected API response structure. Keys: {list(data.keys())}"
            )

        records = data[ts_key]
        rows: list[dict[str, Any]] = []
        for ts_str, values in records.items():
            row = {"timestamp": ts_str}
            for k, v in values.items():
                clean_key = k.split(".", 1)[-1].strip().lower()
                if "adjusted close" in clean_key:
                    row["close"] = float(v)
                elif "open" in clean_key:
                    row["open"] = float(v)
                elif "high" in clean_key:
                    row["high"] = float(v)
                elif "low" in clean_key:
                    row["low"] = float(v)
                elif "close" in clean_key and "close" not in row:
                    row["close"] = float(v)
                elif "volume" in clean_key:
                    row["volume"] = float(v)
            rows.append(row)

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]
