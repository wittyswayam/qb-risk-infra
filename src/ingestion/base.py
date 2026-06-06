"""Abstract interface for all data ingestion adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class BaseIngestionAdapter(ABC):
    """Contract that every ingestion backend must satisfy.

    Implementations exist for CSV files and HTTP APIs. The backtesting
    engine depends only on this interface so adapters are interchangeable.
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return an OHLCV DataFrame for *symbol* over [start, end].

        The returned DataFrame must have:
        - DatetimeIndex (UTC-aware)
        - columns: open, high, low, close, volume  (all float64)
        - no duplicate index entries
        - index sorted ascending

        Args:
            symbol: Ticker string, e.g. "AAPL".
            start: Inclusive start datetime.
            end: Inclusive end datetime.
            interval: Bar interval string ("1min", "5min", "1h", "1d").

        Returns:
            Clean OHLCV DataFrame.

        Raises:
            IngestionError: On network failures, missing files, or empty results.
            DataIntegrityError: When data fails structural validation.
        """

    @abstractmethod
    def available_symbols(self) -> list[str]:
        """Return a list of symbols that this adapter can provide data for."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
