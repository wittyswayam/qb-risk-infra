"""Abstract strategy interface and signal container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.core.types import Order


@dataclass
class Signal:
    """Output of a strategy for a single bar.

    Attributes:
        symbol: Ticker the signal applies to.
        direction: +1 (long), -1 (short), 0 (flat / no position).
        strength: Optional normalised signal strength in [-1, 1].
        metadata: Arbitrary diagnostic values (indicator values, etc.).
    """

    symbol: str
    direction: int  # +1, -1, or 0
    strength: float = 0.0
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"Signal direction must be -1, 0, or 1; got {self.direction}")
        self.metadata = self.metadata or {}


class BaseStrategy(ABC):
    """Contract that every trading strategy must implement.

    A strategy consumes a window of OHLCV bars and emits a Signal.  The
    backtesting engine calls :py:meth:`on_bar` for each new bar and feeds
    the returned signal to the order router.

    Concrete strategies must implement:
    - :py:meth:`on_bar`  – primary signal generation.
    - :py:meth:`reset`   – clear any in-memory state for a new run.
    - :py:attr:`required_history` – minimum bars needed before signals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name used in reports."""

    @property
    @abstractmethod
    def required_history(self) -> int:
        """Minimum number of bars needed to emit a non-neutral signal."""

    @abstractmethod
    def on_bar(
        self,
        symbol: str,
        history: pd.DataFrame,
        current_bar: pd.Series,
    ) -> Signal:
        """Generate a signal given the current bar and preceding history.

        Args:
            symbol: Ticker string.
            history: OHLCV DataFrame up to and including *current_bar*.
                     Guaranteed to have at least :py:attr:`required_history` rows.
            current_bar: The most recent OHLCV bar as a Series.

        Returns:
            A Signal with direction +1, 0, or -1.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset any in-memory state. Called before each backtest run."""

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyperparameters. Used for logging and walk-forward."""
        return {}

    def set_params(self, params: dict[str, Any]) -> None:
        """Apply hyperparameter dict. Used during walk-forward optimisation."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.name}({params})"
