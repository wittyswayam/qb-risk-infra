"""Domain-specific exception hierarchy for the backtesting platform."""

from __future__ import annotations


class QBError(Exception):
    """Base exception for all platform errors."""


class IngestionError(QBError):
    """Raised when market data ingestion fails validation or I/O."""


class DataIntegrityError(IngestionError):
    """Raised when ingested data fails structural or statistical checks."""


class StrategyError(QBError):
    """Raised when a strategy cannot be initialised or executed."""


class BacktestError(QBError):
    """Raised during backtesting execution."""


class InsufficientDataError(BacktestError):
    """Raised when available bars are too few for a given lookback window."""


class OrderExecutionError(BacktestError):
    """Raised when the simulated order book cannot fill an order."""


class WalkForwardError(QBError):
    """Raised during walk-forward slicing or re-training."""


class SimulationError(QBError):
    """Raised during Monte Carlo path generation."""


class RiskCalculationError(QBError):
    """Raised when a risk metric cannot be computed (e.g., insufficient data)."""


class ConfigurationError(QBError):
    """Raised when configuration values are invalid or missing."""


class RepositoryError(QBError):
    """Raised on database I/O failures."""
