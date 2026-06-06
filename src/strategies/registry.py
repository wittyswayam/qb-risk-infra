"""Strategy registry for dynamic strategy lookup and instantiation."""

from __future__ import annotations

from typing import Any, Type

from src.strategies.base import BaseStrategy
from src.strategies.moving_average_crossover import MovingAverageCrossover
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy

_REGISTRY: dict[str, Type[BaseStrategy]] = {
    "moving_average_crossover": MovingAverageCrossover,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "volatility_breakout": VolatilityBreakoutStrategy,
}


def get_strategy(name: str, params: dict[str, Any] | None = None) -> BaseStrategy:
    """Instantiate a strategy by registry key.

    Args:
        name: Registry key, e.g. 'momentum'.
        params: Constructor keyword arguments to pass to the strategy.

    Returns:
        Initialised strategy instance.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Strategy '{name}' not registered. Available: {list(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    return cls(**(params or {}))


def register_strategy(name: str, cls: Type[BaseStrategy]) -> None:
    """Register a custom strategy class.

    Args:
        name: Unique registry key.
        cls: Strategy class implementing BaseStrategy.
    """
    if not issubclass(cls, BaseStrategy):
        raise TypeError(f"{cls} must subclass BaseStrategy")
    _REGISTRY[name] = cls


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    return sorted(_REGISTRY.keys())
