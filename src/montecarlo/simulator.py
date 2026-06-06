"""Monte Carlo simulation engine for portfolio stress testing.

Generates synthetic return paths using a bootstrapped or parametric model and
computes probabilistic loss distributions (VaR, CVaR, percentile bands).
The goal is portfolio robustness analysis under uncertain future conditions,
not point forecasting.

Two sampling methods are supported:
- 'parametric': Assumes returns are normally distributed with historical
  mean and covariance. Fast but ignores fat tails.
- 'bootstrap': Resamples historical return blocks. Preserves empirical
  distribution shape including skewness and kurtosis.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.core.types import MonteCarloResult
from src.core.exceptions import SimulationError

logger = logging.getLogger(__name__)

SamplingMethod = Literal["parametric", "bootstrap"]


class MonteCarloSimulator:
    """Monte Carlo path simulation for a single return series.

    Args:
        n_simulations: Number of independent paths to generate.
        horizon_days: Number of days to project forward.
        method: Sampling method ('parametric' or 'bootstrap').
        confidence_levels: List of confidence levels for VaR / CVaR.
        block_size: Block size for block bootstrap (if method='bootstrap').
        random_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        horizon_days: int = 252,
        method: SamplingMethod = "bootstrap",
        confidence_levels: list[float] | None = None,
        block_size: int = 10,
        random_seed: int = 42,
    ) -> None:
        if n_simulations < 100:
            raise SimulationError("n_simulations must be >= 100 for reliable estimates.")
        self._n_sims = n_simulations
        self._horizon = horizon_days
        self._method = method
        self._confidence_levels = confidence_levels or [0.95, 0.99]
        self._block_size = block_size
        self._rng = np.random.default_rng(random_seed)

    def simulate(
        self,
        returns: pd.Series,
        initial_value: float = 1.0,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on a historical returns series.

        Args:
            returns: Historical daily return series (decimal, not percent).
            initial_value: Starting portfolio value (default 1.0 = relative).

        Returns:
            MonteCarloResult with paths, terminal values, VaR, CVaR.

        Raises:
            SimulationError: If returns series is too short for simulation.
        """
        if len(returns) < 30:
            raise SimulationError(
                f"Need at least 30 historical returns for simulation; got {len(returns)}."
            )

        returns_clean = returns.dropna()
        logger.info(
            "Running Monte Carlo: method=%s n=%d horizon=%d days",
            self._method,
            self._n_sims,
            self._horizon,
        )

        if self._method == "parametric":
            sampled = self._parametric_sample(returns_clean)
        elif self._method == "bootstrap":
            sampled = self._bootstrap_sample(returns_clean)
        else:
            raise SimulationError(f"Unknown sampling method: {self._method}")

        # sampled shape: (n_simulations, horizon_days)
        # Compute cumulative portfolio paths
        paths = initial_value * np.cumprod(1 + sampled, axis=1)
        terminal_values = paths[:, -1]

        var_results: dict[float, float] = {}
        cvar_results: dict[float, float] = {}
        for conf in self._confidence_levels:
            terminal_returns = (terminal_values / initial_value) - 1.0
            var_results[conf] = float(-np.percentile(terminal_returns, (1 - conf) * 100))
            tail_mask = terminal_returns <= -var_results[conf]
            if tail_mask.any():
                cvar_results[conf] = float(-terminal_returns[tail_mask].mean())
            else:
                cvar_results[conf] = var_results[conf]

        percentile_keys = [5, 10, 25, 50, 75, 90, 95]
        percentiles = {
            p: float(np.percentile(terminal_values, p)) for p in percentile_keys
        }

        logger.info(
            "MC complete: median_terminal=%.4f VaR_95=%.4f CVaR_95=%.4f",
            percentiles[50],
            var_results.get(0.95, 0),
            cvar_results.get(0.95, 0),
        )

        return MonteCarloResult(
            n_simulations=self._n_sims,
            horizon_days=self._horizon,
            paths=paths,
            terminal_values=terminal_values,
            var=var_results,
            cvar=cvar_results,
            percentiles=percentiles,
        )

    def stress_test(
        self,
        returns: pd.Series,
        shocks: list[dict],
        initial_value: float = 1.0,
    ) -> dict[str, MonteCarloResult]:
        """Apply named market shock scenarios before simulation.

        Each shock is defined as a dict with:
            - name: str label
            - return_shock: float, one-time instantaneous return shock (e.g. -0.10)
            - vol_multiplier: float, multiply historical vol by this factor

        Args:
            returns: Historical daily return series.
            shocks: List of shock scenario dicts.
            initial_value: Starting portfolio value.

        Returns:
            Dict of scenario_name -> MonteCarloResult.
        """
        results: dict[str, MonteCarloResult] = {}

        # Baseline
        results["baseline"] = self.simulate(returns, initial_value)

        for shock in shocks:
            name = shock.get("name", "unnamed_shock")
            return_shock = float(shock.get("return_shock", 0.0))
            vol_multiplier = float(shock.get("vol_multiplier", 1.0))

            shocked_returns = returns.copy()
            shocked_returns = shocked_returns * vol_multiplier

            # Apply instantaneous shock to first day of each path via returns adjustment
            shocked_returns = pd.concat([
                pd.Series([return_shock]),
                shocked_returns,
            ], ignore_index=True)

            logger.info(
                "Stress test '%s': shock=%.2f%% vol_mult=%.2f",
                name, return_shock * 100, vol_multiplier
            )
            results[name] = self.simulate(shocked_returns, initial_value)

        return results

    def _parametric_sample(self, returns: pd.Series) -> np.ndarray:
        """Sample from a fitted normal distribution."""
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        return self._rng.normal(
            loc=mu, scale=sigma, size=(self._n_sims, self._horizon)
        )

    def _bootstrap_sample(self, returns: pd.Series) -> np.ndarray:
        """Block bootstrap to preserve autocorrelation structure."""
        arr = returns.values
        n = len(arr)
        b = min(self._block_size, n // 4)
        n_blocks = self._horizon // b + 2

        sampled = np.empty((self._n_sims, self._horizon))
        for i in range(self._n_sims):
            starts = self._rng.integers(0, n - b + 1, size=n_blocks)
            blocks = [arr[s: s + b] for s in starts]
            path = np.concatenate(blocks)[:self._horizon]
            sampled[i] = path

        return sampled
