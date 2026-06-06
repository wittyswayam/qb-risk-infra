"""Unit tests for the Monte Carlo simulation engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import SimulationError
from src.montecarlo.simulator import MonteCarloSimulator


@pytest.fixture
def returns_series() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0003, 0.015, 500))


class TestMonteCarloSimulator:
    def test_bootstrap_produces_correct_shape(self, returns_series):
        sim = MonteCarloSimulator(
            n_simulations=200, horizon_days=100, method="bootstrap", random_seed=42
        )
        result = sim.simulate(returns_series)
        assert result.paths.shape == (200, 100)
        assert len(result.terminal_values) == 200

    def test_parametric_produces_correct_shape(self, returns_series):
        sim = MonteCarloSimulator(
            n_simulations=150, horizon_days=63, method="parametric", random_seed=0
        )
        result = sim.simulate(returns_series)
        assert result.paths.shape == (150, 63)

    def test_cvar_ge_var(self, returns_series):
        sim = MonteCarloSimulator(n_simulations=500, horizon_days=252, random_seed=1)
        result = sim.simulate(returns_series)
        assert result.cvar[0.95] >= result.var[0.95]

    def test_var_99_ge_var_95(self, returns_series):
        sim = MonteCarloSimulator(
            n_simulations=500, horizon_days=252,
            confidence_levels=[0.95, 0.99], random_seed=2
        )
        result = sim.simulate(returns_series)
        assert result.var[0.99] >= result.var[0.95]

    def test_insufficient_returns_raises(self):
        sim = MonteCarloSimulator(n_simulations=100, horizon_days=10)
        with pytest.raises(SimulationError, match="at least 30"):
            sim.simulate(pd.Series([0.01] * 5))

    def test_too_few_simulations_raises(self):
        with pytest.raises(SimulationError, match=">= 100"):
            MonteCarloSimulator(n_simulations=10)

    def test_median_terminal_near_one_for_zero_drift(self):
        # Pure white noise around zero: median terminal value should be near 1.0
        rng = np.random.default_rng(42)
        zero_drift = pd.Series(rng.normal(0, 0.01, 1000))
        sim = MonteCarloSimulator(n_simulations=1000, horizon_days=252, random_seed=42)
        result = sim.simulate(zero_drift, initial_value=1.0)
        median_terminal = result.percentiles[50]
        # With daily vol 1% over 252 days, expected median near 1 (log-normal centred)
        assert 0.7 < median_terminal < 1.3

    def test_stress_test_returns_baseline_and_scenarios(self, returns_series):
        sim = MonteCarloSimulator(n_simulations=100, horizon_days=63, random_seed=0)
        shocks = [
            {"name": "crash_10pct", "return_shock": -0.10, "vol_multiplier": 1.5},
        ]
        results = sim.stress_test(returns_series, shocks)
        assert "baseline" in results
        assert "crash_10pct" in results

    def test_shock_reduces_median_terminal(self, returns_series):
        sim = MonteCarloSimulator(n_simulations=500, horizon_days=252, random_seed=7)
        shocks = [{"name": "shock", "return_shock": -0.20, "vol_multiplier": 2.0}]
        results = sim.stress_test(returns_series, shocks)
        baseline_median = results["baseline"].percentiles[50]
        shocked_median = results["shock"].percentiles[50]
        # Shocked scenario should have lower expected terminal value
        assert shocked_median < baseline_median
