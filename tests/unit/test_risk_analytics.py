"""Unit tests for risk analytics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import RiskCalculationError
from src.risk.analytics import (
    correlation_analysis,
    estimate_beta,
    rolling_cvar,
    rolling_var,
    rolling_volatility,
    compute_risk_metrics,
)


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(0.0003, 0.012, 500))


@pytest.fixture
def benchmark() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0002, 0.010, 500))


class TestRollingVolatility:
    def test_output_length(self, returns):
        vol = rolling_volatility(returns, window=21)
        assert len(vol) == len(returns)

    def test_annualised_is_larger_than_daily(self, returns):
        ann = rolling_volatility(returns, window=21, annualise=True)
        daily = rolling_volatility(returns, window=21, annualise=False)
        # Annualised should be scaled up by sqrt(252)
        ratio = ann.dropna() / daily.dropna()
        assert ratio.mean() == pytest.approx(np.sqrt(252), rel=0.01)

    def test_zero_returns_have_zero_vol(self):
        zero = pd.Series([0.0] * 100)
        vol = rolling_volatility(zero, window=21, annualise=False)
        assert vol.dropna().abs().max() < 1e-12


class TestRollingVaR:
    def test_output_length(self, returns):
        var = rolling_var(returns, window=100)
        assert len(var) == len(returns)

    def test_var_positive(self, returns):
        var = rolling_var(returns, window=100, confidence=0.95)
        assert var.dropna().min() >= 0

    def test_var_99_ge_var_95(self, returns):
        v95 = rolling_var(returns, window=100, confidence=0.95)
        v99 = rolling_var(returns, window=100, confidence=0.99)
        common = v95.dropna().index.intersection(v99.dropna().index)
        assert (v99.loc[common] >= v95.loc[common]).all()


class TestRollingCVaR:
    def test_cvar_ge_var(self, returns):
        var = rolling_var(returns, window=100, confidence=0.95)
        cvar = rolling_cvar(returns, window=100, confidence=0.95)
        common = var.dropna().index.intersection(cvar.dropna().index)
        assert (cvar.loc[common] >= var.loc[common]).all()


class TestBetaEstimation:
    def test_beta_near_one_for_same_series(self, returns):
        beta = estimate_beta(returns, returns, method="ols")
        assert beta == pytest.approx(1.0, rel=0.001)

    def test_beta_near_zero_for_uncorrelated(self):
        rng = np.random.default_rng(1)
        port = pd.Series(rng.normal(0, 0.01, 500))
        bench = pd.Series(rng.normal(0, 0.01, 500))
        beta = estimate_beta(port, bench)
        assert abs(beta) < 0.2  # Should be near zero for uncorrelated noise

    def test_ols_and_cov_methods_agree(self, returns, benchmark):
        beta_ols = estimate_beta(returns, benchmark, method="ols")
        beta_cov = estimate_beta(returns, benchmark, method="cov")
        assert abs(beta_ols - beta_cov) < 0.01

    def test_insufficient_data_raises(self):
        with pytest.raises(RiskCalculationError):
            estimate_beta(pd.Series([0.01] * 5), pd.Series([0.01] * 5))


class TestCorrelationAnalysis:
    def test_self_correlation_is_one(self, returns):
        corr = correlation_analysis({"A": returns, "B": returns})
        assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-10)

    def test_matrix_is_symmetric(self, returns, benchmark):
        corr = correlation_analysis({"portfolio": returns, "benchmark": benchmark})
        assert corr.loc["portfolio", "benchmark"] == pytest.approx(
            corr.loc["benchmark", "portfolio"], abs=1e-10
        )

    def test_diagonal_is_one(self, returns, benchmark):
        corr = correlation_analysis({"A": returns, "B": benchmark})
        assert corr.loc["A", "A"] == pytest.approx(1.0)
        assert corr.loc["B", "B"] == pytest.approx(1.0)


class TestComputeRiskMetrics:
    def test_all_fields_populated(self, returns):
        metrics = compute_risk_metrics(returns, symbol="TEST", period="full")
        assert metrics.symbol == "TEST"
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.var_95 >= 0
        assert metrics.cvar_95 >= metrics.var_95

    def test_beta_computed_with_benchmark(self, returns, benchmark):
        metrics = compute_risk_metrics(returns, "TEST", benchmark_returns=benchmark)
        assert metrics.beta is not None
        assert metrics.correlation_to_benchmark is not None

    def test_insufficient_data_raises(self):
        with pytest.raises(RiskCalculationError):
            compute_risk_metrics(pd.Series([0.01, 0.02]), symbol="X")
