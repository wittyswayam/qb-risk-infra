"""Volatility and risk analytics module.

Provides rolling volatility estimation, VaR, CVaR, beta estimation,
and correlation analysis for a return series against a benchmark.
All functions are stateless and operate on pandas Series/DataFrames.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.core.types import RiskMetrics
from src.core.exceptions import RiskCalculationError

logger = logging.getLogger(__name__)


def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    annualise: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Compute rolling realised volatility.

    Uses the sample standard deviation of returns over a rolling window.
    This is a common estimate of short-term realised volatility used
    in options pricing and risk management.

    Args:
        returns: Daily return series.
        window: Rolling window length in periods.
        annualise: If True, multiply by sqrt(trading_days).
        trading_days: Trading days per year for annualisation.

    Returns:
        Series of rolling volatility values.
    """
    roll_std = returns.rolling(window, min_periods=max(window // 2, 5)).std(ddof=1)
    if annualise:
        return roll_std * np.sqrt(trading_days)
    return roll_std


def rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
) -> pd.Series:
    """Rolling historical Value at Risk.

    At each point, VaR is estimated from the empirical distribution of the
    preceding *window* returns.

    Args:
        returns: Daily return series.
        window: Rolling lookback window.
        confidence: Confidence level (e.g., 0.95 = 95th percentile).

    Returns:
        Series of VaR values (positive numbers representing losses).
    """
    quantile = 1.0 - confidence

    def _var(x: pd.Series) -> float:
        return float(-np.percentile(x, quantile * 100))

    return returns.rolling(window, min_periods=window // 2).apply(_var, raw=True)


def rolling_cvar(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
) -> pd.Series:
    """Rolling historical Conditional VaR (Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR. It is a
    coherent risk measure (Artzner et al., 1999) and captures tail risk
    better than VaR alone.

    Args:
        returns: Daily return series.
        window: Rolling lookback window.
        confidence: Confidence level.

    Returns:
        Series of CVaR values (positive numbers representing expected tail loss).
    """
    quantile = 1.0 - confidence

    def _cvar(x: pd.Series) -> float:
        var_threshold = np.percentile(x, quantile * 100)
        tail = x[x <= var_threshold]
        return float(-tail.mean()) if len(tail) > 0 else float(-var_threshold)

    return returns.rolling(window, min_periods=window // 2).apply(_cvar, raw=True)


def estimate_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    method: str = "ols",
) -> float:
    """Estimate portfolio beta relative to a benchmark.

    Beta measures the sensitivity of portfolio returns to benchmark returns.
    Beta = Cov(R_p, R_b) / Var(R_b)

    Args:
        returns: Portfolio return series.
        benchmark_returns: Benchmark return series (same frequency).
        method: 'ols' for OLS regression or 'cov' for covariance ratio.

    Returns:
        Beta estimate as a float.
    """
    aligned = pd.DataFrame({"portfolio": returns, "benchmark": benchmark_returns}).dropna()
    if len(aligned) < 10:
        raise RiskCalculationError(
            "Need at least 10 aligned observations to estimate beta."
        )

    if method == "ols":
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned["benchmark"].values, aligned["portfolio"].values
        )
        logger.debug(
            "Beta OLS: beta=%.4f alpha=%.6f r2=%.4f p=%.4f",
            slope, intercept, r_value**2, p_value,
        )
        return float(slope)
    elif method == "cov":
        cov_matrix = aligned.cov()
        return float(cov_matrix.loc["portfolio", "benchmark"] / cov_matrix.loc["benchmark", "benchmark"])
    else:
        raise RiskCalculationError(f"Unknown beta estimation method: {method}")


def correlation_analysis(
    returns_dict: dict[str, pd.Series],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute pairwise correlation matrix for multiple return series.

    Args:
        returns_dict: Mapping of label -> return series.
        method: Correlation method ('pearson', 'spearman', 'kendall').

    Returns:
        Correlation matrix as a DataFrame.
    """
    df = pd.DataFrame(returns_dict).dropna()
    if df.empty:
        raise RiskCalculationError("All return series are empty after alignment.")
    return df.corr(method=method)


def compute_risk_metrics(
    returns: pd.Series,
    symbol: str,
    period: str = "full",
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> RiskMetrics:
    """Compute a comprehensive RiskMetrics snapshot for a return series.

    Args:
        returns: Daily return series.
        symbol: Label for reporting.
        period: Descriptive period string.
        benchmark_returns: Optional benchmark for beta/correlation.
        risk_free_rate: Annualised risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Populated RiskMetrics dataclass.
    """
    if returns.empty or len(returns) < 5:
        raise RiskCalculationError(
            f"Insufficient returns to compute risk metrics for {symbol} ({len(returns)} bars)."
        )

    from src.backtesting.metrics import (
        annualised_return,
        annualised_volatility,
        sharpe_ratio,
        sortino_ratio,
        var_historical,
        cvar_historical,
    )

    ann_ret = annualised_return(returns, trading_days)
    ann_vol = annualised_volatility(returns, trading_days)
    sr = sharpe_ratio(returns, risk_free_rate, trading_days)
    so = sortino_ratio(returns, risk_free_rate, trading_days)
    var_95 = var_historical(returns, 0.95)
    cvar_95 = cvar_historical(returns, 0.95)
    var_99 = var_historical(returns, 0.99)
    cvar_99 = cvar_historical(returns, 0.99)

    equity = (1 + returns).cumprod()
    from src.backtesting.metrics import max_drawdown
    mdd = max_drawdown(equity)

    beta: Optional[float] = None
    corr: Optional[float] = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            beta = estimate_beta(returns, benchmark_returns)
            aligned = pd.DataFrame({"p": returns, "b": benchmark_returns}).dropna()
            if len(aligned) >= 2:
                corr = float(aligned.corr().loc["p", "b"])
        except Exception as exc:
            logger.warning("Could not compute beta/correlation: %s", exc)

    return RiskMetrics(
        symbol=symbol,
        period=period,
        annualised_return=ann_ret,
        annualised_volatility=ann_vol,
        sharpe_ratio=sr,
        sortino_ratio=so,
        max_drawdown=mdd,
        var_95=var_95,
        cvar_95=cvar_95,
        var_99=var_99,
        cvar_99=cvar_99,
        beta=beta,
        correlation_to_benchmark=corr,
    )
