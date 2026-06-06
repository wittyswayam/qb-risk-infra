"""Performance metric computations for backtesting results.

All functions operate on a pandas Series of periodic returns (e.g., daily).
Annualisation assumes the caller provides the correct trading_days_per_year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def annualised_return(returns: pd.Series, trading_days: int = 252) -> float:
    """Compound annualised growth rate from a daily returns series."""
    if returns.empty or len(returns) < 2:
        return 0.0
    total = (1 + returns).prod()
    n_years = len(returns) / trading_days
    if n_years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def annualised_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualised standard deviation of returns."""
    if returns.empty or len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Sharpe = (E[R] - Rf) / sigma(R)

    where E[R] and sigma(R) are both annualised.

    Args:
        returns: Daily return series.
        risk_free_rate: Annualised risk-free rate.
        trading_days: Trading days per year for annualisation.
    """
    ann_ret = annualised_return(returns, trading_days)
    ann_vol = annualised_volatility(returns, trading_days)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> float:
    """Annualised Sortino ratio.

    Uses the downside deviation (semideviation of negative returns) as the
    denominator rather than total standard deviation, penalising only losses.

    Sortino = (E[R] - Rf) / downside_dev
    """
    ann_ret = annualised_return(returns, trading_days)
    downside = returns[returns < 0]
    if len(downside) < 2:
        return float("inf") if ann_ret > risk_free_rate else 0.0
    downside_dev = float(downside.std(ddof=1) * np.sqrt(trading_days))
    if downside_dev == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / downside_dev)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction (negative number).

    Args:
        equity_curve: Cumulative equity series (not returns).

    Returns:
        MDD as a non-positive float. E.g. -0.25 means -25%.
    """
    if equity_curve.empty:
        return 0.0
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the full drawdown series (fraction below running peak)."""
    roll_max = equity_curve.cummax()
    return (equity_curve - roll_max) / roll_max


def calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    trading_days: int = 252,
) -> float:
    """Calmar ratio: annualised return divided by absolute max drawdown."""
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return float("inf")
    return float(annualised_return(returns, trading_days) / mdd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive return."""
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Ratio of gross profits to gross losses (positive sign).

    A value > 1 means total profits exceed total losses.
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def skewness(returns: pd.Series) -> float:
    """Sample skewness of the returns distribution."""
    if len(returns) < 3:
        return 0.0
    return float(returns.skew())


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (Fisher definition) of the returns distribution."""
    if len(returns) < 4:
        return 0.0
    return float(returns.kurtosis())


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at the given confidence level (positive number).

    Represents the loss not exceeded with probability *confidence*.
    """
    if returns.empty:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR (Expected Shortfall) at the given confidence level.

    CVaR = mean of returns below the VaR quantile (expected tail loss).
    """
    if returns.empty:
        return 0.0
    var = var_historical(returns, confidence)
    tail = returns[returns <= -var]
    if tail.empty:
        return var
    return float(-tail.mean())


def compute_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> dict[str, float]:
    """Compute and return all standard performance metrics.

    Args:
        returns: Daily return series.
        equity_curve: Cumulative equity series corresponding to *returns*.
        risk_free_rate: Annualised risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Dictionary of metric name -> value.
    """
    return {
        "annualised_return": annualised_return(returns, trading_days),
        "annualised_volatility": annualised_volatility(returns, trading_days),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, trading_days),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, trading_days),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(returns, equity_curve, trading_days),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
        "skewness": skewness(returns),
        "kurtosis": kurtosis(returns),
        "var_95": var_historical(returns, 0.95),
        "cvar_95": cvar_historical(returns, 0.95),
        "var_99": var_historical(returns, 0.99),
        "cvar_99": cvar_historical(returns, 0.99),
        "n_periods": float(len(returns)),
    }
