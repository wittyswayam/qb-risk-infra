"""Pydantic request/response schemas for the FastAPI layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    db_connected: bool
    redis_connected: bool


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g. AAPL")
    strategy: str = Field(..., description="Strategy registry key")
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    start_date: str = Field(..., description="ISO date YYYY-MM-DD")
    end_date: str = Field(..., description="ISO date YYYY-MM-DD")
    interval: str = Field(default="1d")
    initial_capital: float = Field(default=100_000.0, gt=0)
    commission_bps: float = Field(default=5.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)
    data_source: Literal["csv", "api"] = Field(default="csv")
    persist: bool = Field(default=True, description="Persist result to database")

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
        return v


class MetricsResponse(BaseModel):
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    profit_factor: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    n_periods: float


class FillResponse(BaseModel):
    symbol: str
    side: str
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    timestamp: datetime


class BacktestResponse(BaseModel):
    run_id: Optional[str]
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    metrics: MetricsResponse
    n_fills: int
    equity_curve: List[Dict[str, Any]]  # [{timestamp, equity}]
    fills: List[FillResponse]


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------

class WalkForwardRequest(BaseModel):
    symbol: str
    strategy: str
    param_grid: List[Dict[str, Any]] = Field(..., min_length=1)
    start_date: str
    end_date: str
    interval: str = Field(default="1d")
    train_window_days: int = Field(default=252, gt=0)
    test_window_days: int = Field(default=63, gt=0)
    step_days: int = Field(default=21, gt=0)
    initial_capital: float = Field(default=100_000.0, gt=0)
    commission_bps: float = Field(default=5.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v


class WalkForwardResponse(BaseModel):
    run_id: Optional[str]
    strategy_name: str
    symbol: str
    n_windows: int
    oos_metrics: MetricsResponse
    param_stability: Dict[str, float]
    windows: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class ShockScenario(BaseModel):
    name: str
    return_shock: float = Field(
        ..., ge=-1.0, le=0.0,
        description="Instantaneous return shock (negative). E.g. -0.10 = -10%."
    )
    vol_multiplier: float = Field(
        default=1.0, gt=0,
        description="Multiply historical volatility by this factor."
    )


class MonteCarloRequest(BaseModel):
    run_id: str = Field(..., description="Backtest run ID to simulate from")
    n_simulations: int = Field(default=1000, ge=100, le=10_000)
    horizon_days: int = Field(default=252, gt=0)
    method: Literal["parametric", "bootstrap"] = Field(default="bootstrap")
    confidence_levels: List[float] = Field(default=[0.95, 0.99])
    stress_scenarios: List[ShockScenario] = Field(default_factory=list)


class MonteCarloResponse(BaseModel):
    run_id: str
    n_simulations: int
    horizon_days: int
    method: str
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    median_terminal: float
    p5_terminal: float
    p95_terminal: float
    stress_results: Dict[str, Dict[str, float]]


# ---------------------------------------------------------------------------
# Risk Analytics
# ---------------------------------------------------------------------------

class RiskAnalyticsRequest(BaseModel):
    run_id: str = Field(..., description="Backtest run ID")
    benchmark_run_id: Optional[str] = Field(
        None,
        description="Optional benchmark backtest run ID for beta/correlation",
    )
    rolling_window: int = Field(default=21, gt=0)


class RiskAnalyticsResponse(BaseModel):
    symbol: str
    period: str
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    beta: Optional[float]
    correlation_to_benchmark: Optional[float]
    rolling_vol: List[Dict[str, Any]]  # [{date, vol}]
    rolling_var: List[Dict[str, Any]]  # [{date, var}]


# ---------------------------------------------------------------------------
# Strategies list
# ---------------------------------------------------------------------------

class StrategyListResponse(BaseModel):
    strategies: List[str]
