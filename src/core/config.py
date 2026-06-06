"""Central configuration management via YAML and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="qb_risk")
    user: str = Field(default="postgres")
    password: str = Field(default="")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class RedisConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = None
    ttl_seconds: int = Field(default=3600)

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class IngestionConfig(BaseModel):
    data_dir: Path = Field(default=Path("data/raw"))
    supported_intervals: list[str] = Field(
        default=["1min", "5min", "15min", "1h", "1d"]
    )
    missing_value_strategy: str = Field(default="ffill")  # ffill | drop | interpolate
    max_gap_days: int = Field(default=5)
    api_timeout_seconds: int = Field(default=30)
    alpha_vantage_key: Optional[str] = None

    @field_validator("missing_value_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"ffill", "drop", "interpolate"}
        if v not in allowed:
            raise ValueError(f"missing_value_strategy must be one of {allowed}")
        return v


class BacktestConfig(BaseModel):
    initial_capital: float = Field(default=100_000.0)
    commission_bps: float = Field(default=5.0)  # basis points per trade
    slippage_bps: float = Field(default=2.0)
    risk_free_rate: float = Field(default=0.04)  # annualised
    trading_days_per_year: int = Field(default=252)
    max_position_size: float = Field(default=0.25)  # fraction of portfolio
    allow_short: bool = Field(default=False)


class WalkForwardConfig(BaseModel):
    train_window_days: int = Field(default=252)
    test_window_days: int = Field(default=63)
    step_days: int = Field(default=21)
    min_train_samples: int = Field(default=100)
    param_stability_threshold: float = Field(default=0.3)  # CV threshold


class MonteCarloConfig(BaseModel):
    n_simulations: int = Field(default=1000)
    simulation_horizon_days: int = Field(default=252)
    confidence_levels: list[float] = Field(default=[0.95, 0.99])
    shock_scenarios: list[dict[str, Any]] = Field(default_factory=list)
    random_seed: int = Field(default=42)


class RiskConfig(BaseModel):
    var_confidence: float = Field(default=0.95)
    cvar_confidence: float = Field(default=0.95)
    rolling_window_days: int = Field(default=21)
    benchmark_symbol: str = Field(default="SPY")
    min_periods: int = Field(default=20)


class APIConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)
    log_level: str = Field(default="info")
    cors_origins: list[str] = Field(default=["*"])
    api_key: Optional[str] = None


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(default="json")  # json | text
    file: Optional[str] = None
    rotation: str = Field(default="100 MB")
    retention: str = Field(default="30 days")


class AppConfig(BaseModel):
    env: str = Field(default="development")
    debug: bool = Field(default=False)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _override_from_env(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides using double-underscore nesting.

    Example: QB__DATABASE__HOST overrides config['database']['host']
    """
    prefix = "QB__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        node = config_dict
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return config_dict


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML file, then apply env overrides."""
    if path is None:
        path = Path(os.getenv("QB_CONFIG", "config/config.yaml"))

    raw: dict[str, Any] = {}
    if path.exists():
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}

    raw = _override_from_env(raw)
    return AppConfig(**raw)


# Module-level singleton — imported across the codebase.
settings: AppConfig = load_config()
