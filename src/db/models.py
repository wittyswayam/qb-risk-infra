"""SQLAlchemy ORM models for persisting backtest results and market data."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class OHLCVRecord(Base):
    """Persisted OHLCV bar for a single symbol and interval."""

    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "timestamp", name="uq_ohlcv_symbol_interval_ts"),
        Index("ix_ohlcv_symbol_interval_ts", "symbol", "interval", "timestamp"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    interval = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class BacktestRun(Base):
    """Top-level record for a single backtest execution."""

    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), unique=True, nullable=False)  # UUID
    strategy_name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    interval = Column(String(10), nullable=False, default="1d")
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    commission_bps = Column(Float, nullable=False)
    slippage_bps = Column(Float, nullable=False)
    strategy_params = Column(JSON)
    metrics = Column(JSON)  # Sharpe, Sortino, MDD, etc.
    created_at = Column(DateTime, server_default=func.now())

    fills = relationship("FillRecord", back_populates="run", cascade="all, delete-orphan")


class FillRecord(Base):
    """Individual trade fill associated with a backtest run."""

    __tablename__ = "fills"
    __table_args__ = (
        Index("ix_fills_run_id", "run_id"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)  # BUY | SELL
    quantity = Column(Float, nullable=False)
    fill_price = Column(Float, nullable=False)
    commission = Column(Float, nullable=False)
    slippage = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    run = relationship("BacktestRun", back_populates="fills")


class WalkForwardRun(Base):
    """Walk-forward validation run record."""

    __tablename__ = "walkforward_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), unique=True, nullable=False)
    strategy_name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    param_grid = Column(JSON)
    train_window_days = Column(Integer, nullable=False)
    test_window_days = Column(Integer, nullable=False)
    n_windows = Column(Integer)
    oos_metrics = Column(JSON)
    param_stability = Column(JSON)
    window_details = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())


class MonteCarloRun(Base):
    """Monte Carlo simulation run record."""

    __tablename__ = "montecarlo_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), unique=True, nullable=False)
    backtest_run_id = Column(String(36), nullable=True)
    n_simulations = Column(Integer, nullable=False)
    horizon_days = Column(Integer, nullable=False)
    method = Column(String(20), nullable=False)
    var_95 = Column(Float)
    cvar_95 = Column(Float)
    var_99 = Column(Float)
    cvar_99 = Column(Float)
    median_terminal = Column(Float)
    p5_terminal = Column(Float)
    p95_terminal = Column(Float)
    stress_scenarios = Column(JSON)  # {scenario_name: {var, cvar, ...}}
    created_at = Column(DateTime, server_default=func.now())
