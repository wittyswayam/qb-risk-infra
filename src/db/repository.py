"""Repository pattern implementations for database I/O.

The repository layer isolates all SQLAlchemy session usage from the service
and API layers. Business logic never constructs queries directly; it delegates
to a repository method that returns domain objects or raises RepositoryError.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from src.db.models import BacktestRun, FillRecord, OHLCVRecord, WalkForwardRun, MonteCarloRun
from src.core.types import BacktestResult, WalkForwardResult, MonteCarloResult, Fill
from src.core.exceptions import RepositoryError

logger = logging.getLogger(__name__)


class OHLCVRepository:
    """Read/write OHLCV records to the database."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> int:
        """Insert OHLCV bars, skipping rows that already exist.

        Returns the number of newly inserted rows.
        """
        inserted = 0
        for ts, row in df.iterrows():
            existing = self._session.execute(
                select(OHLCVRecord).where(
                    and_(
                        OHLCVRecord.symbol == symbol,
                        OHLCVRecord.interval == interval,
                        OHLCVRecord.timestamp == ts,
                    )
                )
            ).scalar_one_or_none()

            if existing is None:
                record = OHLCVRecord(
                    symbol=symbol,
                    interval=interval,
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                self._session.add(record)
                inserted += 1

        try:
            self._session.commit()
        except Exception as exc:
            self._session.rollback()
            raise RepositoryError(f"OHLCV upsert failed: {exc}") from exc

        logger.info("OHLCVRepository: inserted %d bars for %s @ %s", inserted, symbol, interval)
        return inserted

    def fetch_dataframe(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Load OHLCV records as a DataFrame."""
        rows = self._session.execute(
            select(OHLCVRecord).where(
                and_(
                    OHLCVRecord.symbol == symbol,
                    OHLCVRecord.interval == interval,
                    OHLCVRecord.timestamp >= start,
                    OHLCVRecord.timestamp <= end,
                )
            ).order_by(OHLCVRecord.timestamp)
        ).scalars().all()

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        data = [
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
        df = pd.DataFrame(data).set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index, tz="UTC")
        return df


class BacktestRepository:
    """Persist and retrieve backtest run results."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, result: BacktestResult, config: dict) -> str:
        """Persist a BacktestResult and its fills. Returns the run_id UUID."""
        run_id = str(uuid.uuid4())
        run = BacktestRun(
            run_id=run_id,
            strategy_name=result.strategy_name,
            symbol=result.symbol,
            interval=config.get("interval", "1d"),
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=config.get("initial_capital", 0),
            commission_bps=config.get("commission_bps", 0),
            slippage_bps=config.get("slippage_bps", 0),
            strategy_params=config.get("strategy_params"),
            metrics={
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in result.metrics.items()
            },
        )
        self._session.add(run)
        self._session.flush()  # get run.id

        for fill in result.fills:
            self._session.add(FillRecord(
                run_id=run.id,
                symbol=fill.order.symbol,
                side=fill.order.side.value,
                quantity=fill.fill_quantity,
                fill_price=fill.fill_price,
                commission=fill.commission,
                slippage=fill.slippage,
                timestamp=fill.timestamp,
            ))

        try:
            self._session.commit()
        except Exception as exc:
            self._session.rollback()
            raise RepositoryError(f"BacktestRun save failed: {exc}") from exc

        logger.info("BacktestRepository: saved run %s (%s/%s)", run_id, result.strategy_name, result.symbol)
        return run_id

    def get_by_run_id(self, run_id: str) -> Optional[BacktestRun]:
        return self._session.execute(
            select(BacktestRun).where(BacktestRun.run_id == run_id)
        ).scalar_one_or_none()

    def list_runs(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> list[BacktestRun]:
        q = select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit)
        if strategy_name:
            q = q.where(BacktestRun.strategy_name == strategy_name)
        if symbol:
            q = q.where(BacktestRun.symbol == symbol)
        return list(self._session.execute(q).scalars().all())


class WalkForwardRepository:
    """Persist walk-forward validation run results."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, result: WalkForwardResult, config: dict) -> str:
        run_id = str(uuid.uuid4())
        run = WalkForwardRun(
            run_id=run_id,
            strategy_name=result.strategy_name,
            symbol=config.get("symbol", ""),
            param_grid=config.get("param_grid"),
            train_window_days=config.get("train_window_days", 0),
            test_window_days=config.get("test_window_days", 0),
            n_windows=len(result.windows),
            oos_metrics={
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in result.oos_metrics.items()
            },
            param_stability=result.param_stability,
            window_details=result.windows,
        )
        self._session.add(run)
        try:
            self._session.commit()
        except Exception as exc:
            self._session.rollback()
            raise RepositoryError(f"WalkForwardRun save failed: {exc}") from exc
        return run_id
