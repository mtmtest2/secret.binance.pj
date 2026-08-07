"""Async SQLite persistence layer (SQLAlchemy 2.0 + aiosqlite).

Design notes
------------
* **Everything is awaited.**  There is not a single synchronous DB call, so the
  5-minute trading loop and the FastAPI panel never block the event loop.
* **Idempotent writes.**  Market data is written with SQLite ``INSERT ... ON
  CONFLICT DO UPDATE`` so re-running a cycle (or replaying a healed block) can
  never create duplicates - the ``(symbol, timeframe, timestamp)`` uniqueness
  constraint is the source of truth.
* **Read path optimised for Module B.**  :meth:`load_ohlcv_dataframe` returns a
  ``DatetimeIndex``-ed frame straight from SQL with a ``LIMIT`` applied on the
  *descending* ordering, so pulling "the last 3000 candles" never scans the
  whole table.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Any, Final, Sequence

import pandas as pd
from sqlalchemy import Select, delete, desc, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Result
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config.settings import Settings
from core.exceptions import DatabaseError
from core.logger import get_logger
from module_a_data.db_models import (
    AuditLogRow,
    Base,
    EquityRow,
    FuturesMetricsRow,
    OHLCVRow,
    OrderBookRow,
    SystemStateRow,
    TradeRow,
)
from module_a_data.models import FuturesMetrics, OHLCVCandle, OrderBookSnapshot

_LOGGER = get_logger(__name__)

_OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class DatabaseHandler:
    """Owns the async engine and exposes every persistence operation."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        """Create the engine, apply SQLite PRAGMAs and materialise the schema."""
        if self._engine is not None:
            return

        self._settings.db.path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_async_engine(
            self._settings.db.url,
            echo=self._settings.db.echo,
            future=True,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

        try:
            async with self._engine.begin() as connection:
                # WAL lets the web panel read while the trading loop writes.
                await connection.exec_driver_sql(
                    f"PRAGMA journal_mode={self._settings.db.journal_mode};"
                )
                await connection.exec_driver_sql("PRAGMA synchronous=NORMAL;")
                await connection.exec_driver_sql("PRAGMA foreign_keys=ON;")
                await connection.exec_driver_sql(
                    f"PRAGMA busy_timeout={self._settings.db.busy_timeout_ms};"
                )
                await connection.run_sync(Base.metadata.create_all)
        except SQLAlchemyError as error:
            raise DatabaseError("failed to initialise the database", reason=str(error)) from error

        _LOGGER.info("Database ready at %s", self._settings.db.path)

    async def close(self) -> None:
        """Dispose of the engine and its connection pool."""
        if self._engine is None:
            return
        await self._engine.dispose()
        self._engine = None
        self._session_factory = None
        _LOGGER.info("Database connections closed")

    async def __aenter__(self) -> "DatabaseHandler":
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def _factory(self) -> async_sessionmaker[AsyncSession]:
        """Return the session factory, failing loudly if not initialised."""
        if self._session_factory is None:
            raise DatabaseError("DatabaseHandler.initialize() has not been awaited")
        return self._session_factory

    # ------------------------------------------------------------------
    # Market-data writes
    # ------------------------------------------------------------------
    async def upsert_candles(self, candles: Sequence[OHLCVCandle]) -> int:
        """Bulk-upsert validated candles.

        Returns:
            The number of rows submitted (SQLite does not report affected rows
            reliably for multi-row upserts).
        """
        if not candles:
            return 0

        payload: list[dict[str, Any]] = [
            {
                "symbol": candle.symbol,
                "timeframe": candle.timeframe,
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in candles
        ]

        statement = sqlite_insert(OHLCVRow).values(payload)
        statement = statement.on_conflict_do_update(
            index_elements=[OHLCVRow.symbol, OHLCVRow.timeframe, OHLCVRow.timestamp],
            set_={
                "open": statement.excluded.open,
                "high": statement.excluded.high,
                "low": statement.excluded.low,
                "close": statement.excluded.close,
                "volume": statement.excluded.volume,
            },
        )

        try:
            async with self._factory()() as session:
                async with session.begin():
                    await session.execute(statement)
        except SQLAlchemyError as error:
            raise DatabaseError("candle upsert failed", rows=len(payload)) from error
        return len(payload)

    async def upsert_order_book(self, snapshot: OrderBookSnapshot) -> None:
        """Persist a reduced order-book snapshot (idempotent per timestamp)."""
        values: dict[str, Any] = {
            "symbol": snapshot.symbol,
            "timestamp": snapshot.timestamp,
            "best_bid": snapshot.best_bid,
            "best_ask": snapshot.best_ask,
            "spread": snapshot.spread,
            "spread_bps": snapshot.spread_bps,
            "bid_volume": snapshot.bid_volume,
            "ask_volume": snapshot.ask_volume,
            "imbalance": snapshot.imbalance,
            "microprice": snapshot.microprice,
            "levels": snapshot.levels,
        }
        statement = sqlite_insert(OrderBookRow).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[OrderBookRow.symbol, OrderBookRow.timestamp],
            set_={key: statement.excluded[key] for key in values if key not in ("symbol", "timestamp")},
        )
        try:
            async with self._factory()() as session:
                async with session.begin():
                    await session.execute(statement)
        except SQLAlchemyError as error:
            raise DatabaseError("order book upsert failed", symbol=snapshot.symbol) from error

    async def upsert_futures_metrics(self, metrics: FuturesMetrics) -> None:
        """Persist funding / OI / positioning / liquidation metrics."""
        values: dict[str, Any] = {
            "symbol": metrics.symbol,
            "timestamp": metrics.timestamp,
            "funding_rate": metrics.funding_rate,
            "next_funding_time": metrics.next_funding_time,
            "open_interest": metrics.open_interest,
            "open_interest_value": metrics.open_interest_value,
            "long_short_ratio": metrics.long_short_ratio,
            "top_trader_long_short_ratio": metrics.top_trader_long_short_ratio,
            "taker_buy_sell_ratio": metrics.taker_buy_sell_ratio,
            "liquidation_buy_volume": metrics.liquidation_buy_volume,
            "liquidation_sell_volume": metrics.liquidation_sell_volume,
            "mark_price": metrics.mark_price,
            "index_price": metrics.index_price,
        }
        statement = sqlite_insert(FuturesMetricsRow).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[FuturesMetricsRow.symbol, FuturesMetricsRow.timestamp],
            set_={
                key: statement.excluded[key]
                for key in values
                if key not in ("symbol", "timestamp")
            },
        )
        try:
            async with self._factory()() as session:
                async with session.begin():
                    await session.execute(statement)
        except SQLAlchemyError as error:
            raise DatabaseError("futures metrics upsert failed", symbol=metrics.symbol) from error

    # ------------------------------------------------------------------
    # Market-data reads
    # ------------------------------------------------------------------
    async def load_ohlcv_dataframe(
        self,
        symbol: str,
        limit: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        timeframe: str | None = None,
    ) -> pd.DataFrame:
        """Load candles as a UTC-indexed OHLCV frame ready for Module B.

        The ``limit`` is applied to a *descending* query and the result is
        re-sorted ascending, so "the most recent N candles" is an index seek
        rather than a table scan.

        Args:
            timeframe: Defaults to the configured trading timeframe (``"5m"``).
                Pass e.g. ``"1m"`` to read the labeler's cached intra-candle
                refinement data instead.
        """
        timeframe = timeframe if timeframe is not None else self._settings.data.timeframe
        query: Select[Any] = select(
            OHLCVRow.timestamp,
            OHLCVRow.open,
            OHLCVRow.high,
            OHLCVRow.low,
            OHLCVRow.close,
            OHLCVRow.volume,
        ).where(OHLCVRow.symbol == symbol, OHLCVRow.timeframe == timeframe)

        if start_ms is not None:
            query = query.where(OHLCVRow.timestamp >= start_ms)
        if end_ms is not None:
            query = query.where(OHLCVRow.timestamp <= end_ms)

        query = query.order_by(desc(OHLCVRow.timestamp)) if limit else query.order_by(
            OHLCVRow.timestamp
        )
        if limit:
            query = query.limit(limit)

        try:
            async with self._factory()() as session:
                result: Result[Any] = await session.execute(query)
                rows: list[Any] = result.all()
        except SQLAlchemyError as error:
            raise DatabaseError("OHLCV read failed", symbol=symbol) from error

        if not rows:
            empty: pd.DataFrame = pd.DataFrame(columns=list(_OHLCV_COLUMNS))
            empty.index = pd.DatetimeIndex([], name="open_time", tz="UTC")
            return empty

        frame: pd.DataFrame = pd.DataFrame(rows, columns=list(_OHLCV_COLUMNS))
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame.index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame.index.name = "open_time"
        return frame

    async def load_candles(
        self,
        symbol: str,
        limit: int,
    ) -> list[OHLCVCandle]:
        """Load the most recent ``limit`` candles as Pydantic models."""
        frame: pd.DataFrame = await self.load_ohlcv_dataframe(symbol, limit=limit)
        candles: list[OHLCVCandle] = []
        for record in frame.to_dict(orient="records"):
            candles.append(
                OHLCVCandle(
                    symbol=symbol,
                    timeframe=self._settings.data.timeframe,
                    timestamp=int(record["timestamp"]),
                    open=float(record["open"]),
                    high=float(record["high"]),
                    low=float(record["low"]),
                    close=float(record["close"]),
                    volume=float(record["volume"]),
                )
            )
        return candles

    async def latest_candle_timestamp(self, symbol: str) -> int | None:
        """Return the newest stored candle open time for ``symbol``."""
        query: Select[Any] = select(func.max(OHLCVRow.timestamp)).where(
            OHLCVRow.symbol == symbol,
            OHLCVRow.timeframe == self._settings.data.timeframe,
        )
        async with self._factory()() as session:
            result: Result[Any] = await session.execute(query)
            value: Any = result.scalar_one_or_none()
        return int(value) if value is not None else None

    async def candle_count(self, symbol: str) -> int:
        """Return how many candles are stored for ``symbol``."""
        query: Select[Any] = select(func.count()).select_from(OHLCVRow).where(
            OHLCVRow.symbol == symbol,
            OHLCVRow.timeframe == self._settings.data.timeframe,
        )
        async with self._factory()() as session:
            result: Result[Any] = await session.execute(query)
            return int(result.scalar_one())

    async def load_latest_order_book(self, symbol: str) -> dict[str, Any] | None:
        """Return the newest stored order-book snapshot as a plain dict."""
        query: Select[Any] = (
            select(
                OrderBookRow.timestamp,
                OrderBookRow.spread_bps,
                OrderBookRow.bid_volume,
                OrderBookRow.ask_volume,
                OrderBookRow.imbalance,
                OrderBookRow.microprice,
            )
            .where(OrderBookRow.symbol == symbol)
            .order_by(desc(OrderBookRow.timestamp))
            .limit(1)
        )
        async with self._factory()() as session:
            result: Result[Any] = await session.execute(query)
            row: Any = result.first()
        if row is None:
            return None
        return {
            "timestamp": int(row.timestamp),
            "spread_bps": float(row.spread_bps),
            "bid_volume": float(row.bid_volume),
            "ask_volume": float(row.ask_volume),
            "imbalance": float(row.imbalance),
            "microprice": float(row.microprice),
        }

    async def load_futures_metrics_frame(self, symbol: str, limit: int = 1_000) -> pd.DataFrame:
        """Load recent futures metrics as a timestamp-indexed frame."""
        query: Select[Any] = (
            select(
                FuturesMetricsRow.timestamp,
                FuturesMetricsRow.funding_rate,
                FuturesMetricsRow.open_interest,
                FuturesMetricsRow.long_short_ratio,
                FuturesMetricsRow.taker_buy_sell_ratio,
                FuturesMetricsRow.liquidation_buy_volume,
                FuturesMetricsRow.liquidation_sell_volume,
            )
            .where(FuturesMetricsRow.symbol == symbol)
            .order_by(desc(FuturesMetricsRow.timestamp))
            .limit(limit)
        )
        async with self._factory()() as session:
            result: Result[Any] = await session.execute(query)
            rows: list[Any] = result.all()

        columns: list[str] = [
            "timestamp",
            "funding_rate",
            "open_interest",
            "long_short_ratio",
            "taker_buy_sell_ratio",
            "liquidation_buy_volume",
            "liquidation_sell_volume",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame: pd.DataFrame = pd.DataFrame(rows, columns=columns)
        return frame.sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------
    async def insert_audit_log(self, record: dict[str, Any]) -> None:
        """Insert one audit record (already flattened by the Audit Engine)."""
        try:
            async with self._factory()() as session:
                async with session.begin():
                    session.add(AuditLogRow(**record))
        except SQLAlchemyError as error:
            raise DatabaseError("audit log insert failed") from error

    async def insert_audit_logs(self, records: Sequence[dict[str, Any]]) -> int:
        """Bulk-insert audit records in a single transaction."""
        if not records:
            return 0
        try:
            async with self._factory()() as session:
                async with session.begin():
                    session.add_all([AuditLogRow(**record) for record in records])
        except SQLAlchemyError as error:
            raise DatabaseError("audit log bulk insert failed", rows=len(records)) from error
        return len(records)

    async def fetch_audit_logs(
        self,
        limit: int = 100,
        symbol: str | None = None,
        verdict: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the newest audit rows as JSON-serialisable dictionaries."""
        query: Select[Any] = select(AuditLogRow).order_by(desc(AuditLogRow.id)).limit(limit)
        if symbol:
            query = query.where(AuditLogRow.symbol == symbol)
        if verdict:
            query = query.where(AuditLogRow.verdict == verdict)

        async with self._factory()() as session:
            result: Result[Any] = await session.execute(query)
            rows: Sequence[AuditLogRow] = result.scalars().all()

        return [
            {
                "id": row.id,
                "decision_id": row.decision_id,
                "cycle_id": row.cycle_id,
                "symbol": row.symbol,
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "candle_timestamp": row.candle_timestamp,
                "verdict": row.verdict,
                "action": row.action,
                "rule_triggered": row.rule_triggered,
                "reason": row.reason,
                "hmm_regime": row.hmm_regime,
                "garch_volatility": row.garch_volatility,
                "garch_vol_percentile": row.garch_vol_percentile,
                "kama_slope": row.kama_slope,
                "fdi": row.fdi,
                "atr": row.atr,
                "order_book_imbalance": row.order_book_imbalance,
                "close_price": row.close_price,
                "prob_long": row.prob_long,
                "prob_short": row.prob_short,
                "prob_no_trade": row.prob_no_trade,
                "direction_confidence": row.direction_confidence,
                "entry_probability": row.entry_probability,
                "entry_allowed": row.entry_allowed,
                "take_profit_pct": row.take_profit_pct,
                "stop_loss_pct": row.stop_loss_pct,
                "trailing_trigger_pct": row.trailing_trigger_pct,
                "recommended_leverage": row.recommended_leverage,
                "capital_allocation_pct": row.capital_allocation_pct,
                "risk_tier": row.risk_tier,
                "risk_guard_state": row.risk_guard_state,
                "latency_ms": row.latency_ms,
            }
            for row in rows
        ]

    async def purge_audit_logs(self, older_than_days: int) -> int:
        """Delete audit rows older than ``older_than_days``; returns rows removed."""
        cutoff: datetime = datetime.now(tz=timezone.utc) - timedelta(days=older_than_days)
        async with self._factory()() as session:
            async with session.begin():
                result: Result[Any] = await session.execute(
                    delete(AuditLogRow).where(AuditLogRow.created_at < cutoff)
                )
        return int(result.rowcount or 0)

    # ------------------------------------------------------------------
    # Trades & equity
    # ------------------------------------------------------------------
    async def insert_trade(self, record: dict[str, Any]) -> int:
        """Insert a newly opened trade; returns its primary key."""
        try:
            async with self._factory()() as session:
                async with session.begin():
                    row: TradeRow = TradeRow(**record)
                    session.add(row)
                await session.refresh(row)
                return int(row.id)
        except SQLAlchemyError as error:
            raise DatabaseError("trade insert failed", decision_id=record.get("decision_id")) from error

    async def update_trade(self, decision_id: str, updates: dict[str, Any]) -> bool:
        """Patch an existing trade row identified by its ``decision_id``."""
        try:
            async with self._factory()() as session:
                async with session.begin():
                    result: Result[Any] = await session.execute(
                        select(TradeRow).where(TradeRow.decision_id == decision_id)
                    )
                    row: TradeRow | None = result.scalar_one_or_none()
                    if row is None:
                        return False
                    for key, value in updates.items():
                        setattr(row, key, value)
            return True
        except SQLAlchemyError as error:
            raise DatabaseError("trade update failed", decision_id=decision_id) from error

    async def fetch_trades(
        self,
        status: str | None = None,
        mode: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return trades as dictionaries, newest first."""
        query: Select[Any] = select(TradeRow).order_by(desc(TradeRow.id)).limit(limit)
        if status:
            query = query.where(TradeRow.status == status)
        if mode:
            query = query.where(TradeRow.mode == mode)

        async with self._factory()() as session:
            result: Result[Any] = await session.execute(query)
            rows: Sequence[TradeRow] = result.scalars().all()

        return [
            {
                "id": row.id,
                "decision_id": row.decision_id,
                "symbol": row.symbol,
                "mode": row.mode,
                "side": row.side,
                "status": row.status,
                "leverage": row.leverage,
                "quantity": row.quantity,
                "notional": row.notional,
                "margin": row.margin,
                "entry_price": row.entry_price,
                "exit_price": row.exit_price,
                "take_profit": row.take_profit,
                "stop_loss": row.stop_loss,
                "trailing_trigger": row.trailing_trigger,
                "trailing_stop": row.trailing_stop,
                "liquidation_price": row.liquidation_price,
                "fees_paid": row.fees_paid,
                "funding_paid": row.funding_paid,
                "realized_pnl": row.realized_pnl,
                "opened_at": row.opened_at.isoformat() if row.opened_at else "",
                "closed_at": row.closed_at.isoformat() if row.closed_at else "",
                "close_reason": row.close_reason,
            }
            for row in rows
        ]

    async def insert_equity_point(self, record: dict[str, Any]) -> None:
        """Append a point to the equity curve."""
        try:
            async with self._factory()() as session:
                async with session.begin():
                    session.add(EquityRow(**record))
        except SQLAlchemyError as error:
            raise DatabaseError("equity insert failed") from error

    async def fetch_equity_curve(self, mode: str, limit: int = 500) -> list[dict[str, Any]]:
        """Return the most recent equity-curve points in chronological order."""
        query: Select[Any] = (
            select(EquityRow)
            .where(EquityRow.mode == mode)
            .order_by(desc(EquityRow.timestamp))
            .limit(limit)
        )
        async with self._factory()() as session:
            result: Result[Any] = await session.execute(query)
            rows: Sequence[EquityRow] = result.scalars().all()

        points: list[dict[str, Any]] = [
            {
                "timestamp": row.timestamp,
                "balance": row.balance,
                "equity": row.equity,
                "unrealized_pnl": row.unrealized_pnl,
                "open_positions": row.open_positions,
                "drawdown_pct": row.drawdown_pct,
            }
            for row in rows
        ]
        return list(reversed(points))

    # ------------------------------------------------------------------
    # Durable key/value state
    # ------------------------------------------------------------------
    async def set_state(self, key: str, value: dict[str, Any]) -> None:
        """Upsert a durable state blob (Risk Guard state, cycle metadata, ...)."""
        statement = sqlite_insert(SystemStateRow).values(key=key, value=value)
        statement = statement.on_conflict_do_update(
            index_elements=[SystemStateRow.key],
            set_={"value": statement.excluded.value, "updated_at": datetime.now(tz=timezone.utc)},
        )
        try:
            async with self._factory()() as session:
                async with session.begin():
                    await session.execute(statement)
        except SQLAlchemyError as error:
            raise DatabaseError("state upsert failed", key=key) from error

    async def get_state(self, key: str) -> dict[str, Any] | None:
        """Read a durable state blob, or ``None`` when the key is absent."""
        async with self._factory()() as session:
            result: Result[Any] = await session.execute(
                select(SystemStateRow.value).where(SystemStateRow.key == key)
            )
            value: Any = result.scalar_one_or_none()
        return dict(value) if isinstance(value, dict) else None
