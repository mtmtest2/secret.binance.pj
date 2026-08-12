"""SQLAlchemy 2.0 ORM table definitions backing the whole system.

SQLite was chosen deliberately: the system must stay portable on a small VPS
without Docker or a database server.  Every write path goes through the async
engine (``aiosqlite``) and WAL journalling, which lets the FastAPI panel read
while the trading loop writes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Timezone-aware UTC now, used as the default for audit columns."""
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""

    type_annotation_map = {dict[str, Any]: JSON}


class OHLCVRow(Base):
    """One validated 5-minute candle."""

    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_ohlcv_symbol_tf_ts"),
        Index("ix_ohlcv_symbol_ts", "symbol", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(40), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="5m")
    timestamp: Mapped[int] = mapped_column(Integer, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    inserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class OrderBookRow(Base):
    """A reduced L2 order-book snapshot (micro-structure statistics only)."""

    __tablename__ = "order_book_snapshots"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_book_symbol_ts"),
        Index("ix_book_symbol_ts", "symbol", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(40), nullable=False)
    timestamp: Mapped[int] = mapped_column(Integer, nullable=False)
    best_bid: Mapped[float] = mapped_column(Float, nullable=False)
    best_ask: Mapped[float] = mapped_column(Float, nullable=False)
    spread: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spread_bps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bid_volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ask_volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    imbalance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    microprice: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    levels: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    inserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AggTradeFlowRow(Base):
    """Aggressive buy/sell volume for one closed 5-minute bucket.

    Derived from Binance Futures ``aggTrades`` using ``isBuyerMaker``; this is
    the storage behind the ``order_flow_imbalance_5m`` / ``volume_delta_5m`` /
    ``relative_volume_5m`` feature block.  Its ``timestamp`` shares the 5-minute
    grid with :class:`OHLCVRow`, so the join to candles is an exact-key merge.
    """

    __tablename__ = "agg_trade_flow"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_flow_symbol_tf_ts"),
        Index("ix_flow_symbol_ts", "symbol", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(40), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="5m")
    timestamp: Mapped[int] = mapped_column(Integer, nullable=False)
    buy_volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sell_volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    buy_quote_volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sell_quote_volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class FuturesMetricsRow(Base):
    """Perpetual-specific state: funding, open interest, liquidations.

    The Binance positioning ratios (global/top-trader long-short account ratio,
    taker buy/sell ratio) used to live here.  They were removed with the
    features that consumed them: Binance retains only ~30 days of those series,
    which is far short of the training period, so they can never be backfilled
    honestly.  Aggressive flow is measured from aggTrades instead - see
    :class:`AggTradeFlowRow`.
    """

    __tablename__ = "futures_metrics"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_futures_symbol_ts"),
        Index("ix_futures_symbol_ts", "symbol", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(40), nullable=False)
    timestamp: Mapped[int] = mapped_column(Integer, nullable=False)
    funding_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    next_funding_time: Mapped[int | None] = mapped_column(Integer, nullable=True)
    open_interest: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    open_interest_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    liquidation_buy_volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    liquidation_sell_volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mark_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    index_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    inserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AuditLogRow(Base):
    """One full decision-cycle record produced by the Audit Engine."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_symbol_ts", "symbol", "created_at"),
        Index("ix_audit_decision", "decision_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(String(36), nullable=False)
    cycle_id: Mapped[str] = mapped_column(String(36), nullable=False, default="")
    symbol: Mapped[str] = mapped_column(String(40), nullable=False)
    candle_timestamp: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    verdict: Mapped[str] = mapped_column(String(24), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False, default="NO_TRADE")
    rule_triggered: Mapped[str] = mapped_column(String(96), nullable=False, default="")
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # --- Feature snapshot -------------------------------------------------
    hmm_regime: Mapped[int] = mapped_column(Integer, nullable=False, default=-1)
    garch_volatility: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    garch_vol_percentile: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    kama_slope: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fdi: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    atr: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    order_book_imbalance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    close_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # --- Model outputs ----------------------------------------------------
    prob_long: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prob_short: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prob_no_trade: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    direction_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    entry_probability: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    entry_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    take_profit_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stop_loss_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trailing_trigger_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recommended_leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    capital_allocation_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_tier: Mapped[str] = mapped_column(String(16), nullable=False, default="UNKNOWN")

    risk_guard_state: Mapped[str] = mapped_column(String(8), nullable=False, default="GREEN")
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class TradeRow(Base):
    """A paper or live trade, from signal to closure."""

    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_symbol_status", "symbol", "status"),
        Index("ix_trades_opened", "opened_at"),
        UniqueConstraint("decision_id", name="uq_trades_decision"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(String(36), nullable=False)
    symbol: Mapped[str] = mapped_column(String(40), nullable=False)
    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="paper")
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="OPEN")

    leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    notional: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    margin: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    entry_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    take_profit: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trailing_trigger: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trailing_stop: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    liquidation_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    fees_paid: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    funding_paid: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_adverse_excursion: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_favorable_excursion: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    close_reason: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    exchange_order_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class EquityRow(Base):
    """A point on the equity curve (one per cycle, per mode)."""

    __tablename__ = "equity_curve"
    __table_args__ = (Index("ix_equity_mode_ts", "mode", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="paper")
    timestamp: Mapped[int] = mapped_column(Integer, nullable=False)
    balance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    equity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SystemStateRow(Base):
    """Durable key/value store for state that must survive a restart."""

    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
