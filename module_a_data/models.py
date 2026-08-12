"""Strict Pydantic schemas for every piece of raw market data.

These models form the *first* line of defence of the QC pipeline: structural and
per-row logical invariants (``high >= low``, ``volume >= 0``, aligned
timestamps, ...) are enforced here at parse time, so no malformed row can ever
reach the feature engineering stage.  Series-level checks (gaps, spikes,
staleness) are the job of :mod:`module_a_data.qc_validator`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from core.utils import ms_to_datetime

_MIN_PLAUSIBLE_TIMESTAMP_MS: int = 1_262_304_000_000  # 2010-01-01, sanity floor.


class QCSeverity(str, Enum):
    """Severity ladder used by the QC gatekeeper."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class QCIssueCode(str, Enum):
    """Machine-readable identifiers for every QC failure mode."""

    EMPTY_DATASET = "EMPTY_DATASET"
    TOO_FEW_ROWS = "TOO_FEW_ROWS"
    MISALIGNED_TIMESTAMP = "MISALIGNED_TIMESTAMP"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    UNSORTED_TIMESTAMPS = "UNSORTED_TIMESTAMPS"
    MISSING_CANDLES = "MISSING_CANDLES"
    FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"
    STALE_DATA = "STALE_DATA"
    PRICE_LOGIC_VIOLATION = "PRICE_LOGIC_VIOLATION"
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    EXCESSIVE_ZERO_VOLUME = "EXCESSIVE_ZERO_VOLUME"
    RETURN_OUTLIER = "RETURN_OUTLIER"
    NAN_VALUES = "NAN_VALUES"
    ORDERBOOK_CROSSED = "ORDERBOOK_CROSSED"
    ORDERBOOK_EMPTY = "ORDERBOOK_EMPTY"


class OHLCVCandle(BaseModel):
    """A single, fully closed 5-minute candle.

    Invariants enforced at construction time:

    * ``timestamp`` is the *open* time in epoch milliseconds, on the 5m grid.
    * ``high >= max(open, close)`` and ``low <= min(open, close)``.
    * ``high >= low`` and every price is strictly positive.
    * ``volume >= 0``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=3)
    timeframe: str = Field(default="5m")
    timestamp: int = Field(ge=_MIN_PLAUSIBLE_TIMESTAMP_MS, description="Candle open time (ms).")
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    quote_volume: float = Field(default=0.0, ge=0.0)
    trades: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def open_time(self) -> datetime:
        """Candle open time as a timezone-aware UTC datetime."""
        return ms_to_datetime(self.timestamp)

    @model_validator(mode="after")
    def _validate_price_logic(self) -> "OHLCVCandle":
        """Reject candles that violate elementary OHLC geometry."""
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"high ({self.high}) below open/close ({self.open}/{self.close})")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"low ({self.low}) above open/close ({self.open}/{self.close})")
        return self

    @classmethod
    def from_ccxt(
        cls,
        row: Sequence[Any],
        symbol: str,
        timeframe: str = "5m",
    ) -> "OHLCVCandle":
        """Build a candle from a raw ccxt ``[ts, o, h, l, c, v]`` row."""
        if len(row) < 6:
            raise ValueError(f"malformed ccxt OHLCV row for {symbol}: {row!r}")
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )


class OrderBookSnapshot(BaseModel):
    """Aggregated L2 order-book state at a point in time.

    Rather than storing the entire book, we persist the derived micro-structure
    statistics the ML models actually consume: spread, depth-weighted volumes
    and the resulting order-flow imbalance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=3)
    timestamp: int = Field(ge=_MIN_PLAUSIBLE_TIMESTAMP_MS)
    best_bid: float = Field(gt=0.0)
    best_ask: float = Field(gt=0.0)
    bid_volume: float = Field(ge=0.0, description="Summed size over the top N bid levels.")
    ask_volume: float = Field(ge=0.0, description="Summed size over the top N ask levels.")
    levels: int = Field(ge=1, description="Number of levels aggregated per side.")

    @model_validator(mode="after")
    def _validate_book(self) -> "OrderBookSnapshot":
        """Reject crossed books (``bid >= ask``), which indicate a stale snapshot."""
        if self.best_bid >= self.best_ask:
            raise ValueError(f"crossed book: bid {self.best_bid} >= ask {self.best_ask}")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid_price(self) -> float:
        """Arithmetic mid price."""
        return (self.best_bid + self.best_ask) / 2.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> float:
        """Absolute bid/ask spread."""
        return self.best_ask - self.best_bid

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread_bps(self) -> float:
        """Relative spread in basis points of the mid price."""
        mid: float = self.mid_price
        return 0.0 if mid <= 0.0 else (self.spread / mid) * 10_000.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def imbalance(self) -> float:
        """Order-book imbalance in ``[-1, 1]``.

        ``(bid_volume - ask_volume) / (bid_volume + ask_volume)``.  Positive
        values mean resting buy-side pressure dominates.
        """
        total: float = self.bid_volume + self.ask_volume
        return 0.0 if total <= 0.0 else (self.bid_volume - self.ask_volume) / total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def microprice(self) -> float:
        """Size-weighted mid price - a better short-horizon fair-value estimate."""
        total: float = self.bid_volume + self.ask_volume
        if total <= 0.0:
            return self.mid_price
        return (self.best_bid * self.ask_volume + self.best_ask * self.bid_volume) / total


class FuturesMetrics(BaseModel):
    """Perpetual-futures specific state for a symbol at a point in time.

    The positioning ratios Binance publishes under ``futures/data``
    (``longShortRatio``, ``takerlongshortRatio``) are deliberately **not**
    modelled here: Binance only retains ~30 days of them, so they cannot be
    reconstructed across the training period and any historical value would be a
    fabrication.  Aggressive buy/sell pressure is instead measured directly from
    aggTrades - see :class:`AggTradeFlow`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=3)
    timestamp: int = Field(ge=_MIN_PLAUSIBLE_TIMESTAMP_MS)
    funding_rate: float = Field(default=0.0, description="Current/most recent funding rate.")
    next_funding_time: int | None = Field(default=None)
    open_interest: float = Field(default=0.0, ge=0.0, description="OI in base contracts.")
    open_interest_value: float = Field(default=0.0, ge=0.0, description="OI notional in USDT.")
    liquidation_buy_volume: float = Field(default=0.0, ge=0.0)
    liquidation_sell_volume: float = Field(default=0.0, ge=0.0)
    mark_price: float = Field(default=0.0, ge=0.0)
    index_price: float = Field(default=0.0, ge=0.0)

    @field_validator("funding_rate")
    @classmethod
    def _sanity_check_funding(cls, value: float) -> float:
        """Binance caps funding at +/-2 % per interval; anything beyond is a glitch."""
        if abs(value) > 0.05:
            raise ValueError(f"implausible funding rate: {value}")
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_liquidation_volume(self) -> float:
        """Signed liquidation flow (buy-side liquidations minus sell-side)."""
        return self.liquidation_buy_volume - self.liquidation_sell_volume


class AggTradeFlow(BaseModel):
    """Aggressive buy/sell volume for one fully closed 5-minute bucket.

    Built by folding Binance Futures ``aggTrades`` onto the 5-minute grid.  Each
    aggregated trade carries ``isBuyerMaker``:

    * ``isBuyerMaker == False`` - the buyer *took* the offer, so the aggressor
      was a buyer and the quantity counts as **aggressive buy volume**.
    * ``isBuyerMaker == True`` - the seller took the bid, so the quantity counts
      as **aggressive sell volume**.

    ``timestamp`` is the bucket's open time and is aligned to the same 5-minute
    grid as :class:`OHLCVCandle`, which is what lets the two be joined on an
    exact key rather than an as-of match.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=3)
    timestamp: int = Field(ge=_MIN_PLAUSIBLE_TIMESTAMP_MS, description="Bucket open time (ms).")
    buy_volume: float = Field(default=0.0, ge=0.0, description="Aggressive buy base volume.")
    sell_volume: float = Field(default=0.0, ge=0.0, description="Aggressive sell base volume.")
    buy_quote_volume: float = Field(default=0.0, ge=0.0)
    sell_quote_volume: float = Field(default=0.0, ge=0.0)
    trades: int = Field(default=0, ge=0, description="Aggregated trades in the bucket.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def volume_delta(self) -> float:
        """Signed aggressive flow: ``buy_volume - sell_volume``."""
        return self.buy_volume - self.sell_volume

    @computed_field  # type: ignore[prop-decorator]
    @property
    def order_flow_imbalance(self) -> float:
        """Normalised flow in ``[-1, 1]``; ``0.0`` for an empty bucket."""
        total: float = self.buy_volume + self.sell_volume
        return 0.0 if total <= 0.0 else (self.buy_volume - self.sell_volume) / total


class QCIssue(BaseModel):
    """A single problem detected by the QC gatekeeper."""

    model_config = ConfigDict(frozen=True)

    code: QCIssueCode
    severity: QCSeverity
    message: str
    symbol: str
    timestamps: tuple[int, ...] = Field(default=())
    healable: bool = Field(default=False, description="Can a targeted re-fetch fix this?")

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"[{self.severity.value}/{self.code.value}] {self.symbol}: {self.message}"


class QCReport(BaseModel):
    """Aggregated verdict for one symbol's fetched data block."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    checked_rows: int = Field(ge=0)
    issues: tuple[QCIssue, ...] = Field(default=())
    missing_timestamps: tuple[int, ...] = Field(default=())
    first_timestamp: int | None = Field(default=None)
    last_timestamp: int | None = Field(default=None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """``True`` when no CRITICAL issue was recorded."""
        return not any(issue.severity is QCSeverity.CRITICAL for issue in self.issues)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def healable(self) -> bool:
        """``True`` when every CRITICAL issue can plausibly be fixed by re-fetching."""
        criticals: list[QCIssue] = [
            issue for issue in self.issues if issue.severity is QCSeverity.CRITICAL
        ]
        return bool(criticals) and all(issue.healable for issue in criticals)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def critical_codes(self) -> tuple[str, ...]:
        """Distinct CRITICAL issue codes, in first-seen order."""
        seen: list[str] = []
        for issue in self.issues:
            if issue.severity is QCSeverity.CRITICAL and issue.code.value not in seen:
                seen.append(issue.code.value)
        return tuple(seen)

    def summary(self) -> str:
        """Human-readable one-line summary for logs and the audit trail."""
        if self.passed:
            return f"{self.symbol}: OK ({self.checked_rows} rows)"
        return f"{self.symbol}: FAILED {', '.join(self.critical_codes)}"


class MarketDataBundle(BaseModel):
    """Everything Module A produces for a single symbol in one 5m cycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str
    candles: tuple[OHLCVCandle, ...]
    order_book: OrderBookSnapshot | None = Field(default=None)
    futures: FuturesMetrics | None = Field(default=None)
    qc_report: QCReport | None = Field(default=None)
    fetched_at_ms: int = Field(default=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        """``True`` when candles exist and the QC gatekeeper approved them."""
        return bool(self.candles) and (self.qc_report is None or self.qc_report.passed)

    def to_dataframe(self) -> pd.DataFrame:
        """Materialise the candles as an OHLCV ``DataFrame`` indexed by open time.

        The frame is sorted ascending by timestamp and carries a
        ``DatetimeIndex`` in UTC, which is what Module B expects.
        """
        if not self.candles:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            ).set_index(pd.DatetimeIndex([], name="open_time", tz="UTC"))

        records: list[dict[str, Any]] = [
            {
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in self.candles
        ]
        frame: pd.DataFrame = pd.DataFrame.from_records(records)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame.index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame.index.name = "open_time"
        return frame
