"""Shared execution-layer data structures.

The same :class:`Position` object is used by the live executor, the paper trader
and the backtester.  That is deliberate: identical PnL, fee, funding and
trailing-stop arithmetic in all three engines is what makes a backtest result
comparable to a paper result comparable to a live result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from module_c_ml.schemas import TradeAction, TradeSignal


class PositionStatus(str, Enum):
    """Lifecycle of a position."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"
    FAILED = "FAILED"


class CloseReason(str, Enum):
    """Why a position was closed - recorded on every trade row."""

    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    LIQUIDATION = "LIQUIDATION"
    KILL_SWITCH = "KILL_SWITCH"
    MANUAL = "MANUAL"
    TIMEOUT = "TIMEOUT"
    SHUTDOWN = "SHUTDOWN"


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(tz=timezone.utc)


@dataclass(slots=True)
class Position:
    """An open (or recently closed) leveraged futures position."""

    decision_id: str
    symbol: str
    action: TradeAction
    entry_price: float
    quantity: float
    leverage: int
    margin: float
    take_profit: float
    stop_loss: float
    trailing_trigger: float
    trailing_distance_pct: float
    mode: str = "paper"
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = field(default_factory=_utcnow)
    closed_at: datetime | None = None
    close_reason: str = ""

    trailing_active: bool = False
    trailing_stop: float = 0.0
    liquidation_price: float = 0.0

    fees_paid: float = 0.0
    funding_paid: float = 0.0
    realized_pnl: float = 0.0
    exit_price: float = 0.0

    max_favorable_price: float = 0.0
    max_adverse_price: float = 0.0
    last_funding_ms: int = 0

    risk_tier: str = "UNKNOWN"
    confidence: float = 0.0
    exchange_order_ids: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def is_long(self) -> bool:
        """``True`` for a long position."""
        return self.action is TradeAction.LONG

    @property
    def direction(self) -> int:
        """``+1`` for long, ``-1`` for short - the sign used in PnL arithmetic."""
        return 1 if self.is_long else -1

    @property
    def notional(self) -> float:
        """Position notional at entry, in USDT."""
        return self.quantity * self.entry_price

    @property
    def side(self) -> str:
        """ccxt side that *opens* this position."""
        return "buy" if self.is_long else "sell"

    @property
    def reduce_side(self) -> str:
        """ccxt side that *closes* this position."""
        return "sell" if self.is_long else "buy"

    def unrealized_pnl(self, mark_price: float) -> float:
        """Mark-to-market PnL in USDT, excluding fees already paid."""
        return (mark_price - self.entry_price) * self.quantity * self.direction

    def unrealized_pnl_pct(self, mark_price: float) -> float:
        """Mark-to-market PnL as a fraction of the committed margin."""
        if self.margin <= 0.0:
            return 0.0
        return self.unrealized_pnl(mark_price) / self.margin

    def effective_stop(self) -> float:
        """The stop currently in force - the trailing stop once it has armed."""
        return self.trailing_stop if self.trailing_active else self.stop_loss

    # ------------------------------------------------------------------
    # Path tracking
    # ------------------------------------------------------------------
    def update_excursions(self, high: float, low: float) -> None:
        """Record the best and worst prices seen while the position is open."""
        if self.max_favorable_price == 0.0:
            self.max_favorable_price = self.entry_price
            self.max_adverse_price = self.entry_price

        if self.is_long:
            self.max_favorable_price = max(self.max_favorable_price, high)
            self.max_adverse_price = min(self.max_adverse_price, low)
        else:
            self.max_favorable_price = min(self.max_favorable_price, low)
            self.max_adverse_price = max(self.max_adverse_price, high)

    @property
    def max_favorable_excursion(self) -> float:
        """Best unrealised PnL reached, in USDT."""
        if self.max_favorable_price == 0.0:
            return 0.0
        return (self.max_favorable_price - self.entry_price) * self.quantity * self.direction

    @property
    def max_adverse_excursion(self) -> float:
        """Worst unrealised PnL reached, in USDT (negative or zero)."""
        if self.max_adverse_price == 0.0:
            return 0.0
        return (self.max_adverse_price - self.entry_price) * self.quantity * self.direction

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------
    def should_arm_trailing(self, price: float) -> bool:
        """``True`` when price has moved past the trailing trigger for the first time."""
        if self.trailing_active:
            return False
        return price >= self.trailing_trigger if self.is_long else price <= self.trailing_trigger

    def compute_trailing_stop(self, price: float) -> float:
        """Trailing stop level implied by ``price`` and the trailing distance."""
        offset: float = price * self.trailing_distance_pct
        return price - offset if self.is_long else price + offset

    def advance_trailing(self, price: float) -> float | None:
        """Ratchet the trailing stop toward ``price``.

        The stop only ever moves in the profitable direction - a trailing stop
        that could loosen would be a bug that silently increases risk.

        Returns:
            The new stop level when it moved, otherwise ``None``.
        """
        candidate: float = self.compute_trailing_stop(price)
        if not self.trailing_active:
            if not self.should_arm_trailing(price):
                return None
            self.trailing_active = True
            self.trailing_stop = candidate
            return candidate

        improved: bool = (
            candidate > self.trailing_stop if self.is_long else candidate < self.trailing_stop
        )
        if not improved:
            return None
        self.trailing_stop = candidate
        return candidate

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        """Flatten into the column layout of the ``trades`` table."""
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "mode": self.mode,
            "side": self.action.value,
            "status": self.status.value,
            "leverage": self.leverage,
            "quantity": self.quantity,
            "notional": self.notional,
            "margin": self.margin,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "trailing_trigger": self.trailing_trigger,
            "trailing_stop": self.trailing_stop,
            "liquidation_price": self.liquidation_price,
            "fees_paid": self.fees_paid,
            "funding_paid": self.funding_paid,
            "realized_pnl": self.realized_pnl,
            "max_adverse_excursion": self.max_adverse_excursion,
            "max_favorable_excursion": self.max_favorable_excursion,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "close_reason": self.close_reason,
            "exchange_order_id": self.exchange_order_ids.get("entry", ""),
            "payload": {
                "risk_tier": self.risk_tier,
                "confidence": self.confidence,
                "trailing_active": self.trailing_active,
                **self.metadata,
            },
        }

    def to_view(self, mark_price: float | None = None) -> dict[str, Any]:
        """Compact representation for the web dashboard."""
        price: float = mark_price if mark_price is not None else self.entry_price
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "side": self.action.value,
            "status": self.status.value,
            "leverage": self.leverage,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "mark_price": price,
            "take_profit": self.take_profit,
            "stop_loss": self.effective_stop(),
            "trailing_active": self.trailing_active,
            "liquidation_price": self.liquidation_price,
            "margin": self.margin,
            "unrealized_pnl": self.unrealized_pnl(price),
            "unrealized_pnl_pct": self.unrealized_pnl_pct(price),
            "risk_tier": self.risk_tier,
            "opened_at": self.opened_at.isoformat(timespec="seconds"),
        }

    @classmethod
    def from_signal(
        cls,
        signal: TradeSignal,
        equity: float,
        entry_price: float,
        quantity: float,
        maintenance_margin_rate: float,
        mode: str,
    ) -> "Position":
        """Materialise a position from a validated signal and a fill.

        The liquidation price uses the standard isolated-margin approximation::

            long  : entry * (1 - 1/leverage + mmr)
            short : entry * (1 + 1/leverage - mmr)

        It ignores accrued fees and funding, so it is a *conservative-by-a-hair*
        estimate that the engines re-check against the exchange's own value where
        one is available.
        """
        leverage: float = float(max(1, signal.leverage))
        if signal.action is TradeAction.LONG:
            liquidation: float = entry_price * (1.0 - 1.0 / leverage + maintenance_margin_rate)
        else:
            liquidation = entry_price * (1.0 + 1.0 / leverage - maintenance_margin_rate)

        return cls(
            decision_id=signal.decision_id,
            symbol=signal.symbol,
            action=signal.action,
            entry_price=entry_price,
            quantity=quantity,
            leverage=signal.leverage,
            margin=signal.margin_for(equity),
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            trailing_trigger=signal.trailing_trigger,
            trailing_distance_pct=signal.trailing_distance_pct,
            mode=mode,
            liquidation_price=max(0.0, liquidation),
            max_favorable_price=entry_price,
            max_adverse_price=entry_price,
            risk_tier=signal.risk_tier,
            confidence=signal.confidence,
            metadata=dict(signal.metadata),
        )


@dataclass(slots=True)
class AccountState:
    """Snapshot of account equity used by the Risk Guard and the dashboard."""

    mode: str
    balance: float
    equity: float
    unrealized_pnl: float
    used_margin: float
    open_positions: int
    timestamp_ms: int = 0

    @property
    def free_margin(self) -> float:
        """Equity not committed as initial margin."""
        return max(0.0, self.equity - self.used_margin)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "mode": self.mode,
            "balance": self.balance,
            "equity": self.equity,
            "unrealized_pnl": self.unrealized_pnl,
            "used_margin": self.used_margin,
            "free_margin": self.free_margin,
            "open_positions": self.open_positions,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass(slots=True)
class ExecutionReport:
    """Result of attempting to act on a :class:`TradeSignal`."""

    decision_id: str
    symbol: str
    accepted: bool
    position: Position | None = None
    error: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for the audit log."""
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "accepted": self.accepted,
            "error": self.error,
            "entry_price": self.position.entry_price if self.position else 0.0,
            "quantity": self.position.quantity if self.position else 0.0,
            **self.detail,
        }
