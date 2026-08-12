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

from config.settings import TakeProfitSettings
from module_c_ml.schemas import TradeAction, TradeSignal
from module_e_execution.tp_ladder import LadderEvent, StopProtection, TakeProfitLadder


class PositionStatus(str, Enum):
    """Lifecycle of a position."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"
    FAILED = "FAILED"


class CloseReason(str, Enum):
    """Why a position (or one ladder leg) was closed."""

    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_1 = "TAKE_PROFIT_1"
    TAKE_PROFIT_2 = "TAKE_PROFIT_2"
    TAKE_PROFIT_3 = "TAKE_PROFIT_3"
    STOP_LOSS = "STOP_LOSS"
    #: Stopped after TP1 moved the stop to entry - the trade gave back its open
    #: profit but not its capital.
    BREAKEVEN_STOP = "BREAKEVEN_STOP"
    #: Stopped after TP2 moved the stop to TP1 - the trade banked a profit.
    PROFIT_STOP = "PROFIT_STOP"
    TRAILING_STOP = "TRAILING_STOP"
    LIQUIDATION = "LIQUIDATION"
    KILL_SWITCH = "KILL_SWITCH"
    MANUAL = "MANUAL"
    TIMEOUT = "TIMEOUT"
    SHUTDOWN = "SHUTDOWN"


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(tz=timezone.utc)


#: Ladder take-profit event -> the close reason it is booked under.
_LADDER_REASONS: dict[str, CloseReason] = {
    "TAKE_PROFIT_1": CloseReason.TAKE_PROFIT_1,
    "TAKE_PROFIT_2": CloseReason.TAKE_PROFIT_2,
    "TAKE_PROFIT_3": CloseReason.TAKE_PROFIT_3,
}

#: Which stop was in force -> the close reason a stop-out is booked under.
_STOP_REASONS: dict[StopProtection, CloseReason] = {
    StopProtection.INITIAL: CloseReason.STOP_LOSS,
    StopProtection.BREAKEVEN: CloseReason.BREAKEVEN_STOP,
    StopProtection.TP1_LOCKED: CloseReason.PROFIT_STOP,
}


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

    #: Three-stage take-profit ladder.  ``None`` restores the original
    #: single-target behaviour, so the ladder can be switched off end-to-end.
    ladder: TakeProfitLadder | None = None
    #: Quantity at entry.  ``quantity`` shrinks as ladder legs fill, so PnL,
    #: fees and funding on the remainder all need the original as their base.
    initial_quantity: float = 0.0
    #: One record per partially closed leg, for the ladder statistics.
    partial_fills: list[dict[str, Any]] = field(default_factory=list)
    #: Which stop was in force when the position was finally stopped out.
    stop_protection: str = ""

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
        """The stop currently in force across every mechanism that can set one.

        Three things can move a stop: the signal's own level, the trailing stop
        once it arms, and the take-profit ladder's stage.  They are combined by
        taking the *most protective* level rather than by letting the last writer
        win - a ladder that has locked in TP1 must never be loosened by a
        trailing stop that happens to sit lower, and vice versa.
        """
        candidates: list[float] = [self.stop_loss]
        if self.trailing_active and self.trailing_stop > 0.0:
            candidates.append(self.trailing_stop)
        if self.ladder is not None:
            candidates.append(self.ladder.current_stop())
        return max(candidates) if self.is_long else min(candidates)

    def stop_close_reason(self) -> CloseReason:
        """Which close reason a stop-out should be booked under.

        The distinction matters for the report: "stopped at breakeven" and
        "stopped with TP1 locked in" are outcomes the ladder is *designed* to
        produce, and lumping them in with ``STOP_LOSS`` would hide whether it
        works.
        """
        if self.ladder is not None:
            protection: StopProtection = self.ladder.protection
            if protection is StopProtection.TP1_LOCKED:
                return CloseReason.PROFIT_STOP
            if protection is StopProtection.BREAKEVEN:
                return CloseReason.BREAKEVEN_STOP
        return CloseReason.TRAILING_STOP if self.trailing_active else CloseReason.STOP_LOSS

    # ------------------------------------------------------------------
    # Partial closes
    # ------------------------------------------------------------------
    @property
    def risk_per_unit(self) -> float:
        """Distance from entry to the *original* stop - the position's 1R."""
        return abs(self.entry_price - self.initial_stop_price)

    @property
    def initial_stop_price(self) -> float:
        """The stop the position opened with, before any ladder ratcheting."""
        return self.ladder.initial_stop if self.ladder is not None else self.stop_loss

    @property
    def realized_r(self) -> float:
        """Realised PnL expressed in units of the original risk.

        ``0`` risk (a degenerate signal that should never pass validation) yields
        ``0`` rather than an infinity that would poison every average downstream.
        """
        risk: float = self.risk_per_unit * max(self.initial_quantity, 0.0)
        return 0.0 if risk <= 0.0 else self.realized_pnl / risk

    def quantity_for_fraction(self, fraction: float) -> float:
        """Contracts corresponding to a share of the *original* position."""
        return max(0.0, self.initial_quantity * max(0.0, min(1.0, fraction)))

    def book_partial_close(
        self,
        exit_price: float,
        fraction: float,
        fee_rate: float,
        reason: CloseReason,
        timestamp_ms: int = 0,
    ) -> float:
        """Close ``fraction`` of the original position and return the cash delta.

        The caller applies slippage before calling: this method is pure
        accounting, shared by the backtester, the paper trader and the live
        executor so that a ladder leg costs the same everywhere.

        Returns:
            ``gross - exit_fee`` for the leg.  Funding is settled once, on the
            final close, because it accrues on the position rather than the leg.
        """
        quantity: float = min(self.quantity, self.quantity_for_fraction(fraction))
        if quantity <= 0.0 or exit_price <= 0.0:
            return 0.0

        gross: float = (exit_price - self.entry_price) * quantity * self.direction
        fee: float = exit_price * quantity * fee_rate
        self.fees_paid += fee
        self.realized_pnl += gross - fee
        self.quantity = max(0.0, self.quantity - quantity)
        self.partial_fills.append(
            {
                "reason": reason.value,
                "price": float(exit_price),
                "quantity": float(quantity),
                "fraction": float(fraction),
                "pnl": float(gross - fee),
                "timestamp_ms": int(timestamp_ms),
            }
        )
        return gross - fee

    def apply_ladder_event(
        self,
        event: LadderEvent,
        exit_price: float,
        fee_rate: float,
        timestamp_ms: int = 0,
    ) -> tuple[float, CloseReason]:
        """Book one ladder event and report its cash delta and close reason.

        The reason comes from the protection recorded *on the event*, not from
        the ladder's current state: by the time this runs the ladder has already
        advanced to CLOSED, and reading its stage back would report every stop as
        a TP1-locked one.
        """
        if event.is_stop:
            protection: StopProtection = event.protection or StopProtection.INITIAL
            self.stop_protection = protection.value
            reason: CloseReason = _STOP_REASONS[protection]
        else:
            reason = _LADDER_REASONS[event.kind]
        delta: float = self.book_partial_close(
            exit_price=exit_price,
            fraction=event.fraction,
            fee_rate=fee_rate,
            reason=reason,
            timestamp_ms=timestamp_ms,
        )
        return delta, reason

    @property
    def is_flat(self) -> bool:
        """``True`` once every contract has been closed."""
        return self.quantity <= 1e-12

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
                "realized_r": self.realized_r,
                "stop_protection": self.stop_protection,
                "partial_fills": list(self.partial_fills),
                "ladder": self.ladder.to_dict() if self.ladder is not None else None,
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
        take_profit_config: TakeProfitSettings | None = None,
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

        # The ladder is anchored to the *fill*, not to the decision-bar close, so
        # the geometry the models asked for is the geometry the position carries.
        ladder: TakeProfitLadder | None = None
        if take_profit_config is not None and take_profit_config.enabled:
            ladder = TakeProfitLadder.build(
                is_long=signal.action is TradeAction.LONG,
                entry_price=entry_price,
                take_profit_pct=signal.take_profit_pct,
                stop_loss_pct=signal.stop_loss_pct,
                config=take_profit_config,
            )

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
            ladder=ladder,
            initial_quantity=quantity,
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
