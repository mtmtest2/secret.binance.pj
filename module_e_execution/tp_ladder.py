"""The three-stage take-profit ladder and its stage-driven stop loss.

One state machine, three engines
--------------------------------
Backtest, paper and live all drive :class:`TakeProfitLadder`.  That is not tidy
-code aesthetics: a ladder re-implemented per engine is a ladder that behaves
differently in the backtest than in production, and the divergence only shows up
as unexplained live underperformance months later.  The engines differ *only* in
what they feed it:

* the backtester calls :meth:`TakeProfitLadder.on_bar` with an OHLC bar,
* the paper trader and the live monitor call :meth:`TakeProfitLadder.on_tick`
  with an observed price.

Both paths run the same transition table:

==========  ================================  ================================
Stage       Fills                             Stop moves to
==========  ================================  ================================
``INITIAL``  -                                the signal's stop loss
``TP1``     ``close_fractions[0]``            entry (breakeven)
``TP2``     ``close_fractions[1]``            the TP1 price (profit locked)
``TP3``     the remainder - position closed   -
==========  ================================  ================================

Intrabar ordering - the conservative rule
-----------------------------------------
5-minute OHLCV records four numbers per bar and **cannot** say whether the high
came before the low.  Every ordering question is therefore resolved against the
position, and the rule is applied identically in labelling, backtesting and
reporting:

1. **The stop is tested first, against the bar's adverse extreme.**  If the bar
   could have taken the stop out, it did - even when the same bar also reached a
   take-profit level.  A bar that touches both TP1 and the initial stop is booked
   as a stop, never as "TP1 first, then a protected exit".
2. **Only then may the ladder advance**, and it advances strictly in order
   (TP1 -> TP2 -> TP3): those levels are monotone in the favourable direction, so
   filling them in sequence assumes nothing about intrabar order.
3. **After advancing, the tightened stop is re-tested against the same bar's
   adverse extreme.**  A bar that ran up to TP2 and then collapsed gives back the
   TP1-locked stop inside that bar rather than escaping to the next one.

Rule 1 is the pessimistic assumption and rule 3 closes the loophole rule 2 would
otherwise open.  Where intrabar data (aggTrades) is available the sequence could
be proven rather than assumed; until it is fed into the backtester, this module
assumes the unfavourable ordering and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Final

from config.settings import TakeProfitSettings

_EPSILON: Final[float] = 1e-12


class LadderStage(IntEnum):
    """How far up the ladder the position has climbed."""

    INITIAL = 0
    TP1_FILLED = 1
    TP2_FILLED = 2
    CLOSED = 3


class StopProtection(str, Enum):
    """Which stop was in force when a position was stopped out.

    Recorded on every trade so the report can answer "what share of trades was
    stopped at breakeven rather than at the original risk?" without re-deriving
    it from prices.
    """

    INITIAL = "INITIAL"
    BREAKEVEN = "BREAKEVEN"
    TP1_LOCKED = "TP1_LOCKED"


@dataclass(slots=True)
class LadderEvent:
    """One thing the ladder decided to do at a price."""

    kind: str
    """``TAKE_PROFIT_1`` / ``TAKE_PROFIT_2`` / ``TAKE_PROFIT_3`` / ``STOP``."""

    price: float
    """Level the fill is booked at - the ladder price, not the observed one."""

    fraction: float
    """Share of the *original* position this event closes."""

    stage: int
    """Ladder stage after the event."""

    protection: StopProtection | None = None
    """For ``STOP`` events, which stop was in force."""

    @property
    def is_stop(self) -> bool:
        """``True`` for a stop-out rather than a take-profit fill."""
        return self.kind == "STOP"


@dataclass(slots=True)
class TakeProfitLadder:
    """Three-stage take-profit ladder with a stage-driven stop.

    The ladder is expressed in absolute prices so that the backtester, the paper
    trader and the live executor all compare like with like, and it tracks the
    *remaining fraction of the original position* rather than a quantity, so it
    is independent of contract-size rounding.
    """

    is_long: bool
    entry_price: float
    initial_stop: float
    tp_prices: tuple[float, float, float]
    close_fractions: tuple[float, float, float]
    breakeven_stop: float
    stage: int = int(LadderStage.INITIAL)
    remaining_fraction: float = 1.0
    filled: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        is_long: bool,
        entry_price: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        config: TakeProfitSettings,
    ) -> "TakeProfitLadder":
        """Derive the ladder from the geometry the Exit model produced.

        The levels are fractions of the model's own take-profit *distance*, so a
        volatile symbol gets a proportionally wider ladder rather than a fixed
        one bolted on top of a volatility-scaled target.  TP3 is the model's
        target exactly, which keeps the reward/risk the Decision Engine checked
        in R5 the reward/risk the position actually carries.
        """
        direction: int = 1 if is_long else -1
        tp_distance: float = abs(entry_price) * max(take_profit_pct, 0.0)
        stop_distance: float = abs(entry_price) * max(stop_loss_pct, 0.0)

        prices: tuple[float, float, float] = (
            entry_price + direction * tp_distance * config.level_fractions[0],
            entry_price + direction * tp_distance * config.level_fractions[1],
            entry_price + direction * tp_distance * config.level_fractions[2],
        )
        breakeven: float = entry_price * (
            1.0 + direction * config.breakeven_offset_pct
        )
        return cls(
            is_long=is_long,
            entry_price=entry_price,
            initial_stop=entry_price - direction * stop_distance,
            tp_prices=prices,
            close_fractions=cls._merged_fractions(config),
            breakeven_stop=breakeven,
        )

    @staticmethod
    def _merged_fractions(config: TakeProfitSettings) -> tuple[float, float, float]:
        """Fold dust-sized legs forward so no leg is too small to submit.

        A 2 % leg on a 12 USDT position rounds to zero contracts on most pairs;
        the exchange rejects it and the ladder silently stalls.  Merging forward
        keeps the ladder's *total* allocation intact while guaranteeing every leg
        it actually submits is tradeable.
        """
        raw: list[float] = list(config.close_fractions)
        minimum: float = config.min_leg_fraction
        for index in range(len(raw) - 1):
            if raw[index] < minimum:
                raw[index + 1] += raw[index]
                raw[index] = 0.0
        return (raw[0], raw[1], raw[2])

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def is_closed(self) -> bool:
        """``True`` once the final leg has been taken."""
        return self.stage >= int(LadderStage.CLOSED) or self.remaining_fraction <= _EPSILON

    @property
    def protection(self) -> StopProtection:
        """Which stop is currently in force."""
        if self.stage >= int(LadderStage.TP2_FILLED):
            return StopProtection.TP1_LOCKED
        if self.stage >= int(LadderStage.TP1_FILLED):
            return StopProtection.BREAKEVEN
        return StopProtection.INITIAL

    def current_stop(self) -> float:
        """The stop price the current stage implies."""
        if self.stage >= int(LadderStage.TP2_FILLED):
            return self.tp_prices[0]
        if self.stage >= int(LadderStage.TP1_FILLED):
            return self.breakeven_stop
        return self.initial_stop

    def next_level(self) -> int | None:
        """Index of the next unfilled take-profit level, or ``None``."""
        return None if self.stage >= 3 else int(self.stage)

    def reached(self, level_index: int, price: float) -> bool:
        """``True`` when ``price`` has traded through take-profit ``level_index``."""
        target: float = self.tp_prices[level_index]
        return price >= target if self.is_long else price <= target

    def stop_breached(self, price: float) -> bool:
        """``True`` when ``price`` has traded through the in-force stop."""
        stop: float = self.current_stop()
        return price <= stop if self.is_long else price >= stop

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def on_tick(self, price: float) -> list[LadderEvent]:
        """Advance the ladder on a single observed price (paper / live).

        A tick carries no ambiguity about ordering - it *is* the order - so the
        only pessimism applied here is the same stop-before-target precedence the
        rest of the system uses when a single observation satisfies both.
        """
        if self.is_closed:
            return []

        if self.stop_breached(price):
            return [self._stop_event()]

        events: list[LadderEvent] = []
        while not self.is_closed:
            level: int | None = self.next_level()
            if level is None or not self.reached(level, price):
                break
            events.append(self._fill_event(level))
        if events and not self.is_closed and self.stop_breached(price):
            # The tightened stop is already breached by this same price.
            events.append(self._stop_event())
        return events

    def on_bar(self, high: float, low: float) -> list[LadderEvent]:
        """Advance the ladder across one OHLC bar, pessimistically.

        See the module docstring for the full rule.  In short: adverse extreme
        first, then ordered take-profit fills, then the tightened stop re-tested
        against the same bar.
        """
        if self.is_closed:
            return []

        adverse: float = low if self.is_long else high
        favourable: float = high if self.is_long else low

        # Rule 1 - the bar could have stopped us out before anything else, so it did.
        if self.stop_breached(adverse):
            return [self._stop_event()]

        # Rule 2 - fill the levels this bar reached, strictly in order.
        events: list[LadderEvent] = []
        while not self.is_closed:
            level: int | None = self.next_level()
            if level is None or not self.reached(level, favourable):
                break
            events.append(self._fill_event(level))

        # Rule 3 - re-test the newly tightened stop against this same bar.
        if events and not self.is_closed and self.stop_breached(adverse):
            events.append(self._stop_event())
        return events

    # ------------------------------------------------------------------
    # Event construction
    # ------------------------------------------------------------------
    def _fill_event(self, level_index: int) -> LadderEvent:
        """Book a take-profit leg and move the stage (and the stop) up."""
        fraction: float = min(self.remaining_fraction, self.close_fractions[level_index])
        if level_index == 2:
            # The last leg always closes whatever is left, so rounding across the
            # earlier legs can never strand a residual position.
            fraction = self.remaining_fraction

        price: float = self.tp_prices[level_index]
        self.remaining_fraction = max(0.0, self.remaining_fraction - fraction)
        # Filling TP3 sets the stage to CLOSED on its own; nothing else may.
        self.stage = level_index + 1

        self.filled.append({"level": level_index + 1, "price": price, "fraction": fraction})
        return LadderEvent(
            kind=f"TAKE_PROFIT_{level_index + 1}",
            price=price,
            fraction=fraction,
            stage=self.stage,
        )

    def _stop_event(self) -> LadderEvent:
        """Close the remainder at the in-force stop.

        The **stage is deliberately left where it is**.  It records how far up
        the ladder the trade actually climbed, and that is what the "stopped at
        breakeven" / "stopped with TP1 locked" statistics are read from; forcing
        it to CLOSED here would make every stopped trade look as though it had
        reached TP2.  ``remaining_fraction`` reaching zero is what closes the
        ladder.
        """
        protection: StopProtection = self.protection
        fraction: float = self.remaining_fraction
        price: float = self.current_stop()
        self.remaining_fraction = 0.0
        return LadderEvent(
            kind="STOP",
            price=price,
            fraction=fraction,
            stage=int(LadderStage.CLOSED),
            protection=protection,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialisable snapshot recorded on the trade row."""
        return {
            "stage": int(self.stage),
            "remaining_fraction": float(self.remaining_fraction),
            "tp1": float(self.tp_prices[0]),
            "tp2": float(self.tp_prices[1]),
            "tp3": float(self.tp_prices[2]),
            "initial_stop": float(self.initial_stop),
            "current_stop": float(self.current_stop()),
            "protection": self.protection.value,
            "filled": list(self.filled),
            # Derived from the legs that actually filled, never from the stage:
            # a stop-out also advances the stage to CLOSED, so reading the stage
            # back would report every stopped trade as having reached TP1 and TP2.
            **{
                f"reached_tp{level}": any(leg["level"] == level for leg in self.filled)
                for level in (1, 2, 3)
            },
        }
