"""The Risk Guard - a stateful supervisor that can shut the whole system down.

Every other module is allowed to be wrong.  This one exists on the assumption
that they *will* be: models drift, exchanges wedge, and a losing streak looks
exactly like a working strategy right up until the account is gone.

State machine
-------------
======  ================================================================
GREEN   Normal operation, full sizing.
YELLOW  Warning band - sizing is throttled by ``yellow_size_multiplier``.
RED     Halt.  All orders cancelled, all positions flattened, execution
        disabled.  **Requires manual intervention to clear** when
        ``risk.require_manual_reset`` is set (the default).
======  ================================================================

RED triggers
------------
* Daily drawdown from the session's starting equity breaches
  ``daily_drawdown_red_pct`` (default -5 %).
* Peak-to-trough drawdown breaches ``total_drawdown_red_pct``.
* ``consecutive_losses_red`` losing trades in a row.
* ``api_errors_red`` exchange errors inside ``api_error_window_seconds``.
* An operator hits the kill switch on the panel.

The guard persists its state to SQLite, so a RED latch survives a process
restart: a crash-loop must not silently re-enable trading.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Final, Sequence

from config.settings import RiskSettings, Settings
from core.logger import get_logger
from core.utils import utc_now, utc_now_ms
from module_a_data.db_handler import DatabaseHandler

_LOGGER = get_logger(__name__)

_STATE_KEY: Final[str] = "risk_guard_state"

#: Async callback invoked when the guard trips RED: ``(reason) -> None``.
HaltCallback = Callable[[str], Awaitable[None]]


class SystemState(str, Enum):
    """Global trading permission level."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class RiskLadder:
    """The state ladder, as a pure synchronous function of observed history.

    :class:`RiskGuard` is the live object: async, SQLite-backed, and it rolls its
    trading day off the wall clock.  None of that suits a bar-by-bar replay,
    where "today" is a property of the bar being replayed and there is no event
    loop to await.  Reimplementing the thresholds inside the backtester would
    have meant two copies of the rule that decides when trading stops - and the
    copy the backtest used would inevitably drift from the one production runs.

    So the ladder lives here, owns the thresholds, and both callers drive it.
    """

    def __init__(self, settings: Settings) -> None:
        self._config: RiskSettings = settings.risk
        self.state: SystemState = SystemState.GREEN
        self.halt_reason: str = ""
        self.consecutive_losses: int = 0
        self.trades_today: int = 0
        self.peak_equity: float = float(settings.risk.starting_equity)
        self.day_start_equity: float = float(settings.risk.starting_equity)
        self._current_day: date | None = None

    def daily_drawdown_pct(self, equity: float) -> float:
        if self.day_start_equity <= 0.0:
            return 0.0
        return max(0.0, (self.day_start_equity - equity) / self.day_start_equity)

    def total_drawdown_pct(self, equity: float) -> float:
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    @property
    def size_multiplier(self) -> float:
        if self.state is SystemState.RED:
            return 0.0
        if self.state is SystemState.YELLOW:
            return self._config.yellow_size_multiplier
        return 1.0

    @property
    def can_trade(self) -> bool:
        return self.state is not SystemState.RED and self.trades_today < self._config.max_daily_trades

    def observe(
        self,
        *,
        equity: float,
        realised_pnls: Sequence[float],
        day: date,
    ) -> tuple[SystemState, str]:
        """Advance the ladder by one bar. Returns ``(state, reason_if_changed)``.

        RED latches: ``require_manual_reset`` means a live deployment stops and
        stays stopped until a human intervenes, so a replay that trips RED must
        stop too.  Reporting the equity curve it *would* have had afterwards is
        the difference between "the strategy returned 52%" and "the strategy
        returned 52% and would have been halted on day 40".
        """
        if self._current_day is None:
            self._current_day = day
            self.day_start_equity = equity
        elif day != self._current_day:
            self._current_day = day
            self.day_start_equity = equity
            self.trades_today = 0

        self.peak_equity = max(self.peak_equity, equity)
        previous: SystemState = self.state

        for pnl in realised_pnls:
            self.trades_today += 1
            if pnl < 0.0:
                self.consecutive_losses += 1
            elif pnl > 0.0:
                self.consecutive_losses = 0

        if self.state is SystemState.RED and self._config.require_manual_reset:
            return self.state, ""

        daily: float = self.daily_drawdown_pct(equity)
        total: float = self.total_drawdown_pct(equity)
        reason: str = ""
        if daily >= self._config.daily_drawdown_red_pct:
            self.state = SystemState.RED
            reason = f"daily drawdown {daily:.2%} breached the {self._config.daily_drawdown_red_pct:.2%} limit"
        elif total >= self._config.total_drawdown_red_pct:
            self.state = SystemState.RED
            reason = (
                f"peak-to-trough drawdown {total:.2%} breached the "
                f"{self._config.total_drawdown_red_pct:.2%} limit"
            )
        elif self.consecutive_losses >= self._config.consecutive_losses_red:
            self.state = SystemState.RED
            reason = (
                f"{self.consecutive_losses} consecutive losing trades "
                f"(limit {self._config.consecutive_losses_red})"
            )
        elif (
            daily >= self._config.daily_drawdown_yellow_pct
            or self.consecutive_losses >= self._config.consecutive_losses_yellow
        ):
            self.state = SystemState.YELLOW
            reason = f"daily drawdown {daily:.2%} / streak {self.consecutive_losses}"
        else:
            self.state = SystemState.GREEN
            reason = "warning conditions cleared" if previous is SystemState.YELLOW else ""

        if self.state is SystemState.RED and previous is not SystemState.RED:
            self.halt_reason = reason
        return self.state, (reason if self.state is not previous else "")


class RiskGuard:
    """Monitors equity, trade outcomes and API health; halts the system on breach."""

    def __init__(self, settings: Settings, database: DatabaseHandler | None = None) -> None:
        self._settings: Settings = settings
        self._config: RiskSettings = settings.risk
        self._db: DatabaseHandler | None = database
        self._lock: asyncio.Lock = asyncio.Lock()
        self._halt_callbacks: list[HaltCallback] = []

        starting: float = settings.risk.starting_equity
        self.state: SystemState = SystemState.GREEN
        self.halt_reason: str = ""
        self.halted_at: datetime | None = None

        self.session_start_equity: float = starting
        self.day_start_equity: float = starting
        self.peak_equity: float = starting
        self.current_equity: float = starting

        self.consecutive_losses: int = 0
        self.trades_today: int = 0
        self.wins_today: int = 0
        self.losses_today: int = 0
        self.realized_pnl_today: float = 0.0

        self._current_day: date = utc_now().date()
        self._api_errors: Deque[float] = deque(maxlen=256)
        self.total_api_errors: int = 0

    # ------------------------------------------------------------------
    # Lifecycle / persistence
    # ------------------------------------------------------------------
    def register_halt_callback(self, callback: HaltCallback) -> None:
        """Register a coroutine to run when the guard trips RED.

        The execution engine registers its "cancel everything and flatten"
        routine here.  Keeping it a callback (rather than a direct dependency)
        means the guard has no knowledge of Module E's internals.
        """
        self._halt_callbacks.append(callback)

    async def load(self) -> None:
        """Restore a persisted RED latch (and the day's counters) after a restart."""
        if self._db is None:
            return
        stored: dict[str, Any] | None = await self._db.get_state(_STATE_KEY)
        if not stored:
            return

        try:
            self.state = SystemState(str(stored.get("state", SystemState.GREEN.value)))
        except ValueError:
            self.state = SystemState.GREEN

        self.halt_reason = str(stored.get("halt_reason", ""))
        self.day_start_equity = float(stored.get("day_start_equity", self.day_start_equity))
        self.peak_equity = float(stored.get("peak_equity", self.peak_equity))
        self.consecutive_losses = int(stored.get("consecutive_losses", 0))
        self.realized_pnl_today = float(stored.get("realized_pnl_today", 0.0))
        self.trades_today = int(stored.get("trades_today", 0))

        stored_day: str = str(stored.get("day", ""))
        if stored_day and stored_day != self._current_day.isoformat():
            # A new UTC day started while we were down: roll the counters over.
            self._roll_day(self.current_equity)

        if self.state is SystemState.RED:
            _LOGGER.critical(
                "Risk Guard restored in RED state (%s) - manual reset required", self.halt_reason
            )

    async def persist(self) -> None:
        """Write the guard's state to SQLite so a RED latch survives a restart."""
        if self._db is None:
            return
        await self._db.set_state(
            _STATE_KEY,
            {
                "state": self.state.value,
                "halt_reason": self.halt_reason,
                "halted_at": self.halted_at.isoformat() if self.halted_at else "",
                "day": self._current_day.isoformat(),
                "day_start_equity": self.day_start_equity,
                "peak_equity": self.peak_equity,
                "current_equity": self.current_equity,
                "consecutive_losses": self.consecutive_losses,
                "trades_today": self.trades_today,
                "realized_pnl_today": self.realized_pnl_today,
                "updated_ms": utc_now_ms(),
            },
        )

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------
    @property
    def is_halted(self) -> bool:
        """``True`` when no new risk may be taken."""
        return self.state is SystemState.RED

    @property
    def can_trade(self) -> bool:
        """``True`` when the engine may open new positions."""
        return self.state is not SystemState.RED and self.trades_today < self._config.max_daily_trades

    @property
    def size_multiplier(self) -> float:
        """Sizing throttle applied by the Decision Engine."""
        if self.state is SystemState.RED:
            return 0.0
        if self.state is SystemState.YELLOW:
            return self._config.yellow_size_multiplier
        return 1.0

    @property
    def daily_drawdown_pct(self) -> float:
        """Drawdown from the day's opening equity (positive number = loss)."""
        if self.day_start_equity <= 0.0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.current_equity) / self.day_start_equity)

    @property
    def total_drawdown_pct(self) -> float:
        """Peak-to-trough drawdown (positive number = loss)."""
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)

    def snapshot(self) -> dict[str, Any]:
        """Everything the dashboard needs to render the risk panel."""
        return {
            "state": self.state.value,
            "halt_reason": self.halt_reason,
            "halted_at": self.halted_at.isoformat(timespec="seconds") if self.halted_at else "",
            "can_trade": self.can_trade,
            "size_multiplier": self.size_multiplier,
            "equity": self.current_equity,
            "day_start_equity": self.day_start_equity,
            "peak_equity": self.peak_equity,
            "daily_drawdown_pct": self.daily_drawdown_pct,
            "total_drawdown_pct": self.total_drawdown_pct,
            "daily_drawdown_limit": self._config.daily_drawdown_red_pct,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_losses_limit": self._config.consecutive_losses_red,
            "trades_today": self.trades_today,
            "max_daily_trades": self._config.max_daily_trades,
            "wins_today": self.wins_today,
            "losses_today": self.losses_today,
            "realized_pnl_today": self.realized_pnl_today,
            "api_errors_recent": len(self._recent_api_errors()),
            "api_errors_total": self.total_api_errors,
        }

    # ------------------------------------------------------------------
    # Event intake
    # ------------------------------------------------------------------
    async def update_equity(self, equity: float) -> SystemState:
        """Feed the current account equity and re-evaluate the state ladder.

        Called once per cycle by the orchestrator.  Returns the state *after*
        evaluation so the caller can react immediately.
        """
        async with self._lock:
            today: date = utc_now().date()
            if today != self._current_day:
                self._roll_day(equity)

            self.current_equity = equity
            self.peak_equity = max(self.peak_equity, equity)

            if self.state is SystemState.RED:
                return self.state

            breach: str | None = self._evaluate_red_conditions()
            if breach is not None:
                await self._trip(breach)
                return self.state

            self._evaluate_yellow_conditions()

        await self.persist()
        return self.state

    async def record_trade_result(self, realized_pnl: float, symbol: str = "") -> SystemState:
        """Register a closed trade and update the losing-streak counter."""
        async with self._lock:
            self.trades_today += 1
            self.realized_pnl_today += realized_pnl

            if realized_pnl < 0.0:
                self.consecutive_losses += 1
                self.losses_today += 1
            elif realized_pnl > 0.0:
                self.consecutive_losses = 0
                self.wins_today += 1

            _LOGGER.info(
                "Trade closed %s pnl=%.4f USDT | streak=%d | day pnl=%.4f",
                symbol,
                realized_pnl,
                self.consecutive_losses,
                self.realized_pnl_today,
            )

            if self.state is not SystemState.RED:
                if self.consecutive_losses >= self._config.consecutive_losses_red:
                    await self._trip(
                        f"{self.consecutive_losses} consecutive losing trades "
                        f"(limit {self._config.consecutive_losses_red})"
                    )
                elif self.consecutive_losses >= self._config.consecutive_losses_yellow:
                    self._set_state(
                        SystemState.YELLOW,
                        f"{self.consecutive_losses} consecutive losses - sizing throttled",
                    )

        await self.persist()
        return self.state

    async def record_api_error(self, detail: str = "") -> SystemState:
        """Register an exchange/network failure and check the error budget.

        Errors are counted inside a sliding window so an outage that recovers
        does not leave a permanent scar, while a genuinely degraded API - the
        situation where blind order submission is most dangerous - trips RED.
        """
        async with self._lock:
            now: float = utc_now_ms() / 1_000.0
            self._api_errors.append(now)
            self.total_api_errors += 1
            recent: int = len(self._recent_api_errors())

            _LOGGER.warning("API error recorded (%d in window): %s", recent, detail)

            if self.state is SystemState.RED:
                return self.state

            if recent >= self._config.api_errors_red:
                await self._trip(
                    f"{recent} exchange errors within "
                    f"{self._config.api_error_window_seconds:.0f}s - API considered unreliable"
                )
            elif recent >= self._config.api_errors_yellow:
                self._set_state(SystemState.YELLOW, f"{recent} recent exchange errors")

        await self.persist()
        return self.state

    def record_api_success(self) -> None:
        """Clear the sliding error window after a clean round-trip."""
        self._api_errors.clear()

    async def trigger_kill_switch(self, reason: str = "manual kill switch") -> SystemState:
        """Operator panic button - trip RED immediately."""
        async with self._lock:
            await self._trip(reason)
        await self.persist()
        return self.state

    async def reset(self, operator: str = "operator") -> SystemState:
        """Manually clear a RED latch.

        Deliberately *not* automatic: a drawdown halt means something the system
        does not understand happened, and a human should look at it before
        capital is at risk again.
        """
        async with self._lock:
            previous: str = self.halt_reason
            self.state = SystemState.GREEN
            self.halt_reason = ""
            self.halted_at = None
            self.consecutive_losses = 0
            self._api_errors.clear()
            self.day_start_equity = self.current_equity
            self.peak_equity = max(self.peak_equity, self.current_equity)
            _LOGGER.warning("Risk Guard reset by %s (was: %s)", operator, previous or "n/a")

        await self.persist()
        return self.state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _recent_api_errors(self) -> list[float]:
        """API error timestamps inside the sliding window."""
        cutoff: float = utc_now_ms() / 1_000.0 - self._config.api_error_window_seconds
        return [stamp for stamp in self._api_errors if stamp >= cutoff]

    def _evaluate_red_conditions(self) -> str | None:
        """Return the breach description when a RED condition is met."""
        daily: float = self.daily_drawdown_pct
        if daily >= self._config.daily_drawdown_red_pct:
            return (
                f"daily drawdown {daily:.2%} breached the "
                f"{self._config.daily_drawdown_red_pct:.2%} limit"
            )

        total: float = self.total_drawdown_pct
        if total >= self._config.total_drawdown_red_pct:
            return (
                f"peak-to-trough drawdown {total:.2%} breached the "
                f"{self._config.total_drawdown_red_pct:.2%} limit"
            )
        return None

    def _evaluate_yellow_conditions(self) -> None:
        """Promote to (or demote from) YELLOW based on the warning thresholds."""
        daily: float = self.daily_drawdown_pct
        warning: bool = (
            daily >= self._config.daily_drawdown_yellow_pct
            or self.consecutive_losses >= self._config.consecutive_losses_yellow
            or len(self._recent_api_errors()) >= self._config.api_errors_yellow
        )

        if warning and self.state is SystemState.GREEN:
            self._set_state(
                SystemState.YELLOW,
                f"daily drawdown {daily:.2%} / streak {self.consecutive_losses}",
            )
        elif not warning and self.state is SystemState.YELLOW:
            self._set_state(SystemState.GREEN, "warning conditions cleared")

    def _set_state(self, state: SystemState, reason: str) -> None:
        """Transition between GREEN and YELLOW (RED goes through :meth:`_trip`)."""
        if self.state is state:
            return
        _LOGGER.warning("Risk Guard %s -> %s (%s)", self.state.value, state.value, reason)
        self.state = state

    async def _trip(self, reason: str) -> None:
        """Latch RED and fan out to every registered halt callback.

        Callback failures are logged but never propagated: a broken flatten
        routine must not prevent the halt itself from taking effect.
        """
        if self.state is SystemState.RED:
            return

        self.state = SystemState.RED
        self.halt_reason = reason
        self.halted_at = datetime.now(tz=timezone.utc)
        _LOGGER.critical("=" * 78)
        _LOGGER.critical("KILL SWITCH ENGAGED - TRADING HALTED: %s", reason)
        _LOGGER.critical("Manual reset required via /api/reset_risk_guard or the panel")
        _LOGGER.critical("=" * 78)

        for callback in self._halt_callbacks:
            try:
                await callback(reason)
            except Exception as error:  # pragma: no cover - callback isolation
                _LOGGER.error("Halt callback failed: %s", error, exc_info=True)

    def _roll_day(self, equity: float) -> None:
        """Reset the daily counters at the UTC date boundary."""
        _LOGGER.info(
            "UTC day rollover: %d trades, %.4f realised PnL, resetting daily counters",
            self.trades_today,
            self.realized_pnl_today,
        )
        self._current_day = utc_now().date()
        self.day_start_equity = equity if equity > 0.0 else self.current_equity
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.realized_pnl_today = 0.0
