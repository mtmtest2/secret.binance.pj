"""Paper-trading engine - the live executor with the network calls intercepted.

It deliberately mirrors :class:`~module_e_execution.executor.LiveExecutor`'s
public surface exactly (``start``, ``execute``, ``close_position``,
``emergency_flatten``, ``fetch_account_state``, ``positions``), so the
orchestrator can swap engines without a single conditional in the trading loop.

What it does *not* do is pretend that simulated trading is free:

* **Fees.**  Entry and exit are charged at the Binance USDT-M taker rate
  (0.05 %) for market orders and the maker rate (0.02 %) for resting limit
  orders - the two round trips are what turn a marginal edge negative.
* **Slippage.**  Market fills are penalised by ``slippage_bps`` in the
  unfavourable direction, on both entry and exit.
* **Funding.**  Open positions accrue funding every 8 hours using the funding
  rate actually recorded by Module A for that symbol.
* **Liquidation.**  The isolated-margin liquidation price is checked on every
  tick, before the take-profit is.
* **Conservative ordering.**  When a tick could resolve more than one barrier,
  the loss is always booked first.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Final, Sequence

from config.settings import ExecutionSettings, Settings
from core.exceptions import KillSwitchEngaged
from core.logger import get_logger
from core.utils import safe_float, utc_now_ms
from module_a_data.db_handler import DatabaseHandler
from module_a_data.fetcher import BinanceDataFetcher
from module_c_ml.schemas import TradeAction, TradeSignal
from module_e_execution.models import (
    AccountState,
    CloseReason,
    ExecutionReport,
    Position,
    PositionStatus,
)
from module_e_execution.risk_guard import RiskGuard
from module_e_execution.tp_ladder import LadderEvent

_LOGGER = get_logger(__name__)

_MS_PER_HOUR: Final[int] = 3_600_000


class PaperTrader:
    """Full-fidelity simulation of the live executor against real market prices."""

    mode: str = "paper"

    def __init__(
        self,
        settings: Settings,
        fetcher: BinanceDataFetcher,
        database: DatabaseHandler,
        risk_guard: RiskGuard,
    ) -> None:
        self._settings: Settings = settings
        self._config: ExecutionSettings = settings.execution
        self._fetcher: BinanceDataFetcher = fetcher
        self._db: DatabaseHandler = database
        self._risk: RiskGuard = risk_guard

        self._balance: float = settings.execution.paper_starting_balance
        self._positions: dict[str, Position] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._last_prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Restore the virtual balance and start the barrier-monitor task."""
        if self._running:
            return
        await self._restore_state()
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(), name="paper-monitor")
        _LOGGER.info("Paper trader started with a virtual balance of %.2f USDT", self._balance)

    async def stop(self) -> None:
        """Stop the monitor task and persist the virtual balance."""
        self._running = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        await self._persist_state()
        _LOGGER.info("Paper trader stopped (balance %.2f USDT)", self._balance)

    @property
    def positions(self) -> dict[str, Position]:
        """Currently open virtual positions keyed by symbol."""
        return dict(self._positions)

    @property
    def open_symbols(self) -> frozenset[str]:
        """Symbols with an open virtual position."""
        return frozenset(self._positions)

    @property
    def balance(self) -> float:
        """Realised virtual balance, excluding open-position PnL."""
        return self._balance

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    async def fetch_account_state(self) -> AccountState:
        """Mark the virtual book to market using live ticker prices."""
        prices: dict[str, float] = await self._prices(list(self._positions))
        unrealized: float = 0.0
        used_margin: float = 0.0
        for symbol, position in self._positions.items():
            price: float = prices.get(symbol, position.entry_price)
            unrealized += position.unrealized_pnl(price) - position.funding_paid
            used_margin += position.margin

        return AccountState(
            mode=self.mode,
            balance=self._balance,
            equity=self._balance + unrealized,
            unrealized_pnl=unrealized,
            used_margin=used_margin,
            open_positions=len(self._positions),
            timestamp_ms=utc_now_ms(),
        )

    async def _prices(self, symbols: Sequence[str]) -> dict[str, float]:
        """Fetch live prices, falling back to the last known values."""
        if not symbols:
            return {}
        try:
            fresh: dict[str, float] = await self._fetcher.fetch_tickers(list(symbols))
            self._last_prices.update(fresh)
            return {symbol: self._last_prices.get(symbol, 0.0) for symbol in symbols}
        except Exception as error:
            _LOGGER.warning("Paper price fetch failed, using cached prices: %s", error)
            await self._risk.record_api_error(f"paper ticker fetch: {error}")
            return {symbol: self._last_prices.get(symbol, 0.0) for symbol in symbols}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def execute(self, signal: TradeSignal) -> ExecutionReport:
        """Open a virtual position, charging slippage and the taker fee."""
        if self._risk.is_halted:
            raise KillSwitchEngaged("Risk Guard is RED", reason=self._risk.halt_reason)
        if not self._risk.can_trade:
            return ExecutionReport(
                decision_id=signal.decision_id,
                symbol=signal.symbol,
                accepted=False,
                error="daily trade cap reached",
            )

        async with self._lock:
            if signal.symbol in self._positions:
                return ExecutionReport(
                    decision_id=signal.decision_id,
                    symbol=signal.symbol,
                    accepted=False,
                    error="a position is already open on this symbol",
                )

            account: AccountState = await self.fetch_account_state()
            equity: float = account.equity
            margin: float = signal.margin_for(equity)
            if margin <= 0.0 or margin > account.free_margin:
                return ExecutionReport(
                    decision_id=signal.decision_id,
                    symbol=signal.symbol,
                    accepted=False,
                    error=(
                        f"insufficient free margin ({account.free_margin:.2f} USDT available, "
                        f"{margin:.2f} required)"
                    ),
                )

            market_price: float = await self._entry_price(signal)
            fill_price: float = self._apply_slippage(market_price, signal.action, is_entry=True)
            quantity: float = (margin * signal.leverage) / fill_price
            if quantity <= 0.0:
                return ExecutionReport(
                    decision_id=signal.decision_id,
                    symbol=signal.symbol,
                    accepted=False,
                    error="computed quantity is zero",
                )

            position: Position = Position.from_signal(
                signal,
                equity=equity,
                entry_price=fill_price,
                quantity=quantity,
                maintenance_margin_rate=self._config.maintenance_margin_rate,
                mode=self.mode,
                take_profit_config=self._settings.take_profit,
            )
            position.fees_paid = fill_price * quantity * self._entry_fee_rate()
            position.last_funding_ms = utc_now_ms()

            # The entry fee is realised immediately, exactly as on the exchange.
            self._balance -= position.fees_paid
            self._positions[signal.symbol] = position
            self._last_prices[signal.symbol] = market_price

            await self._persist_open(position)
            _LOGGER.info(
                "[PAPER] OPEN %s %s qty=%.6f @ %.6f (slippage from %.6f) | %dx | "
                "TP %.6f SL %.6f liq %.6f | fee %.4f",
                position.action.value,
                position.symbol,
                position.quantity,
                position.entry_price,
                market_price,
                position.leverage,
                position.take_profit,
                position.stop_loss,
                position.liquidation_price,
                position.fees_paid,
            )
            return ExecutionReport(
                decision_id=signal.decision_id,
                symbol=signal.symbol,
                accepted=True,
                position=position,
                detail={"simulated": True, "slippage_from": market_price},
            )

    async def _entry_price(self, signal: TradeSignal) -> float:
        """Live price to fill against, falling back to the signal's reference."""
        try:
            return await self._fetcher.fetch_last_price(signal.symbol)
        except Exception as error:
            _LOGGER.warning(
                "Paper entry price fetch failed for %s (%s) - using the signal reference",
                signal.symbol,
                error,
            )
            return signal.reference_price

    def _apply_slippage(self, price: float, action: TradeAction, is_entry: bool) -> float:
        """Push the fill price against us by ``slippage_bps``.

        Entering long and exiting short both buy, so both are penalised upward;
        the reverse pair is penalised downward.
        """
        penalty: float = self._config.slippage_bps / 10_000.0
        buying: bool = (action is TradeAction.LONG) == is_entry
        return price * (1.0 + penalty) if buying else price * (1.0 - penalty)

    def _entry_fee_rate(self) -> float:
        """Maker rate for resting limit orders, taker rate for market orders."""
        return (
            self._config.taker_fee
            if self._config.order_type == "market"
            else self._config.maker_fee
        )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    async def _monitor_loop(self) -> None:
        """Poll live prices and resolve virtual barriers."""
        interval: float = self._config.position_monitor_interval_seconds
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._positions:
                    continue
                await self.mark_to_market()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # pragma: no cover - the monitor must never die
                _LOGGER.error("Paper monitor iteration failed: %s", error, exc_info=True)

    async def mark_to_market(self) -> None:
        """Apply funding, advance trailing stops and resolve barriers.

        Barrier precedence is deliberately pessimistic: liquidation, then stop,
        then take-profit.  A tick that could plausibly have hit more than one
        always books the worse outcome.
        """
        async with self._lock:
            prices: dict[str, float] = await self._prices(list(self._positions))
            now_ms: int = utc_now_ms()

            for symbol, position in list(self._positions.items()):
                price: float = prices.get(symbol, 0.0)
                if price <= 0.0:
                    continue

                position.update_excursions(price, price)
                await self._accrue_funding(position, price, now_ms)

                if self._hit_liquidation(position, price):
                    await self._close(position, position.liquidation_price, CloseReason.LIQUIDATION)
                    continue

                if position.ladder is not None:
                    # Same state machine the backtester drives, fed ticks instead
                    # of bars.  Nothing about the ladder's behaviour is
                    # re-implemented here.
                    await self._apply_ladder(position, price, now_ms)
                    continue

                position.advance_trailing(price)
                stop: float = position.effective_stop()
                if self._hit_stop(position, price, stop):
                    reason: CloseReason = position.stop_close_reason()
                    await self._close(position, stop, reason)
                    continue

                if self._hit_take_profit(position, price):
                    await self._close(position, position.take_profit, CloseReason.TAKE_PROFIT)

    async def _apply_ladder(self, position: Position, price: float, now_ms: int) -> None:
        """Book whatever the ladder decided at this observed price."""
        events: list[LadderEvent] = position.ladder.on_tick(price) if position.ladder else []
        if not events:
            return

        last_reason: CloseReason = CloseReason.TAKE_PROFIT
        for event in events:
            fill_price: float = self._apply_slippage(
                event.price, position.action, is_entry=False
            )
            delta, last_reason = position.apply_ladder_event(
                event,
                exit_price=fill_price,
                fee_rate=self._config.taker_fee,
                timestamp_ms=now_ms,
            )
            self._balance += delta
            _LOGGER.info(
                "[PAPER] %s %s %.6f @ %.6f (%.0f%% of the position) | pnl=%.4f",
                last_reason.value,
                position.symbol,
                position.quantity_for_fraction(event.fraction),
                fill_price,
                event.fraction * 100.0,
                delta,
            )

        if position.is_flat:
            await self._finalise_ladder_close(position, last_reason)

    async def _finalise_ladder_close(self, position: Position, reason: CloseReason) -> None:
        """Settle funding and retire a position whose ladder ran to completion."""
        self._balance -= position.funding_paid
        position.realized_pnl -= position.funding_paid
        position.exit_price = (
            float(position.partial_fills[-1]["price"]) if position.partial_fills else position.entry_price
        )
        position.closed_at = datetime.now(tz=timezone.utc)
        position.close_reason = reason.value
        position.status = PositionStatus.CLOSED

        self._positions.pop(position.symbol, None)
        await self._persist_close(position)
        await self._risk.record_trade_result(position.realized_pnl, position.symbol)
        _LOGGER.info(
            "[PAPER] CLOSE %s %s | %s | pnl=%.4f USDT (%.2fR) | balance=%.2f",
            position.action.value,
            position.symbol,
            reason.value,
            position.realized_pnl,
            position.realized_r,
            self._balance,
        )

    @staticmethod
    def _hit_liquidation(position: Position, price: float) -> bool:
        """``True`` when the mark price has reached the liquidation level."""
        if position.liquidation_price <= 0.0:
            return False
        return (
            price <= position.liquidation_price
            if position.is_long
            else price >= position.liquidation_price
        )

    @staticmethod
    def _hit_stop(position: Position, price: float, stop: float) -> bool:
        """``True`` when the in-force stop has been breached."""
        return price <= stop if position.is_long else price >= stop

    @staticmethod
    def _hit_take_profit(position: Position, price: float) -> bool:
        """``True`` when the take-profit level has been reached."""
        return (
            price >= position.take_profit
            if position.is_long
            else price <= position.take_profit
        )

    async def _accrue_funding(self, position: Position, price: float, now_ms: int) -> None:
        """Charge (or credit) funding once per funding interval.

        Longs pay shorts when the funding rate is positive, which is the normal
        state of a bull market and a real, compounding cost of carry.
        """
        interval_ms: int = self._config.funding_interval_hours * _MS_PER_HOUR
        if position.last_funding_ms <= 0:
            position.last_funding_ms = now_ms
            return
        if now_ms - position.last_funding_ms < interval_ms:
            return

        rate: float = await self._latest_funding_rate(position.symbol)
        notional: float = position.quantity * price
        payment: float = notional * rate * position.direction
        position.funding_paid += payment
        position.last_funding_ms = now_ms
        _LOGGER.info(
            "[PAPER] funding on %s: rate=%.6f payment=%.4f USDT",
            position.symbol,
            rate,
            payment,
        )

    async def _latest_funding_rate(self, symbol: str) -> float:
        """Read the most recent funding rate Module A recorded for the symbol."""
        try:
            frame = await self._db.load_futures_metrics_frame(symbol, limit=1)
            if frame.empty:
                return 0.0
            return safe_float(frame["funding_rate"].iloc[-1], 0.0)
        except Exception as error:  # pragma: no cover - degraded-mode path
            _LOGGER.debug("Funding rate lookup failed for %s: %s", symbol, error)
            return 0.0

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------
    async def close_position(
        self,
        symbol: str,
        reason: CloseReason = CloseReason.MANUAL,
    ) -> ExecutionReport:
        """Close one virtual position at the current market price."""
        async with self._lock:
            position: Position | None = self._positions.get(symbol)
            if position is None:
                return ExecutionReport(
                    decision_id="", symbol=symbol, accepted=False, error="no open position"
                )
            prices: dict[str, float] = await self._prices([symbol])
            price: float = prices.get(symbol) or position.entry_price
            await self._close(position, price, reason)
            return ExecutionReport(
                decision_id=position.decision_id,
                symbol=symbol,
                accepted=True,
                position=position,
                detail={"close_reason": reason.value, "exit_price": position.exit_price},
            )

    async def _close(
        self,
        position: Position,
        raw_exit_price: float,
        reason: CloseReason,
    ) -> None:
        """Book a virtual closure, charging exit slippage and the taker fee.

        Liquidations are *not* given slippage relief: the position is wiped at
        the liquidation price and the entire margin is lost, which is what
        actually happens on the exchange.
        """
        if reason is CloseReason.LIQUIDATION:
            exit_price: float = raw_exit_price
        else:
            exit_price = self._apply_slippage(raw_exit_price, position.action, is_entry=False)

        remaining: float = position.quantity
        gross: float = (exit_price - position.entry_price) * remaining * position.direction
        exit_fee: float = exit_price * remaining * self._config.taker_fee
        position.fees_paid += exit_fee
        position.quantity = 0.0

        net: float = gross - exit_fee - position.funding_paid
        if reason is CloseReason.LIQUIDATION:
            # Cannot lose more than the isolated margin that was committed - and
            # profit already banked by earlier ladder legs is not clawed back.
            net = max(net, -position.margin - position.realized_pnl)

        # ``realized_pnl`` may already carry closed ladder legs, so this adds to
        # it rather than replacing it.
        position.realized_pnl += net
        position.exit_price = exit_price
        position.closed_at = datetime.now(tz=timezone.utc)
        position.close_reason = reason.value
        position.status = (
            PositionStatus.LIQUIDATED if reason is CloseReason.LIQUIDATION else PositionStatus.CLOSED
        )

        # The entry fee was already deducted when the position opened.
        self._balance += net
        if reason is CloseReason.LIQUIDATION:
            self._balance = max(self._balance, 0.0)

        self._positions.pop(position.symbol, None)
        await self._persist_close(position)
        await self._risk.record_trade_result(position.realized_pnl, position.symbol)

        _LOGGER.info(
            "[PAPER] CLOSE %s %s @ %.6f | %s | pnl=%.4f USDT | balance=%.2f",
            position.action.value,
            position.symbol,
            exit_price,
            reason.value,
            position.realized_pnl,
            self._balance,
        )

    async def cancel_all_orders(self) -> None:
        """No resting orders exist in simulation; kept for interface parity."""
        return None

    async def reconcile(self) -> None:
        """Interface parity with the live executor: re-price the virtual book."""
        await self.mark_to_market()

    async def emergency_flatten(self, reason: str) -> None:
        """Risk Guard halt callback: close every virtual position at market."""
        _LOGGER.critical("[PAPER] EMERGENCY FLATTEN: %s", reason)
        for symbol in list(self._positions):
            try:
                await self.close_position(symbol, CloseReason.KILL_SWITCH)
            except Exception as error:  # pragma: no cover - flatten must be total
                _LOGGER.error("Paper flatten failed for %s: %s", symbol, error)
        await self._persist_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    async def _persist_open(self, position: Position) -> None:
        """Insert the opened virtual trade."""
        try:
            await self._db.insert_trade(position.to_row())
        except Exception as error:  # pragma: no cover
            _LOGGER.error("Could not persist the opened paper trade: %s", error)

    async def _persist_close(self, position: Position) -> None:
        """Update the virtual trade row and append an equity-curve point."""
        row: dict[str, Any] = position.to_row()
        updates: dict[str, Any] = {
            key: row[key]
            for key in (
                "status",
                "exit_price",
                "trailing_stop",
                "fees_paid",
                "funding_paid",
                "realized_pnl",
                "max_adverse_excursion",
                "max_favorable_excursion",
                "closed_at",
                "close_reason",
            )
        }
        try:
            updated: bool = await self._db.update_trade(position.decision_id, updates)
            if not updated:
                await self._db.insert_trade(row)
            await self._persist_state()
        except Exception as error:  # pragma: no cover
            _LOGGER.error("Could not persist the closed paper trade: %s", error)

    async def record_equity_point(self) -> AccountState:
        """Append the current equity to the curve; called once per cycle."""
        account: AccountState = await self.fetch_account_state()
        peak: float = max(self._risk.peak_equity, account.equity)
        drawdown: float = 0.0 if peak <= 0.0 else max(0.0, (peak - account.equity) / peak)
        try:
            await self._db.insert_equity_point(
                {
                    "mode": self.mode,
                    "timestamp": account.timestamp_ms,
                    "balance": account.balance,
                    "equity": account.equity,
                    "unrealized_pnl": account.unrealized_pnl,
                    "open_positions": account.open_positions,
                    "drawdown_pct": drawdown,
                }
            )
        except Exception as error:  # pragma: no cover
            _LOGGER.error("Could not persist the equity point: %s", error)
        return account

    async def _persist_state(self) -> None:
        """Store the virtual balance so it survives a restart."""
        try:
            await self._db.set_state(
                "paper_trader_state",
                {"balance": self._balance, "updated_ms": utc_now_ms()},
            )
        except Exception as error:  # pragma: no cover
            _LOGGER.error("Could not persist the paper balance: %s", error)

    async def _restore_state(self) -> None:
        """Reload the virtual balance from a previous run, if any."""
        try:
            stored: dict[str, Any] | None = await self._db.get_state("paper_trader_state")
        except Exception as error:  # pragma: no cover
            _LOGGER.warning("Could not restore the paper balance: %s", error)
            return
        if stored and "balance" in stored:
            self._balance = safe_float(stored["balance"], self._balance)
            _LOGGER.info("Restored paper balance: %.2f USDT", self._balance)
