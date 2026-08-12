"""Live execution engine for Binance USDT-M perpetual futures.

Operating assumption: **the market and the API are hostile.**  Concretely that
means

* the pre-flight (margin mode + leverage) is always set explicitly before an
  order is sent, because a stale 20x leverage from a previous session would
  silently multiply the position size;
* protective orders are placed *immediately* after the entry fills, and if the
  protective leg cannot be placed the position is closed at market rather than
  left naked;
* every ccxt call is classified - ``RateLimitExceeded`` backs off,
  ``NetworkError`` retries, ``ExchangeError`` aborts - and every failure is
  reported to the Risk Guard, which trips RED once the API looks unreliable;
* the engine only ever acts on a validated :class:`TradeSignal`.  It has no
  opinion about the market and no access to the models.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Final, Sequence

import ccxt.async_support as ccxt

from config.settings import ExecutionSettings, Settings
from core.exceptions import ExecutionError, KillSwitchEngaged
from core.logger import get_logger
from core.utils import async_retry, safe_float, utc_now_ms
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

_TRANSIENT: Final[tuple[type[BaseException], ...]] = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
    ccxt.OnMaintenance,
    asyncio.TimeoutError,
)
_PERMANENT: Final[tuple[type[BaseException], ...]] = (
    ccxt.AuthenticationError,
    ccxt.PermissionDenied,
    ccxt.InsufficientFunds,
    ccxt.InvalidOrder,
    ccxt.BadSymbol,
)


class LiveExecutor:
    """Translates :class:`TradeSignal` objects into real Binance orders."""

    mode: str = "live"

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

        self._positions: dict[str, Position] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._cached_balance: float = settings.risk.starting_equity

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Load markets, reconcile with the exchange and start the monitor task."""
        if self._running:
            return
        await self._fetcher.load_markets()
        self._running = True
        await self.reconcile()
        self._monitor_task = asyncio.create_task(self._monitor_loop(), name="position-monitor")
        _LOGGER.info("Live executor started (%d position(s) reconciled)", len(self._positions))

    async def stop(self) -> None:
        """Stop the monitor task.  Open positions are *not* auto-closed."""
        self._running = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        _LOGGER.info("Live executor stopped")

    @property
    def exchange(self) -> ccxt.binance:
        """The shared ccxt client."""
        return self._fetcher.exchange

    @property
    def positions(self) -> dict[str, Position]:
        """Currently tracked open positions keyed by symbol."""
        return dict(self._positions)

    @property
    def open_symbols(self) -> frozenset[str]:
        """Symbols with an open position - consumed by the Decision Engine."""
        return frozenset(self._positions)

    # ------------------------------------------------------------------
    # ccxt call wrapper
    # ------------------------------------------------------------------
    async def _call(self, label: str, operation: Any, critical: bool = True) -> Any:
        """Execute a ccxt call with retry, classification and Risk Guard reporting.

        Args:
            label: Operation name for logs.
            operation: Zero-argument callable returning an awaitable.
            critical: When ``True``, failures are reported to the Risk Guard.

        Raises:
            ExecutionError: On any unrecoverable failure.
        """

        def _log(attempt: int, error: BaseException, delay: float) -> None:
            _LOGGER.warning(
                "%s failed (attempt %d): %s - retry in %.2fs", label, attempt + 1, error, delay
            )

        try:
            result: Any = await async_retry(
                operation,
                attempts=self._settings.exchange.max_retries,
                base_seconds=self._settings.exchange.backoff_base_seconds,
                max_seconds=self._settings.exchange.backoff_max_seconds,
                jitter=self._settings.exchange.backoff_jitter,
                retry_on=_TRANSIENT,
                give_up_on=_PERMANENT,
                on_error=_log,
            )
        except ccxt.RateLimitExceeded as error:
            if critical:
                await self._risk.record_api_error(f"{label}: rate limited")
            raise ExecutionError(f"{label}: rate limit exceeded", reason=str(error)) from error
        except _TRANSIENT as error:  # type: ignore[misc]
            if critical:
                await self._risk.record_api_error(f"{label}: {type(error).__name__}")
            raise ExecutionError(f"{label}: network failure", reason=str(error)) from error
        except _PERMANENT as error:  # type: ignore[misc]
            if critical:
                await self._risk.record_api_error(f"{label}: {type(error).__name__}")
            raise ExecutionError(f"{label}: rejected by exchange", reason=str(error)) from error
        except ccxt.ExchangeError as error:
            if critical:
                await self._risk.record_api_error(f"{label}: ExchangeError")
            raise ExecutionError(f"{label}: exchange error", reason=str(error)) from error

        self._risk.record_api_success()
        return result

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    async def fetch_account_state(self) -> AccountState:
        """Fetch balance and mark-to-market equity from the exchange."""
        try:
            balance: dict[str, Any] = await self._call(
                "fetch_balance", lambda: self.exchange.fetch_balance()
            )
        except ExecutionError as error:
            _LOGGER.error("Balance fetch failed, using cached equity: %s", error)
            return AccountState(
                mode=self.mode,
                balance=self._cached_balance,
                equity=self._cached_balance,
                unrealized_pnl=0.0,
                used_margin=sum(position.margin for position in self._positions.values()),
                open_positions=len(self._positions),
                timestamp_ms=utc_now_ms(),
            )

        usdt: dict[str, Any] = balance.get("USDT") or {}
        free: float = safe_float(usdt.get("free"), 0.0)
        total: float = safe_float(usdt.get("total"), free)
        self._cached_balance = total

        marks: dict[str, float] = await self._safe_prices(list(self._positions))
        unrealized: float = sum(
            position.unrealized_pnl(marks.get(symbol, position.entry_price))
            for symbol, position in self._positions.items()
        )
        used_margin: float = sum(position.margin for position in self._positions.values())

        return AccountState(
            mode=self.mode,
            balance=total,
            equity=total + unrealized,
            unrealized_pnl=unrealized,
            used_margin=used_margin,
            open_positions=len(self._positions),
            timestamp_ms=utc_now_ms(),
        )

    async def _safe_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        """Batch ticker fetch that degrades to an empty mapping on failure."""
        if not symbols:
            return {}
        try:
            return await self._fetcher.fetch_tickers(list(symbols))
        except Exception as error:  # pragma: no cover - degraded-mode path
            _LOGGER.warning("Ticker batch failed: %s", error)
            return {}

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    async def execute(self, signal: TradeSignal) -> ExecutionReport:
        """Open a position for ``signal``, protective orders included.

        Raises:
            KillSwitchEngaged: When the Risk Guard is RED.
        """
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

            try:
                return await self._open_position(signal)
            except KillSwitchEngaged:
                raise
            except ExecutionError as error:
                _LOGGER.error("Execution failed for %s: %s", signal.symbol, error)
                return ExecutionReport(
                    decision_id=signal.decision_id,
                    symbol=signal.symbol,
                    accepted=False,
                    error=str(error),
                )

    async def _open_position(self, signal: TradeSignal) -> ExecutionReport:
        """Pre-flight, entry order, protective orders, bookkeeping."""
        account: AccountState = await self.fetch_account_state()
        equity: float = account.equity

        quantity, reason = await self._resolve_quantity(signal, equity)
        if quantity <= 0.0:
            return ExecutionReport(
                decision_id=signal.decision_id,
                symbol=signal.symbol,
                accepted=False,
                error=reason,
            )

        await self._preflight(signal)

        order: dict[str, Any] = await self._place_entry_order(signal, quantity)
        fill_price: float = safe_float(order.get("average") or order.get("price"), 0.0)
        filled: float = safe_float(order.get("filled"), 0.0)
        if fill_price <= 0.0:
            fill_price = await self._fetcher.fetch_last_price(signal.symbol)
        if filled <= 0.0:
            filled = quantity

        position: Position = Position.from_signal(
            signal,
            equity=equity,
            entry_price=fill_price,
            quantity=filled,
            maintenance_margin_rate=self._config.maintenance_margin_rate,
            mode=self.mode,
            take_profit_config=self._settings.take_profit,
        )
        position.exchange_order_ids["entry"] = str(order.get("id", ""))
        position.fees_paid = self._extract_fee(order, fill_price, filled)
        position.last_funding_ms = utc_now_ms()

        try:
            await self._place_protective_orders(position)
        except ExecutionError as error:
            # A naked leveraged position is the single worst state to be in.
            _LOGGER.critical(
                "Protective orders failed for %s (%s) - flattening immediately",
                signal.symbol,
                error,
            )
            await self._market_close(position, CloseReason.MANUAL, note="protective-order failure")
            return ExecutionReport(
                decision_id=signal.decision_id,
                symbol=signal.symbol,
                accepted=False,
                error=f"protective orders failed, position flattened: {error}",
            )

        self._positions[signal.symbol] = position
        await self._persist_open(position)

        _LOGGER.info(
            "OPEN %s %s qty=%s @ %.6f | %dx | TP %.6f SL %.6f | liq %.6f",
            position.action.value,
            position.symbol,
            f"{position.quantity:g}",
            position.entry_price,
            position.leverage,
            position.take_profit,
            position.stop_loss,
            position.liquidation_price,
        )
        return ExecutionReport(
            decision_id=signal.decision_id,
            symbol=signal.symbol,
            accepted=True,
            position=position,
            detail={"order_id": position.exchange_order_ids.get("entry", "")},
        )

    async def _preflight(self, signal: TradeSignal) -> None:
        """Force ISOLATED margin and the signal's leverage before any order.

        Binance rejects a margin-mode change while a position is open on the
        symbol and returns "No need to change margin type" when it is already
        correct - both are benign and swallowed here.
        """
        symbol: str = signal.symbol
        try:
            await self._call(
                f"set_margin_mode[{symbol}]",
                lambda: self.exchange.set_margin_mode(self._config.margin_mode, symbol),
                critical=False,
            )
        except ExecutionError as error:
            message: str = str(error).lower()
            if "no need to change" not in message and "margin type cannot be changed" not in message:
                _LOGGER.warning("Could not set margin mode for %s: %s", symbol, error)

        await self._call(
            f"set_leverage[{symbol}]",
            lambda: self.exchange.set_leverage(signal.leverage, symbol),
        )

    async def _resolve_quantity(self, signal: TradeSignal, equity: float) -> tuple[float, str]:
        """Size the order and validate it against the market's own limits."""
        raw_quantity: float = signal.quantity_for(equity)
        if raw_quantity <= 0.0:
            return 0.0, "computed quantity is zero"

        markets: dict[str, Any] = self.exchange.markets or {}
        market: dict[str, Any] | None = markets.get(signal.symbol)
        if market is None:
            return 0.0, f"unknown market {signal.symbol}"

        try:
            quantity: float = float(self.exchange.amount_to_precision(signal.symbol, raw_quantity))
        except Exception as error:  # pragma: no cover - precision metadata gaps
            _LOGGER.warning("Precision rounding failed for %s: %s", signal.symbol, error)
            quantity = raw_quantity

        limits: dict[str, Any] = market.get("limits") or {}
        min_amount: float = safe_float((limits.get("amount") or {}).get("min"), 0.0)
        min_cost: float = safe_float((limits.get("cost") or {}).get("min"), 0.0)

        if min_amount > 0.0 and quantity < min_amount:
            return 0.0, f"quantity {quantity} below the exchange minimum {min_amount}"

        notional: float = quantity * signal.reference_price
        if min_cost > 0.0 and notional < min_cost:
            return 0.0, f"notional {notional:.2f} below the exchange minimum {min_cost:.2f}"

        if signal.margin_for(equity) > equity:
            return 0.0, "required margin exceeds available equity"

        return quantity, ""

    async def _place_entry_order(self, signal: TradeSignal, quantity: float) -> dict[str, Any]:
        """Send the entry order (market, or limit with a timeout fallback)."""
        symbol: str = signal.symbol
        side: str = signal.side

        if self._config.order_type == "market":
            return await self._call(
                f"create_market_order[{symbol}]",
                lambda: self.exchange.create_order(
                    symbol, "market", side, quantity, None, {"reduceOnly": False}
                ),
            )

        offset: float = self._config.limit_offset_bps / 10_000.0
        raw_price: float = (
            signal.reference_price * (1.0 - offset)
            if signal.action is TradeAction.LONG
            else signal.reference_price * (1.0 + offset)
        )
        limit_price: float = float(self.exchange.price_to_precision(symbol, raw_price))

        order: dict[str, Any] = await self._call(
            f"create_limit_order[{symbol}]",
            lambda: self.exchange.create_order(
                symbol, "limit", side, quantity, limit_price, {"reduceOnly": False}
            ),
        )
        return await self._await_limit_fill(symbol, order, quantity, side)

    async def _await_limit_fill(
        self,
        symbol: str,
        order: dict[str, Any],
        quantity: float,
        side: str,
    ) -> dict[str, Any]:
        """Poll a limit order; convert to market if it has not filled in time.

        A partially filled order is kept (the remainder is cancelled) - chasing
        the rest with a market order at a worse price is how a small edge becomes
        a negative one.
        """
        order_id: str = str(order.get("id", ""))
        deadline: float = utc_now_ms() / 1_000.0 + self._config.limit_order_timeout_seconds

        while utc_now_ms() / 1_000.0 < deadline:
            await asyncio.sleep(2.0)
            current: dict[str, Any] = await self._call(
                f"fetch_order[{symbol}]",
                lambda: self.exchange.fetch_order(order_id, symbol),
            )
            if str(current.get("status", "")).lower() in {"closed", "filled"}:
                return current
            order = current

        _LOGGER.warning("Limit order %s on %s did not fill within the timeout", order_id, symbol)
        try:
            await self._call(
                f"cancel_order[{symbol}]",
                lambda: self.exchange.cancel_order(order_id, symbol),
                critical=False,
            )
        except ExecutionError as error:
            _LOGGER.warning("Could not cancel stale limit order: %s", error)

        filled: float = safe_float(order.get("filled"), 0.0)
        if filled > 0.0:
            return order
        raise ExecutionError("limit order expired unfilled", symbol=symbol, order_id=order_id)

    async def _place_protective_orders(self, position: Position) -> None:
        """Place the reduce-only take-profit ladder and the stop-loss order.

        ``workingType=MARK_PRICE`` matches how Binance evaluates liquidation, so
        the stop cannot be triggered by a last-price wick that never touched the
        mark.

        With the ladder enabled this places one reduce-only ``TAKE_PROFIT_MARKET``
        per level, each sized to that level's share of the position, so the
        exchange executes the scale-out even if this process dies.  The *stop*
        stays a single order covering the whole remaining position and is
        re-placed by :meth:`_sync_ladder_stop` as legs fill.
        """
        symbol: str = position.symbol
        side: str = position.reduce_side
        params: dict[str, Any] = {"reduceOnly": True, "workingType": "MARK_PRICE"}

        if position.ladder is not None:
            for level, (price, fraction) in enumerate(
                zip(position.ladder.tp_prices, position.ladder.close_fractions, strict=True),
                start=1,
            ):
                if fraction <= 0.0:
                    continue
                leg_quantity: float = float(
                    self.exchange.amount_to_precision(
                        symbol, position.quantity_for_fraction(fraction)
                    )
                )
                if leg_quantity <= 0.0:
                    continue
                leg_price: float = float(self.exchange.price_to_precision(symbol, price))
                leg_order: dict[str, Any] = await self._call(
                    f"create_tp{level}[{symbol}]",
                    lambda leg_price=leg_price, leg_quantity=leg_quantity: self.exchange.create_order(
                        symbol,
                        "TAKE_PROFIT_MARKET",
                        side,
                        leg_quantity,
                        None,
                        {**params, "stopPrice": leg_price},
                    ),
                )
                position.exchange_order_ids[f"take_profit_{level}"] = str(leg_order.get("id", ""))
        else:
            take_profit: float = float(
                self.exchange.price_to_precision(symbol, position.take_profit)
            )
            tp_order: dict[str, Any] = await self._call(
                f"create_tp[{symbol}]",
                lambda: self.exchange.create_order(
                    symbol,
                    "TAKE_PROFIT_MARKET",
                    side,
                    position.quantity,
                    None,
                    {**params, "stopPrice": take_profit},
                ),
            )
            position.exchange_order_ids["take_profit"] = str(tp_order.get("id", ""))

        quantity: float = position.quantity
        stop_loss: float = float(self.exchange.price_to_precision(symbol, position.effective_stop()))
        sl_order: dict[str, Any] = await self._call(
            f"create_sl[{symbol}]",
            lambda: self.exchange.create_order(
                symbol,
                "STOP_MARKET",
                side,
                quantity,
                None,
                {**params, "stopPrice": stop_loss},
            ),
        )
        position.exchange_order_ids["stop_loss"] = str(sl_order.get("id", ""))

    # ------------------------------------------------------------------
    # Position monitoring & trailing stops
    # ------------------------------------------------------------------
    async def _monitor_loop(self) -> None:
        """Background task: trail stops and detect exchange-side closures."""
        interval: float = self._config.position_monitor_interval_seconds
        reconcile_every: float = self._config.reconcile_interval_seconds
        last_reconcile: float = utc_now_ms() / 1_000.0

        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._positions:
                    continue

                prices: dict[str, float] = await self._safe_prices(list(self._positions))
                for symbol, position in list(self._positions.items()):
                    price: float | None = prices.get(symbol)
                    if price is None or price <= 0.0:
                        continue
                    position.update_excursions(price, price)
                    if position.ladder is not None:
                        await self._maybe_advance_ladder(position, price)
                    else:
                        await self._maybe_advance_trailing(position, price)

                now: float = utc_now_ms() / 1_000.0
                if now - last_reconcile >= reconcile_every:
                    await self.reconcile()
                    last_reconcile = now
            except asyncio.CancelledError:
                raise
            except Exception as error:  # pragma: no cover - the monitor must never die
                _LOGGER.error("Position monitor iteration failed: %s", error, exc_info=True)

    async def _maybe_advance_ladder(self, position: Position, price: float) -> None:
        """Track ladder progress on the exchange and move the stop behind it.

        The take-profit legs are resting reduce-only orders, so the *exchange*
        executes the scale-out - this method does not send them.  What it does is
        keep the local state machine in step (it advances on the same price
        crossings the resting orders trigger on) and re-place the single stop
        order at the new stage's level and the new remaining size.

        Stop breaches are deliberately not handled here: the resting stop order
        owns that, and :meth:`reconcile` books the closure.  Racing the exchange
        with a second closing order is how positions get double-closed.
        """
        ladder = position.ladder
        if ladder is None or ladder.is_closed or ladder.stop_breached(price):
            return

        previous_stage: int = ladder.stage
        filled: list[LadderEvent] = [
            event for event in ladder.on_tick(price) if not event.is_stop
        ]
        if not filled or ladder.stage == previous_stage:
            return

        for event in filled:
            position.quantity = max(
                0.0, position.quantity - position.quantity_for_fraction(event.fraction)
            )
            position.partial_fills.append(
                {
                    "reason": event.kind,
                    "price": float(event.price),
                    "fraction": float(event.fraction),
                    "timestamp_ms": utc_now_ms(),
                }
            )
            _LOGGER.info(
                "%s reached on %s at %.6f - %.0f%% of the position scaled out by the exchange",
                event.kind,
                position.symbol,
                event.price,
                event.fraction * 100.0,
            )

        if position.is_flat:
            # TP3 filled: the exchange has closed us out; reconcile will book it.
            return
        await self._sync_ladder_stop(position)

    async def _sync_ladder_stop(self, position: Position) -> None:
        """Cancel and re-place the stop at the stage's level and remaining size.

        The cancel happens first: briefly having no stop is safer than briefly
        having two, which would double-close the position.  If the replacement
        fails the position is flattened at market rather than left naked.
        """
        symbol: str = position.symbol
        old_order_id: str = position.exchange_order_ids.get("stop_loss", "")
        if old_order_id:
            try:
                await self._call(
                    f"cancel_sl[{symbol}]",
                    lambda: self.exchange.cancel_order(old_order_id, symbol),
                    critical=False,
                )
            except ExecutionError as error:
                _LOGGER.warning("Could not cancel the previous stop on %s: %s", symbol, error)

        try:
            stop_price: float = float(
                self.exchange.price_to_precision(symbol, position.effective_stop())
            )
            quantity: float = float(
                self.exchange.amount_to_precision(symbol, position.quantity)
            )
            order: dict[str, Any] = await self._call(
                f"replace_sl[{symbol}]",
                lambda: self.exchange.create_order(
                    symbol,
                    "STOP_MARKET",
                    position.reduce_side,
                    quantity,
                    None,
                    {"reduceOnly": True, "workingType": "MARK_PRICE", "stopPrice": stop_price},
                ),
            )
            position.exchange_order_ids["stop_loss"] = str(order.get("id", ""))
            _LOGGER.info(
                "Stop moved on %s -> %.6f (%s protection, %g remaining)",
                symbol,
                stop_price,
                position.ladder.protection.value if position.ladder else "NONE",
                quantity,
            )
        except ExecutionError as error:
            _LOGGER.critical(
                "Failed to re-place the stop on %s after cancelling it (%s) - flattening",
                symbol,
                error,
            )
            await self._market_close(position, CloseReason.MANUAL, note="stop replacement failed")

    async def _maybe_advance_trailing(self, position: Position, price: float) -> None:
        """Move the stop up (long) / down (short) as the trade goes in our favour.

        The exchange has no native "trailing stop on a mark-price trigger" that
        matches our geometry, so the old stop order is cancelled and a new one is
        placed.  The cancel is done first: briefly having no stop is safer than
        briefly having *two*, which would double-close the position.
        """
        new_stop: float | None = position.advance_trailing(price)
        if new_stop is None:
            return

        symbol: str = position.symbol
        old_order_id: str = position.exchange_order_ids.get("stop_loss", "")

        try:
            if old_order_id:
                await self._call(
                    f"cancel_sl[{symbol}]",
                    lambda: self.exchange.cancel_order(old_order_id, symbol),
                    critical=False,
                )
        except ExecutionError as error:
            _LOGGER.warning("Could not cancel the previous stop on %s: %s", symbol, error)

        try:
            rounded: float = float(self.exchange.price_to_precision(symbol, new_stop))
            order: dict[str, Any] = await self._call(
                f"replace_sl[{symbol}]",
                lambda: self.exchange.create_order(
                    symbol,
                    "STOP_MARKET",
                    position.reduce_side,
                    position.quantity,
                    None,
                    {"reduceOnly": True, "workingType": "MARK_PRICE", "stopPrice": rounded},
                ),
            )
            position.exchange_order_ids["stop_loss"] = str(order.get("id", ""))
            position.trailing_stop = rounded
            _LOGGER.info(
                "Trailing stop advanced on %s -> %.6f (price %.6f)", symbol, rounded, price
            )
        except ExecutionError as error:
            _LOGGER.critical(
                "Failed to re-place the stop on %s after cancelling it (%s) - flattening",
                symbol,
                error,
            )
            await self._market_close(position, CloseReason.MANUAL, note="stop replacement failed")

    async def reconcile(self) -> None:
        """Sync tracked positions against the exchange's own view.

        Anything we track that the exchange no longer reports has been closed by
        a protective order (or liquidated); the closure is booked here.
        """
        try:
            raw: list[dict[str, Any]] = await self._call(
                "fetch_positions",
                lambda: self.exchange.fetch_positions(),
                critical=False,
            )
        except ExecutionError as error:
            _LOGGER.warning("Reconciliation skipped: %s", error)
            return

        live_symbols: set[str] = set()
        for entry in raw:
            contracts: float = safe_float(entry.get("contracts"), 0.0)
            if abs(contracts) <= 0.0:
                continue
            symbol: str = str(entry.get("symbol", ""))
            live_symbols.add(symbol)
            if symbol not in self._positions:
                _LOGGER.warning(
                    "Untracked exchange position on %s (%s contracts) - not managed by this engine",
                    symbol,
                    contracts,
                )

        for symbol in list(self._positions):
            if symbol not in live_symbols:
                await self._book_external_close(self._positions[symbol])

    async def _book_external_close(self, position: Position) -> None:
        """Record a closure that happened on the exchange (TP/SL/liquidation)."""
        reason: CloseReason = position.stop_close_reason()
        exit_price: float = position.effective_stop()

        candidates: list[tuple[str, CloseReason]] = [
            ("take_profit_3", CloseReason.TAKE_PROFIT_3),
            ("take_profit_2", CloseReason.TAKE_PROFIT_2),
            ("take_profit_1", CloseReason.TAKE_PROFIT_1),
            ("take_profit", CloseReason.TAKE_PROFIT),
            ("stop_loss", position.stop_close_reason()),
        ]
        for key, candidate in candidates:
            order_id: str = position.exchange_order_ids.get(key, "")
            if not order_id:
                continue
            try:
                order: dict[str, Any] = await self._call(
                    f"fetch_order[{position.symbol}]",
                    lambda: self.exchange.fetch_order(order_id, position.symbol),
                    critical=False,
                )
            except ExecutionError:
                continue
            if str(order.get("status", "")).lower() in {"closed", "filled"}:
                reason = candidate
                exit_price = safe_float(
                    order.get("average") or order.get("price"), exit_price
                ) or exit_price
                break

        await self._finalise(position, exit_price, reason)
        await self._cancel_symbol_orders(position.symbol)

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------
    async def close_position(
        self,
        symbol: str,
        reason: CloseReason = CloseReason.MANUAL,
    ) -> ExecutionReport:
        """Close one position at market and cancel its resting orders."""
        async with self._lock:
            position: Position | None = self._positions.get(symbol)
            if position is None:
                return ExecutionReport(
                    decision_id="", symbol=symbol, accepted=False, error="no open position"
                )
            return await self._market_close(position, reason)

    async def _market_close(
        self,
        position: Position,
        reason: CloseReason,
        note: str = "",
    ) -> ExecutionReport:
        """Send a reduce-only market order and book the result."""
        symbol: str = position.symbol
        await self._cancel_symbol_orders(symbol)

        try:
            order: dict[str, Any] = await self._call(
                f"close_market[{symbol}]",
                lambda: self.exchange.create_order(
                    symbol,
                    "market",
                    position.reduce_side,
                    position.quantity,
                    None,
                    {"reduceOnly": True},
                ),
            )
            exit_price: float = safe_float(order.get("average") or order.get("price"), 0.0)
        except ExecutionError as error:
            _LOGGER.critical("MARKET CLOSE FAILED on %s: %s", symbol, error)
            return ExecutionReport(
                decision_id=position.decision_id,
                symbol=symbol,
                accepted=False,
                error=f"close failed: {error}",
            )

        if exit_price <= 0.0:
            try:
                exit_price = await self._fetcher.fetch_last_price(symbol)
            except Exception:  # pragma: no cover - last-resort valuation
                exit_price = position.entry_price

        await self._finalise(position, exit_price, reason, note=note)
        return ExecutionReport(
            decision_id=position.decision_id,
            symbol=symbol,
            accepted=True,
            position=position,
            detail={"close_reason": reason.value, "exit_price": exit_price},
        )

    async def _finalise(
        self,
        position: Position,
        exit_price: float,
        reason: CloseReason,
        note: str = "",
    ) -> None:
        """Compute realised PnL, persist the trade and notify the Risk Guard."""
        realized_legs: float = sum(
            (float(leg["price"]) - position.entry_price)
            * position.quantity_for_fraction(float(leg["fraction"]))
            * position.direction
            for leg in position.partial_fills
        )
        leg_fees: float = sum(
            float(leg["price"])
            * position.quantity_for_fraction(float(leg["fraction"]))
            * self._config.taker_fee
            for leg in position.partial_fills
        )
        gross: float = (exit_price - position.entry_price) * position.quantity * position.direction
        exit_fee: float = exit_price * position.quantity * self._config.taker_fee
        position.fees_paid += exit_fee + leg_fees
        position.realized_pnl = (
            gross + realized_legs - position.fees_paid - position.funding_paid
        )
        position.exit_price = exit_price
        position.closed_at = datetime.now(tz=timezone.utc)
        position.close_reason = reason.value if not note else f"{reason.value}:{note}"
        position.status = (
            PositionStatus.LIQUIDATED if reason is CloseReason.LIQUIDATION else PositionStatus.CLOSED
        )

        self._positions.pop(position.symbol, None)
        await self._persist_close(position)
        await self._risk.record_trade_result(position.realized_pnl, position.symbol)

        _LOGGER.info(
            "CLOSE %s %s @ %.6f | reason=%s | pnl=%.4f USDT (fees %.4f, funding %.4f)",
            position.action.value,
            position.symbol,
            exit_price,
            position.close_reason,
            position.realized_pnl,
            position.fees_paid,
            position.funding_paid,
        )

    async def _cancel_symbol_orders(self, symbol: str) -> None:
        """Cancel every resting order on a symbol (best effort)."""
        try:
            await self._call(
                f"cancel_all[{symbol}]",
                lambda: self.exchange.cancel_all_orders(symbol),
                critical=False,
            )
        except ExecutionError as error:
            _LOGGER.warning("Could not cancel open orders on %s: %s", symbol, error)

    async def cancel_all_orders(self) -> None:
        """Cancel resting orders across every symbol we track."""
        for symbol in list(self._positions):
            await self._cancel_symbol_orders(symbol)

    async def emergency_flatten(self, reason: str) -> None:
        """Risk Guard halt callback: cancel everything, close everything.

        Failures are logged and the loop continues - one symbol that refuses to
        close must not leave the remaining positions open.
        """
        _LOGGER.critical("EMERGENCY FLATTEN requested: %s", reason)
        await self.cancel_all_orders()
        for symbol in list(self._positions):
            try:
                position: Position | None = self._positions.get(symbol)
                if position is not None:
                    await self._market_close(position, CloseReason.KILL_SWITCH, note=reason[:40])
            except Exception as error:  # pragma: no cover - flatten must be total
                _LOGGER.critical("Emergency close failed for %s: %s", symbol, error, exc_info=True)
        _LOGGER.critical("Emergency flatten complete; %d position(s) remain", len(self._positions))

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    async def _persist_open(self, position: Position) -> None:
        """Insert the opened trade row."""
        try:
            await self._db.insert_trade(position.to_row())
        except Exception as error:  # pragma: no cover - DB must not break trading
            _LOGGER.error("Could not persist the opened trade: %s", error)

    async def _persist_close(self, position: Position) -> None:
        """Update the trade row on closure."""
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
        except Exception as error:  # pragma: no cover - DB must not break trading
            _LOGGER.error("Could not persist the closed trade: %s", error)

    @staticmethod
    def _extract_fee(order: dict[str, Any], price: float, quantity: float) -> float:
        """Read the fee the exchange actually charged, or estimate it."""
        fee: dict[str, Any] = order.get("fee") or {}
        cost: float = safe_float(fee.get("cost"), 0.0)
        if cost > 0.0:
            return cost
        fees: list[dict[str, Any]] = order.get("fees") or []
        total: float = sum(safe_float(entry.get("cost"), 0.0) for entry in fees)
        if total > 0.0:
            return total
        return price * quantity * 0.0005
