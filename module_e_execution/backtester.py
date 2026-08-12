"""Event-driven backtester for the 5-minute pipeline.

Realism is the whole point of this file.  A backtester that is easy to make
profitable is worthless, so every shortcut that flatters results has been closed:

* **Signals fill on the next bar's open, never on the close that produced them.**
  A decision made at the close of bar ``t`` could not have been executed inside
  bar ``t``; filling at ``close[t]`` is the most common way backtests
  manufacture returns that do not exist.
* **Worst-case intra-candle ordering.**  OHLC data cannot say whether the high
  or the low came first, so when a bar could resolve several barriers the
  adverse one is always booked: liquidation/stop before take-profit.
* **Full cost model.**  Taker fees on both legs, slippage on every market fill,
  and funding charged at the real 00:00 / 08:00 / 16:00 UTC boundaries using the
  funding rates Module A actually recorded.
* **Isolated-margin liquidation.**  Checked on every bar; a liquidated position
  loses its entire margin, no more and no less.
* **Portfolio-level constraints.**  The same concurrent-position cap and
  one-position-per-symbol rule the live engine enforces.

Features are precomputed once per symbol over the whole window.  That is exactly
equivalent to computing them bar by bar - and safe - *because* every transform in
Module B is causal; the property is verified there rather than assumed here.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from config.settings import ExecutionSettings, Settings
from core.exceptions import InsufficientDataError
from core.logger import get_logger
from core.utils import ms_to_datetime
from module_a_data.db_handler import DatabaseHandler
from module_b_features.features import FEATURE_COLUMNS, FeatureService
from module_b_features.processor import InferencePayload
from module_c_ml.decision_engine import DecisionContext, DecisionEngine
from module_c_ml.ml_models import MLSubsystem
from module_c_ml.schemas import DecisionResult, TradeAction, TradeSignal
from module_e_execution.models import CloseReason, Position, PositionStatus
from module_e_execution.tp_ladder import LadderEvent

_LOGGER = get_logger(__name__)

#: 5-minute bars in a 365-day year - the annualisation factor for Sharpe.
_BARS_PER_YEAR: Final[int] = 365 * 24 * 12
_MS_PER_HOUR: Final[int] = 3_600_000


@dataclass(slots=True)
class BacktestReport:
    """Quantitative summary of a backtest run."""

    start: datetime | None
    end: datetime | None
    initial_equity: float
    final_equity: float
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    symbols: tuple[str, ...] = field(default=())
    signals_generated: int = 0
    signals_rejected: int = 0

    def summary(self) -> str:
        """Multi-line, human-readable report for logs and the CLI."""
        lines: list[str] = [
            "=" * 66,
            "BACKTEST REPORT",
            "=" * 66,
            f"Period            : {self.start} -> {self.end}",
            f"Symbols           : {len(self.symbols)}",
            f"Initial equity    : {self.initial_equity:,.2f} USDT",
            f"Final equity      : {self.final_equity:,.2f} USDT",
            f"Total return      : {self.metrics.get('total_return_pct', 0.0):.2%}",
            f"Trades            : {int(self.metrics.get('total_trades', 0))}",
            f"Win rate          : {self.metrics.get('win_rate', 0.0):.2%}",
            f"Profit factor     : {self.metrics.get('profit_factor', 0.0):.3f}",
            f"Expectancy        : {self.metrics.get('expectancy', 0.0):,.4f} USDT/trade",
            f"Max drawdown      : {self.metrics.get('max_drawdown_pct', 0.0):.2%}",
            f"Sharpe ratio      : {self.metrics.get('sharpe_ratio', 0.0):.3f}",
            f"Calmar ratio      : {self.metrics.get('calmar_ratio', 0.0):.3f}",
            f"Sortino ratio     : {self.metrics.get('sortino_ratio', 0.0):.3f}",
            f"Avg win / loss    : {self.metrics.get('average_win', 0.0):,.4f} / "
            f"{self.metrics.get('average_loss', 0.0):,.4f}",
            f"Fees paid         : {self.metrics.get('total_fees', 0.0):,.4f} USDT",
            f"Funding paid      : {self.metrics.get('total_funding', 0.0):,.4f} USDT",
            f"Liquidations      : {int(self.metrics.get('liquidations', 0))}",
            f"Signals (gen/rej) : {self.signals_generated} / {self.signals_rejected}",
            "-" * 66,
            f"LONG  trades/win  : {int(self.metrics.get('long_trades', 0))} / "
            f"{self.metrics.get('long_win_rate', 0.0):.2%} "
            f"(PnL {self.metrics.get('long_net_profit', 0.0):,.4f})",
            f"SHORT trades/win  : {int(self.metrics.get('short_trades', 0))} / "
            f"{self.metrics.get('short_win_rate', 0.0):.2%} "
            f"(PnL {self.metrics.get('short_net_profit', 0.0):,.4f})",
            "-" * 66,
            f"Reached TP1/2/3   : {self.metrics.get('pct_reached_tp1', 0.0):.2%} / "
            f"{self.metrics.get('pct_reached_tp2', 0.0):.2%} / "
            f"{self.metrics.get('pct_reached_tp3', 0.0):.2%}",
            f"Stopped SL/BE/TP1 : {self.metrics.get('pct_stopped_initial', 0.0):.2%} / "
            f"{self.metrics.get('pct_stopped_breakeven', 0.0):.2%} / "
            f"{self.metrics.get('pct_stopped_tp1_locked', 0.0):.2%}",
            f"Average R / return: {self.metrics.get('average_r', 0.0):.3f} / "
            f"{self.metrics.get('average_trade_return', 0.0):.2%}",
            "=" * 66,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "start": self.start.isoformat() if self.start else "",
            "end": self.end.isoformat() if self.end else "",
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "symbols": list(self.symbols),
            "metrics": self.metrics,
            "signals_generated": self.signals_generated,
            "signals_rejected": self.signals_rejected,
            "trades": self.trades[-500:],
            "equity_curve": self.equity_curve[-2_000:],
        }


class Backtester:
    """Replays history bar by bar through the full decision pipeline."""

    mode: str = "backtest"

    def __init__(
        self,
        settings: Settings,
        database: DatabaseHandler,
        feature_service: FeatureService,
        ml_subsystem: MLSubsystem,
        decision_engine: DecisionEngine,
    ) -> None:
        self._settings: Settings = settings
        self._config: ExecutionSettings = settings.execution
        self._db: DatabaseHandler = database
        self._features: FeatureService = feature_service
        self._ml: MLSubsystem = ml_subsystem
        self._decisions: DecisionEngine = decision_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(
        self,
        symbols: Sequence[str] | None = None,
        max_candles: int | None = None,
        initial_equity: float | None = None,
        warmup_bars: int | None = None,
    ) -> BacktestReport:
        """Run a full backtest across the requested universe.

        Args:
            symbols: Universe (defaults to the configured one).
            max_candles: History depth per symbol.
            initial_equity: Starting virtual equity.
            warmup_bars: Bars skipped at the start so slow features are warm.

        Returns:
            A :class:`BacktestReport` with the standard quantitative metrics.
        """
        universe: list[str] = list(symbols) if symbols else list(self._settings.data.symbols)
        depth: int = max_candles or self._settings.data.history_bootstrap_candles
        equity: float = initial_equity or self._config.paper_starting_balance

        featured: dict[str, pd.DataFrame] = await self._prepare(universe, depth)
        if not featured:
            raise InsufficientDataError("no symbol produced a usable feature frame")

        funding: dict[str, pd.DataFrame] = {
            symbol: await self._db.load_futures_metrics_frame(symbol, limit=depth)
            for symbol in featured
        }

        skip: int = warmup_bars if warmup_bars is not None else 0
        return await asyncio.to_thread(self._simulate, featured, funding, equity, skip)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    async def _prepare(self, symbols: Sequence[str], depth: int) -> dict[str, pd.DataFrame]:
        """Load candles and precompute the causal feature matrix per symbol."""
        prepared: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            ohlcv: pd.DataFrame = await self._db.load_ohlcv_dataframe(symbol, limit=depth)
            if len(ohlcv) < 200:
                _LOGGER.warning("Skipping %s: only %d candles stored", symbol, len(ohlcv))
                continue

            futures: pd.DataFrame = await self._db.load_futures_metrics_frame(symbol, limit=depth)
            flow: pd.DataFrame = await self._db.load_agg_trade_flow_frame(symbol, limit=depth)
            try:
                frame: pd.DataFrame = await self._features.build(
                    ohlcv,
                    futures if not futures.empty else None,
                    None,
                    flow if not flow.empty else None,
                )
            except Exception as error:
                _LOGGER.error("Feature build failed for %s: %s", symbol, error)
                continue

            usable: pd.DataFrame = frame.replace([np.inf, -np.inf], np.nan).dropna(
                subset=list(FEATURE_COLUMNS)
            )
            if usable.empty:
                _LOGGER.warning("Skipping %s: every feature row is still warming up", symbol)
                continue
            prepared[symbol] = usable
            _LOGGER.info("Prepared %s: %d usable bars", symbol, len(usable))
        return prepared

    # ------------------------------------------------------------------
    # Simulation core
    # ------------------------------------------------------------------
    def _simulate(
        self,
        featured: dict[str, pd.DataFrame],
        funding: dict[str, pd.DataFrame],
        initial_equity: float,
        warmup_bars: int,
    ) -> BacktestReport:
        """Bar-by-bar replay.  Runs off the event loop (called via ``to_thread``)."""
        timeline: list[int] = sorted(
            {int(value) for frame in featured.values() for value in frame["timestamp"]}
        )
        if warmup_bars > 0:
            timeline = timeline[warmup_bars:]
        if not timeline:
            raise InsufficientDataError("timeline is empty after warm-up")

        indexed: dict[str, dict[int, pd.Series]] = {
            symbol: {int(row["timestamp"]): row for _, row in frame.iterrows()}
            for symbol, frame in featured.items()
        }
        funding_lookup: dict[str, dict[int, float]] = {
            symbol: self._funding_lookup(frame) for symbol, frame in funding.items()
        }

        balance: float = initial_equity
        peak_equity: float = initial_equity
        positions: dict[str, Position] = {}
        pending: list[TradeSignal] = []
        closed: list[dict[str, Any]] = []
        curve: list[dict[str, float]] = []
        generated: int = 0
        rejected: int = 0

        for timestamp in timeline:
            # --- 1. Resolve barriers on open positions with THIS bar ----------
            for symbol in list(positions):
                row: pd.Series | None = indexed.get(symbol, {}).get(timestamp)
                if row is None:
                    continue
                position: Position = positions[symbol]
                self._accrue_funding(position, row, funding_lookup.get(symbol, {}), timestamp)
                balance += self._advance_bar(position, row, timestamp)
                if position.status is not PositionStatus.OPEN:
                    closed.append(position.to_row() | {"closed_ts": timestamp})
                    positions.pop(symbol, None)

            # --- 2. Fill signals raised on the PREVIOUS bar -------------------
            # A signal is good for exactly one bar: if it cannot be filled on the
            # very next open it is dropped rather than chased at a worse price.
            equity: float = balance + self._unrealized(positions, indexed, timestamp)
            for signal in pending:
                fill_row: pd.Series | None = indexed.get(signal.symbol, {}).get(timestamp)
                if fill_row is None or signal.symbol in positions:
                    continue
                if len(positions) >= self._settings.decision.max_concurrent_positions:
                    continue
                opened: Position | None = self._fill(signal, fill_row, equity)
                if opened is None:
                    continue
                balance -= opened.fees_paid  # the entry fee realises immediately
                positions[signal.symbol] = opened
            pending = []

            # --- 3. Generate new signals from THIS bar's close ----------------
            equity = balance + self._unrealized(positions, indexed, timestamp)
            context = DecisionContext(
                risk_guard_state="GREEN",
                trading_enabled=True,
                trading_mode="backtest",
                open_positions=len(positions),
                open_symbols=frozenset(positions),
                equity=equity,
                size_multiplier=1.0,
            )
            for symbol, rows in indexed.items():
                row = rows.get(timestamp)
                if row is None or symbol in positions:
                    continue
                if len(positions) + len(pending) >= self._settings.decision.max_concurrent_positions:
                    break
                decision: DecisionResult = self._decide(symbol, row, context)
                generated += 1
                if decision.is_executable and decision.signal is not None:
                    pending.append(decision.signal)
                else:
                    rejected += 1

            # --- 4. Mark to market -------------------------------------------
            equity = balance + self._unrealized(positions, indexed, timestamp)
            peak_equity = max(peak_equity, equity)
            drawdown: float = 0.0 if peak_equity <= 0.0 else (peak_equity - equity) / peak_equity
            curve.append(
                {
                    "timestamp": float(timestamp),
                    "equity": equity,
                    "balance": balance,
                    "drawdown_pct": drawdown,
                    "open_positions": float(len(positions)),
                }
            )

            if equity <= 0.0:
                _LOGGER.error("Account wiped out at %s - halting the backtest", ms_to_datetime(timestamp))
                break

        # --- Force-close whatever is still open at the end --------------------
        final_timestamp: int = timeline[-1]
        for symbol in list(positions):
            position = positions[symbol]
            row = indexed.get(symbol, {}).get(final_timestamp)
            price: float = float(row["close"]) if row is not None else position.entry_price
            balance += self._book_close(position, price, CloseReason.SHUTDOWN, final_timestamp)
            closed.append(position.to_row() | {"closed_ts": final_timestamp})
            positions.pop(symbol, None)

        report = BacktestReport(
            start=ms_to_datetime(timeline[0]),
            end=ms_to_datetime(final_timestamp),
            initial_equity=initial_equity,
            final_equity=balance,
            trades=closed,
            equity_curve=curve,
            symbols=tuple(featured),
            signals_generated=generated,
            signals_rejected=rejected,
        )
        report.metrics = self._compute_metrics(report)
        return report

    # ------------------------------------------------------------------
    # Bar mechanics
    # ------------------------------------------------------------------
    def _decide(
        self,
        symbol: str,
        row: pd.Series,
        context: DecisionContext,
    ) -> DecisionResult:
        """Run inference plus the decision cascade for one bar."""
        features: pd.DataFrame = row[list(FEATURE_COLUMNS)].to_frame().T.astype(float)
        payload = InferencePayload(
            symbol=symbol,
            timestamp=int(row["timestamp"]),
            close=float(row["close"]),
            features=features,
            snapshot={
                "hmm_regime": float(row.get("hmm_regime", -1.0)),
                "garch_volatility": float(row.get("garch_volatility", 0.0)),
                "garch_vol_rank": float(row.get("garch_vol_rank", 0.5)),
                "kama_slope": float(row.get("kama_slope", 0.0)),
                "fdi": float(row.get("fdi", 1.5)),
                "atr": float(row.get("atr", 0.0)),
            },
        )
        inference = self._ml.infer_sync(payload)
        return self._decisions.evaluate(inference, context)

    def _fill(self, signal: TradeSignal, row: pd.Series, equity: float) -> Position | None:
        """Fill a pending signal at this bar's **open**, with slippage.

        Filling at the open of the bar *after* the decision is the single most
        important realism constraint in this file.
        """
        bar_open: float = float(row["open"])
        if bar_open <= 0.0:
            return None

        penalty: float = self._config.slippage_bps / 10_000.0
        fill_price: float = (
            bar_open * (1.0 + penalty)
            if signal.action is TradeAction.LONG
            else bar_open * (1.0 - penalty)
        )

        margin: float = signal.margin_for(equity)
        if margin <= 0.0 or margin > equity:
            return None
        quantity: float = (margin * signal.leverage) / fill_price
        if quantity <= 0.0:
            return None

        position: Position = Position.from_signal(
            signal,
            equity=equity,
            entry_price=fill_price,
            quantity=quantity,
            maintenance_margin_rate=self._config.maintenance_margin_rate,
            mode=self.mode,
            take_profit_config=self._settings.take_profit,
        )
        # The barriers were computed off the decision-bar close; re-anchor them to
        # the actual fill so the geometry the models asked for is preserved.
        if signal.action is TradeAction.LONG:
            position.take_profit = fill_price * (1.0 + signal.take_profit_pct)
            position.stop_loss = fill_price * (1.0 - signal.stop_loss_pct)
            position.trailing_trigger = fill_price * (
                1.0 + signal.take_profit_pct * 0.5
            )
        else:
            position.take_profit = fill_price * (1.0 - signal.take_profit_pct)
            position.stop_loss = fill_price * (1.0 + signal.stop_loss_pct)
            position.trailing_trigger = fill_price * (1.0 - signal.take_profit_pct * 0.5)

        position.fees_paid = fill_price * quantity * self._config.taker_fee
        position.last_funding_ms = int(row["timestamp"])
        return position

    def _advance_bar(self, position: Position, row: pd.Series, timestamp: int) -> float:
        """Run one bar through the position and return the cash delta it produced.

        Precedence within a bar is fixed and always unfavourable to the position:

        1. **Liquidation** - checked before anything else, because a liquidated
           position cannot go on to take a profit.
        2. **The ladder**, which applies its own adverse-first rule (see
           :mod:`module_e_execution.tp_ladder`): the in-force stop is tested
           against the bar's adverse extreme *before* any take-profit level is
           allowed to fill, and re-tested afterwards against the same bar.
        3. **The trailing stop**, which may only ratchet on a bar that resolved
           nothing, and is then immediately re-tested against that bar's adverse
           extreme.

        With OHLCV alone the true intrabar sequence is unknowable, so the
        favourable ordering is never assumed anywhere in this chain.
        """
        high: float = float(row["high"])
        low: float = float(row["low"])
        position.update_excursions(high, low)

        liquidation: float = position.liquidation_price
        stop: float = position.effective_stop()
        if liquidation > 0.0:
            liquidation_hit: bool = low <= liquidation if position.is_long else high >= liquidation
            liquidation_first: bool = (
                liquidation >= stop if position.is_long else liquidation <= stop
            )
            if liquidation_hit and liquidation_first:
                return self._book_close(position, liquidation, CloseReason.LIQUIDATION, timestamp)

        if position.ladder is not None:
            events: list[LadderEvent] = position.ladder.on_bar(high, low)
            if events:
                return self._book_ladder_events(position, events, timestamp)
            return 0.0

        # --- Ladder disabled: the original single-target behaviour ----------
        if position.is_long:
            adverse_hit: bool = low <= stop
            target_hit: bool = high >= position.take_profit
        else:
            adverse_hit = high >= stop
            target_hit = low <= position.take_profit

        if adverse_hit:
            return self._book_close(position, stop, position.stop_close_reason(), timestamp)
        if target_hit:
            return self._book_close(
                position, position.take_profit, CloseReason.TAKE_PROFIT, timestamp
            )

        advanced: float | None = position.advance_trailing(high if position.is_long else low)
        if advanced is None:
            return 0.0
        breached: bool = low <= advanced if position.is_long else high >= advanced
        if breached:
            return self._book_close(position, advanced, CloseReason.TRAILING_STOP, timestamp)
        return 0.0

    def _book_ladder_events(
        self,
        position: Position,
        events: list[LadderEvent],
        timestamp: int,
    ) -> float:
        """Book every leg the ladder produced on this bar, with costs."""
        delta: float = 0.0
        last_reason: CloseReason = CloseReason.TAKE_PROFIT
        for event in events:
            fill_price: float = self._exit_price(position, event.price)
            leg_delta, last_reason = position.apply_ladder_event(
                event,
                exit_price=fill_price,
                fee_rate=self._config.taker_fee,
                timestamp_ms=timestamp,
            )
            delta += leg_delta

        if position.is_flat:
            delta -= position.funding_paid
            position.exit_price = self._exit_price(position, events[-1].price)
            position.close_reason = last_reason.value
            position.closed_at = datetime.fromtimestamp(timestamp / 1_000.0, tz=timezone.utc)
            position.status = PositionStatus.CLOSED
        return delta

    def _exit_price(self, position: Position, raw_price: float) -> float:
        """Apply exit slippage in the unfavourable direction."""
        penalty: float = self._config.slippage_bps / 10_000.0
        return raw_price * (1.0 - penalty) if position.is_long else raw_price * (1.0 + penalty)

    def _accrue_funding(
        self,
        position: Position,
        row: pd.Series,
        rates: dict[int, float],
        timestamp: int,
    ) -> None:
        """Charge funding when this bar crosses a funding boundary.

        Binance settles at 00:00, 08:00 and 16:00 UTC; a bar whose open time
        falls on one of those boundaries carries the settlement.
        """
        interval_ms: int = self._config.funding_interval_hours * _MS_PER_HOUR
        if timestamp % interval_ms != 0:
            return
        if position.last_funding_ms > 0 and timestamp - position.last_funding_ms < interval_ms:
            return

        rate: float = rates.get(timestamp, self._nearest_rate(rates, timestamp))
        notional: float = position.quantity * float(row["close"])
        position.funding_paid += notional * rate * position.direction
        position.last_funding_ms = timestamp

    @staticmethod
    def _funding_lookup(frame: pd.DataFrame) -> dict[int, float]:
        """Index the recorded funding rates by timestamp."""
        if frame.empty or "funding_rate" not in frame.columns:
            return {}
        return {
            int(timestamp): float(rate)
            for timestamp, rate in zip(frame["timestamp"], frame["funding_rate"], strict=False)
        }

    @staticmethod
    def _nearest_rate(rates: dict[int, float], timestamp: int) -> float:
        """Most recent funding rate at or before ``timestamp`` (0.0 if none)."""
        if not rates:
            return 0.0
        earlier: list[int] = [key for key in rates if key <= timestamp]
        return rates[max(earlier)] if earlier else 0.0

    def _book_close(
        self,
        position: Position,
        raw_price: float,
        reason: CloseReason,
        timestamp: int,
    ) -> float:
        """Close the remaining position and return the balance delta.

        The entry fee was already deducted at fill time, so this returns
        ``gross - exit_fee - funding`` for whatever quantity is still open.  With
        a partially filled ladder that is the *remainder*; the legs already taken
        were booked as they filled.
        """
        penalty: float = self._config.slippage_bps / 10_000.0
        if reason is CloseReason.LIQUIDATION:
            # A liquidation is filled by the exchange at the liquidation price;
            # there is no slippage relief and no better fill to hope for.
            exit_price: float = raw_price
        elif position.is_long:
            exit_price = raw_price * (1.0 - penalty)
        else:
            exit_price = raw_price * (1.0 + penalty)

        remaining: float = position.quantity
        gross: float = (exit_price - position.entry_price) * remaining * position.direction
        exit_fee: float = exit_price * remaining * self._config.taker_fee
        position.fees_paid += exit_fee
        position.quantity = 0.0

        delta: float = gross - exit_fee - position.funding_paid
        if reason is CloseReason.LIQUIDATION:
            # Isolated margin: the loss is capped at the committed margin, and
            # any profit already banked by the ladder is not clawed back.
            delta = max(delta, -position.margin - position.realized_pnl)

        position.realized_pnl += delta
        position.exit_price = exit_price
        position.close_reason = reason.value
        position.closed_at = datetime.fromtimestamp(timestamp / 1_000.0, tz=timezone.utc)
        position.status = (
            PositionStatus.LIQUIDATED if reason is CloseReason.LIQUIDATION else PositionStatus.CLOSED
        )
        return delta

    @staticmethod
    def _unrealized(
        positions: dict[str, Position],
        indexed: dict[str, dict[int, pd.Series]],
        timestamp: int,
    ) -> float:
        """Mark open positions to this bar's close."""
        total: float = 0.0
        for symbol, position in positions.items():
            row: pd.Series | None = indexed.get(symbol, {}).get(timestamp)
            price: float = float(row["close"]) if row is not None else position.entry_price
            total += position.unrealized_pnl(price) - position.funding_paid
        return total

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self, report: BacktestReport) -> dict[str, float]:
        """Compute the standard quantitative performance metrics."""
        trades: list[dict[str, Any]] = report.trades
        pnls: np.ndarray = np.asarray(
            [float(trade.get("realized_pnl", 0.0)) for trade in trades], dtype=np.float64
        )

        wins: np.ndarray = pnls[pnls > 0.0]
        losses: np.ndarray = pnls[pnls < 0.0]
        gross_profit: float = float(wins.sum()) if wins.size else 0.0
        gross_loss: float = float(-losses.sum()) if losses.size else 0.0

        equity_values: np.ndarray = np.asarray(
            [point["equity"] for point in report.equity_curve], dtype=np.float64
        )
        max_drawdown: float = self._max_drawdown(equity_values)

        periods: float = max(1.0, float(equity_values.size))
        total_return: float = (
            (report.final_equity - report.initial_equity) / report.initial_equity
            if report.initial_equity > 0.0
            else 0.0
        )
        years: float = periods / float(_BARS_PER_YEAR)
        if years > 0.0 and report.initial_equity > 0.0 and report.final_equity > 0.0:
            annualised: float = (report.final_equity / report.initial_equity) ** (1.0 / years) - 1.0
        else:
            annualised = 0.0

        return {
            **self._ladder_metrics(trades),
            **self._side_metrics(trades),
            "total_trades": float(pnls.size),
            "winning_trades": float(wins.size),
            "losing_trades": float(losses.size),
            "win_rate": float(wins.size / pnls.size) if pnls.size else 0.0,
            "profit_factor": (
                gross_profit / gross_loss if gross_loss > 0.0 else (math.inf if gross_profit > 0.0 else 0.0)
            ),
            "expectancy": float(pnls.mean()) if pnls.size else 0.0,
            "average_win": float(wins.mean()) if wins.size else 0.0,
            "average_loss": float(losses.mean()) if losses.size else 0.0,
            "largest_win": float(wins.max()) if wins.size else 0.0,
            "largest_loss": float(losses.min()) if losses.size else 0.0,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": float(pnls.sum()) if pnls.size else 0.0,
            "total_return_pct": total_return,
            "annualised_return_pct": annualised,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": self._sharpe(equity_values),
            "sortino_ratio": self._sortino(equity_values),
            "calmar_ratio": (annualised / max_drawdown) if max_drawdown > 0.0 else 0.0,
            "total_fees": float(sum(float(trade.get("fees_paid", 0.0)) for trade in trades)),
            "total_funding": float(sum(float(trade.get("funding_paid", 0.0)) for trade in trades)),
            "liquidations": float(
                sum(1 for trade in trades if trade.get("close_reason") == CloseReason.LIQUIDATION.value)
            ),
        }

    @staticmethod
    def _ladder_metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
        """How the three-stage ladder actually behaved.

        These are the numbers that say whether the ladder is doing its job:
        what share of trades got far enough to move the stop to breakeven, what
        share banked a TP1-locked profit after reversing, and what the average
        realised R and trade return look like once partial closes are counted.
        """
        total: int = len(trades)
        if total == 0:
            return {}

        reached: dict[str, int] = {"tp1": 0, "tp2": 0, "tp3": 0}
        stopped: dict[str, int] = {"INITIAL": 0, "BREAKEVEN": 0, "TP1_LOCKED": 0}
        realized_r: list[float] = []
        returns: list[float] = []

        for trade in trades:
            payload: dict[str, Any] = trade.get("payload") or {}
            ladder: dict[str, Any] | None = payload.get("ladder")
            if ladder:
                reached["tp1"] += int(bool(ladder.get("reached_tp1")))
                reached["tp2"] += int(bool(ladder.get("reached_tp2")))
                reached["tp3"] += int(bool(ladder.get("reached_tp3")))
            protection: str = str(payload.get("stop_protection") or "")
            if protection in stopped:
                stopped[protection] += 1
            realized_r.append(float(payload.get("realized_r", 0.0) or 0.0))
            margin: float = float(trade.get("margin", 0.0) or 0.0)
            if margin > 0.0:
                returns.append(float(trade.get("realized_pnl", 0.0)) / margin)

        return {
            "pct_reached_tp1": reached["tp1"] / total,
            "pct_reached_tp2": reached["tp2"] / total,
            "pct_reached_tp3": reached["tp3"] / total,
            "pct_stopped_initial": stopped["INITIAL"] / total,
            "pct_stopped_breakeven": stopped["BREAKEVEN"] / total,
            "pct_stopped_tp1_locked": stopped["TP1_LOCKED"] / total,
            "average_r": float(np.mean(realized_r)) if realized_r else 0.0,
            "average_trade_return": float(np.mean(returns)) if returns else 0.0,
        }

    @staticmethod
    def _side_metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
        """LONG and SHORT performance reported separately.

        A strategy whose aggregate looks acceptable while one side bleeds is a
        strategy with one working half; the split is the only way to see it.
        """
        metrics: dict[str, float] = {}
        for side in ("LONG", "SHORT"):
            subset: list[dict[str, Any]] = [
                trade for trade in trades if str(trade.get("side")) == side
            ]
            pnls: np.ndarray = np.asarray(
                [float(trade.get("realized_pnl", 0.0)) for trade in subset], dtype=np.float64
            )
            prefix: str = side.lower()
            wins: np.ndarray = pnls[pnls > 0.0]
            losses: np.ndarray = pnls[pnls < 0.0]
            gross_loss: float = float(-losses.sum()) if losses.size else 0.0
            metrics[f"{prefix}_trades"] = float(pnls.size)
            metrics[f"{prefix}_win_rate"] = float(wins.size / pnls.size) if pnls.size else 0.0
            metrics[f"{prefix}_net_profit"] = float(pnls.sum()) if pnls.size else 0.0
            metrics[f"{prefix}_expectancy"] = float(pnls.mean()) if pnls.size else 0.0
            metrics[f"{prefix}_profit_factor"] = (
                float(wins.sum()) / gross_loss
                if gross_loss > 0.0
                else (math.inf if wins.size else 0.0)
            )
        return metrics

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        """Maximum peak-to-trough decline as a positive fraction."""
        if equity.size == 0:
            return 0.0
        running_peak: np.ndarray = np.maximum.accumulate(equity)
        safe_peak: np.ndarray = np.where(running_peak > 0.0, running_peak, 1.0)
        drawdowns: np.ndarray = (running_peak - equity) / safe_peak
        return float(np.max(drawdowns)) if drawdowns.size else 0.0

    @staticmethod
    def _period_returns(equity: np.ndarray) -> np.ndarray:
        """Simple per-bar returns of the equity curve."""
        if equity.size < 2:
            return np.array([], dtype=np.float64)
        previous: np.ndarray = equity[:-1]
        safe_previous: np.ndarray = np.where(previous != 0.0, previous, np.nan)
        returns: np.ndarray = (equity[1:] - previous) / safe_previous
        return returns[np.isfinite(returns)]

    @classmethod
    def _sharpe(cls, equity: np.ndarray) -> float:
        """Annualised Sharpe ratio at a zero risk-free rate."""
        returns: np.ndarray = cls._period_returns(equity)
        if returns.size < 2:
            return 0.0
        deviation: float = float(returns.std(ddof=1))
        if deviation <= 0.0:
            return 0.0
        return float(returns.mean() / deviation * math.sqrt(_BARS_PER_YEAR))

    @classmethod
    def _sortino(cls, equity: np.ndarray) -> float:
        """Annualised Sortino ratio (downside deviation only)."""
        returns: np.ndarray = cls._period_returns(equity)
        if returns.size < 2:
            return 0.0
        downside: np.ndarray = returns[returns < 0.0]
        if downside.size == 0:
            return 0.0
        deviation: float = float(np.sqrt(np.mean(np.square(downside))))
        if deviation <= 0.0:
            return 0.0
        return float(returns.mean() / deviation * math.sqrt(_BARS_PER_YEAR))
