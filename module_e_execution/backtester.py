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
from module_b_features.features import (
    FEATURE_COLUMNS,
    REQUIRED_FEATURE_COLUMNS,
    FeatureService,
)
from module_b_features.processor import InferencePayload
from module_c_ml.decision_engine import DecisionContext, DecisionEngine
from module_c_ml.ml_models import MLSubsystem
from module_c_ml.schemas import DecisionResult, ModelInferenceResult, TradeAction, TradeSignal
from module_e_execution.models import CloseReason, Position, PositionStatus
from module_e_execution.risk_guard import RiskLadder

_LOGGER = get_logger(__name__)

#: 5-minute bars in a 365-day year - the annualisation factor for Sharpe.
_BARS_PER_YEAR: Final[int] = 365 * 24 * 12
_MS_PER_HOUR: Final[int] = 3_600_000

#: Feature-snapshot keys the Decision Engine reads, with the neutral value used
#: when a bar does not carry one. Declared once rather than spelled out at the
#: call site: `atr_pct` was missing from the backtester's hand-written dict, so
#: R5's stop_vs_labelled_atr telemetry - added specifically to stop a geometry
#: mismatch from being reintroduced silently - evaluated to NaN on every
#: backtest bar, in the one environment where it would first have been exercised.
_SNAPSHOT_KEYS: Final[dict[str, float]] = {
    "hmm_regime": -1.0,
    "garch_volatility": 0.0,
    "garch_vol_rank": 0.5,
    "kama_slope": 0.0,
    "fdi": 1.5,
    "atr": 0.0,
    "atr_pct": 0.0,
}


def _accumulate_rule_counts(
    decision: "DecisionResult",
    independent: dict[str, int],
    evaluations: dict[str, dict[str, int]],
) -> None:
    """Count every rule that was evaluated and every one that objected.

    `rejection_breakdown` records only the rule that stopped a signal, because
    `evaluate` returns on the first failure. That makes a late rule look
    harmless when it is merely rarely reached. Each DecisionResult already
    carries the full `checks` list, so the independent view costs nothing but
    the bookkeeping.
    """
    for check in decision.checks or []:
        rule = str(check.get("rule", ""))
        if not rule:
            continue
        counts = evaluations.setdefault(rule, {"reached": 0, "passed": 0})
        counts["reached"] += 1
        if check.get("passed"):
            counts["passed"] += 1
        else:
            independent[rule] = independent.get(rule, 0) + 1



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
    #: Count of rejected signals per Decision Engine rule (``Rule.*`` id ->
    #: count), so "why were 99.9% of signals rejected" has a real, measured
    #: answer instead of a guess - see ``module_c_ml.decision_engine.Rule``.
    rejection_breakdown: dict[str, int] = field(default_factory=dict)
    #: How often each rule would have objected *on its own*, independent of
    #: which rule happened to fire first. ``rejection_breakdown`` records only
    #: the rule that stopped a signal, so a rule sitting late in the cascade
    #: reads as harmless purely because it is rarely reached: the audited run
    #: showed R4 rejecting 568 signals under live thresholds and 1,268,257 under
    #: looser ones, a factor of 2,233 in the *opposite* direction to intuition.
    #: This breakdown is the one that supports tuning.
    rejection_breakdown_independent: dict[str, int] = field(default_factory=dict)
    #: ``rule -> {"reached": n, "passed": n}``: a rule's pass rate conditional on
    #: being evaluated at all.
    rule_evaluation_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Set by ``TradingSystem._run_validation_backtest`` (never by ``run()``
    #: itself, which has no notion of a train/validation split) to disclose
    #: what fraction of this replay window is genuinely out-of-sample versus
    #: overlapping the model's own training data. ``None`` for a backtest run
    #: outside that diagnostic path (e.g. the plain CLI ``backtest`` command).
    oos_disclosure: dict[str, Any] | None = field(default=None)
    #: Field-by-field difference from the live DecisionSettings, set on the
    #: relaxed diagnostic replay so any unintended divergence is visible in the
    #: report rather than only in the source.
    settings_delta: dict[str, Any] | None = field(default=None)
    #: Risk Guard state transitions during the replay, with the equity and
    #: reason that triggered each. A run that would have halted is a result.
    risk_guard_transitions: list[dict[str, Any]] = field(default_factory=list)
    #: Set when the Risk Guard halted the replay and never released it.
    halted_at: str | None = field(default=None)

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
        ]
        if self.rejection_breakdown:
            lines.append("Rejected by rule  :")
            for rule, count in sorted(
                self.rejection_breakdown.items(), key=lambda item: item[1], reverse=True
            ):
                pct: float = count / self.signals_rejected if self.signals_rejected else 0.0
                lines.append(f"  {rule:32s} {count:8d} ({pct:.1%})")
        lines.append("=" * 66)
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
            "rejection_breakdown": self.rejection_breakdown,
            "rejection_breakdown_independent": self.rejection_breakdown_independent,
            "rule_evaluation_counts": self.rule_evaluation_counts,
            "oos_disclosure": self.oos_disclosure,
            "settings_delta": self.settings_delta,
            "risk_guard_transitions": self.risk_guard_transitions,
            "halted_at": self.halted_at,
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
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> BacktestReport:
        """Run a full backtest across the requested universe.

        Args:
            symbols: Universe (defaults to the configured one).
            max_candles: History depth per symbol to load.
            initial_equity: Starting virtual equity.
            warmup_bars: Bars skipped at the start so slow features are warm.
                Ignored when ``start_ms`` is given - the explicit window already
                says where the replay begins, and applying both trims the front
                of the window twice.
            start_ms: Replay only bars at or after this timestamp.
            end_ms: Replay only bars at or before this timestamp.

        ``start_ms``/``end_ms`` pin the replay to a known window.  Without them
        the window is "the most recent N bars", which is a moving target while
        the ingestion pipeline keeps writing: two replays launched minutes apart
        cover different periods, which is how a strict and a relaxed pass
        described as "the same out-of-sample window" ended up hours apart at
        both ends with different signal counts.

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

        # `_prepare` has already dropped un-warmed rows via the REQUIRED feature
        # gate, so the explicit-window path needs no further trim.
        skip: int = 0 if start_ms is not None else (warmup_bars if warmup_bars is not None else 0)
        return await asyncio.to_thread(
            self._simulate, featured, funding, equity, skip, start_ms, end_ms
        )

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
            try:
                frame: pd.DataFrame = await self._features.build(
                    ohlcv, futures if not futures.empty else None, None
                )
            except Exception as error:
                _LOGGER.error("Feature build failed for %s: %s", symbol, error)
                continue

            # Gate on the required block only, matching the live inference path
            # (DatasetProcessor.build_inference_payload) and the training path.
            # Dropping on the optional micro-structure columns too would make the
            # backtest silently skip any symbol whose book archive starts later
            # than its klines - i.e. exactly the symbols worth checking.
            usable: pd.DataFrame = frame.replace([np.inf, -np.inf], np.nan).dropna(
                subset=list(REQUIRED_FEATURE_COLUMNS)
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
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> BacktestReport:
        """Bar-by-bar replay.  Runs off the event loop (called via ``to_thread``)."""
        timeline: list[int] = sorted(
            {int(value) for frame in featured.values() for value in frame["timestamp"]}
        )
        if start_ms is not None:
            timeline = [ts for ts in timeline if ts >= start_ms]
        if end_ms is not None:
            timeline = [ts for ts in timeline if ts <= end_ms]
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
        rejection_breakdown: dict[str, int] = {}
        rejection_breakdown_independent: dict[str, int] = {}
        rule_evaluation_counts: dict[str, dict[str, int]] = {}
        closed_pnl_this_bar: list[float] = []
        report_transitions: list[dict[str, Any]] = []
        halted_at: str | None = None
        guard: RiskLadder = RiskLadder(self._settings)

        for timestamp in timeline:
            # --- 1. Fill signals raised on the PREVIOUS bar, at THIS bar's open -
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

            # --- 2. Resolve barriers against THIS bar, newly-filled included ---
            # Filling before resolving is the whole point: a position opened at
            # this bar's open is exposed to this bar's high and low, and must be
            # tested against them.  Resolving first gave every new position a
            # free bar of adverse movement - a systematic optimism landing
            # precisely on the trades that would otherwise stop out immediately,
            # which flatters win rate and average loss at the same time.
            for symbol in list(positions):
                row: pd.Series | None = indexed.get(symbol, {}).get(timestamp)
                if row is None:
                    continue
                position: Position = positions[symbol]
                self._accrue_funding(position, row, funding_lookup.get(symbol, {}), timestamp)
                resolution: tuple[CloseReason, float] | None = self._resolve_bar(position, row)
                if resolution is not None:
                    reason, price = resolution
                    realised = self._book_close(position, price, reason, timestamp)
                    balance += realised
                    closed.append(position.to_row() | {"closed_ts": timestamp})
                    positions.pop(symbol, None)
                    closed_pnl_this_bar.append(realised)

            # --- 3. Risk Guard: the policy a live deployment would run under ---
            equity = balance + self._unrealized(positions, indexed, timestamp)
            guard_state, size_multiplier = self._guard_state(
                guard, equity, closed_pnl_this_bar, timestamp, report_transitions
            )
            closed_pnl_this_bar = []
            if guard_state == "RED" and halted_at is None:
                halted_at = ms_to_datetime(timestamp).isoformat()

            # --- 4. Generate new signals from THIS bar's close ------------------
            context = DecisionContext(
                risk_guard_state=guard_state,
                trading_enabled=True,
                trading_mode="backtest",
                open_positions=len(positions),
                open_symbols=frozenset(positions),
                positions_per_symbol={symbol: 1 for symbol in positions},
                equity=equity,
                size_multiplier=size_multiplier,
            )
            # Every eligible symbol is scored, then the batch is ranked by
            # directional confidence and filled through the *same*
            # `evaluate_many` the live path uses.  Iterating a dict and breaking
            # at the concurrency cap allocated slots by insertion order, so the
            # backtest measured a selection policy nobody runs - and stopped
            # counting candidates the moment the book filled, making
            # `signals_generated` a function of how fast positions opened rather
            # than of how many opportunities existed.
            batch: list[ModelInferenceResult] = []
            for symbol, rows in indexed.items():
                row = rows.get(timestamp)
                if row is None or symbol in positions:
                    continue
                batch.append(self._infer(symbol, row))
            if batch:
                generated += len(batch)
                for decision in self._decisions.evaluate_many(batch, context):
                    if decision.is_executable and decision.signal is not None:
                        pending.append(decision.signal)
                    else:
                        rejected += 1
                        rejection_breakdown[decision.rule_triggered] = (
                            rejection_breakdown.get(decision.rule_triggered, 0) + 1
                        )
                    _accumulate_rule_counts(
                        decision, rejection_breakdown_independent, rule_evaluation_counts
                    )

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
            rejection_breakdown=rejection_breakdown,
            rejection_breakdown_independent=rejection_breakdown_independent,
            rule_evaluation_counts=rule_evaluation_counts,
            risk_guard_transitions=report_transitions,
            halted_at=halted_at,
        )
        report.metrics = self._compute_metrics(report)
        return report

    # ------------------------------------------------------------------
    # Bar mechanics
    # ------------------------------------------------------------------
    def _guard_state(
        self,
        guard: RiskLadder,
        equity: float,
        realised: Sequence[float],
        timestamp: int,
        transitions: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """Advance the risk ladder and record any state change."""
        state, reason = guard.observe(
            equity=equity,
            realised_pnls=realised,
            day=ms_to_datetime(timestamp).date(),
        )
        if reason:
            transitions.append(
                {
                    "timestamp": timestamp,
                    "at": ms_to_datetime(timestamp).isoformat(),
                    "state": state.value,
                    "equity": equity,
                    "reason": reason,
                }
            )
            _LOGGER.info("Backtest risk guard -> %s at %s: %s", state.value, timestamp, reason)
        return state.value, guard.size_multiplier

    def _infer(self, symbol: str, row: pd.Series) -> ModelInferenceResult:
        """Run all four heads on one bar."""
        features: pd.DataFrame = row[list(FEATURE_COLUMNS)].to_frame().T.astype(float)
        payload = InferencePayload(
            symbol=symbol,
            timestamp=int(row["timestamp"]),
            close=float(row["close"]),
            features=features,
            snapshot={key: float(row.get(key, default)) for key, default in _SNAPSHOT_KEYS.items()},
        )
        return self._ml.infer_sync(payload)

    def _decide(
        self,
        symbol: str,
        row: pd.Series,
        context: DecisionContext,
    ) -> DecisionResult:
        """Run inference plus the decision cascade for one bar."""
        return self._decisions.evaluate(self._infer(symbol, row), context)

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
        )
        # The barriers were computed off the decision-bar close; re-anchor them to
        # the actual fill so the geometry the models asked for is preserved.
        # Re-anchored from the *model's* activation distance, not a hardcoded
        # half of the take-profit. `_assemble` clamps trailing activation into
        # [TP*0.25, TP*0.95] and the live path arms against that value, so
        # substituting 0.5*TP here made the backtest trail on a different rule
        # from production - on a quarter of all exits.
        trailing_pct: float = signal.trailing_activation_pct
        if signal.action is TradeAction.LONG:
            position.take_profit = fill_price * (1.0 + signal.take_profit_pct)
            position.stop_loss = fill_price * (1.0 - signal.stop_loss_pct)
            position.trailing_trigger = fill_price * (1.0 + trailing_pct)
        else:
            position.take_profit = fill_price * (1.0 - signal.take_profit_pct)
            position.stop_loss = fill_price * (1.0 + signal.stop_loss_pct)
            position.trailing_trigger = fill_price * (1.0 - trailing_pct)

        position.fees_paid = fill_price * quantity * self._config.taker_fee
        position.last_funding_ms = int(row["timestamp"])
        # Simulated time, not wall clock. `Position.opened_at` defaults to
        # datetime.now(), while `_book_close` stamps `closed_at` from the bar -
        # so every backtested trade recorded a close *before* its open and a
        # negative holding period, making any duration or time-of-day analysis
        # (and the exported trades CSV) meaningless.
        position.opened_at = ms_to_datetime(int(row["timestamp"]))
        return position

    def _resolve_bar(self, position: Position, row: pd.Series) -> tuple[CloseReason, float] | None:
        """Decide whether this bar closes the position, pessimistically.

        The adverse side is evaluated first.  For a long, the *higher* of the
        stop and the liquidation price is reached first on the way down, so that
        level - and its reason - wins.  Only if neither adverse level was touched
        is the take-profit considered, and only then is the trailing stop allowed
        to ratchet on the bar's favourable extreme.
        """
        high: float = float(row["high"])
        low: float = float(row["low"])
        position.update_excursions(high, low)

        stop: float = position.effective_stop()
        liquidation: float = position.liquidation_price

        if position.is_long:
            adverse_level: float = max(stop, liquidation)
            adverse_hit: bool = low <= adverse_level
            target_hit: bool = high >= position.take_profit
        else:
            adverse_level = min(stop, liquidation) if liquidation > 0.0 else stop
            adverse_hit = high >= adverse_level
            target_hit = low <= position.take_profit

        if adverse_hit:
            liquidation_first: bool = (
                liquidation > 0.0
                and (liquidation >= stop if position.is_long else liquidation <= stop)
            )
            if liquidation_first:
                return CloseReason.LIQUIDATION, liquidation
            reason: CloseReason = (
                CloseReason.TRAILING_STOP if position.trailing_active else CloseReason.STOP_LOSS
            )
            return reason, stop

        if target_hit:
            return CloseReason.TAKE_PROFIT, position.take_profit

        # Neither pre-existing barrier was touched, so the trail may ratchet on
        # this bar's favourable extreme.  Having done so, the bar's *adverse*
        # extreme must be re-tested against the tightened stop: a bar that ran up
        # and then gave it all back would otherwise escape until the next bar, an
        # optimism that compounds across a backtest.
        advanced: float | None = position.advance_trailing(high if position.is_long else low)
        if advanced is None:
            return None
        breached: bool = low <= advanced if position.is_long else high >= advanced
        if breached:
            return CloseReason.TRAILING_STOP, advanced
        return None

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
        """Close a position and return the balance delta (excluding the entry fee).

        The entry fee was already deducted at fill time, so this returns
        ``gross - exit_fee - funding``.
        """
        penalty: float = self._config.slippage_bps / 10_000.0
        # Liquidation used to be exempt from slippage, which has it backwards:
        # a forced close during the move that triggered it is the fill most
        # likely to be worse than its trigger price, not better.
        if position.is_long:
            exit_price: float = raw_price * (1.0 - penalty)
        else:
            exit_price = raw_price * (1.0 + penalty)

        gross: float = (exit_price - position.entry_price) * position.quantity * position.direction
        exit_fee: float = exit_price * position.quantity * self._config.taker_fee
        position.fees_paid += exit_fee

        delta: float = gross - exit_fee - position.funding_paid
        if reason is CloseReason.LIQUIDATION:
            delta = max(delta, -position.margin)

        position.realized_pnl = delta
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
            # The bar-based ratios above annualise ~105k near-degenerate,
            # heavily autocorrelated per-bar observations per year as if they
            # were independent draws, when the effective sample is the trade
            # count. These report the same thing on the sample that actually
            # exists, so a headline Sharpe cannot be read without its n.
            "trades_per_year": float(pnls.size / years) if years > 0.0 else 0.0,
            "per_trade_sharpe": (
                float(pnls.mean() / pnls.std(ddof=1)) if pnls.size > 1 and pnls.std(ddof=1) > 0 else 0.0
            ),
            "annualised_sharpe_from_trades": (
                float(pnls.mean() / pnls.std(ddof=1) * math.sqrt(pnls.size / years))
                if pnls.size > 1 and pnls.std(ddof=1) > 0 and years > 0.0
                else 0.0
            ),
            "equity_curve_points": float(equity_values.size),
            "nonzero_return_bars": float(np.count_nonzero(self._period_returns(equity_values))),
            "calmar_ratio": (annualised / max_drawdown) if max_drawdown > 0.0 else 0.0,
            "total_fees": float(sum(float(trade.get("fees_paid", 0.0)) for trade in trades)),
            "total_funding": float(sum(float(trade.get("funding_paid", 0.0)) for trade in trades)),
            "liquidations": float(
                sum(1 for trade in trades if trade.get("close_reason") == CloseReason.LIQUIDATION.value)
            ),
        }

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
        # Downside deviation divides by the total number of periods, not by the
        # count of negative ones. Dividing by `downside.size` inflates the
        # deviation by sqrt(N_total / N_down) and deflates Sortino by the same
        # factor - which is why the audited run reported Sortino *below* Sharpe
        # on a distribution with a 2.4:1 win/loss payoff, where the opposite
        # relationship is the only coherent one.
        if not np.any(returns < 0.0):
            return 0.0
        downside: np.ndarray = np.minimum(returns, 0.0)
        deviation: float = float(np.sqrt(np.mean(np.square(downside))))
        if deviation <= 0.0:
            return 0.0
        return float(returns.mean() / deviation * math.sqrt(_BARS_PER_YEAR))
