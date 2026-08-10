"""System orchestrator - the entry point that wires Modules A through F together.

Two long-running things share one event loop:

1. **The trading loop**, driven by an ``AsyncIOScheduler`` cron job that fires on
   the 5-minute grid (with a small second-offset, because firing at exactly
   ``:00`` races Binance's own candle close and reliably drops the last bar).
2. **The uvicorn server** hosting the FastAPI panel.

Both are launched with ``asyncio.create_task`` and awaited via
``asyncio.gather``, and ``SIGINT``/``SIGTERM`` are trapped so shutdown is
orderly: stop accepting new signals, stop the scheduler, drain the audit queue,
stop the execution engine, close the exchange session, close the database.

Usage::

    python main.py                          # THE single-run path (see below)
    python main.py universe                 # print the screened pair list
    python main.py bootstrap                # backfill historical candles only
    python main.py train                    # build the dataset and fit all 4 heads
    python main.py backtest --candles 5000  # replay history through the pipeline
    python main.py cycle                    # run exactly one trading cycle

The single-run path
-------------------
``python main.py`` binds the panel immediately, then walks itself through setup
in the background:

1. If no universe has been saved, it parks in ``AWAITING_UNIVERSE`` and waits for
   the operator to tick pairs at ``/universe`` (the list is discovered live from
   Binance and screened for liquidity, spread, history and small-account fit).
2. Saving a selection triggers data collection for exactly those pairs.
3. Then the four models are trained on them.
4. The system settles at ``READY`` - and stops there.  Trading is armed by the
   operator from the panel, never automatically, and paper can be swapped for
   live at any time from the same control.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Final, Sequence

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import DecisionSettings, Settings, get_settings
from core.exceptions import KillSwitchEngaged, QuantSystemError
from core.logger import configure_logging, get_logger
from core.utils import utc_now, utc_now_ms
from module_a_data.db_handler import DatabaseHandler
from module_a_data.fetcher import BinanceDataFetcher
from module_a_data.models import MarketDataBundle
from module_a_data.pipeline import DataPipeline
from module_a_data.qc_validator import QCValidator
from module_a_data.universe import SymbolCandidate, UniverseManager
from module_b_features.features import FeatureService
from module_b_features.processor import DatasetProcessor, InferencePayload, ProcessedDataset
from module_c_ml.decision_engine import DecisionContext, DecisionEngine
from module_c_ml.ml_models import MLSubsystem
from module_c_ml.schemas import DecisionResult, ModelInferenceResult
from module_e_execution.backtester import BacktestReport, Backtester
from module_e_execution.executor import LiveExecutor
from module_e_execution.models import AccountState, ExecutionReport
from module_e_execution.paper_trader import PaperTrader
from module_e_execution.risk_guard import RiskGuard, SystemState
from module_f_panel import diagnostics
from module_f_panel.audit_engine import AuditEngine
from module_f_panel.setup_state import SetupProgress, SystemPhase
from module_f_panel.web_app import build_app

_LOGGER = get_logger("main")

_CYCLE_MINUTES: Final[str] = "0,5,10,15,20,25,30,35,40,45,50,55"

#: Rolling history cap for persisted per-cycle timing telemetry.
_MAX_STORED_CYCLE_TIMINGS: Final[int] = 500

#: Decision-cascade thresholds for the *diagnostic-only* relaxed backtest run
#: by ``TradingSystem._run_validation_backtest`` - never used for real trading.
#: The live ``DecisionSettings`` defaults (min_gate_confidence=0.55,
#: min_direction_given_trade_confidence=0.60, min_entry_probability=0.55, ...)
#: are intentionally strict and, combined with a single ~8-week out-of-sample
#: window, routinely leave the strict backtest with a single-digit trade
#: count - too small for win rate/profit factor/Sharpe to mean anything. This
#: loosened copy replays the *same* window to check whether the strategy's
#: edge is visible at all with a larger sample, without ever touching the
#: thresholds that gate real orders.
_RELAXED_DECISION_SETTINGS: Final[DecisionSettings] = DecisionSettings(
    min_gate_confidence=0.50,
    min_direction_given_trade_confidence=0.52,
    max_no_trade_probability=0.50,
    min_entry_probability=0.50,
    min_reward_risk_ratio=1.0,
)


def _diagnostic_backtest_window(
    *, oos_validation_rows: int, diagnostic_backtest_bars: int, symbol_count: int
) -> tuple[int, dict[str, Any]]:
    """Compute the diagnostic backtest's per-symbol replay depth and the
    genuinely-out-of-sample-vs-in-sample split for that window.

    Pure and side-effect free (no DB/network) so it is directly unit
    testable - see ``TradingSystem._run_validation_backtest`` for the caller
    and its docstring for why this split matters and must never be hidden.

    Returns ``(diagnostic_bars_per_symbol, oos_disclosure)``.
    """
    if symbol_count <= 0:
        raise ValueError("symbol_count must be positive")
    # Genuinely out-of-sample tail, per symbol (the model's own
    # train/validation split boundary), spread evenly like validation_index
    # itself is spread across the combined multi-symbol dataset.
    oos_bars_per_symbol: int = -(-oos_validation_rows // symbol_count)  # ceil
    # Target diagnostic replay length, per symbol - decoupled from the OOS
    # tail above, see MLSettings.diagnostic_backtest_bars.
    diagnostic_bars_per_symbol: int = -(-diagnostic_backtest_bars // symbol_count)  # ceil

    out_of_sample_bars_per_symbol: int = min(oos_bars_per_symbol, diagnostic_bars_per_symbol)
    in_sample_bars_per_symbol: int = max(
        diagnostic_bars_per_symbol - out_of_sample_bars_per_symbol, 0
    )
    oos_fraction: float = (
        out_of_sample_bars_per_symbol / diagnostic_bars_per_symbol
        if diagnostic_bars_per_symbol > 0
        else 0.0
    )
    oos_disclosure: dict[str, Any] = {
        "note": (
            "Only the out_of_sample_bars_per_symbol portion of this replay window "
            "was never seen in training (the model's own validation_fraction tail, "
            "purged by purge_bars). The remaining in_sample_bars_per_symbol bars "
            "overlap the training set and will tend to read better than genuine "
            "live performance - treat this backtest as a blend, not a clean "
            "holdout, whenever oos_fraction < 1.0."
        ),
        "diagnostic_backtest_bars_per_symbol": diagnostic_bars_per_symbol,
        "out_of_sample_bars_per_symbol": out_of_sample_bars_per_symbol,
        "in_sample_bars_per_symbol": in_sample_bars_per_symbol,
        "oos_fraction": oos_fraction,
    }
    return diagnostic_bars_per_symbol, oos_disclosure


def _headline_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Small, fixed-size summary of a training report for frequent polling.

    ``report`` (from ``MLSubsystem.train_all``) now carries full per-head
    metrics - confusion matrices, threshold sweeps, feature importance - which
    is exactly what the ML diagnostic report needs but is far too large to
    push through the ``/api/setup/status`` endpoint on every poll. The full
    report is always available via the diagnostic report export instead.
    """
    headline: dict[str, Any] = {}
    for head, metrics in report.items():
        if "error" in metrics:
            headline[head] = {"error": metrics["error"]}
            continue
        keys = {
            "direction": ("accuracy", "balanced_accuracy", "log_loss"),
            "entry": ("precision", "recall", "roc_auc"),
            "risk": ("mae", "r2"),
        }.get(head)
        if keys is not None:
            headline[head] = {key: metrics[key] for key in keys if key in metrics}
        elif head == "exit":
            headline[head] = {
                target: {"mae": target_metrics.get("mae")}
                for target, target_metrics in metrics.items()
                if isinstance(target_metrics, dict)
            }
        else:  # pragma: no cover - defensive default
            headline[head] = {}
    return headline

#: Either concrete engine satisfies the same interface; the loop never branches.
ExecutionEngine = PaperTrader | LiveExecutor


class TradingSystem:
    """Owns every component and drives the 5-minute decision cycle."""

    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings

        # --- Module A --------------------------------------------------
        self.database: DatabaseHandler = DatabaseHandler(settings)
        self.fetcher: BinanceDataFetcher = BinanceDataFetcher(settings)
        self.validator: QCValidator = QCValidator(settings)
        self.pipeline: DataPipeline = DataPipeline(
            settings, self.fetcher, self.validator, self.database
        )
        self.universe: UniverseManager = UniverseManager(settings, self.fetcher, self.database)

        # --- Module B --------------------------------------------------
        self.features: FeatureService = FeatureService(settings)
        self.processor: DatasetProcessor = DatasetProcessor(
            settings, self.database, self.features
        )

        # --- Modules C & D ---------------------------------------------
        self.ml: MLSubsystem = MLSubsystem(settings)
        self.decisions: DecisionEngine = DecisionEngine(settings)

        # --- Module E ---------------------------------------------------
        self.risk_guard: RiskGuard = RiskGuard(settings, self.database)
        self.engine: ExecutionEngine = self._build_engine(settings.trading_mode)

        # --- Module F ---------------------------------------------------
        self.audit: AuditEngine = AuditEngine(self.database)

        # --- Runtime state -----------------------------------------------
        self.scheduler: AsyncIOScheduler = AsyncIOScheduler(timezone="UTC")
        self.trading_enabled: bool = False  # armed by the operator, never at boot
        self.trading_mode: str = settings.trading_mode
        self._cycle_lock: asyncio.Lock = asyncio.Lock()
        self._shutdown: asyncio.Event = asyncio.Event()

        self.phase: SystemPhase = SystemPhase.STARTING
        self.progress: SetupProgress = SetupProgress()
        self.active_symbols: list[str] = []
        self._setup_task: asyncio.Task[None] | None = None
        self._setup_lock: asyncio.Lock = asyncio.Lock()

        self.last_cycle_at: datetime | None = None
        self.last_cycle_duration_s: float = 0.0
        self.last_cycle_symbols_ok: int = 0
        self.last_cycle_decisions: int = 0
        self.last_cycle_executed: int = 0
        self.last_cycle_error: str = ""
        self.cycles_completed: int = 0
        #: Per-stage wall-clock timings (seconds) for the most recently
        #: completed cycle - ingestion, features, prediction, decision,
        #: execution and database/audit, plus the overall total. Surfaced on
        #: the ML diagnostic report's Pipeline Timing section.
        self.last_cycle_timings: dict[str, float] = {}
        #: Cycles skipped because the previous one was still running when the
        #: next 5m slot fired - direct evidence for whether cycles are
        #: overrunning their scheduling interval.
        self.cycles_skipped_overlap: int = 0
        #: Run ID of the most recently generated ML diagnostic report, if any.
        self.latest_ml_report_id: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_engine(self, mode: str) -> ExecutionEngine:
        """Instantiate the execution engine for ``mode``."""
        if mode == "live":
            _LOGGER.warning("LIVE TRADING MODE - real funds are at risk")
            return LiveExecutor(self.settings, self.fetcher, self.database, self.risk_guard)
        return PaperTrader(self.settings, self.fetcher, self.database, self.risk_guard)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def startup(self) -> None:
        """Initialise every component in dependency order.

        Deliberately fast and network-tolerant: the panel must come up even when
        the exchange is unreachable, so the operator can see *why*.  The slow
        work (backfill, training) happens afterwards in a background task.
        """
        _LOGGER.info("=" * 70)
        _LOGGER.info("AI Quant Trading System - Binance USDT-M Perpetuals (5m)")
        _LOGGER.info("=" * 70)

        await self.database.initialize()
        await self.audit.start()
        await self.risk_guard.load()
        await self.fetcher.load_markets()

        loaded: dict[str, bool] = self.ml.load_all()
        if not all(loaded.values()):
            _LOGGER.info(
                "Model artifacts missing (%s) - they will be trained during setup",
                ", ".join(name for name, ok in loaded.items() if not ok),
            )

        self.active_symbols = await self.universe.get_selection()
        self.risk_guard.register_halt_callback(self.engine.emergency_flatten)
        await self.engine.start()

        if self.risk_guard.is_halted:
            _LOGGER.critical(
                "Risk Guard is RED on startup (%s) - no trades until it is reset",
                self.risk_guard.halt_reason,
            )

    # ------------------------------------------------------------------
    # Automated setup pipeline (the "single run" path)
    # ------------------------------------------------------------------
    def launch_setup(self, force_retrain: bool = False) -> bool:
        """Start the setup pipeline in the background.

        Returns:
            ``False`` when a setup run is already in flight, ``True`` otherwise.
        """
        if self._setup_task is not None and not self._setup_task.done():
            _LOGGER.info("Setup is already running - ignoring the duplicate request")
            return False
        self._setup_task = asyncio.create_task(
            self.run_setup(force_retrain), name="setup-pipeline"
        )
        return True

    async def run_setup(self, force_retrain: bool = False) -> None:
        """Collect data and train the models, publishing progress throughout.

        Idempotent and safe to re-run: the backfill only fetches the missing
        tail, and training is skipped when every artifact is already present
        and the universe has not changed.
        """
        async with self._setup_lock:
            if self.phase.is_trading:
                _LOGGER.warning("Setup requested while trading is armed - disarming first")
                await self.stop_trading()

            try:
                symbols: list[str] = await self.universe.get_selection()
                if not symbols:
                    self.phase = SystemPhase.AWAITING_UNIVERSE
                    self.progress.begin(
                        SystemPhase.AWAITING_UNIVERSE,
                        "awaiting universe",
                        "Pick the perpetual futures pairs to trade in the web panel.",
                    )
                    _LOGGER.warning(
                        "No universe selected. Open http://%s:%d/universe and choose the pairs.",
                        self.settings.web.host,
                        self.settings.web.port,
                    )
                    return

                self.active_symbols = symbols
                await self._setup_collect(symbols)
                await self._setup_train(symbols, force_retrain)

                self.phase = SystemPhase.READY
                self.progress.finish(
                    SystemPhase.READY,
                    "ready",
                    f"{len(symbols)} symbol(s) prepared. Start paper trading when you are ready.",
                )
                _LOGGER.info("=" * 70)
                _LOGGER.info("SETUP COMPLETE - system is READY")
                _LOGGER.info("Arm paper trading from the panel or POST /api/trading/start")
                _LOGGER.info("=" * 70)

                if self.settings.autostart_paper_trading:
                    await self.start_trading("paper")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.progress.fail(str(error))
                self.phase = SystemPhase.SETUP_FAILED
                _LOGGER.error("Setup pipeline failed: %s", error, exc_info=True)
                await self.audit.log_system_event(
                    "setup", "ERROR", f"setup failed: {error}", rule="SETUP_FAILURE"
                )

    async def _setup_collect(self, symbols: list[str]) -> None:
        """Stage 1 - backfill historical candles and derivatives/positioning history."""
        self.phase = SystemPhase.COLLECTING_DATA
        self.progress.begin(
            SystemPhase.COLLECTING_DATA,
            "collecting data",
            f"Backfilling {self.settings.data.history_bootstrap_candles} candles "
            f"for {len(symbols)} symbol(s)",
        )
        self.progress.advance(0, len(symbols))

        def report(symbol: str, done: int, total: int) -> None:
            self.progress.advance(done, total, f"{symbol} ({done}/{total})")

        written: dict[str, int] = await self.pipeline.bootstrap_history(symbols, progress=report)
        self.progress.data_summary = written

        stored: dict[str, int] = {
            symbol: await self.database.candle_count(symbol) for symbol in symbols
        }
        usable: list[str] = [symbol for symbol, count in stored.items() if count >= 500]
        if not usable:
            raise QuantSystemError(
                "no symbol has enough stored history to train on - check exchange connectivity"
            )
        if len(usable) < len(symbols):
            _LOGGER.warning(
                "%d symbol(s) have too little history and will be skipped: %s",
                len(symbols) - len(usable),
                ", ".join(symbol for symbol in symbols if symbol not in usable),
            )
        _LOGGER.info(
            "Data collection complete: %d candles written, %d symbol(s) usable",
            sum(written.values()),
            len(usable),
        )

        # Backfill funding-rate/open-interest/positioning history so the
        # micro-structure/derivatives features are not stuck at their neutral
        # default for the whole training window - see
        # `DataPipeline.backfill_futures_metrics` for why this is a separate
        # pass from the candle backfill above (order book has no historical
        # endpoint at all; the rest are capped by Binance's own retention).
        self.progress.advance(0, len(usable), "backfilling derivatives/positioning history")
        futures_written: dict[str, int] = await self.pipeline.backfill_futures_metrics(
            usable, progress=report
        )
        _LOGGER.info(
            "Futures-metrics backfill complete: %d row(s) across %d symbol(s)",
            sum(futures_written.values()),
            len(futures_written),
        )

    async def _run_validation_backtest(
        self, dataset: ProcessedDataset
    ) -> tuple[BacktestReport | None, BacktestReport | None]:
        """Replay the out-of-sample validation window through the full pipeline.

        The ML diagnostic report's trading-level fields (win rate, profit
        factor, expectancy, max drawdown, Sharpe) were previously always
        ``NOT_AVAILABLE`` because nothing ever ran a backtest and passed it
        in. The replay window's length is controlled by
        ``MLSettings.diagnostic_backtest_bars`` (default: a full year of 5m
        bars, 105_120) - a deliberately separate, explicitly-named setting
        from ``validation_fraction`` (see its docstring). Decoupling the two
        lets the diagnostic backtest span a full year for a large enough
        sample size, independent of how large the model's own OOS validation
        tail happens to be.

        Honesty requirement - READ BEFORE TRUSTING THIS NUMBER: only the
        portion of the replay window that falls inside the model's actual
        validation split (``validation_fraction``'s tail, purged by
        ``purge_bars``) is genuinely out-of-sample. When
        ``diagnostic_backtest_bars`` exceeds that validation tail's bar
        count (e.g. a full-year window against ``validation_fraction=0.2``
        over a 1.5-year collection, whose OOS tail is only ~0.3 years), the
        remainder of the replay overlaps rows the model was actually trained
        on - an in-sample replay, not a real holdout test, that will tend to
        read *better* than genuine live performance. This function computes
        and returns that split explicitly (see ``oos_disclosure`` below) so
        it is measured and logged on every run, not silently implied away by
        a big, good-looking window. Treat metrics from the in-sample portion
        as inflated; only the OOS-tail portion is a trustworthy estimate.

        Caveat, stated honestly rather than hidden: when the Direction/Entry
        heads' production isotonic calibrators are wired in
        (``BaseModelHead._fit_production_calibrator``), they are fit on this
        same validation block. This is therefore the best available
        approximation of a clean holdout, not a third, fully untouched split -
        treat the resulting numbers as directionally informative rather than
        a certified live-performance estimate.

        Returns ``(strict, relaxed)``: ``strict`` replays with the live
        ``DecisionSettings`` and is the number that matters operationally. A
        highly selective cascade (min direction confidence, entry
        probability, reward/risk floor, ...) can still leave it with a
        single-digit trade count, which is too small for its own
        win-rate/profit-factor/Sharpe to mean anything (see
        ``module_f_panel.diagnostics._backtest_reliability``). ``relaxed``
        re-runs the *same* window with those thresholds loosened (never the
        live config) purely to see whether the strategy's edge is
        directionally visible at all with a larger trade count - it is a
        diagnostic reference only, never a performance estimate to trade on.
        Both reports carry identical ``oos_disclosure`` metadata since both
        replay the same window.

        Never raises: a backtest failure must not block training or the
        diagnostic report it enriches.
        """
        if not dataset.symbols:
            return None, None
        try:
            _, validation_index = dataset.train_validation_split(
                self.settings.ml.validation_fraction, self.settings.ml.purge_bars
            )
            if len(validation_index) == 0:
                return None, None

            symbols: list[str] = list(dataset.symbols)
            warmup_padding: int = self.features.engineer.minimum_rows()
            diagnostic_bars_per_symbol, oos_disclosure = _diagnostic_backtest_window(
                oos_validation_rows=len(validation_index),
                diagnostic_backtest_bars=self.settings.ml.diagnostic_backtest_bars,
                symbol_count=len(symbols),
            )
            max_candles: int = diagnostic_bars_per_symbol + warmup_padding
            oos_fraction: float = oos_disclosure["oos_fraction"]
            if oos_fraction < 1.0:
                _LOGGER.warning(
                    "Diagnostic backtest window (%d bars/symbol) exceeds the genuinely "
                    "out-of-sample validation tail (%d bars/symbol) - only %.0f%% of the "
                    "replay is true holdout; the rest overlaps training data.",
                    diagnostic_bars_per_symbol,
                    oos_disclosure["out_of_sample_bars_per_symbol"],
                    oos_fraction * 100,
                )

            backtester = Backtester(
                self.settings, self.database, self.features, self.ml, self.decisions
            )
            strict: BacktestReport = await backtester.run(
                symbols=symbols, max_candles=max_candles, warmup_bars=warmup_padding
            )
            strict.oos_disclosure = oos_disclosure

            relaxed: BacktestReport | None = None
            try:
                relaxed_settings: Settings = self.settings.model_copy(
                    update={"decision": _RELAXED_DECISION_SETTINGS}
                )
                relaxed_engine = DecisionEngine(relaxed_settings)
                relaxed_backtester = Backtester(
                    relaxed_settings, self.database, self.features, self.ml, relaxed_engine
                )
                relaxed = await relaxed_backtester.run(
                    symbols=symbols, max_candles=max_candles, warmup_bars=warmup_padding
                )
                relaxed.oos_disclosure = oos_disclosure
            except Exception as error:  # noqa: BLE001 - diagnostic-only pass, never fatal
                _LOGGER.error("Relaxed diagnostic backtest failed: %s", error, exc_info=True)

            return strict, relaxed
        except Exception as error:  # noqa: BLE001 - a backtest failure must not block training
            _LOGGER.error("Validation backtest failed: %s", error, exc_info=True)
            return None, None

    async def _setup_train(self, symbols: list[str], force_retrain: bool) -> None:
        """Stage 2 - build the dataset and fit the four heads."""
        if not self.settings.auto_train and not force_retrain:
            _LOGGER.info("Automatic training is disabled - keeping the existing artifacts")
            return

        trained_on: list[str] = await self._trained_universe()
        artifacts_ready: bool = self.ml.all_loaded
        universe_changed: bool = sorted(trained_on) != sorted(symbols)

        if artifacts_ready and not universe_changed and not force_retrain:
            _LOGGER.info("Models are current for this universe - skipping training")
            self.progress.training_summary = {"skipped": "models already trained for this universe"}
            return

        reason: str = (
            "forced" if force_retrain
            else "no artifacts" if not artifacts_ready
            else "universe changed"
        )
        self.phase = SystemPhase.TRAINING
        self.progress.begin(SystemPhase.TRAINING, "building dataset", f"retraining ({reason})")
        self.progress.advance(0, 4, "engineering features and labels")

        # Idempotent - only fetches the missing tail - but must run again here
        # regardless of what `_setup_collect` already did upstream, so that any
        # future call path that reaches `_setup_train` without first going
        # through `_setup_collect` still gets funding_rate/open_interest/
        # long_short_ratio/taker_buy_sell_ratio history before training reads it.
        await self.pipeline.backfill_futures_metrics(symbols)

        dataset: ProcessedDataset = await self.processor.build_training_dataset(symbols=symbols)
        if dataset.is_empty:
            raise QuantSystemError(
                "training dataset is empty after feature/label generation - "
                "not enough clean history"
            )

        self.progress.set_step("fitting models")
        self.progress.advance(1, 4, f"fitting 4 models on {len(dataset)} rows")
        _LOGGER.info("Training on %d rows | %s", len(dataset), dataset.class_distribution())
        report: dict[str, Any] = await self.ml.train_all(dataset)
        self.progress.advance(2, 4, "models fitted")

        failed: list[str] = [head for head, metrics in report.items() if "error" in metrics]
        if failed:
            raise QuantSystemError(f"model training failed for: {', '.join(failed)}")

        self.progress.training_summary = {
            "rows": len(dataset),
            "distribution": dataset.class_distribution(),
            "metrics": _headline_metrics(report),
        }
        await self.database.set_state(
            "trained_universe",
            {"symbols": symbols, "trained_ms": utc_now_ms(), "rows": len(dataset)},
        )
        _LOGGER.info("Training complete: %s", _headline_metrics(report))

        run_id: str = str(uuid.uuid4())
        self.progress.set_step("running validation backtest", "replaying out-of-sample bars")
        backtest_report: BacktestReport | None
        relaxed_backtest_report: BacktestReport | None
        backtest_report, relaxed_backtest_report = await self._run_validation_backtest(dataset)
        self.progress.advance(3, 4, "validation backtest complete")
        try:
            self.progress.set_step(
                "generating diagnostic report", "walk-forward validation and feature analytics"
            )
            diagnostic_report: dict[str, Any] = await diagnostics.build_report(
                settings=self.settings,
                database=self.database,
                ml=self.ml,
                dataset=dataset,
                run_id=run_id,
                backtest=backtest_report,
                relaxed_backtest=relaxed_backtest_report,
            )
            self.latest_ml_report_id = run_id
            self.progress.advance(4, 4, "diagnostic report ready")
            _LOGGER.info(
                "ML diagnostic report %s: status=%s",
                run_id,
                diagnostic_report["ai_summary"]["overall_status"],
            )
        except Exception as error:  # pragma: no cover - reporting must not break training
            _LOGGER.error("Could not build the ML diagnostic report: %s", error, exc_info=True)

    async def _trained_universe(self) -> list[str]:
        """Universe the current artifacts were trained on (empty when unknown)."""
        stored: dict[str, Any] | None = await self.database.get_state("trained_universe")
        if not stored or not isinstance(stored.get("symbols"), list):
            return []
        return [str(item) for item in stored["symbols"]]

    # ------------------------------------------------------------------
    # Arming and disarming trading
    # ------------------------------------------------------------------
    async def start_trading(self, mode: str) -> dict[str, Any]:
        """Arm paper or live trading.

        Refuses unless setup has completed, the Risk Guard is clear, and - for
        live - every model is a genuinely trained artifact rather than a
        heuristic fallback.
        """
        if mode not in {"paper", "live"}:
            raise ValueError("mode must be 'paper' or 'live'")
        if not self.phase.can_arm_trading:
            raise ValueError(
                f"cannot start trading from phase {self.phase.value} - setup must finish first"
            )
        if self.risk_guard.is_halted:
            raise ValueError(
                f"Risk Guard is RED ({self.risk_guard.halt_reason}) - reset it before trading"
            )
        if mode == "live" and not self.ml.all_loaded:
            raise ValueError(
                "live trading requires all four trained models; heuristic fallbacks are blocked"
            )
        if not self.active_symbols:
            raise ValueError("no universe selected")

        if mode != self.trading_mode:
            await self._switch_engine(mode)

        self.trading_enabled = True
        self.phase = SystemPhase.PAPER_TRADING if mode == "paper" else SystemPhase.LIVE_TRADING
        _LOGGER.warning("TRADING ARMED in %s mode on %d symbol(s)", mode.upper(), len(self.active_symbols))
        await self.audit.log_system_event(
            "control", "NO_TRADE", f"{mode} trading armed by the operator", rule="OPERATOR_ARM"
        )
        return {
            "phase": self.phase.value,
            "trading_mode": self.trading_mode,
            "trading_enabled": True,
            "symbols": len(self.active_symbols),
        }

    async def stop_trading(self, flatten: bool = False) -> dict[str, Any]:
        """Disarm trading.

        By default open positions keep being managed to their own TP/SL - pulling
        the strategy does not mean abandoning live risk.  Pass ``flatten=True``
        to close everything at market instead.
        """
        was_trading: bool = self.phase.is_trading
        self.trading_enabled = False
        if flatten:
            await self.engine.emergency_flatten("operator stopped trading")
        if was_trading:
            self.phase = SystemPhase.READY
            _LOGGER.warning(
                "TRADING DISARMED (open positions %s)",
                "flattened" if flatten else "still managed",
            )
            await self.audit.log_system_event(
                "control", "BLOCKED", "trading disarmed by the operator", rule="OPERATOR_DISARM"
            )
        return {
            "phase": self.phase.value,
            "trading_enabled": False,
            "flattened": flatten,
            "open_positions": len(self.engine.positions),
        }

    async def _switch_engine(self, mode: str) -> None:
        """Replace the execution engine, flattening the outgoing one first.

        A paper position has no counterpart on the exchange and vice versa, so
        carrying one across the switch would leave it unmanaged forever.
        """
        _LOGGER.warning("Switching execution engine: %s -> %s", self.trading_mode, mode)
        await self.engine.emergency_flatten(f"mode switch to {mode}")
        await self.engine.stop()

        self.trading_mode = mode
        self.engine = self._build_engine(mode)
        self.risk_guard.register_halt_callback(self.engine.emergency_flatten)
        await self.engine.start()

    async def shutdown(self) -> None:
        """Tear everything down in reverse dependency order."""
        _LOGGER.info("Shutting down...")
        self.trading_enabled = False

        if self._setup_task is not None and not self._setup_task.done():
            self._setup_task.cancel()
            try:
                await self._setup_task
            except (asyncio.CancelledError, Exception):  # noqa: B014 - shutdown best effort
                pass

        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

        try:
            await self.engine.stop()
        except Exception as error:  # pragma: no cover - shutdown best effort
            _LOGGER.error("Engine shutdown failed: %s", error)

        try:
            await self.audit.stop(drain=True)
        except Exception as error:  # pragma: no cover
            _LOGGER.error("Audit shutdown failed: %s", error)

        try:
            await self.features.shutdown()
        except Exception as error:  # pragma: no cover
            _LOGGER.error("Feature service shutdown failed: %s", error)

        await self.fetcher.close()
        await self.database.close()
        _LOGGER.info("Shutdown complete")

    def request_shutdown(self) -> None:
        """Signal handler hook - unblocks :meth:`run_forever`."""
        _LOGGER.warning("Shutdown signal received")
        self._shutdown.set()

    # ------------------------------------------------------------------
    # The 5-minute trading cycle
    # ------------------------------------------------------------------
    async def trading_cycle(self) -> dict[str, Any]:
        """Run one full A -> F pass.

        Overlapping runs are impossible: the lock is acquired without waiting, so
        a cycle that overruns its slot is skipped rather than queued behind the
        previous one.
        """
        if self._cycle_lock.locked():
            self.cycles_skipped_overlap += 1
            _LOGGER.warning(
                "Previous cycle is still running - skipping this slot (%d skipped so far)",
                self.cycles_skipped_overlap,
            )
            return {"skipped": True}

        async with self._cycle_lock:
            cycle_id: str = str(uuid.uuid4())
            started: float = utc_now_ms() / 1_000.0
            _LOGGER.info("--- cycle %s start ---", cycle_id[:8])

            try:
                summary: dict[str, Any] = await self._run_cycle(cycle_id)
                self.last_cycle_error = ""
                self.cycles_completed += 1
            except QuantSystemError as error:
                self.last_cycle_error = str(error)
                _LOGGER.error("Cycle %s failed: %s", cycle_id[:8], error)
                await self.audit.log_system_event(
                    cycle_id, "ERROR", f"cycle failed: {error}", rule="CYCLE_FAILURE"
                )
                summary = {"error": str(error)}
            except Exception as error:  # pragma: no cover - the loop must never die
                self.last_cycle_error = str(error)
                _LOGGER.error("Cycle %s crashed: %s", cycle_id[:8], error, exc_info=True)
                await self.risk_guard.record_api_error(f"cycle crash: {error}")
                summary = {"error": str(error)}

            self.last_cycle_at = utc_now()
            self.last_cycle_duration_s = utc_now_ms() / 1_000.0 - started
            _LOGGER.info(
                "--- cycle %s done in %.2fs ---", cycle_id[:8], self.last_cycle_duration_s
            )
            if self.last_cycle_timings:
                await self._persist_cycle_timings(cycle_id, self.last_cycle_timings)
            return summary

    async def _run_cycle(self, cycle_id: str) -> dict[str, Any]:
        """The A -> F pipeline for one 5-minute bar.

        Ingestion (Module A) runs whenever a universe exists, even while trading
        is disarmed: keeping the candle history warm means that arming paper
        trading takes effect on the very next bar instead of waiting for the slow
        features to spin up.  Modules B-F only run once trading is armed.
        """
        if not self.active_symbols:
            _LOGGER.debug("Cycle skipped: no universe selected")
            return {"skipped": "no universe"}

        timings: dict[str, float] = {}
        cycle_started: float = time.perf_counter()

        def _mark(stage: str, since: float) -> float:
            """Record ``stage``'s duration and return a fresh checkpoint."""
            now: float = time.perf_counter()
            timings[stage] = now - since
            return now

        checkpoint: float = cycle_started

        # --- A: ingestion + QC -------------------------------------------
        bundles: dict[str, MarketDataBundle] = await self.pipeline.run_cycle(self.active_symbols)
        checkpoint = _mark("ingestion_and_qc", checkpoint)
        self.last_cycle_symbols_ok = len(bundles)
        if not bundles:
            await self.audit.log_system_event(
                cycle_id, "NO_TRADE", "no symbol passed QC this cycle", rule="QC_ALL_FAILED"
            )
            timings["total"] = time.perf_counter() - cycle_started
            self.last_cycle_timings = timings
            return {"symbols": 0}

        if not self.phase.is_trading or not self.trading_enabled:
            _LOGGER.info(
                "Data refreshed for %d symbol(s); trading is disarmed (phase=%s)",
                len(bundles),
                self.phase.value,
            )
            timings["total"] = time.perf_counter() - cycle_started
            self.last_cycle_timings = timings
            return {"symbols": len(bundles), "trading": False, "phase": self.phase.value}

        # --- Equity refresh + Risk Guard evaluation -----------------------
        account: AccountState = await self._account_state()
        state: SystemState = await self.risk_guard.update_equity(account.equity)
        checkpoint = _mark("risk_guard", checkpoint)

        # --- B: features ---------------------------------------------------
        payloads: dict[str, InferencePayload] = await self.processor.build_inference_payloads(
            list(bundles)
        )
        checkpoint = _mark("feature_generation", checkpoint)
        if not payloads:
            await self.audit.log_system_event(
                cycle_id, "NO_TRADE", "no symbol produced a warm feature row", rule="FEATURES_COLD"
            )
            timings["total"] = time.perf_counter() - cycle_started
            self.last_cycle_timings = timings
            return {"symbols": len(bundles), "payloads": 0}

        # --- C: inference ---------------------------------------------------
        inferences: dict[str, ModelInferenceResult] = await self.ml.infer_many(
            list(payloads.values())
        )
        checkpoint = _mark("prediction", checkpoint)

        # --- D: decisions ----------------------------------------------------
        context = DecisionContext(
            risk_guard_state=state.value,
            trading_enabled=self.trading_enabled and self.risk_guard.can_trade,
            trading_mode=self.trading_mode,
            open_positions=len(self.engine.positions),
            open_symbols=self.engine.open_symbols,
            equity=account.equity,
            size_multiplier=self.risk_guard.size_multiplier,
        )
        decisions: list[DecisionResult] = self.decisions.evaluate_many(
            list(inferences.values()), context
        )
        self.last_cycle_decisions = len(decisions)
        checkpoint = _mark("decision", checkpoint)

        # --- E: execution -----------------------------------------------------
        executed: int = 0
        for decision in decisions:
            execution_detail: dict[str, Any] | None = None
            if decision.is_executable and decision.signal is not None:
                execution_detail = await self._execute(decision)
                if execution_detail.get("accepted"):
                    executed += 1
            # --- F: audit (every decision, executed or not) -------------------
            await self.audit.log_decision(
                decision,
                cycle_id=cycle_id,
                risk_guard_state=state.value,
                execution_detail=execution_detail,
            )
        self.last_cycle_executed = executed
        checkpoint = _mark("execution_and_audit", checkpoint)

        # --- Post-cycle bookkeeping ---------------------------------------------
        await self._record_equity()
        checkpoint = _mark("database", checkpoint)

        timings["total"] = time.perf_counter() - cycle_started
        self.last_cycle_timings = timings

        _LOGGER.info(
            "Cycle summary: %d symbols | %d decisions | %d executed | guard=%s | equity=%.2f | "
            "timings=%s",
            len(bundles),
            len(decisions),
            executed,
            state.value,
            account.equity,
            {key: round(value, 3) for key, value in timings.items()},
        )
        return {
            "cycle_id": cycle_id,
            "symbols": len(bundles),
            "decisions": len(decisions),
            "executed": executed,
            "risk_guard": state.value,
            "equity": account.equity,
            "timings": timings,
        }

    async def _execute(self, decision: DecisionResult) -> dict[str, Any]:
        """Hand a signal to the execution engine, isolating its failures."""
        assert decision.signal is not None  # guaranteed by `is_executable`
        try:
            report: ExecutionReport = await self.engine.execute(decision.signal)
            return report.to_dict()
        except KillSwitchEngaged as error:
            _LOGGER.warning("Execution blocked by the kill switch: %s", error)
            return {"accepted": False, "error": "kill switch engaged"}
        except QuantSystemError as error:
            _LOGGER.error("Execution failed for %s: %s", decision.symbol, error)
            return {"accepted": False, "error": str(error)}

    async def _account_state(self) -> AccountState:
        """Fetch the engine's account state, degrading gracefully on failure."""
        try:
            return await self.engine.fetch_account_state()
        except Exception as error:  # pragma: no cover - degraded-mode path
            _LOGGER.error("Account state unavailable: %s", error)
            await self.risk_guard.record_api_error(f"account state: {error}")
            return AccountState(
                mode=self.trading_mode,
                balance=self.risk_guard.current_equity,
                equity=self.risk_guard.current_equity,
                unrealized_pnl=0.0,
                used_margin=0.0,
                open_positions=len(self.engine.positions),
                timestamp_ms=utc_now_ms(),
            )

    async def _record_equity(self) -> None:
        """Append an equity-curve point (paper engine keeps its own bookkeeping)."""
        try:
            if isinstance(self.engine, PaperTrader):
                await self.engine.record_equity_point()
                return
            account: AccountState = await self._account_state()
            peak: float = max(self.risk_guard.peak_equity, account.equity)
            drawdown: float = 0.0 if peak <= 0.0 else max(0.0, (peak - account.equity) / peak)
            await self.database.insert_equity_point(
                {
                    "mode": self.trading_mode,
                    "timestamp": account.timestamp_ms,
                    "balance": account.balance,
                    "equity": account.equity,
                    "unrealized_pnl": account.unrealized_pnl,
                    "open_positions": account.open_positions,
                    "drawdown_pct": drawdown,
                }
            )
        except Exception as error:  # pragma: no cover
            _LOGGER.error("Could not record the equity point: %s", error)

    async def _persist_cycle_timings(self, cycle_id: str, timings: dict[str, float]) -> None:
        """Append this cycle's per-stage timings to a bounded rolling history.

        Feeds the ML diagnostic report's Pipeline Timing section with real,
        measured durations across many cycles rather than just the latest one.
        Best-effort: a persistence failure must not affect trading.
        """
        try:
            stored: dict[str, Any] | None = await self.database.get_state("cycle_timings_history")
            history: list[Any] = list((stored or {}).get("records", []))
            history.append({"cycle_id": cycle_id, "at": utc_now_ms(), **timings})
            history = history[-_MAX_STORED_CYCLE_TIMINGS:]
            await self.database.set_state("cycle_timings_history", {"records": history})
        except Exception as error:  # pragma: no cover - telemetry must not break the loop
            _LOGGER.error("Could not persist cycle timings: %s", error)

    # ------------------------------------------------------------------
    # SystemController implementation (consumed by the web panel)
    # ------------------------------------------------------------------
    async def status_snapshot(self) -> dict[str, Any]:
        """Everything the dashboard renders, in one call."""
        account: AccountState = await self._account_state()
        positions: list[dict[str, Any]] = [
            position.to_view() for position in self.engine.positions.values()
        ]
        return {
            "app": self.settings.app_name,
            "phase": self.phase.value,
            "setup": self.progress.to_dict(),
            "trading_mode": self.trading_mode,
            "trading_enabled": self.trading_enabled,
            "universe": {
                "symbols": self.active_symbols,
                "count": len(self.active_symbols),
                "selected": bool(self.active_symbols),
            },
            "risk_guard": self.risk_guard.snapshot(),
            "account": account.to_dict(),
            "positions": positions,
            "last_cycle_at": (
                self.last_cycle_at.isoformat(timespec="seconds") if self.last_cycle_at else ""
            ),
            "last_cycle_duration_s": self.last_cycle_duration_s,
            "last_cycle_symbols_ok": self.last_cycle_symbols_ok,
            "last_cycle_decisions": self.last_cycle_decisions,
            "last_cycle_executed": self.last_cycle_executed,
            "last_cycle_error": self.last_cycle_error,
            "cycles_completed": self.cycles_completed,
            "health": {
                **{f"model:{name}": ready for name, ready in self.ml.loaded_heads.items()},
                "phase": self.phase.value,
                "scheduler": "running" if self.scheduler.running else "stopped",
                "audit_queue": str(self.audit.stats()["queued"]),
                "audit_written": str(self.audit.stats()["written"]),
                "universe_size": str(len(self.active_symbols)),
                "universe_ok": str(self.last_cycle_symbols_ok),
            },
        }

    # --- Universe -----------------------------------------------------
    async def list_universe_candidates(self, refresh: bool = False) -> dict[str, Any]:
        """Every USDT-M perpetual on Binance, screened and scored for the panel."""
        candidates: list[SymbolCandidate] = await self.universe.discover(force_refresh=refresh)
        selected: list[str] = await self.universe.get_selection()
        config = self.settings.universe
        return {
            "rows": [candidate.to_dict() for candidate in candidates],
            "selected": selected,
            "eligible_count": sum(1 for item in candidates if item.eligible),
            "total_count": len(candidates),
            "criteria": {
                "min_quote_volume_24h": config.min_quote_volume_24h,
                "max_spread_bps": config.max_spread_bps,
                "min_history_days": config.min_history_days,
                "reference_equity": config.reference_equity,
                "target_count": config.target_count,
            },
        }

    async def suggest_universe(self, limit: int | None = None) -> dict[str, Any]:
        """Top-scoring eligible symbols, used to pre-tick the selection table."""
        symbols: list[str] = await self.universe.suggest(limit)
        return {"symbols": symbols, "count": len(symbols)}

    async def save_universe(
        self,
        symbols: Sequence[str],
        start_setup: bool = True,
        operator: str = "web-panel",
    ) -> dict[str, Any]:
        """Persist the operator's pair selection and kick off setup.

        Changing the universe invalidates the trained models (they were fitted on
        a different cross-section), so a changed selection forces a retrain.
        """
        result: dict[str, Any] = await self.universe.save_selection(symbols, operator)
        self.active_symbols = result["symbols"]
        await self.audit.log_system_event(
            "control",
            "NO_TRADE",
            f"universe saved: {len(self.active_symbols)} symbol(s)",
            rule="OPERATOR_UNIVERSE",
            payload={"symbols": self.active_symbols},
        )
        if start_setup:
            result["setup_started"] = self.launch_setup(force_retrain=result["changed"])
        return result

    # --- Setup --------------------------------------------------------
    async def setup_status(self) -> dict[str, Any]:
        """Live progress of the data-collection and training pipeline."""
        payload: dict[str, Any] = self.progress.to_dict()
        payload["phase"] = self.phase.value
        payload["models"] = self.ml.loaded_heads
        payload["universe_size"] = len(self.active_symbols)
        return payload

    async def start_setup(self, force_retrain: bool = False) -> dict[str, Any]:
        """Re-run collection and (optionally forced) training."""
        started: bool = self.launch_setup(force_retrain=force_retrain)
        return {"started": started, "phase": self.phase.value, "force_retrain": force_retrain}

    # --- Trading control ----------------------------------------------
    async def set_trading_enabled(self, enabled: bool) -> dict[str, Any]:
        """Arm/disarm trading, keeping the current execution mode."""
        if enabled:
            return await self.start_trading(self.trading_mode)
        return await self.stop_trading()

    async def set_trading_mode(self, mode: str) -> dict[str, Any]:
        """Switch between paper and live, re-arming if trading was already on."""
        if mode == self.trading_mode:
            return {"trading_mode": self.trading_mode, "changed": False}
        was_trading: bool = self.phase.is_trading
        if was_trading:
            return await self.start_trading(mode)
        await self._switch_engine(mode)
        return {"trading_mode": self.trading_mode, "changed": True, "armed": False}

    async def engage_kill_switch(self, reason: str) -> dict[str, Any]:
        """Trip RED: the guard's halt callback flattens and disables execution."""
        state: SystemState = await self.risk_guard.trigger_kill_switch(reason)
        self.trading_enabled = False
        if self.phase.is_trading:
            self.phase = SystemPhase.READY
        await self.audit.log_system_event(
            "control", "BLOCKED", f"kill switch engaged: {reason}", rule="OPERATOR_KILL_SWITCH"
        )
        return {"risk_guard_state": state.value, "reason": reason, "trading_enabled": False}

    async def reset_risk_guard(self, operator: str) -> dict[str, Any]:
        """Clear a RED latch.  Trading stays disarmed until explicitly re-armed."""
        state: SystemState = await self.risk_guard.reset(operator)
        await self.audit.log_system_event(
            "control", "NO_TRADE", f"risk guard reset by {operator}", rule="OPERATOR_RESET"
        )
        return {
            "risk_guard_state": state.value,
            "trading_enabled": self.trading_enabled,
            "phase": self.phase.value,
        }

    async def recent_audit(
        self,
        limit: int,
        symbol: str | None,
        verdict: str | None,
    ) -> list[dict[str, Any]]:
        """Proxy to the Audit Engine."""
        return await self.audit.recent(limit=limit, symbol=symbol, verdict=verdict)

    async def recent_trades(self, limit: int, status_filter: str | None) -> list[dict[str, Any]]:
        """Proxy to the trades table."""
        return await self.database.fetch_trades(status=status_filter, limit=limit)

    async def ml_diagnostics(self) -> dict[str, Any]:
        """The full ML diagnostic report for the most recent training run.

        Read back from the durable state pointer + JSON file written by
        :func:`module_f_panel.diagnostics.build_report`, so this reflects the
        real last-completed run even across a panel restart - never
        recomputed or approximated here.
        """
        report: dict[str, Any] | None = await diagnostics.load_latest_report(self.database)
        if report is None:
            return {"status": "NOT_AVAILABLE", "reason": "no training run has completed yet"}
        return report

    async def ml_diagnostics_markdown(self) -> str:
        """The same report, rendered as human-readable Markdown."""
        report: dict[str, Any] = await self.ml_diagnostics()
        if report.get("status") == "NOT_AVAILABLE":
            return "# ML Diagnostic Report\n\nNo training run has completed yet."
        return diagnostics.render_markdown(report)

    # ------------------------------------------------------------------
    # Runners
    # ------------------------------------------------------------------
    def _schedule(self) -> None:
        """Register the 5-minute cron job."""
        self.scheduler.add_job(
            self.trading_cycle,
            trigger=CronTrigger(
                minute=_CYCLE_MINUTES,
                second=self.settings.data.cycle_second_offset,
                timezone="UTC",
            ),
            id="trading_cycle",
            name="5m trading cycle",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=90,
            replace_existing=True,
        )
        self.scheduler.start()
        _LOGGER.info(
            "Scheduler armed: every 5 minutes at second %d (UTC)",
            self.settings.data.cycle_second_offset,
        )

    async def run_forever(self) -> None:
        """Run the scheduler, the setup pipeline and the web server together.

        The panel binds first so the operator can watch (and steer) setup as it
        happens, rather than staring at a terminal for the several minutes a
        30-symbol backfill and training run takes.
        """
        await self.startup()
        self._schedule()

        if self.settings.auto_setup_on_start:
            self.launch_setup()
        else:
            self.phase = (
                SystemPhase.READY if self.active_symbols else SystemPhase.AWAITING_UNIVERSE
            )

        config = uvicorn.Config(
            build_app(self),
            host=self.settings.web.host,
            port=self.settings.web.port,
            log_level=self.settings.log_level.lower(),
            access_log=False,
            log_config=None,
        )
        server = uvicorn.Server(config)
        server_task: asyncio.Task[None] = asyncio.create_task(server.serve(), name="web-server")
        shutdown_task: asyncio.Task[bool] = asyncio.create_task(
            self._shutdown.wait(), name="shutdown-watch"
        )

        _LOGGER.info(
            "Panel available at http://%s:%d/", self.settings.web.host, self.settings.web.port
        )

        try:
            done, pending = await asyncio.wait(
                {server_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            server.should_exit = True
            await asyncio.gather(server_task, return_exceptions=True)
        finally:
            await self.shutdown()

    # ------------------------------------------------------------------
    # One-shot commands
    # ------------------------------------------------------------------
    async def _resolve_cli_universe(self) -> list[str]:
        """Universe for the one-shot CLI commands.

        Prefers the operator's saved selection; falls back to auto-suggesting one
        so the CLI stays usable before anybody has opened the panel.
        """
        symbols: list[str] = await self.universe.get_selection()
        if symbols:
            return symbols
        _LOGGER.warning("No saved universe - auto-selecting the top screened symbols")
        suggested: list[str] = await self.universe.suggest()
        if not suggested:
            raise QuantSystemError("no symbol passes the universe screens")
        await self.universe.save_selection(suggested, operator="cli-auto")
        self.active_symbols = suggested
        return suggested

    async def command_universe(self) -> None:
        """Print the screened universe so it can be reviewed without the panel."""
        await self.database.initialize()
        await self.fetcher.load_markets()
        candidates: list[SymbolCandidate] = await self.universe.discover(force_refresh=True)
        selected: set[str] = set(await self.universe.get_selection())

        header: str = (
            f"{'':2} {'SYMBOL':22} {'PRICE':>12} {'24H VOL':>10} {'SPREAD':>8} "
            f"{'MIN$':>7} {'STEP$':>9} {'DAYS':>6}  ELIGIBLE"
        )
        print(header)
        print("-" * len(header))
        for candidate in candidates[:80]:
            mark: str = "*" if candidate.symbol in selected else " "
            flag: str = "yes" if candidate.eligible else "; ".join(candidate.reasons)[:60]
            print(
                f"{mark:2} {candidate.symbol:22} {candidate.price:12.6f} "
                f"{candidate.quote_volume_24h / 1e6:9.1f}M {candidate.spread_bps:7.2f}b "
                f"{candidate.min_notional:7.2f} {candidate.granularity_usdt:9.2f} "
                f"{candidate.listed_days:6.0f}  {flag}"
            )
        eligible: int = sum(1 for item in candidates if item.eligible)
        print(f"\n{eligible}/{len(candidates)} eligible | * = currently selected")
        await self.fetcher.close()
        await self.database.close()

    async def command_bootstrap(self) -> None:
        """Backfill historical candles for the whole universe."""
        await self.database.initialize()
        await self.fetcher.load_markets()
        symbols: list[str] = await self._resolve_cli_universe()
        written: dict[str, int] = await self.pipeline.bootstrap_history(symbols)
        for symbol, count in sorted(written.items()):
            stored: int = await self.database.candle_count(symbol)
            _LOGGER.info("%-22s +%6d candles (stored: %d)", symbol, count, stored)
        await self.fetcher.close()
        await self.database.close()

    async def command_train(self, max_candles: int | None = None) -> None:
        """Build the training dataset and fit all four heads."""
        await self.database.initialize()
        symbols: list[str] = await self._resolve_cli_universe()
        # `train` is a standalone CLI path - it never runs `_setup_collect`, so
        # without this the funding_rate/open_interest/long_short_ratio/
        # taker_buy_sell_ratio history stays at its neutral default no matter
        # how many times the model is retrained from this entry point.
        await self.pipeline.backfill_futures_metrics(symbols)
        dataset: ProcessedDataset = await self.processor.build_training_dataset(
            symbols=symbols, max_candles_per_symbol=max_candles
        )
        if dataset.is_empty:
            _LOGGER.error("Training aborted: the dataset is empty - run `bootstrap` first")
            await self.database.close()
            return

        _LOGGER.info("Dataset: %d rows | %s", len(dataset), dataset.class_distribution())
        report: dict[str, Any] = await self.ml.train_all(dataset)
        for head, metrics in _headline_metrics(report).items():
            _LOGGER.info("%-10s -> %s", head, metrics)

        run_id: str = str(uuid.uuid4())
        backtest_report: BacktestReport | None
        relaxed_backtest_report: BacktestReport | None
        backtest_report, relaxed_backtest_report = await self._run_validation_backtest(dataset)
        try:
            diagnostic_report: dict[str, Any] = await diagnostics.build_report(
                settings=self.settings,
                database=self.database,
                ml=self.ml,
                dataset=dataset,
                run_id=run_id,
                backtest=backtest_report,
                relaxed_backtest=relaxed_backtest_report,
            )
            self.latest_ml_report_id = run_id
            reports_dir = self.settings.ml.model_dir.parent / diagnostics.REPORTS_DIR_NAME
            print(f"\nML diagnostic report: {reports_dir / f'ml_diagnostic_{run_id}.json'}")
            print(f"Overall status: {diagnostic_report['ai_summary']['overall_status']}")
        except Exception as error:  # pragma: no cover - reporting must not break the CLI
            _LOGGER.error("Could not build the ML diagnostic report: %s", error, exc_info=True)

        await self.features.shutdown()
        await self.database.close()

    async def command_backtest(
        self,
        max_candles: int | None = None,
        equity: float | None = None,
    ) -> BacktestReport:
        """Replay stored history through the full decision pipeline."""
        await self.database.initialize()
        self.ml.load_all()
        backtester = Backtester(
            self.settings, self.database, self.features, self.ml, self.decisions
        )
        symbols: list[str] = await self._resolve_cli_universe()
        report: BacktestReport = await backtester.run(
            symbols=symbols, max_candles=max_candles, initial_equity=equity
        )
        print(report.summary())
        await self.features.shutdown()
        await self.database.close()
        return report

    async def command_single_cycle(self) -> dict[str, Any]:
        """Run exactly one trading cycle, then exit (useful for cron/debugging)."""
        await self.startup()
        try:
            return await self.trading_cycle()
        finally:
            await self.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _install_signal_handlers(system: TradingSystem) -> None:
    """Trap SIGINT/SIGTERM so shutdown is graceful rather than abrupt."""
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    for signal_name in ("SIGINT", "SIGTERM"):
        signal_number: signal.Signals | None = getattr(signal, signal_name, None)
        if signal_number is None:  # pragma: no cover - Windows lacks SIGTERM
            continue
        try:
            loop.add_signal_handler(signal_number, system.request_shutdown)
        except NotImplementedError:  # pragma: no cover - non-POSIX event loops
            signal.signal(signal_number, lambda *_: system.request_shutdown())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="ai-quant",
        description="AI Quant Trading System for Binance USDT-M perpetual futures (5m).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "bootstrap", "train", "backtest", "cycle", "universe"],
        help="run: panel + auto setup + scheduler | universe: print the screened pairs | "
        "bootstrap: backfill | train: fit models | backtest: replay history | "
        "cycle: one cycle then exit",
    )
    parser.add_argument("--candles", type=int, default=None, help="History depth per symbol.")
    parser.add_argument("--equity", type=float, default=None, help="Backtest starting equity.")
    parser.add_argument("--mode", choices=["paper", "live"], default=None, help="Execution mode.")
    return parser.parse_args(argv)


async def _async_main(arguments: argparse.Namespace) -> int:
    """Dispatch the requested command."""
    settings: Settings = get_settings()
    if arguments.mode:
        settings.trading_mode = arguments.mode

    configure_logging(settings.log_level, settings.log_dir)
    system = TradingSystem(settings)

    if arguments.command == "universe":
        await system.command_universe()
        return 0
    if arguments.command == "bootstrap":
        await system.command_bootstrap()
        return 0
    if arguments.command == "train":
        await system.command_train(arguments.candles)
        return 0
    if arguments.command == "backtest":
        await system.command_backtest(arguments.candles, arguments.equity)
        return 0
    if arguments.command == "cycle":
        await system.command_single_cycle()
        return 0

    _install_signal_handlers(system)
    await system.run_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Synchronous entry point."""
    arguments: argparse.Namespace = _parse_args(argv)
    try:
        return asyncio.run(_async_main(arguments))
    except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
        _LOGGER.warning("Interrupted by the user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
