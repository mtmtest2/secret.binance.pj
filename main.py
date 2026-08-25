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
import uuid
from datetime import datetime, timezone
from typing import Any, Final, Sequence

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import Settings, get_settings
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
from module_f_panel.audit_engine import AuditEngine
from module_f_panel.setup_state import SetupProgress, SystemPhase
from module_f_panel.web_app import build_app

_LOGGER = get_logger("main")

_CYCLE_MINUTES: Final[str] = "0,5,10,15,20,25,30,35,40,45,50,55"

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
        """Stage 1 - backfill historical candles for the selected universe."""
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
        self.progress.advance(0, 2, "engineering features and labels")

        dataset: ProcessedDataset = await self.processor.build_training_dataset(symbols=symbols)
        if dataset.is_empty:
            raise QuantSystemError(
                "training dataset is empty after feature/label generation - "
                "not enough clean history"
            )

        self.progress.advance(1, 2, f"fitting 4 models on {len(dataset)} rows")
        _LOGGER.info("Training on %d rows | %s", len(dataset), dataset.class_distribution())
        report: dict[str, Any] = await self.ml.train_all(dataset)
        self.progress.advance(2, 2, "training complete")

        failed: list[str] = [head for head, metrics in report.items() if "error" in metrics]
        if failed:
            raise QuantSystemError(f"model training failed for: {', '.join(failed)}")

        self.progress.training_summary = {
            "rows": len(dataset),
            "distribution": dataset.class_distribution(),
            "metrics": report,
        }
        await self.database.set_state(
            "trained_universe",
            {"symbols": symbols, "trained_ms": utc_now_ms(), "rows": len(dataset)},
        )
        _LOGGER.info("Training complete: %s", report)

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
            _LOGGER.warning("Previous cycle is still running - skipping this slot")
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

        # --- A: ingestion + QC -------------------------------------------
        bundles: dict[str, MarketDataBundle] = await self.pipeline.run_cycle(self.active_symbols)
        self.last_cycle_symbols_ok = len(bundles)
        if not bundles:
            await self.audit.log_system_event(
                cycle_id, "NO_TRADE", "no symbol passed QC this cycle", rule="QC_ALL_FAILED"
            )
            return {"symbols": 0}

        if not self.phase.is_trading or not self.trading_enabled:
            _LOGGER.info(
                "Data refreshed for %d symbol(s); trading is disarmed (phase=%s)",
                len(bundles),
                self.phase.value,
            )
            return {"symbols": len(bundles), "trading": False, "phase": self.phase.value}

        # --- Equity refresh + Risk Guard evaluation -----------------------
        account: AccountState = await self._account_state()
        state: SystemState = await self.risk_guard.update_equity(account.equity)

        # --- B: features ---------------------------------------------------
        payloads: dict[str, InferencePayload] = await self.processor.build_inference_payloads(
            list(bundles)
        )
        if not payloads:
            await self.audit.log_system_event(
                cycle_id, "NO_TRADE", "no symbol produced a warm feature row", rule="FEATURES_COLD"
            )
            return {"symbols": len(bundles), "payloads": 0}

        # --- C: inference ---------------------------------------------------
        inferences: dict[str, ModelInferenceResult] = await self.ml.infer_many(
            list(payloads.values())
        )

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

        # --- Post-cycle bookkeeping ---------------------------------------------
        await self._record_equity()

        _LOGGER.info(
            "Cycle summary: %d symbols | %d decisions | %d executed | guard=%s | equity=%.2f",
            len(bundles),
            len(decisions),
            executed,
            state.value,
            account.equity,
        )
        return {
            "cycle_id": cycle_id,
            "symbols": len(bundles),
            "decisions": len(decisions),
            "executed": executed,
            "risk_guard": state.value,
            "equity": account.equity,
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

    async def ml_report(self) -> dict[str, Any]:
        """Full, downloadable training report: metrics + feature importances.

        Merges the per-head report (which survives restarts, since it is rebuilt
        from the loaded artifacts) with the dataset summary from the last run and
        the universe the artifacts were trained on.
        """
        stored: dict[str, Any] | None = await self.database.get_state("trained_universe")
        summary: dict[str, Any] = dict(self.progress.training_summary or {})
        report: dict[str, Any] = self.ml.training_report()
        report.update(
            {
                "generated_at": utc_now().isoformat(timespec="seconds"),
                "app": self.settings.app_name,
                "trained_universe": stored.get("symbols", []) if stored else [],
                "trained_at": stored.get("trained_ms") if stored else None,
                "dataset": {
                    "rows": summary.get("rows"),
                    "distribution": summary.get("distribution"),
                },
                "history_bootstrap_candles": self.settings.data.history_bootstrap_candles,
            }
        )
        return report

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
        dataset: ProcessedDataset = await self.processor.build_training_dataset(
            symbols=symbols, max_candles_per_symbol=max_candles
        )
        if dataset.is_empty:
            _LOGGER.error("Training aborted: the dataset is empty - run `bootstrap` first")
            await self.database.close()
            return

        _LOGGER.info("Dataset: %d rows | %s", len(dataset), dataset.class_distribution())
        report: dict[str, Any] = await self.ml.train_all(dataset)
        for head, metrics in report.items():
            _LOGGER.info("%-10s -> %s", head, metrics)
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
