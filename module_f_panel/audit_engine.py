"""The Audit Engine - why the system did what it did.

In a quantitative ML system the *reason* for a trade matters at least as much as
the trade.  Six weeks after a drawdown, "the model said so" is not a debuggable
statement; "on 2026-03-14 09:15 UTC BTC was rejected because the direction
confidence was 0.68 against a 0.70 threshold, with the HMM in regime 2 and GARCH
volatility in the 93rd percentile" is.

Every decision cycle is logged - **including every ``NO_TRADE``**, which is where
the interesting failures hide.  Writes are batched through an
:class:`asyncio.Queue` and drained by a background task, so the trading loop
never waits on SQLite.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final

from core.logger import get_logger
from core.utils import utc_now_ms
from module_a_data.db_handler import DatabaseHandler
from module_c_ml.schemas import DecisionResult, DecisionVerdict, ModelInferenceResult, TradeAction

_LOGGER = get_logger(__name__)

_QUEUE_MAXSIZE: Final[int] = 4_000
_FLUSH_BATCH: Final[int] = 64
_FLUSH_INTERVAL_SECONDS: Final[float] = 2.0


@dataclass(slots=True)
class AuditRecord:
    """One fully-flattened decision-cycle record, ready for the ``audit_logs`` table."""

    decision_id: str
    cycle_id: str
    symbol: str
    candle_timestamp: int
    verdict: str
    action: str
    rule_triggered: str
    reason: str

    hmm_regime: int = -1
    garch_volatility: float = 0.0
    garch_vol_percentile: float = 0.0
    kama_slope: float = 0.0
    fdi: float = 0.0
    atr: float = 0.0
    order_book_imbalance: float = 0.0
    close_price: float = 0.0

    prob_long: float = 0.0
    prob_short: float = 0.0
    prob_no_trade: float = 0.0
    direction_confidence: float = 0.0
    entry_probability: float = 0.0
    entry_allowed: bool = False

    take_profit_pct: float = 0.0
    stop_loss_pct: float = 0.0
    trailing_trigger_pct: float = 0.0
    recommended_leverage: int = 0
    capital_allocation_pct: float = 0.0
    risk_tier: str = "UNKNOWN"

    risk_guard_state: str = "GREEN"
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    payload: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Column mapping for :class:`~module_a_data.db_models.AuditLogRow`."""
        return {
            "decision_id": self.decision_id,
            "cycle_id": self.cycle_id,
            "symbol": self.symbol,
            "candle_timestamp": self.candle_timestamp,
            "created_at": self.created_at,
            "verdict": self.verdict,
            "action": self.action,
            "rule_triggered": self.rule_triggered,
            "reason": self.reason,
            "hmm_regime": self.hmm_regime,
            "garch_volatility": self.garch_volatility,
            "garch_vol_percentile": self.garch_vol_percentile,
            "kama_slope": self.kama_slope,
            "fdi": self.fdi,
            "atr": self.atr,
            "order_book_imbalance": self.order_book_imbalance,
            "close_price": self.close_price,
            "prob_long": self.prob_long,
            "prob_short": self.prob_short,
            "prob_no_trade": self.prob_no_trade,
            "direction_confidence": self.direction_confidence,
            "entry_probability": self.entry_probability,
            "entry_allowed": self.entry_allowed,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_trigger_pct": self.trailing_trigger_pct,
            "recommended_leverage": self.recommended_leverage,
            "capital_allocation_pct": self.capital_allocation_pct,
            "risk_tier": self.risk_tier,
            "risk_guard_state": self.risk_guard_state,
            "latency_ms": self.latency_ms,
            "payload": self.payload,
        }


class AuditEngine:
    """Asynchronous, batched writer for the decision audit trail."""

    def __init__(self, database: DatabaseHandler) -> None:
        self._db: DatabaseHandler = database
        self._queue: asyncio.Queue[AuditRecord] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._worker: asyncio.Task[None] | None = None
        self._running: bool = False
        self.records_written: int = 0
        self.records_dropped: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the background flush task."""
        if self._running:
            return
        self._running = True
        self._worker = asyncio.create_task(self._flush_loop(), name="audit-writer")
        _LOGGER.info("Audit engine started")

    async def stop(self, drain: bool = True) -> None:
        """Stop the writer, optionally draining whatever is still queued."""
        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        if drain:
            await self._drain()
        _LOGGER.info(
            "Audit engine stopped (%d written, %d dropped)",
            self.records_written,
            self.records_dropped,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    async def log_decision(
        self,
        decision: DecisionResult,
        cycle_id: str,
        risk_guard_state: str = "GREEN",
        execution_detail: dict[str, Any] | None = None,
    ) -> AuditRecord:
        """Flatten and enqueue one decision, whatever its verdict.

        Args:
            decision: The Decision Engine's result for one symbol.
            cycle_id: UUID shared by every decision in the same 5m cycle.
            risk_guard_state: The guard's state at decision time.
            execution_detail: Optional execution outcome merged into the payload.

        Returns:
            The :class:`AuditRecord` that was enqueued (useful in tests).
        """
        record: AuditRecord = self._build_record(
            decision, cycle_id, risk_guard_state, execution_detail
        )
        self._enqueue(record)
        return record

    async def log_decisions(
        self,
        decisions: list[DecisionResult],
        cycle_id: str,
        risk_guard_state: str = "GREEN",
    ) -> int:
        """Enqueue a whole cycle's worth of decisions."""
        for decision in decisions:
            await self.log_decision(decision, cycle_id, risk_guard_state)
        return len(decisions)

    async def log_system_event(
        self,
        cycle_id: str,
        verdict: str,
        reason: str,
        rule: str = "SYSTEM_EVENT",
        symbol: str = "*",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a non-per-symbol event (kill switch, cycle failure, mode change)."""
        self._enqueue(
            AuditRecord(
                decision_id=f"sys-{utc_now_ms()}",
                cycle_id=cycle_id,
                symbol=symbol,
                candle_timestamp=0,
                verdict=verdict,
                action=TradeAction.NO_TRADE.value,
                rule_triggered=rule,
                reason=reason,
                payload=payload or {},
            )
        )

    def _enqueue(self, record: AuditRecord) -> None:
        """Enqueue without blocking; drop the oldest record when saturated.

        Losing an audit line is bad.  Stalling the trading loop behind SQLite is
        worse, so back-pressure is resolved by dropping and counting.
        """
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.records_dropped += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(record)
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover - racy edge
                pass
            if self.records_dropped % 100 == 1:
                _LOGGER.warning(
                    "Audit queue saturated - %d record(s) dropped so far", self.records_dropped
                )

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------
    async def _flush_loop(self) -> None:
        """Drain the queue in batches until cancelled."""
        while self._running:
            try:
                await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
                await self._drain()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # pragma: no cover - the writer must never die
                _LOGGER.error("Audit flush failed: %s", error, exc_info=True)

    async def _drain(self) -> None:
        """Write everything currently queued, in batches."""
        batch: list[dict[str, Any]] = []
        while not self._queue.empty() and len(batch) < _FLUSH_BATCH:
            try:
                batch.append(self._queue.get_nowait().to_row())
            except asyncio.QueueEmpty:  # pragma: no cover - racy edge
                break

        if not batch:
            return
        try:
            written: int = await self._db.insert_audit_logs(batch)
            self.records_written += written
        except Exception as error:
            _LOGGER.error("Could not persist %d audit record(s): %s", len(batch), error)

        if not self._queue.empty():
            await self._drain()

    async def flush(self) -> None:
        """Force an immediate drain (used on shutdown and by the panel)."""
        await self._drain()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    async def recent(
        self,
        limit: int = 100,
        symbol: str | None = None,
        verdict: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the newest audit rows for the ``/audit`` page."""
        await self.flush()
        return await self._db.fetch_audit_logs(limit=limit, symbol=symbol, verdict=verdict)

    def stats(self) -> dict[str, Any]:
        """Writer health, surfaced on the dashboard."""
        return {
            "queued": self._queue.qsize(),
            "written": self.records_written,
            "dropped": self.records_dropped,
            "running": self._running,
        }

    # ------------------------------------------------------------------
    # Flattening
    # ------------------------------------------------------------------
    @staticmethod
    def _build_record(
        decision: DecisionResult,
        cycle_id: str,
        risk_guard_state: str,
        execution_detail: dict[str, Any] | None,
    ) -> AuditRecord:
        """Project a :class:`DecisionResult` onto the flat audit schema."""
        inference: ModelInferenceResult | None = decision.inference
        record = AuditRecord(
            decision_id=decision.decision_id,
            cycle_id=cycle_id,
            symbol=decision.symbol,
            candle_timestamp=inference.timestamp if inference else 0,
            verdict=decision.verdict.value,
            action=decision.action.value,
            rule_triggered=decision.rule_triggered,
            reason=decision.reason,
            risk_guard_state=risk_guard_state,
        )

        if inference is not None:
            snapshot: dict[str, float] = inference.feature_snapshot
            record.hmm_regime = int(snapshot.get("hmm_regime", -1))
            record.garch_volatility = float(snapshot.get("garch_volatility", 0.0))
            record.garch_vol_percentile = float(snapshot.get("garch_vol_rank", 0.0))
            record.kama_slope = float(snapshot.get("kama_slope", 0.0))
            record.fdi = float(snapshot.get("fdi", 0.0))
            record.atr = float(snapshot.get("atr", 0.0))
            record.order_book_imbalance = float(snapshot.get("ob_imbalance", 0.0))
            record.close_price = inference.close_price

            record.prob_long = inference.direction.long_probability
            record.prob_short = inference.direction.short_probability
            record.prob_no_trade = inference.direction.no_trade_probability
            record.direction_confidence = inference.direction.confidence
            record.entry_probability = inference.entry.probability
            record.entry_allowed = inference.entry.should_enter

            record.take_profit_pct = inference.exit_params.take_profit_pct
            record.stop_loss_pct = inference.exit_params.stop_loss_pct
            record.trailing_trigger_pct = inference.exit_params.trailing_activation_pct
            record.recommended_leverage = inference.risk.leverage
            record.capital_allocation_pct = inference.risk.capital_allocation_pct
            record.risk_tier = inference.risk.risk_tier
            record.latency_ms = inference.latency_ms

            record.payload = {
                "direction_probabilities": inference.direction.probabilities,
                "direction_source": inference.direction.source.value,
                "entry_source": inference.entry.source.value,
                "exit_source": inference.exit_params.source.value,
                "risk_source": inference.risk.source.value,
                "reward_risk_ratio": inference.exit_params.reward_risk_ratio,
                "risk_abort_reason": inference.risk.abort_reason,
                "model_versions": inference.model_versions,
                "any_fallback": inference.any_fallback,
                "checks": decision.checks,
            }
        else:
            record.payload = {"checks": decision.checks}

        if decision.signal is not None:
            record.payload["signal"] = {
                "action": decision.signal.action.value,
                "leverage": decision.signal.leverage,
                "capital_allocation_pct": decision.signal.capital_allocation_pct,
                "take_profit": decision.signal.take_profit,
                "stop_loss": decision.signal.stop_loss,
                "trailing_trigger": decision.signal.trailing_trigger,
                "reward_risk_ratio": decision.signal.reward_risk_ratio,
            }
        if execution_detail:
            record.payload["execution"] = execution_detail

        if decision.verdict is DecisionVerdict.EXECUTE:
            record.reason = record.reason or "Trade Executed"
        return record
