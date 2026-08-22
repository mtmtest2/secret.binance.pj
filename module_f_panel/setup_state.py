"""System lifecycle phases and setup progress reporting.

The system is a state machine, not a script.  A single ``python main.py`` boots
the panel immediately and then walks itself through discovery, data collection
and training in the background, publishing progress the whole way, and stopping
at ``READY`` so the operator arms trading deliberately rather than by accident.

::

    STARTING
        │
        ├── no universe saved ──▶ AWAITING_UNIVERSE ──(operator ticks coins)──┐
        │                                                                      │
        ▼                                                                      │
    COLLECTING_DATA ◀──────────────────────────────────────────────────────────┘
        │
        ▼
    TRAINING ──▶ READY ──(operator arms)──▶ PAPER_TRADING ⇄ LIVE_TRADING
                   ▲                              │
                   └──────────(operator stops)────┘

``SETUP_FAILED`` is terminal until the operator retries; it never silently falls
back to trading with stale models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SystemPhase(str, Enum):
    """Where the system is in its lifecycle."""

    STARTING = "STARTING"
    AWAITING_UNIVERSE = "AWAITING_UNIVERSE"
    COLLECTING_DATA = "COLLECTING_DATA"
    TRAINING = "TRAINING"
    READY = "READY"
    PAPER_TRADING = "PAPER_TRADING"
    LIVE_TRADING = "LIVE_TRADING"
    SETUP_FAILED = "SETUP_FAILED"

    @property
    def is_trading(self) -> bool:
        """``True`` while the decision pipeline is allowed to open positions."""
        return self in (SystemPhase.PAPER_TRADING, SystemPhase.LIVE_TRADING)

    @property
    def is_busy(self) -> bool:
        """``True`` while a setup stage is running."""
        return self in (SystemPhase.COLLECTING_DATA, SystemPhase.TRAINING)

    @property
    def can_arm_trading(self) -> bool:
        """``True`` when the operator may start paper or live trading."""
        return self in (SystemPhase.READY, SystemPhase.PAPER_TRADING, SystemPhase.LIVE_TRADING)


@dataclass(slots=True)
class SetupProgress:
    """Live progress of the automated setup pipeline, rendered by the panel."""

    phase: SystemPhase = SystemPhase.STARTING
    step: str = "initialising"
    detail: str = ""
    percent: float = 0.0
    symbols_total: int = 0
    symbols_done: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str = ""
    data_summary: dict[str, int] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)

    def begin(self, phase: SystemPhase, step: str, detail: str = "") -> None:
        """Enter a new stage, resetting the per-stage counters."""
        self.phase = phase
        self.step = step
        self.detail = detail
        self.error = ""
        self.symbols_done = 0
        self.finished_at = None
        if self.started_at is None:
            self.started_at = datetime.now(tz=timezone.utc)

    def advance(self, done: int, total: int, detail: str = "") -> None:
        """Update the progress bar for the current stage."""
        self.symbols_done = done
        self.symbols_total = total
        if detail:
            self.detail = detail
        self.percent = 0.0 if total <= 0 else min(100.0, 100.0 * done / total)

    def set_step(self, step: str, detail: str = "") -> None:
        """Relabel the current sub-stage without resetting the phase or counters.

        ``advance()`` moves the progress bar but leaves ``step`` untouched, so a
        multi-stage phase (e.g. training: dataset -> fit -> backtest -> report)
        would otherwise show a stale label like "building dataset" long after
        that sub-stage finished.
        """
        self.step = step
        if detail:
            self.detail = detail

    def finish(self, phase: SystemPhase, step: str, detail: str = "") -> None:
        """Mark the pipeline complete."""
        self.phase = phase
        self.step = step
        self.detail = detail
        self.percent = 100.0
        self.error = ""
        self.finished_at = datetime.now(tz=timezone.utc)

    def fail(self, message: str) -> None:
        """Mark the pipeline failed; the phase becomes terminal until retried."""
        self.phase = SystemPhase.SETUP_FAILED
        self.step = "failed"
        self.error = message
        self.finished_at = datetime.now(tz=timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for ``GET /api/setup/status``."""
        return {
            "phase": self.phase.value,
            "step": self.step,
            "detail": self.detail,
            "percent": round(self.percent, 1),
            "symbols_total": self.symbols_total,
            "symbols_done": self.symbols_done,
            "started_at": self.started_at.isoformat(timespec="seconds") if self.started_at else "",
            "finished_at": (
                self.finished_at.isoformat(timespec="seconds") if self.finished_at else ""
            ),
            "error": self.error,
            "is_busy": self.phase.is_busy,
            "is_trading": self.phase.is_trading,
            "can_arm_trading": self.phase.can_arm_trading,
            "data_summary": self.data_summary,
            "training_summary": self.training_summary,
        }
