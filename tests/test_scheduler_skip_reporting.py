"""A skipped 5-minute firing means two opposite things depending on the phase.

During bootstrap a single ingestion cycle legitimately runs for tens of minutes
(a two-year backfill plus healing), so every firing inside it is skipped - 16 of
them in the first live run. Once the system is trading, the identical skip means
live candles are being missed. Logging both as the same warning trains an
operator to ignore the one that matters.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import main as main_module
from main import TradingSystem
from module_f_panel.setup_state import SetupProgress, SystemPhase


def _system_in(phase: SystemPhase) -> TradingSystem:
    """A bare instance carrying only the state the skip path reads."""
    system: TradingSystem = TradingSystem.__new__(TradingSystem)
    progress = SetupProgress()
    progress.phase = phase
    system.progress = progress
    return system


@pytest.mark.parametrize(
    "phase",
    [
        SystemPhase.STARTING,
        SystemPhase.AWAITING_UNIVERSE,
        SystemPhase.COLLECTING_DATA,
        SystemPhase.TRAINING,
    ],
)
def test_setup_phases_are_treated_as_expected_overruns(phase: SystemPhase) -> None:
    assert _system_in(phase)._setup_in_progress() is True


@pytest.mark.parametrize(
    "phase",
    [
        SystemPhase.READY,
        SystemPhase.PAPER_TRADING,
        SystemPhase.LIVE_TRADING,
        SystemPhase.SETUP_FAILED,
    ],
)
def test_non_setup_phases_are_not(phase: SystemPhase) -> None:
    assert _system_in(phase)._setup_in_progress() is False


def test_setup_phases_are_real_enum_values() -> None:
    """Guards against a renamed phase silently turning every skip into a warning."""
    known = {member.value for member in SystemPhase}
    assert main_module._SETUP_PHASES <= known


def test_phase_name_survives_a_system_with_no_progress_yet() -> None:
    """`_on_job_missed` can fire before __init__ finished wiring the progress object."""
    system: TradingSystem = TradingSystem.__new__(TradingSystem)
    assert system._phase_name() == "UNKNOWN"
    assert system._setup_in_progress() is False


def test_skip_during_bootstrap_is_informational(caplog: pytest.LogCaptureFixture) -> None:
    system = _system_in(SystemPhase.COLLECTING_DATA)
    with caplog.at_level(logging.INFO):
        system._on_job_missed(object())

    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert "COLLECTING_DATA" in caplog.records[0].getMessage()


def test_skip_while_trading_is_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    system = _system_in(SystemPhase.LIVE_TRADING)
    with caplog.at_level(logging.INFO):
        system._on_job_missed(object())

    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert "candles are being missed" in caplog.records[0].getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "expected_level"),
    [
        (SystemPhase.COLLECTING_DATA, logging.INFO),
        (SystemPhase.LIVE_TRADING, logging.WARNING),
    ],
)
async def test_overlap_skip_uses_the_same_distinction(
    phase: SystemPhase,
    expected_level: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The in-cycle lock path and the scheduler path must agree on severity.

    They report the same event seen from two places: a firing that never reached
    the coroutine, and one that reached it and found the lock held.
    """
    system = _system_in(phase)
    system._cycle_lock = asyncio.Lock()
    system.cycles_skipped_overlap = 0

    async with system._cycle_lock:
        with caplog.at_level(logging.INFO):
            result = await system.trading_cycle()

    assert result == {"skipped": True}
    assert system.cycles_skipped_overlap == 1
    assert [record.levelno for record in caplog.records] == [expected_level]
