"""Precision/robustness of the QC healing system (module_a_data/qc_validator.py)."""

from __future__ import annotations

import pytest

from config.settings import QCSettings, Settings
from core.exceptions import DataIntegrityError
from core.utils import last_closed_candle_open_ms
from module_a_data.models import OHLCVCandle, QCIssue, QCIssueCode, QCReport, QCSeverity
from module_a_data.qc_validator import QCValidator

TF_MS = 5 * 60 * 1_000
# The freshness check compares the newest candle against "now" at call time, so
# fixtures are anchored there (not a fixed historical epoch) and always end on
# the freshest closed bar unless a test overrides `end`.
_NEWEST_CLOSED_MS = last_closed_candle_open_ms(TF_MS)


def make_candle(ts: int, price: float = 100.0) -> OHLCVCandle:
    return OHLCVCandle(
        symbol="BTC/USDT:USDT",
        timestamp=ts,
        open=price,
        high=price * 1.001,
        low=price * 0.999,
        close=price,
        volume=10.0,
    )


def make_series(n: int, end: int = _NEWEST_CLOSED_MS) -> list[OHLCVCandle]:
    """A clean, grid-aligned run of ``n`` candles ending at ``end`` (default: now)."""
    base: int = end - (n - 1) * TF_MS
    return [make_candle(base + i * TF_MS) for i in range(n)]


def build_validator(**qc_overrides: object) -> tuple[QCValidator, Settings]:
    settings = Settings(qc=QCSettings(**qc_overrides))
    return QCValidator(settings), settings


# ---------------------------------------------------------------------------
# _heal_windows: grouping / precision
# ---------------------------------------------------------------------------
def test_heal_windows_separates_distant_gaps() -> None:
    validator, _ = build_validator(heal_merge_gap_bars=2)
    candles = make_series(50)
    report = QCReport(
        symbol="X",
        checked_rows=50,
        missing_timestamps=(candles[5].timestamp, candles[40].timestamp),
    )
    windows = validator._heal_windows(candles, report)
    assert len(windows) == 2


def test_heal_windows_merges_nearby_gaps() -> None:
    validator, _ = build_validator(heal_merge_gap_bars=3)
    candles = make_series(50)
    report = QCReport(
        symbol="X",
        checked_rows=50,
        missing_timestamps=(candles[10].timestamp, candles[12].timestamp),
    )
    windows = validator._heal_windows(candles, report)
    assert len(windows) == 1


def test_heal_windows_extends_last_window_for_stale_data() -> None:
    validator, _ = build_validator()
    candles = make_series(10)
    stale_ts = candles[-1].timestamp
    issue = QCIssue(
        code=QCIssueCode.STALE_DATA,
        severity=QCSeverity.CRITICAL,
        message="stale",
        symbol="X",
        timestamps=(stale_ts,),
        healable=True,
    )
    report = QCReport(symbol="X", checked_rows=10, issues=(issue,))
    windows = validator._heal_windows(candles, report)
    assert len(windows) == 1
    expected_end = last_closed_candle_open_ms(TF_MS) + TF_MS
    assert windows[0][1] >= expected_end


def test_heal_windows_collapses_when_too_fragmented() -> None:
    validator, _ = build_validator(heal_merge_gap_bars=0, max_heal_window_groups=3)
    candles = make_series(100)
    missing = tuple(candles[i].timestamp for i in (5, 20, 40, 60, 80))
    report = QCReport(symbol="X", checked_rows=100, missing_timestamps=missing)
    windows = validator._heal_windows(candles, report)
    assert len(windows) == 1
    assert windows[0][0] <= candles[5].timestamp
    assert windows[0][1] >= candles[80].timestamp


# ---------------------------------------------------------------------------
# _merge_patches: no erosion of clean boundary candles
# ---------------------------------------------------------------------------
def test_merge_patches_keeps_clean_boundary_when_patch_is_empty() -> None:
    validator, _ = build_validator()
    base = make_series(10)
    # Only candle[5] is actually suspicious; candle[4] and candle[6] are clean
    # boundary candles that happened to fall inside the padded fetch window.
    drop = {base[5].timestamp}
    merged = validator._merge_patches(base, drop, patches=[[]])
    timestamps = {c.timestamp for c in merged}
    assert base[4].timestamp in timestamps
    assert base[6].timestamp in timestamps
    assert base[5].timestamp not in timestamps
    assert len(merged) == 9


def test_merge_patches_applies_fresh_rows() -> None:
    validator, _ = build_validator()
    base = make_series(5)
    drop = {base[2].timestamp}
    replacement = make_candle(base[2].timestamp, price=999.0)
    merged = validator._merge_patches(base, drop, patches=[[replacement]])
    by_ts = {c.timestamp: c for c in merged}
    assert by_ts[base[2].timestamp].close == 999.0
    assert len(merged) == 5


# ---------------------------------------------------------------------------
# validate_and_heal: bounded attempts, quarantine behaviour
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_validate_and_heal_heals_a_fixable_gap() -> None:
    validator, _ = build_validator(max_heal_attempts=3, heal_backoff_seconds=0.001)
    full = make_series(20)
    damaged = [c for i, c in enumerate(full) if i != 10]  # one missing candle

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return [c for c in full if start_ms <= c.timestamp <= end_ms]

    healed, report, heal_attempts = await validator.validate_and_heal("X", damaged, refetch)
    assert report.passed
    assert len(healed) == 20
    assert len(heal_attempts) == 1
    assert heal_attempts[0].result == "resolved"
    assert heal_attempts[0].bars_invalid_after_heal == 0


@pytest.mark.asyncio
async def test_validate_and_heal_raises_without_quarantine_on_unfixable_gap() -> None:
    validator, _ = build_validator(max_heal_attempts=2, heal_backoff_seconds=0.001)
    full = make_series(30)
    damaged = [c for i, c in enumerate(full) if not (5 <= i <= 9)]

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return []  # the exchange genuinely has nothing here (delisting/halt gap)

    with pytest.raises(DataIntegrityError):
        await validator.validate_and_heal("X", damaged, refetch, quarantine_unhealable=False)


@pytest.mark.asyncio
async def test_validate_and_heal_quarantines_unfixable_gap_when_enabled() -> None:
    validator, _ = build_validator(max_heal_attempts=2, heal_backoff_seconds=0.001)
    full = make_series(30)
    damaged = [c for i, c in enumerate(full) if not (5 <= i <= 9)]

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return []

    healed, report, heal_attempts = await validator.validate_and_heal(
        "X", damaged, refetch, quarantine_unhealable=True
    )
    assert report.passed
    # Only the clean trailing run (after the unfixable gap) survives.
    assert healed[0].timestamp == full[10].timestamp
    assert healed[-1].timestamp == full[-1].timestamp
    assert len(healed) == 20
    assert heal_attempts[-1].result == "quarantined"


@pytest.mark.asyncio
async def test_validate_and_heal_respects_wall_clock_budget() -> None:
    validator, _ = build_validator(
        max_heal_attempts=10,
        heal_backoff_seconds=0.05,
        max_heal_duration_seconds=0.03,
    )
    full = make_series(20)
    damaged = [c for i, c in enumerate(full) if i != 10]

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return []  # never actually heals, forcing the loop to keep retrying

    with pytest.raises(DataIntegrityError, match="wall-clock budget"):
        await validator.validate_and_heal("X", damaged, refetch, quarantine_unhealable=False)


@pytest.mark.asyncio
async def test_validate_and_heal_passthrough_when_already_clean() -> None:
    validator, _ = build_validator()
    full = make_series(20)

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        raise AssertionError("refetch must not be called when the block is already clean")

    healed, report, heal_attempts = await validator.validate_and_heal("X", full, refetch)
    assert report.passed
    assert len(healed) == 20
    assert heal_attempts == []
