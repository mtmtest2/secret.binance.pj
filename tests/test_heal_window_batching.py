"""Bounded fallback batching in QCValidator._heal_windows (module_a_data/qc_validator.py).

Before this fix, a heal round with more damaged windows than
``max_heal_window_groups`` collapsed into a single window spanning the whole
damaged range - unbounded in size. This exercises the fix: the fallback now
splits the range into batches capped at ``max_heal_window_bars`` each.
"""

from __future__ import annotations

import pytest

from config.settings import QCSettings, Settings
from core.exceptions import DataIntegrityError
from core.utils import last_closed_candle_open_ms
from module_a_data.models import OHLCVCandle, QCReport
from module_a_data.qc_validator import QCValidator

TF_MS = 5 * 60 * 1_000
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
    base: int = end - (n - 1) * TF_MS
    return [make_candle(base + i * TF_MS) for i in range(n)]


def build_validator(**qc_overrides: object) -> QCValidator:
    return QCValidator(Settings(qc=QCSettings(**qc_overrides)))


def test_fallback_batches_stay_bounded_instead_of_one_giant_window() -> None:
    """Widely scattered damage across a long history must not collapse into
    a single unbounded re-fetch - it should split into capped batches."""
    validator = build_validator(
        heal_merge_gap_bars=0,
        max_heal_window_groups=3,
        max_heal_window_bars=200,
    )
    n = 5_000
    candles = make_series(n)
    # Damage scattered every 500 bars across the whole 5,000-bar history.
    missing = tuple(candles[i].timestamp for i in range(10, n - 10, 500))
    report = QCReport(symbol="X", checked_rows=n, missing_timestamps=missing)

    windows = validator._heal_windows(candles, report)

    assert len(windows) > 1, "damage spanning the whole history must not collapse to one window"
    max_span_ms = 200 * TF_MS
    for start_ms, end_ms in windows:
        assert end_ms - start_ms <= max_span_ms + TF_MS, (
            f"window [{start_ms}, {end_ms}] exceeds the configured max_heal_window_bars cap"
        )
    # Batches must still cover the full damaged range between them.
    assert windows[0][0] <= candles[10].timestamp
    assert windows[-1][1] >= missing[-1]


def test_fallback_collapses_to_one_window_when_span_is_small() -> None:
    """Small total span still collapses to a single window - unchanged behaviour
    for the common case, only the *unbounded* case is fixed."""
    validator = build_validator(heal_merge_gap_bars=0, max_heal_window_groups=3, max_heal_window_bars=2_000)
    candles = make_series(100)
    missing = tuple(candles[i].timestamp for i in (5, 20, 40, 60, 80))
    report = QCReport(symbol="X", checked_rows=100, missing_timestamps=missing)

    windows = validator._heal_windows(candles, report)
    assert len(windows) == 1


@pytest.mark.asyncio
async def test_heal_attempt_telemetry_records_real_counts() -> None:
    """Every heal round now returns real, measured HealAttempt telemetry -
    not just the final pass/fail verdict."""
    validator = build_validator(max_heal_attempts=3, heal_backoff_seconds=0.001)
    full = make_series(20)
    damaged = [c for i, c in enumerate(full) if i != 10]

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return [c for c in full if start_ms <= c.timestamp <= end_ms]

    healed, report, heal_attempts = await validator.validate_and_heal("BTC/USDT:USDT", damaged, refetch)

    assert report.passed
    assert len(heal_attempts) == 1
    attempt = heal_attempts[0]
    assert attempt.symbol == "BTC/USDT:USDT"
    assert attempt.attempt_number == 1
    assert attempt.bars_requested >= 1
    assert attempt.bars_received >= 1
    assert attempt.bars_invalid_after_heal == 0
    assert attempt.duration_seconds >= 0.0
    assert attempt.result == "resolved"


@pytest.mark.asyncio
async def test_heal_attempts_attached_to_raised_error_context() -> None:
    """Telemetry must survive even when healing ultimately fails, so a caller
    can still report what was tried."""
    validator = build_validator(max_heal_attempts=2, heal_backoff_seconds=0.001)
    full = make_series(30)
    damaged = [c for i, c in enumerate(full) if not (5 <= i <= 9)]

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return []

    with pytest.raises(DataIntegrityError) as exc_info:
        await validator.validate_and_heal("X", damaged, refetch, quarantine_unhealable=False)

    attempts = exc_info.value.context.get("heal_attempts")
    assert attempts is not None
    assert len(attempts) == 2
    assert all(record["result"] == "still_invalid" for record in attempts)
