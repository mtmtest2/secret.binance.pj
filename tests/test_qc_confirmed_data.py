"""Regressions for the candle-ingestion failures seen in production.

Every test here reproduces a symptom taken verbatim from a live bootstrap log:

* a market-wide dislocation quarantining ~59 % of a symbol's history,
* an ``EXCESSIVE_ZERO_VOLUME`` verdict degenerating into a 212 401-bar
  whole-history re-fetch that blew the 90 s heal budget,
* four byte-identical heal rounds per doomed symbol.
"""

from __future__ import annotations

import pytest

from config.settings import QCSettings, Settings
from core.exceptions import DataIntegrityError
from core.utils import last_closed_candle_open_ms
from module_a_data.models import OHLCVCandle, QCIssue, QCIssueCode, QCReport, QCSeverity
from module_a_data.qc_validator import QCValidator

TF_MS = 5 * 60 * 1_000
_NEWEST_CLOSED_MS = last_closed_candle_open_ms(TF_MS)


def make_candle(ts: int, price: float = 100.0, volume: float = 10.0) -> OHLCVCandle:
    return OHLCVCandle(
        symbol="BTC/USDT:USDT",
        timestamp=ts,
        open=price,
        high=price * 1.001,
        low=price * 0.999,
        close=price,
        volume=volume,
    )


def make_series(n: int, end: int = _NEWEST_CLOSED_MS) -> list[OHLCVCandle]:
    base: int = end - (n - 1) * TF_MS
    return [make_candle(base + i * TF_MS) for i in range(n)]


def build_validator(**qc_overrides: object) -> QCValidator:
    return QCValidator(Settings(qc=QCSettings(**qc_overrides)))


def echo_refetch(source: list[OHLCVCandle]):
    """A refetch that serves the exchange's own copy back - i.e. the data is real."""
    by_ts: dict[int, OHLCVCandle] = {candle.timestamp: candle for candle in source}

    async def _refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return [by_ts[ts] for ts in sorted(by_ts) if start_ms <= ts <= end_ms]

    return _refetch


# ---------------------------------------------------------------------------
# A confirmed dislocation is market data, not corruption
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exchange_confirmed_crash_is_accepted_not_quarantined() -> None:
    """A 40 % five-minute candle the exchange re-serves unchanged must be kept.

    Production log: two bars at 1760130600000 flagged across nearly the whole
    universe caused ``124841 candle(s) quarantined as unhealable`` per symbol -
    ~59 % of a 212 400-candle history thrown away over a real market event.
    """
    candles = make_series(400)
    crash_index = 150
    # A genuine -45 % bar, then the price stays down (a crash, not a bad print).
    for index in range(crash_index, len(candles)):
        candles[index] = make_candle(candles[index].timestamp, price=55.0)

    validator = build_validator(heal_backoff_seconds=0.001)
    healed, report, attempts = await validator.validate_and_heal(
        "X", candles, echo_refetch(candles), quarantine_unhealable=True
    )

    assert report.passed
    # Nothing quarantined: the full history survives.
    assert len(healed) == len(candles)
    assert [record.result for record in attempts] == ["resolved"]
    # The event is still reported, just not as a blocking failure.
    outlier = [issue for issue in report.issues if issue.code is QCIssueCode.RETURN_OUTLIER]
    assert outlier and all(issue.severity is QCSeverity.WARNING for issue in outlier)


@pytest.mark.asyncio
async def test_unconfirmed_outlier_still_fails_the_block() -> None:
    """A bad print the exchange *corrects* must not be waved through."""
    candles = make_series(400)
    bad_index = 150
    corrupt = make_candle(candles[bad_index].timestamp, price=1_000.0)
    damaged = list(candles)
    damaged[bad_index] = corrupt

    validator = build_validator(heal_backoff_seconds=0.001)
    healed, report, _ = await validator.validate_and_heal(
        "X", damaged, echo_refetch(candles), quarantine_unhealable=False
    )

    assert report.passed
    # The clean copy replaced the glitch rather than the glitch being accepted.
    assert healed[bad_index].close == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# EXCESSIVE_ZERO_VOLUME must name its bars
# ---------------------------------------------------------------------------
def test_zero_volume_issue_names_the_offending_bars() -> None:
    """Without timestamps the healer had nothing to target and re-fetched the
    entire block (``bars_requested: 212401``, ``duration_seconds: 415``)."""
    candles = make_series(100)
    for index in range(0, 40):
        candles[index] = make_candle(candles[index].timestamp, volume=0.0)

    validator = build_validator()
    report = validator.validate_candles("X", candles)

    zero = next(i for i in report.issues if i.code is QCIssueCode.EXCESSIVE_ZERO_VOLUME)
    assert zero.severity is QCSeverity.CRITICAL
    assert len(zero.timestamps) == 40

    windows = validator._heal_windows(candles, report)
    requested = sum(max(1, (end - start) // TF_MS) for start, end in windows)
    assert requested < len(candles)  # targeted, not a whole-history sweep


def test_heal_windows_never_sweeps_the_whole_block() -> None:
    """A CRITICAL verdict naming no bars yields no plan at all."""
    candles = make_series(1_000)
    report = QCReport(
        symbol="X",
        checked_rows=1_000,
        issues=(
            QCIssue(
                code=QCIssueCode.EXCESSIVE_ZERO_VOLUME,
                severity=QCSeverity.CRITICAL,
                message="block-level verdict with no timestamps",
                symbol="X",
                healable=True,
            ),
        ),
    )
    assert build_validator()._heal_windows(candles, report) == []


def test_single_contiguous_run_is_split_to_the_window_cap() -> None:
    """One long damaged stretch used to escape ``max_heal_window_bars`` entirely."""
    validator = build_validator(max_heal_window_bars=50)
    candles = make_series(1_000)
    damaged = tuple(candle.timestamp for candle in candles[100:600])
    report = QCReport(symbol="X", checked_rows=1_000, missing_timestamps=damaged)

    windows = validator._heal_windows(candles, report)
    assert len(windows) > 1
    assert all((end - start) <= 50 * TF_MS for start, end in windows)


@pytest.mark.asyncio
async def test_thin_market_is_kept_once_the_exchange_confirms_it() -> None:
    """ZIL/1INCH/CAKE were failing bootstrap outright over genuinely thin trading."""
    candles = make_series(200)
    for index in range(0, 80):
        candles[index] = make_candle(candles[index].timestamp, volume=0.0)

    validator = build_validator(heal_backoff_seconds=0.001)
    healed, report, _ = await validator.validate_and_heal(
        "ZIL", candles, echo_refetch(candles), quarantine_unhealable=True
    )

    assert report.passed
    assert len(healed) == len(candles)
    zero = next(i for i in report.issues if i.code is QCIssueCode.EXCESSIVE_ZERO_VOLUME)
    assert zero.severity is QCSeverity.WARNING


# ---------------------------------------------------------------------------
# No-progress detection
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_identical_refetch_stops_after_one_round() -> None:
    """The log shows four byte-identical attempts per doomed symbol.

    Structural damage the exchange cannot repair must cost one round, not four.
    """
    candles = make_series(200)
    # Price geometry the exchange keeps re-serving, which healing cannot fix and
    # which confirmation does not excuse: a hard grid gap.
    damaged = [candle for index, candle in enumerate(candles) if index != 100]
    rounds: list[tuple[int, int]] = []

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        rounds.append((start_ms, end_ms))
        # The exchange answers, but the gap genuinely does not exist upstream.
        return [c for c in damaged if start_ms <= c.timestamp <= end_ms]

    validator = build_validator(max_heal_attempts=4, heal_backoff_seconds=0.001)
    with pytest.raises(DataIntegrityError) as exc_info:
        await validator.validate_and_heal("X", damaged, refetch, quarantine_unhealable=False)

    attempts = exc_info.value.context["heal_attempts"]
    assert len(attempts) == 1
    assert len(rounds) == 1


@pytest.mark.asyncio
async def test_empty_reply_is_still_retried() -> None:
    """An empty response is a transient miss, not confirmation - keep the backoff."""
    candles = make_series(200)
    damaged = [candle for index, candle in enumerate(candles) if index != 100]

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return []

    validator = build_validator(max_heal_attempts=3, heal_backoff_seconds=0.001)
    with pytest.raises(DataIntegrityError) as exc_info:
        await validator.validate_and_heal("X", damaged, refetch, quarantine_unhealable=False)

    assert len(exc_info.value.context["heal_attempts"]) == 3


@pytest.mark.asyncio
async def test_bars_invalid_after_heal_is_reported() -> None:
    """``bars_invalid_after_heal: 0`` next to ``result: still_invalid`` was a lie."""
    candles = make_series(200)
    damaged = [candle for index, candle in enumerate(candles) if index != 100]

    async def refetch(symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return [c for c in damaged if start_ms <= c.timestamp <= end_ms]

    validator = build_validator(max_heal_attempts=2, heal_backoff_seconds=0.001)
    with pytest.raises(DataIntegrityError) as exc_info:
        await validator.validate_and_heal("X", damaged, refetch, quarantine_unhealable=False)

    attempt = exc_info.value.context["heal_attempts"][0]
    assert attempt["result"] == "still_invalid"
    assert attempt["bars_invalid_after_heal"] > 0
