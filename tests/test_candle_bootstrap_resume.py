"""Regression: a failed bootstrap must not lock a symbol out of its own history.

Production diagnostic ``a19251cc``, per-symbol row counts::

    SOL   200,574      ZIL   115,600
    ...
    ALGO      469      ATOM      469      PIXEL      469

Three symbols with 1.6 days of history against ~120,000 bars for everyone else.
Their bootstrap had failed on the heal budget, so nothing historical was ever
written - but the live 5-minute cycle kept writing candles afterwards, which put
their *newest* timestamp at "now".  The resume rule read that as "already up to
date" and started the next fetch one bar after it, so the two-year hole was
never requested again, on that run or any future one.

This is the same defect ``test_futures_backfill_resume`` covers for the
derivatives feed; the candle path had it too and nobody had looked.
"""

from __future__ import annotations

import pytest

from config.settings import Settings
from core.utils import last_closed_candle_open_ms
from module_a_data.models import OHLCVCandle, QCReport
from module_a_data.pipeline import DataPipeline

TF_MS = 5 * 60 * 1_000
SYMBOL = "ALGO/USDT:USDT"
_BOOTSTRAP_BARS = 600


class RangeFetcher:
    """Serves a clean, dense candle run for whatever window it is asked for."""

    def __init__(self) -> None:
        self.windows: list[tuple[int, int]] = []

    async def fetch_ohlcv_range(
        self, symbol: str, *, start_ms: int, end_ms: int
    ) -> list[OHLCVCandle]:
        self.windows.append((start_ms, end_ms))
        return [
            OHLCVCandle(
                symbol=symbol,
                timestamp=ts,
                open=100.0,
                high=100.1,
                low=99.9,
                close=100.0,
                volume=10.0,
            )
            for ts in range(start_ms, end_ms + 1, TF_MS)
        ]


class GapDatabase:
    """A database whose stored candles start *after* the requested window."""

    def __init__(
        self,
        oldest: int | None,
        newest: int | None,
        state: dict[str, object] | None = None,
    ) -> None:
        self._oldest = oldest
        self._newest = newest
        self.written: list[OHLCVCandle] = []
        self.state: dict[str, object] = dict(state or {})

    async def earliest_candle_timestamp(self, symbol: str) -> int | None:
        return self._oldest

    async def latest_candle_timestamp(self, symbol: str) -> int | None:
        return self._newest

    async def upsert_candles(self, candles) -> int:
        self.written.extend(candles)
        return len(candles)

    async def get_state(self, key: str):
        return self.state.get(key)

    async def set_state(self, key: str, value) -> None:
        self.state[key] = value


class PassingValidator:
    """QC that always passes, so the test isolates the resume rule."""

    def validate_candles(self, symbol: str, candles, **_: object) -> QCReport:
        return QCReport(symbol=symbol, checked_rows=len(candles))


def build_pipeline(database: GapDatabase) -> tuple[DataPipeline, RangeFetcher]:
    settings = Settings(data={"history_bootstrap_candles": _BOOTSTRAP_BARS})
    fetcher = RangeFetcher()
    pipeline = DataPipeline(settings, fetcher, None, database)
    pipeline._validator = PassingValidator()  # type: ignore[assignment]
    return pipeline, fetcher


@pytest.mark.asyncio
async def test_recent_only_history_is_refetched_from_the_window_start() -> None:
    """The ALGO/ATOM/PIXEL case: live candles at 'now', nothing behind them."""
    now = last_closed_candle_open_ms(TF_MS)
    # 469 bars ending at 'now' - exactly what the live cycle leaves behind.
    database = GapDatabase(oldest=now - 469 * TF_MS, newest=now)
    pipeline, fetcher = build_pipeline(database)

    written = await pipeline.bootstrap_history([SYMBOL])

    assert written[SYMBOL] > 469, "the historical hole was skipped again"
    start_ms, end_ms = fetcher.windows[0]
    assert end_ms - start_ms == pytest.approx(_BOOTSTRAP_BARS * TF_MS, rel=0.01)


@pytest.mark.asyncio
async def test_empty_history_backfills_the_full_window() -> None:
    database = GapDatabase(oldest=None, newest=None)
    pipeline, fetcher = build_pipeline(database)

    await pipeline.bootstrap_history([SYMBOL])

    start_ms, end_ms = fetcher.windows[0]
    assert end_ms - start_ms == pytest.approx(_BOOTSTRAP_BARS * TF_MS, rel=0.01)


@pytest.mark.asyncio
async def test_complete_history_stays_incremental() -> None:
    """A symbol that really is backfilled must not refetch two years every run."""
    now = last_closed_candle_open_ms(TF_MS)
    database = GapDatabase(
        oldest=now - (_BOOTSTRAP_BARS + 50) * TF_MS,
        newest=now - 10 * TF_MS,
    )
    pipeline, fetcher = build_pipeline(database)

    await pipeline.bootstrap_history([SYMBOL])

    start_ms, _ = fetcher.windows[0]
    assert start_ms == now - 9 * TF_MS, "an up-to-date symbol refetched its whole history"


@pytest.mark.asyncio
async def test_fully_current_symbol_fetches_nothing() -> None:
    now = last_closed_candle_open_ms(TF_MS)
    database = GapDatabase(oldest=now - (_BOOTSTRAP_BARS + 50) * TF_MS, newest=now)
    pipeline, fetcher = build_pipeline(database)

    written = await pipeline.bootstrap_history([SYMBOL])

    assert written[SYMBOL] == 0
    assert fetcher.windows == []


@pytest.mark.asyncio
async def test_a_late_listing_is_recorded_and_not_refetched_next_run() -> None:
    """The other half: most of this universe listed after the window opens.

    Their history is short because it does not exist, not because it failed. A
    rule that refetched on "oldest is later than the window start" alone would
    page through their pre-listing months on every single run, forever.
    """
    now = last_closed_candle_open_ms(TF_MS)
    listed_at = now - 200 * TF_MS  # well inside the 600-bar window
    database = GapDatabase(oldest=listed_at, newest=now)
    pipeline, fetcher = build_pipeline(database)

    # First run: no floor recorded, so the full window is attempted once.
    await pipeline.bootstrap_history([SYMBOL])
    assert len(fetcher.windows) == 1
    first_start, _ = fetcher.windows[0]
    assert first_start == now - _BOOTSTRAP_BARS * TF_MS

    # The attempt established where the symbol really starts.
    floors = database.state["bootstrap_history_floors"]["floors"]
    assert floors[SYMBOL] == first_start

    # Second run: the symbol sits at its known floor, so it stays incremental.
    database._oldest = first_start
    pipeline2, fetcher2 = build_pipeline(database)
    await pipeline2.bootstrap_history([SYMBOL])
    assert fetcher2.windows == [], "a late-listed symbol refetched its whole window again"


@pytest.mark.asyncio
async def test_a_failed_download_records_no_floor_so_it_stays_retryable() -> None:
    """A symbol that fetched nothing must not be marked as 'this is all there is'."""
    now = last_closed_candle_open_ms(TF_MS)
    database = GapDatabase(oldest=now - 469 * TF_MS, newest=now)
    pipeline, fetcher = build_pipeline(database)

    async def _empty(symbol: str, *, start_ms: int, end_ms: int):
        fetcher.windows.append((start_ms, end_ms))
        return []

    pipeline._fetcher.fetch_ohlcv_range = _empty  # type: ignore[method-assign]

    await pipeline.bootstrap_history([SYMBOL])

    assert "bootstrap_history_floors" not in database.state
