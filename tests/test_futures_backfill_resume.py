"""Regression: the futures-metrics backfill must not be blocked by live snapshots.

Production log, twice per run::

    Futures-metrics backfill complete: 0 row(s) across 27 symbol(s)

The resume point was taken from the *newest* stored row, but the live 5-minute
cycle writes a "now" snapshot on every pass - so ``newest`` was always the
present moment, ``start_ms`` always exceeded ``end_ms``, and the historical
window was never filled.  ``funding_rate``, ``open_interest_change``,
``long_short_ratio`` and ``taker_buy_sell_ratio`` therefore sat at their neutral
defaults for essentially every training row: 4 of the 50 features, dead.
"""

from __future__ import annotations

import pytest

from config.settings import Settings
from core.utils import last_closed_candle_open_ms
from module_a_data.models import FuturesMetrics
from module_a_data.pipeline import DataPipeline

TF_MS = 5 * 60 * 1_000
SYMBOL = "BTC/USDT:USDT"


class SeriesFetcher:
    """Serves a dense metric history for whatever window it is asked for."""

    def __init__(self) -> None:
        self.windows: list[tuple[int, int]] = []

    def _series(self, start_ms: int, end_ms: int, value: float) -> list[tuple[int, float]]:
        return [(ts, value) for ts in range(start_ms, end_ms + 1, TF_MS)]

    async def fetch_funding_rate_history(self, symbol, start_ms, end_ms):
        self.windows.append((start_ms, end_ms))
        return self._series(start_ms, end_ms, 0.0001)

    async def fetch_open_interest_history(self, symbol, start_ms, end_ms):
        return self._series(start_ms, end_ms, 1_000.0)

    async def fetch_long_short_ratio_history(self, symbol, start_ms, end_ms):
        return self._series(start_ms, end_ms, 1.2)

    async def fetch_taker_ratio_history(self, symbol, start_ms, end_ms):
        return self._series(start_ms, end_ms, 0.9)


class SnapshotDatabase:
    """A database holding only what the live cycle wrote: one row at 'now'."""

    def __init__(self, oldest: int | None, newest: int | None) -> None:
        self._oldest = oldest
        self._newest = newest
        self.written: list[FuturesMetrics] = []

    async def earliest_futures_metrics_timestamp(self, symbol: str) -> int | None:
        return self._oldest

    async def latest_futures_metrics_timestamp(self, symbol: str) -> int | None:
        return self._newest

    async def upsert_futures_metrics_batch(self, records) -> int:
        self.written.extend(records)
        return len(records)


def build_pipeline(database) -> tuple[DataPipeline, SeriesFetcher]:
    settings = Settings(data={"history_bootstrap_candles": 600})
    fetcher = SeriesFetcher()
    return DataPipeline(settings, fetcher, None, database), fetcher


@pytest.mark.asyncio
async def test_live_now_snapshot_does_not_block_the_backfill() -> None:
    """One row at 'now' must still leave the whole history to backfill."""
    now = last_closed_candle_open_ms(TF_MS)
    database = SnapshotDatabase(oldest=now, newest=now)
    pipeline, fetcher = build_pipeline(database)

    written = await pipeline.backfill_futures_metrics([SYMBOL])

    assert written[SYMBOL] > 0
    assert database.written, "the historical window was skipped entirely"
    # The request covered the configured window, not a zero-width slice after 'now'.
    start_ms, end_ms = fetcher.windows[0]
    assert end_ms - start_ms == pytest.approx(600 * TF_MS, rel=0.01)


@pytest.mark.asyncio
async def test_empty_history_backfills_the_full_window() -> None:
    database = SnapshotDatabase(oldest=None, newest=None)
    pipeline, fetcher = build_pipeline(database)

    written = await pipeline.backfill_futures_metrics([SYMBOL])

    assert written[SYMBOL] > 0
    start_ms, end_ms = fetcher.windows[0]
    assert end_ms - start_ms == pytest.approx(600 * TF_MS, rel=0.01)


@pytest.mark.asyncio
async def test_already_covered_history_resumes_at_the_leading_edge() -> None:
    """When the stored history really does reach back far enough, stay incremental."""
    now = last_closed_candle_open_ms(TF_MS)
    database = SnapshotDatabase(oldest=now - 900 * TF_MS, newest=now - 10 * TF_MS)
    pipeline, fetcher = build_pipeline(database)

    await pipeline.backfill_futures_metrics([SYMBOL])

    start_ms, end_ms = fetcher.windows[0]
    assert start_ms == now - 10 * TF_MS + 1
