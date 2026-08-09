"""End-to-end: bootstrap must persist clean history instead of discarding it all.

Exercises module_a_data/pipeline.py::DataPipeline.bootstrap_history against a
fetcher whose data has a permanently unfetchable gap (an exchange halt or a
pre-listing window that can never be healed by re-requesting it).  Before the
quarantine wiring, a single such gap made `validate_and_heal` raise and the
bootstrap discarded *everything* it had fetched for that symbol - every single
run, forever.  It must now persist the clean trailing run instead.
"""

from __future__ import annotations

import pytest

from config.settings import QCSettings, Settings
from core.utils import last_closed_candle_open_ms
from module_a_data.models import OHLCVCandle
from module_a_data.pipeline import DataPipeline
from module_a_data.qc_validator import QCValidator

TF_MS = 5 * 60 * 1_000
SYMBOL = "BTC/USDT:USDT"


def make_candle(ts: int, price: float = 100.0) -> OHLCVCandle:
    return OHLCVCandle(
        symbol=SYMBOL, timestamp=ts, open=price, high=price * 1.001,
        low=price * 0.999, close=price, volume=10.0,
    )


class GappyFetcher:
    """Serves a fixed history with one permanent hole, regardless of the window asked for."""

    def __init__(self, full: list[OHLCVCandle], gap: tuple[int, int]) -> None:
        self._full = full
        self._gap_start, self._gap_end = gap

    async def fetch_ohlcv_range(self, symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        return [
            c for c in self._full
            if start_ms <= c.timestamp <= end_ms and not (self._gap_start <= c.timestamp <= self._gap_end)
        ]


class RecordingDatabase:
    def __init__(self) -> None:
        self.written: dict[str, list[OHLCVCandle]] = {}

    async def latest_candle_timestamp(self, symbol: str) -> int | None:
        return None

    async def upsert_candles(self, candles: list[OHLCVCandle]) -> int:
        self.written.setdefault(SYMBOL, []).extend(candles)
        return len(candles)


@pytest.mark.asyncio
async def test_bootstrap_persists_clean_trailing_run_despite_unfixable_gap() -> None:
    end_ms = last_closed_candle_open_ms(TF_MS)
    n = 500
    base = end_ms - (n - 1) * TF_MS
    full = [make_candle(base + i * TF_MS) for i in range(n)]

    gap_start = base + 20 * TF_MS
    gap_end = base + 29 * TF_MS  # 10 permanently unfetchable candles

    settings = Settings(
        data={"history_bootstrap_candles": n},
        qc=QCSettings(max_heal_attempts=2, heal_backoff_seconds=0.001),
    )
    fetcher = GappyFetcher(full, (gap_start, gap_end))
    validator = QCValidator(settings)
    database = RecordingDatabase()
    pipeline = DataPipeline(settings, fetcher, validator, database)

    written = await pipeline.bootstrap_history([SYMBOL])

    # The whole symbol must not be discarded just because an old window is gone.
    assert written[SYMBOL] > 0
    persisted = database.written[SYMBOL]
    persisted_timestamps = {c.timestamp for c in persisted}
    # The permanently-gapped bars are excluded...
    assert gap_start not in persisted_timestamps
    # ...but the clean run after the gap survives, up to the newest candle.
    assert (base + (n - 1) * TF_MS) in persisted_timestamps
    assert len(persisted) == n - 30  # everything from index 30 onward (0..29 dropped)
