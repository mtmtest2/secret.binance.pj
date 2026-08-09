"""Regression test for a real production failure: bulk-upserting a full
history (tens of thousands of candles) in one `.values(list)` INSERT exceeds
SQLite's bound-parameter ceiling. Uses a real (temp-file) SQLite database via
DatabaseHandler - not a mock - so this proves the fix against actual SQLite
behaviour rather than an assumption about the driver.
"""

from __future__ import annotations

import asyncio

import pytest

from config.settings import DatabaseSettings, Settings
from core.utils import last_closed_candle_open_ms
from module_a_data.db_handler import DatabaseHandler
from module_a_data.models import OHLCVCandle

TF_MS = 5 * 60 * 1_000
SYMBOL = "BTC/USDT:USDT"
SYMBOLS = tuple(f"SYM{i}/USDT:USDT" for i in range(12))


def make_candle(ts: int, price: float = 100.0, symbol: str = SYMBOL) -> OHLCVCandle:
    return OHLCVCandle(
        symbol=symbol, timestamp=ts, open=price, high=price * 1.001,
        low=price * 0.999, close=price, volume=10.0,
    )


@pytest.mark.asyncio
async def test_large_bulk_upsert_succeeds_against_real_sqlite(tmp_path) -> None:
    settings = Settings(db=DatabaseSettings(path=tmp_path / "quant.db"))
    db = DatabaseHandler(settings)
    await db.initialize()
    try:
        # Larger than any single-statement variable ceiling (999 or 32766)
        # once multiplied by 8 columns/row - this reproduces the production
        # failure ("candle upsert failed (rows=87060)") against real SQLite.
        n = 60_000
        end_ms = last_closed_candle_open_ms(TF_MS)
        base = end_ms - (n - 1) * TF_MS
        candles = [make_candle(base + i * TF_MS) for i in range(n)]

        written = await db.upsert_candles(candles)
        assert written == n
        assert await db.candle_count(SYMBOL) == n

        # Idempotent re-upsert (the ON CONFLICT DO UPDATE path) must also
        # survive the same batching without duplicating rows.
        written_again = await db.upsert_candles(candles)
        assert written_again == n
        assert await db.candle_count(SYMBOL) == n
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_bulk_upsert_preserves_all_rows_correctly(tmp_path) -> None:
    """Batching must not drop, duplicate or misassign rows at batch boundaries."""
    settings = Settings(db=DatabaseSettings(path=tmp_path / "quant.db"))
    db = DatabaseHandler(settings)
    await db.initialize()
    try:
        n = 733  # deliberately not a multiple of the batch size
        end_ms = last_closed_candle_open_ms(TF_MS)
        base = end_ms - (n - 1) * TF_MS
        candles = [make_candle(base + i * TF_MS, price=100.0 + i) for i in range(n)]

        await db.upsert_candles(candles)
        frame = await db.load_ohlcv_dataframe(SYMBOL, limit=n)

        assert len(frame) == n
        assert frame["timestamp"].tolist() == [c.timestamp for c in candles]
        assert frame["close"].tolist() == [c.close for c in candles]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_concurrent_large_upserts_never_hit_database_locked(tmp_path) -> None:
    """Regression: chunked upserts hold a write transaction open across many
    sequential statements, so many symbols bootstrapping concurrently (as
    DataPipeline.bootstrap_history does via asyncio.gather) could contend for
    SQLite's single writer lock long enough to exceed busy_timeout_ms and fail
    with "database is locked". A very small busy_timeout here means SQLite's
    own retry-on-busy window is too short to paper over any contention that
    slips past the application-level write lock - if DatabaseHandler did not
    serialise writes itself, this test would fail intermittently.
    """
    settings = Settings(
        db=DatabaseSettings(path=tmp_path / "quant.db", busy_timeout_ms=100)
    )
    db = DatabaseHandler(settings)
    await db.initialize()
    try:
        n = 4_000
        end_ms = last_closed_candle_open_ms(TF_MS)
        base = end_ms - (n - 1) * TF_MS

        async def upsert_one(symbol: str) -> int:
            candles = [make_candle(base + i * TF_MS, symbol=symbol) for i in range(n)]
            return await db.upsert_candles(candles)

        results = await asyncio.gather(*(upsert_one(symbol) for symbol in SYMBOLS))

        assert results == [n] * len(SYMBOLS)
        for symbol in SYMBOLS:
            assert await db.candle_count(symbol) == n
    finally:
        await db.close()
