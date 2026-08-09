"""Regression test for a real production failure: bulk-upserting a full
history (tens of thousands of candles) in one `.values(list)` INSERT exceeds
SQLite's bound-parameter ceiling. Uses a real (temp-file) SQLite database via
DatabaseHandler - not a mock - so this proves the fix against actual SQLite
behaviour rather than an assumption about the driver.
"""

from __future__ import annotations

import pytest

from config.settings import DatabaseSettings, Settings
from core.utils import last_closed_candle_open_ms
from module_a_data.db_handler import DatabaseHandler
from module_a_data.models import OHLCVCandle

TF_MS = 5 * 60 * 1_000
SYMBOL = "BTC/USDT:USDT"


def make_candle(ts: int, price: float = 100.0) -> OHLCVCandle:
    return OHLCVCandle(
        symbol=SYMBOL, timestamp=ts, open=price, high=price * 1.001,
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
