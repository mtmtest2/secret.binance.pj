"""End-to-end: DataPipeline.run_cycle records and persists real QC/heal
telemetry and per-symbol exclusion reasons (module_a_data/pipeline.py).

Before this, a symbol excluded by QC vanished into an ERROR log line with no
structured record of *why* - nothing a diagnostic report could read back.
"""

from __future__ import annotations

from typing import Any

import pytest

from config.settings import QCSettings, Settings
from core.utils import last_closed_candle_open_ms
from module_a_data.models import (
    FuturesMetrics,
    OHLCVCandle,
    OrderBookSnapshot,
)
from module_a_data.pipeline import (
    HEAL_TELEMETRY_STATE_KEY,
    SYMBOL_EXCLUSION_STATE_KEY,
    DataPipeline,
)
from module_a_data.qc_validator import QCValidator

TF_MS = 5 * 60 * 1_000
GOOD_SYMBOL = "BTC/USDT:USDT"
BAD_SYMBOL = "ETH/USDT:USDT"


def make_candle(ts: int, price: float = 100.0) -> OHLCVCandle:
    return OHLCVCandle(
        symbol="XXX", timestamp=ts, open=price, high=price * 1.001,
        low=price * 0.999, close=price, volume=10.0,
    )


def make_series(n: int, end: int) -> list[OHLCVCandle]:
    base = end - (n - 1) * TF_MS
    return [make_candle(base + i * TF_MS) for i in range(n)]


class FakeFetcher:
    """Serves clean history for GOOD_SYMBOL and a permanently unfetchable
    gap for BAD_SYMBOL, so healing is attempted and then genuinely fails."""

    def __init__(self) -> None:
        end_ms = last_closed_candle_open_ms(TF_MS)
        self.good_series = make_series(40, end_ms)
        full_bad = make_series(40, end_ms)
        self.bad_gap = (full_bad[10].timestamp, full_bad[14].timestamp)
        self.bad_series = [c for c in full_bad if not (self.bad_gap[0] <= c.timestamp <= self.bad_gap[1])]

    async def fetch_ohlcv(self, symbol: str) -> list[OHLCVCandle]:
        return self.good_series if symbol == GOOD_SYMBOL else self.bad_series

    async def fetch_agg_trade_flow(self, symbol: str, start_ms: int, end_ms: int) -> list[object]:
        # Order flow is additive to this test's subject (QC telemetry); an
        # empty result exercises the "nothing to write" path without noise.
        return []

    async def fetch_ohlcv_range(self, symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        # The gap is permanently unfetchable - the exchange genuinely has nothing there.
        return [
            c for c in self.good_series
            if symbol == GOOD_SYMBOL and start_ms <= c.timestamp <= end_ms
        ]

    async def fetch_order_book(self, symbol: str) -> OrderBookSnapshot | None:
        return None

    async def fetch_futures_metrics(self, symbol: str) -> FuturesMetrics | None:
        return None


class FakeDatabase:
    """In-memory stand-in for DatabaseHandler's candle/state operations."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}
        self.written: dict[str, int] = {}

    async def upsert_candles(self, candles: list[OHLCVCandle]) -> int:
        if candles:
            self.written[candles[0].symbol] = self.written.get(candles[0].symbol, 0) + len(candles)
        return len(candles)

    async def upsert_order_book(self, book: OrderBookSnapshot) -> None:  # pragma: no cover - unused
        raise AssertionError("order book upsert should not be called in this test")

    async def upsert_futures_metrics(self, metrics: FuturesMetrics) -> None:  # pragma: no cover
        raise AssertionError("futures metrics upsert should not be called in this test")

    async def upsert_agg_trade_flow(self, buckets: list[Any]) -> int:
        # The live cycle always refreshes the trailing closed order-flow
        # buckets; with the fake fetcher returning none, this is a no-op that
        # simply must not raise.
        self.agg_flow_writes = getattr(self, "agg_flow_writes", 0) + len(buckets)
        return len(buckets)

    async def get_state(self, key: str) -> dict[str, Any] | None:
        return self._state.get(key)

    async def set_state(self, key: str, value: dict[str, Any]) -> None:
        self._state[key] = value


@pytest.mark.asyncio
async def test_run_cycle_records_exclusion_and_persists_telemetry() -> None:
    settings = Settings(qc=QCSettings(max_heal_attempts=2, heal_backoff_seconds=0.001))
    fetcher = FakeFetcher()
    validator = QCValidator(settings)
    database = FakeDatabase()
    pipeline = DataPipeline(settings, fetcher, validator, database)

    bundles = await pipeline.run_cycle([GOOD_SYMBOL, BAD_SYMBOL])

    assert GOOD_SYMBOL in bundles
    assert BAD_SYMBOL not in bundles
    assert pipeline.last_cycle_symbols_ok == 1
    assert pipeline.last_cycle_symbols_failed == 1

    # In-memory per-cycle telemetry.
    assert len(pipeline.last_cycle_exclusions) == 1
    exclusion = pipeline.last_cycle_exclusions[0]
    assert exclusion["symbol"] == BAD_SYMBOL
    assert exclusion["reason"]  # a real, non-empty reason string
    assert len(pipeline.last_cycle_heal_attempts) >= 1
    assert all(record["symbol"] == BAD_SYMBOL for record in pipeline.last_cycle_heal_attempts)

    # Persisted, bounded, durable history a diagnostic report can read back.
    heal_state = await database.get_state(HEAL_TELEMETRY_STATE_KEY)
    exclusion_state = await database.get_state(SYMBOL_EXCLUSION_STATE_KEY)
    assert heal_state is not None and heal_state["records"]
    assert exclusion_state is not None
    assert exclusion_state["records"][-1]["symbol"] == BAD_SYMBOL


@pytest.mark.asyncio
async def test_run_cycle_with_no_damage_persists_nothing() -> None:
    """A cycle with nothing to heal must not write empty telemetry blobs."""
    settings = Settings(qc=QCSettings())
    fetcher = FakeFetcher()
    validator = QCValidator(settings)
    database = FakeDatabase()
    pipeline = DataPipeline(settings, fetcher, validator, database)

    await pipeline.run_cycle([GOOD_SYMBOL])

    assert pipeline.last_cycle_heal_attempts == []
    assert pipeline.last_cycle_exclusions == []
    assert await database.get_state(HEAL_TELEMETRY_STATE_KEY) is None
    assert await database.get_state(SYMBOL_EXCLUSION_STATE_KEY) is None
