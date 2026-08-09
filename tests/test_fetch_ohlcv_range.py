"""Robustness of BinanceDataFetcher.fetch_ohlcv_range against silent truncation."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from config.settings import ExchangeSettings, Settings
from core.exceptions import DataFetchError
from module_a_data.fetcher import BinanceDataFetcher

TF_MS = 5 * 60 * 1_000
T0 = (1_700_000_000_000 // TF_MS) * TF_MS


class FakeExchange:
    """Duck-types just enough of ccxt's async binance client for these tests."""

    def __init__(self, responder: Callable[[int, int], list[list[Any]]]) -> None:
        self.markets: dict[str, Any] = {"BTC/USDT:USDT": {"id": "BTCUSDT"}}
        self._responder = responder
        self.calls: list[tuple[int, int]] = []

    async def load_markets(self, reload: bool = False) -> dict[str, Any]:
        return self.markets

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "5m", since: int | None = None, limit: int | None = None
    ) -> list[list[Any]]:
        self.calls.append((since, limit))
        return self._responder(since, limit)


def row(ts: int, price: float = 100.0, volume: float = 10.0) -> list[Any]:
    return [ts, price, price * 1.001, price * 0.999, price, volume]


def _settings(**overrides: object) -> Settings:
    exchange = ExchangeSettings(
        max_retries=2, backoff_base_seconds=0.001, backoff_max_seconds=0.01, request_rate_scale=1.0
    )
    return Settings(exchange=exchange, **overrides)


def build_fetcher(responder: Callable[[int, int], list[list[Any]]]) -> tuple[BinanceDataFetcher, FakeExchange]:
    fake = FakeExchange(responder)
    fetcher = BinanceDataFetcher(_settings(), exchange=fake)
    return fetcher, fake


# ---------------------------------------------------------------------------
# Happy path: full multi-page coverage, no gaps, no duplicates
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_range_is_fetched_completely_across_pages() -> None:
    total_bars = 1_000
    page_size = 300
    end_ts = T0 + (total_bars - 1) * TF_MS

    def responder(since: int, limit: int) -> list[list[Any]]:
        start = max(since, T0)
        return [row(start + i * TF_MS) for i in range(limit) if start + i * TF_MS <= end_ts]

    fetcher, fake = build_fetcher(responder)
    candles = await fetcher.fetch_ohlcv_range("BTC/USDT:USDT", T0, end_ts, page_limit=page_size)

    assert len(candles) == total_bars
    timestamps = [c.timestamp for c in candles]
    assert timestamps == sorted(set(timestamps))  # sorted, no duplicates
    assert timestamps[0] == T0
    assert timestamps[-1] == end_ts
    # Confirms real pagination happened rather than one lucky oversized call.
    assert len(fake.calls) >= total_bars // page_size


# ---------------------------------------------------------------------------
# A genuine exchange-side gap (halt/delisting window) is walked over cleanly
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_genuine_exchange_gap_is_skipped_without_error() -> None:
    end_ts = T0 + 299 * TF_MS
    gap_start = T0 + 100 * TF_MS
    gap_end = T0 + 109 * TF_MS  # candles 100..109 do not exist upstream

    def responder(since: int, limit: int) -> list[list[Any]]:
        start = max(since, T0)
        out: list[list[Any]] = []
        ts = start
        while len(out) < limit and ts <= end_ts:
            if not (gap_start <= ts <= gap_end):
                out.append(row(ts))
            ts += TF_MS
        return out

    fetcher, _ = build_fetcher(responder)
    candles = await fetcher.fetch_ohlcv_range("BTC/USDT:USDT", T0, end_ts, page_limit=50)

    timestamps = {c.timestamp for c in candles}
    for gapped in range(100, 110):
        assert (T0 + gapped * TF_MS) not in timestamps
    assert len(candles) == 300 - 10


# ---------------------------------------------------------------------------
# Rows that exist but all fail validation must not look like "end of history"
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_persistently_malformed_rows_raise_instead_of_truncating() -> None:
    def responder(since: int, limit: int) -> list[list[Any]]:
        # Negative price -> OHLCVCandle validation fails on every row, forever.
        return [row(since, price=-1.0)]

    fetcher, _ = build_fetcher(responder)
    with pytest.raises(DataFetchError, match="malformed"):
        await fetcher.fetch_ohlcv_range(
            "BTC/USDT:USDT", T0, T0 + 50 * TF_MS, page_limit=10
        )


@pytest.mark.asyncio
async def test_transient_malformed_page_recovers_on_retry() -> None:
    state = {"attempts": 0}
    end_ts = T0 + 9 * TF_MS

    def responder(since: int, limit: int) -> list[list[Any]]:
        if since == T0 and state["attempts"] < 2:
            state["attempts"] += 1
            return [row(T0, price=-1.0)]  # fails validation, twice
        start = max(since, T0)
        return [row(start + i * TF_MS) for i in range(limit) if start + i * TF_MS <= end_ts]

    fetcher, _ = build_fetcher(responder)
    candles = await fetcher.fetch_ohlcv_range("BTC/USDT:USDT", T0, end_ts, page_limit=20)
    assert len(candles) == 10
    assert state["attempts"] == 2


# ---------------------------------------------------------------------------
# The page-count safety guard must fail loudly, never return a silent partial
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stuck_pagination_raises_rather_than_truncating_silently() -> None:
    def responder(since: int, limit: int) -> list[list[Any]]:
        # Always advances by exactly one bar regardless of the requested page
        # size, so a huge requested range can never be covered within budget.
        return [row(since)]

    fetcher, _ = build_fetcher(responder)
    with pytest.raises(DataFetchError, match="safety guard"):
        await fetcher.fetch_ohlcv_range(
            "BTC/USDT:USDT", T0, T0 + 100_000 * TF_MS, page_limit=500
        )


# ---------------------------------------------------------------------------
# Genuine end of history (exchange returns nothing) stops cleanly
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_genuine_end_of_history_stops_cleanly() -> None:
    real_end = T0 + 19 * TF_MS

    def responder(since: int, limit: int) -> list[list[Any]]:
        start = max(since, T0)
        return [row(start + i * TF_MS) for i in range(limit) if start + i * TF_MS <= real_end]

    fetcher, _ = build_fetcher(responder)
    # Ask for far more than actually exists; the exchange runs dry and the call
    # must return exactly what's available rather than erroring or hanging.
    candles = await fetcher.fetch_ohlcv_range(
        "BTC/USDT:USDT", T0, T0 + 500 * TF_MS, page_limit=50
    )
    assert len(candles) == 20
    assert candles[-1].timestamp == real_end
