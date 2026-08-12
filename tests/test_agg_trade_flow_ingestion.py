"""aggTrades -> 5-minute order-flow buckets: fetch, bucket, persist, resume.

The aggressor-side convention is the whole point of this source, so it is
pinned explicitly: Binance's ``m`` field is ``isBuyerMaker``, meaning ``true``
is a *seller*-initiated print. Getting that backwards would invert every
order-flow feature while leaving all the shapes and ranges looking perfectly
healthy - a silent sign error is exactly the kind of bug that survives review.
"""

from __future__ import annotations

from typing import Any

import pytest

from config.settings import Settings
from module_a_data.fetcher import BinanceDataFetcher
from module_a_data.models import AggTradeFlow

_BUCKET_MS = 5 * 60 * 1_000
_START = 1_700_000_000_000 // _BUCKET_MS * _BUCKET_MS


class _FakeExchange:
    """Serves canned aggTrades pages and records the requests made."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.requests: list[dict[str, Any]] = []
        self.markets: dict[str, Any] = {"BTC/USDT:USDT": {"id": "BTCUSDT"}}
        self.options: dict[str, Any] = {}
        self.has: dict[str, Any] = {}

    async def load_markets(self, reload: bool = False) -> dict[str, Any]:
        return self.markets

    def set_sandbox_mode(self, enabled: bool) -> None:  # pragma: no cover - unused
        return None

    async def fapiPublicGetAggTrades(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.requests.append(dict(params))
        start, end = int(params["startTime"]), int(params["endTime"])
        return [row for row in self._rows if start <= int(row["T"]) < end]

    async def close(self) -> None:  # pragma: no cover - unused
        return None


def _fetcher(rows: list[dict[str, Any]]) -> tuple[BinanceDataFetcher, _FakeExchange]:
    exchange = _FakeExchange(rows)
    fetcher = BinanceDataFetcher(Settings(), exchange=exchange)  # type: ignore[arg-type]
    fetcher._markets_loaded = True  # noqa: SLF001 - bypass the network in tests
    return fetcher, exchange


def _trade(offset_ms: int, quantity: float, buyer_is_maker: bool, price: float = 100.0) -> dict:
    return {"T": _START + offset_ms, "q": str(quantity), "p": str(price), "m": buyer_is_maker}


# ---------------------------------------------------------------------------
# Aggressor side
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_is_buyer_maker_false_counts_as_aggressive_buying() -> None:
    """``m == False``: the buyer lifted the offer, so it is buy volume."""
    fetcher, _ = _fetcher([_trade(0, 3.0, buyer_is_maker=False)])
    buckets = await fetcher.fetch_agg_trade_flow(
        "BTC/USDT:USDT", start_ms=_START, end_ms=_START + _BUCKET_MS
    )
    assert len(buckets) == 1
    assert buckets[0].buy_volume == pytest.approx(3.0)
    assert buckets[0].sell_volume == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_is_buyer_maker_true_counts_as_aggressive_selling() -> None:
    """``m == True``: the seller hit the bid, so it is sell volume."""
    fetcher, _ = _fetcher([_trade(0, 3.0, buyer_is_maker=True)])
    buckets = await fetcher.fetch_agg_trade_flow(
        "BTC/USDT:USDT", start_ms=_START, end_ms=_START + _BUCKET_MS
    )
    assert buckets[0].sell_volume == pytest.approx(3.0)
    assert buckets[0].buy_volume == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_trades_are_folded_into_their_own_five_minute_bucket() -> None:
    """Prints land in the bucket their timestamp falls in, and nowhere else."""
    rows = [
        _trade(0, 1.0, buyer_is_maker=False),
        _trade(60_000, 2.0, buyer_is_maker=True),
        _trade(_BUCKET_MS + 1_000, 5.0, buyer_is_maker=False),
    ]
    fetcher, _ = _fetcher(rows)
    buckets = await fetcher.fetch_agg_trade_flow(
        "BTC/USDT:USDT", start_ms=_START, end_ms=_START + 2 * _BUCKET_MS
    )

    assert [bucket.timestamp for bucket in buckets] == [_START, _START + _BUCKET_MS]
    assert buckets[0].buy_volume == pytest.approx(1.0)
    assert buckets[0].sell_volume == pytest.approx(2.0)
    assert buckets[0].trades == 2
    assert buckets[0].order_flow_imbalance == pytest.approx((1.0 - 2.0) / 3.0)
    assert buckets[1].buy_volume == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_partially_observed_trailing_bucket_is_never_emitted() -> None:
    """Only fully closed buckets may reach the feature stack.

    A bucket still being filled would otherwise arrive as a partial
    observation - live-only data that the backtest could never reproduce.
    """
    rows = [_trade(0, 1.0, buyer_is_maker=False), _trade(_BUCKET_MS + 10, 9.0, buyer_is_maker=False)]
    fetcher, _ = _fetcher(rows)
    # end_ms lands mid-way through the second bucket, so only the first closes.
    buckets = await fetcher.fetch_agg_trade_flow(
        "BTC/USDT:USDT", start_ms=_START, end_ms=_START + _BUCKET_MS + 60_000
    )
    assert [bucket.timestamp for bucket in buckets] == [_START]


@pytest.mark.asyncio
async def test_empty_intervals_produce_no_bucket() -> None:
    """Nothing traded means no row, which the feature layer reads as zero flow."""
    fetcher, _ = _fetcher([])
    buckets = await fetcher.fetch_agg_trade_flow(
        "BTC/USDT:USDT", start_ms=_START, end_ms=_START + 4 * _BUCKET_MS
    )
    assert buckets == []


@pytest.mark.asyncio
async def test_request_window_never_exceeds_binance_one_hour_cap() -> None:
    """Binance rejects an aggTrades window wider than an hour."""
    fetcher, exchange = _fetcher([])
    await fetcher.fetch_agg_trade_flow(
        "BTC/USDT:USDT", start_ms=_START, end_ms=_START + 6 * 60 * 60 * 1_000
    )
    assert exchange.requests
    for request in exchange.requests:
        assert int(request["endTime"]) - int(request["startTime"]) <= 60 * 60 * 1_000


@pytest.mark.asyncio
async def test_walk_advances_past_quiet_hours_instead_of_stalling() -> None:
    """An illiquid stretch must not pin the cursor in place."""
    far = 5 * 60 * 60 * 1_000  # a print 5 hours in, with silence before it
    fetcher, exchange = _fetcher([_trade(far, 2.0, buyer_is_maker=False)])
    buckets = await fetcher.fetch_agg_trade_flow(
        "BTC/USDT:USDT", start_ms=_START, end_ms=_START + 6 * 60 * 60 * 1_000
    )
    assert [bucket.timestamp for bucket in buckets] == [_START + (far // _BUCKET_MS) * _BUCKET_MS]
    # Bounded work: one request per hourly window, not an unbounded spin.
    assert len(exchange.requests) <= 7


# ---------------------------------------------------------------------------
# Model invariants
# ---------------------------------------------------------------------------
def test_empty_bucket_imbalance_is_zero_not_undefined() -> None:
    bucket = AggTradeFlow(symbol="BTC/USDT:USDT", timestamp=_START)
    assert bucket.order_flow_imbalance == 0.0
    assert bucket.volume_delta == 0.0


def test_imbalance_is_bounded_by_construction() -> None:
    buy_only = AggTradeFlow(symbol="BTC/USDT:USDT", timestamp=_START, buy_volume=10.0)
    sell_only = AggTradeFlow(symbol="BTC/USDT:USDT", timestamp=_START, sell_volume=10.0)
    assert buy_only.order_flow_imbalance == pytest.approx(1.0)
    assert sell_only.order_flow_imbalance == pytest.approx(-1.0)
