"""Binance ``data.binance.vision`` archive loader.

The tests that matter most here are the *convention* tests.  A sign error in
the liquidation side mapping, or a transposition of the bid/ask columns, would
invert the resulting feature while leaving every shape, range and null count
looking perfectly healthy - the model would train happily on an exactly-wrong
signal and nothing downstream would flag it.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_a_data.archive_loader import (
    DATASET_BOOK_TICKER,
    DATASET_LIQUIDATION,
    ArchiveCoverage,
    BinanceArchiveLoader,
    _days_between,
    _read_csv_chunks,
    aggregate_book_ticker,
    aggregate_liquidations,
    coverage_summary,
    merge_microstructure,
    to_archive_symbol,
)

_BUCKET_MS = 300_000
#: 2026-01-05T00:00:00Z, an exact 5m boundary.
_DAY_START = 1767571200000


def _book_chunk(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("unified", "expected"),
    [
        ("SOL/USDT:USDT", "SOLUSDT"),
        ("1000SHIB/USDT:USDT", "1000SHIBUSDT"),
        ("BTCUSDT", "BTCUSDT"),
        ("btc/usdt:usdt", "BTCUSDT"),
    ],
)
def test_to_archive_symbol(unified: str, expected: str) -> None:
    assert to_archive_symbol(unified) == expected


# ---------------------------------------------------------------------------
# bookTicker aggregation
# ---------------------------------------------------------------------------
def test_book_ticker_aggregates_onto_the_5m_grid() -> None:
    chunk = _book_chunk(
        [
            # Two events inside bucket 0, one inside bucket 1.
            {
                "best_bid_price": 100.0,
                "best_bid_qty": 30.0,
                "best_ask_price": 100.1,
                "best_ask_qty": 10.0,
                "transaction_time": _DAY_START + 1_000,
            },
            {
                "best_bid_price": 100.0,
                "best_bid_qty": 10.0,
                "best_ask_price": 100.1,
                "best_ask_qty": 30.0,
                "transaction_time": _DAY_START + 2_000,
            },
            {
                "best_bid_price": 200.0,
                "best_bid_qty": 50.0,
                "best_ask_price": 200.2,
                "best_ask_qty": 50.0,
                "transaction_time": _DAY_START + _BUCKET_MS + 1_000,
            },
        ]
    )
    result = aggregate_book_ticker([chunk])

    assert list(result["timestamp"]) == [_DAY_START, _DAY_START + _BUCKET_MS]
    # Bucket 0 averages (30, 10) and (10, 30) -> (20, 20).
    assert result.loc[0, "bid_qty"] == pytest.approx(20.0)
    assert result.loc[0, "ask_qty"] == pytest.approx(20.0)
    # 0.1 on a 100.05 mid is ~9.995 bps.
    assert result.loc[0, "spread_bps"] == pytest.approx(9.995, abs=0.01)


def test_book_ticker_preserves_the_bid_ask_orientation() -> None:
    """A bid-heavy book must produce bid_qty > ask_qty, not the reverse.

    This is the transposition guard: swapping the two columns leaves the frame
    structurally identical and silently inverts every downstream imbalance.
    """
    chunk = _book_chunk(
        [
            {
                "best_bid_price": 10.0,
                "best_bid_qty": 900.0,
                "best_ask_price": 10.01,
                "best_ask_qty": 100.0,
                "transaction_time": _DAY_START,
            }
        ]
    )
    result = aggregate_book_ticker([chunk])
    assert result.loc[0, "bid_qty"] == pytest.approx(900.0)
    assert result.loc[0, "ask_qty"] == pytest.approx(100.0)

    # And the feature layer's derivation of it must come out positive.
    imbalance = (result.loc[0, "bid_qty"] - result.loc[0, "ask_qty"]) / (
        result.loc[0, "bid_qty"] + result.loc[0, "ask_qty"]
    )
    assert imbalance > 0.0


def test_book_ticker_discards_crossed_books() -> None:
    """A crossed book (ask < bid) is a feed artifact and must not average in."""
    chunk = _book_chunk(
        [
            {
                "best_bid_price": 100.2,
                "best_bid_qty": 5.0,
                "best_ask_price": 100.0,
                "best_ask_qty": 5.0,
                "transaction_time": _DAY_START,
            },
            {
                "best_bid_price": 100.0,
                "best_bid_qty": 7.0,
                "best_ask_price": 100.1,
                "best_ask_qty": 3.0,
                "transaction_time": _DAY_START + 1_000,
            },
        ]
    )
    result = aggregate_book_ticker([chunk])
    assert len(result) == 1
    assert result.loc[0, "bid_qty"] == pytest.approx(7.0)


def test_book_ticker_accumulates_across_chunks() -> None:
    """Chunked reading must not reset a bucket's running mean."""
    first = _book_chunk(
        [
            {
                "best_bid_price": 100.0,
                "best_bid_qty": 30.0,
                "best_ask_price": 100.1,
                "best_ask_qty": 10.0,
                "transaction_time": _DAY_START,
            }
        ]
    )
    second = _book_chunk(
        [
            {
                "best_bid_price": 100.0,
                "best_bid_qty": 10.0,
                "best_ask_price": 100.1,
                "best_ask_qty": 30.0,
                "transaction_time": _DAY_START + 1_000,
            }
        ]
    )
    chunked = aggregate_book_ticker([first, second])
    combined = aggregate_book_ticker([pd.concat([first, second], ignore_index=True)])

    pd.testing.assert_frame_equal(chunked, combined)


def test_book_ticker_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        aggregate_book_ticker([pd.DataFrame({"transaction_time": [_DAY_START]})])


# ---------------------------------------------------------------------------
# liquidationSnapshot aggregation
# ---------------------------------------------------------------------------
def test_liquidation_side_convention_sell_orders_are_longs_being_liquidated() -> None:
    """SELL forced orders close *long* positions; BUY forced orders close shorts.

    ``liquidation_sell_volume`` must therefore accumulate SELL-side notional.
    Getting this backwards inverts ``liquidation_imbalance`` - the feature would
    read "shorts are being squeezed" exactly when longs are being flushed.
    """
    chunk = pd.DataFrame(
        [
            {
                "time": _DAY_START,
                "side": "SELL",
                "accumulated_fill_quantity": 10.0,
                "average_price": 100.0,
            },
            {
                "time": _DAY_START + 1_000,
                "side": "BUY",
                "accumulated_fill_quantity": 2.0,
                "average_price": 100.0,
            },
        ]
    )
    result = aggregate_liquidations([chunk])
    row = result[result["timestamp"] == _DAY_START].iloc[0]

    assert row["liquidation_sell_volume"] == pytest.approx(1_000.0)
    assert row["liquidation_buy_volume"] == pytest.approx(200.0)

    total = row["liquidation_buy_volume"] + row["liquidation_sell_volume"]
    imbalance = (row["liquidation_buy_volume"] - row["liquidation_sell_volume"]) / total
    # Mostly longs liquidated -> negative imbalance -> downward forced selling.
    assert imbalance < 0.0


def test_liquidation_uses_notional_not_contract_count() -> None:
    """Quantity x price, so 1 BTC and 1 DOGE are not treated as equal size."""
    chunk = pd.DataFrame(
        [
            {
                "time": _DAY_START,
                "side": "BUY",
                "accumulated_fill_quantity": 3.0,
                "average_price": 250.0,
            }
        ]
    )
    result = aggregate_liquidations([chunk])
    row = result[result["timestamp"] == _DAY_START].iloc[0]
    assert row["liquidation_buy_volume"] == pytest.approx(750.0)


def test_liquidation_emits_a_full_day_grid_with_real_zeros() -> None:
    """A covered day yields 288 buckets; quiet ones are 0.0, not missing.

    This is what lets the feature layer distinguish "nobody was liquidated in
    this bucket" (real information) from "this day was never downloaded"
    (genuinely unknown, and therefore NaN).
    """
    chunk = pd.DataFrame(
        [
            {
                "time": _DAY_START + 42 * _BUCKET_MS,
                "side": "BUY",
                "accumulated_fill_quantity": 1.0,
                "average_price": 10.0,
            }
        ]
    )
    result = aggregate_liquidations([chunk])

    assert len(result) == 288
    assert result["timestamp"].min() == _DAY_START
    assert result["timestamp"].max() == _DAY_START + 287 * _BUCKET_MS
    # The one active bucket carries the notional...
    active = result[result["timestamp"] == _DAY_START + 42 * _BUCKET_MS].iloc[0]
    assert active["liquidation_buy_volume"] == pytest.approx(10.0)
    # ...and every other bucket is an explicit zero, never NaN.
    assert result["liquidation_buy_volume"].notna().all()
    assert result["liquidation_sell_volume"].notna().all()
    assert result["liquidation_buy_volume"].sum() == pytest.approx(10.0)


def test_liquidation_falls_back_to_original_quantity() -> None:
    """Older archive days lack accumulated_fill_quantity."""
    chunk = pd.DataFrame(
        [
            {
                "time": _DAY_START,
                "side": "SELL",
                "original_quantity": 4.0,
                "price": 50.0,
            }
        ]
    )
    result = aggregate_liquidations([chunk])
    row = result[result["timestamp"] == _DAY_START].iloc[0]
    assert row["liquidation_sell_volume"] == pytest.approx(200.0)


def test_liquidation_without_side_column_raises() -> None:
    with pytest.raises(ValueError, match="side"):
        aggregate_liquidations([pd.DataFrame({"time": [_DAY_START], "price": [1.0]})])


# ---------------------------------------------------------------------------
# CSV reading: header present or absent
# ---------------------------------------------------------------------------
def test_read_csv_chunks_detects_a_header_row() -> None:
    raw = (
        b"update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,"
        b"transaction_time,event_time\n"
        b"1,100.0,5.0,100.1,6.0,1767571200000,1767571200000\n"
    )
    chunks = list(_read_csv_chunks(raw, ("update_id", "best_bid_price", "best_bid_qty",
                                         "best_ask_price", "best_ask_qty",
                                         "transaction_time", "event_time")))
    assert len(chunks) == 1
    assert chunks[0].loc[0, "best_bid_qty"] == pytest.approx(5.0)
    assert len(chunks[0]) == 1  # the header was not read as data


def test_read_csv_chunks_handles_headerless_files() -> None:
    """Older archive days ship without a header; both generations coexist."""
    raw = b"1,100.0,5.0,100.1,6.0,1767571200000,1767571200000\n"
    chunks = list(_read_csv_chunks(raw, ("update_id", "best_bid_price", "best_bid_qty",
                                         "best_ask_price", "best_ask_qty",
                                         "transaction_time", "event_time")))
    assert len(chunks) == 1
    assert chunks[0].loc[0, "best_bid_qty"] == pytest.approx(5.0)
    assert chunks[0].loc[0, "best_ask_qty"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Merging / coverage
# ---------------------------------------------------------------------------
def test_merge_microstructure_outer_joins_and_keeps_gaps_as_nan() -> None:
    book = pd.DataFrame(
        {"timestamp": [_DAY_START], "bid_qty": [1.0], "ask_qty": [2.0], "spread_bps": [3.0]}
    )
    liquidations = pd.DataFrame(
        {
            "timestamp": [_DAY_START + _BUCKET_MS],
            "liquidation_buy_volume": [5.0],
            "liquidation_sell_volume": [6.0],
        }
    )
    merged = merge_microstructure(book, liquidations)

    assert len(merged) == 2
    # The book-only bucket has no liquidation data and must stay NaN.
    assert np.isnan(merged.loc[0, "liquidation_buy_volume"])
    # The liquidation-only bucket has no book data and must stay NaN.
    assert np.isnan(merged.loc[1, "bid_qty"])


def test_merge_microstructure_handles_both_empty() -> None:
    merged = merge_microstructure(pd.DataFrame(), pd.DataFrame())
    assert merged.empty
    assert "liquidation_buy_volume" in merged.columns


def test_coverage_summary_reports_the_shortfall() -> None:
    reports = [
        ArchiveCoverage("SOL/USDT:USDT", DATASET_BOOK_TICKER, days_requested=10, days_downloaded=10),
        ArchiveCoverage("PIXEL/USDT:USDT", DATASET_BOOK_TICKER, days_requested=10, days_downloaded=2),
    ]
    summary = coverage_summary(reports)
    assert summary["days_requested"] == 20
    assert summary["days_downloaded"] == 12
    assert summary["overall_coverage_pct"] == pytest.approx(0.6)
    assert summary["per_symbol"][1]["coverage_pct"] == pytest.approx(0.2)


def test_coverage_summary_without_reports_is_not_available() -> None:
    assert coverage_summary([])["status"] == "NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# Day enumeration
# ---------------------------------------------------------------------------
def test_days_between_stops_before_today() -> None:
    """The archive only publishes a day once it has fully closed."""
    now = datetime.now(tz=timezone.utc)
    start_ms = int((now - timedelta(days=3)).timestamp() * 1_000)
    end_ms = int(now.timestamp() * 1_000)

    days = list(_days_between(start_ms, end_ms))
    assert days, "expected at least one closed day"
    assert max(days) <= now.date() - timedelta(days=1)


def test_days_between_empty_when_window_is_inverted() -> None:
    assert list(_days_between(2_000, 1_000)) == []


# ---------------------------------------------------------------------------
# End-to-end: a real ZIP through parse + aggregate
# ---------------------------------------------------------------------------
def _zip_bytes(name: str, body: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, body)
    return buffer.getvalue()


def test_parse_and_aggregate_unzips_a_book_ticker_archive(tmp_path) -> None:
    settings = Settings()
    loader = BinanceArchiveLoader(settings, cache_dir=tmp_path)
    body = (
        b"update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,"
        b"transaction_time,event_time\n"
        b"1,100.0,80.0,100.1,20.0,1767571200000,1767571200000\n"
        b"2,100.0,60.0,100.1,40.0,1767571201000,1767571201000\n"
    )
    payload = _zip_bytes("SOLUSDT-bookTicker-2026-01-05.csv", body)

    result = loader._parse_and_aggregate(payload, DATASET_BOOK_TICKER)

    assert list(result["timestamp"]) == [_DAY_START]
    assert result.loc[0, "bid_qty"] == pytest.approx(70.0)
    assert result.loc[0, "ask_qty"] == pytest.approx(30.0)


def test_parse_and_aggregate_unzips_a_liquidation_archive(tmp_path) -> None:
    settings = Settings()
    loader = BinanceArchiveLoader(settings, cache_dir=tmp_path)
    body = (
        b"time,symbol,side,order_type,time_in_force,original_quantity,price,"
        b"average_price,order_status,last_fill_quantity,accumulated_fill_quantity\n"
        b"1767571200000,SOLUSDT,SELL,LIMIT,IOC,5.0,100.0,100.0,FILLED,5.0,5.0\n"
    )
    payload = _zip_bytes("SOLUSDT-liquidationSnapshot-2026-01-05.csv", body)

    result = loader._parse_and_aggregate(payload, DATASET_LIQUIDATION)
    row = result[result["timestamp"] == _DAY_START].iloc[0]
    assert row["liquidation_sell_volume"] == pytest.approx(500.0)
    assert row["liquidation_buy_volume"] == pytest.approx(0.0)


def test_parse_rejects_an_archive_with_no_csv(tmp_path) -> None:
    loader = BinanceArchiveLoader(Settings(), cache_dir=tmp_path)
    payload = _zip_bytes("README.txt", b"not a csv")
    with pytest.raises(ValueError, match="no CSV member"):
        loader._parse_and_aggregate(payload, DATASET_BOOK_TICKER)


def test_unsupported_dataset_is_rejected(tmp_path) -> None:
    loader = BinanceArchiveLoader(Settings(), cache_dir=tmp_path)
    with pytest.raises(ValueError, match="unsupported archive dataset"):
        import asyncio

        asyncio.run(loader.load_5m_buckets("SOL/USDT:USDT", "klines", 0, 1))
