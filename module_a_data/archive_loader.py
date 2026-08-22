"""Binance public-data archive loader (``data.binance.vision``).

Why this module exists
----------------------
Binance serves the same market data through two completely different channels
with completely different retention:

* The **REST API** (``/futures/data/...``), which is what
  :class:`~module_a_data.fetcher.BinanceDataFetcher` uses.  Several endpoints
  here are capped at roughly the last 30 days - ``openInterestHist`` and the
  positioning ratios say so explicitly in Binance's own documentation - and
  order-book depth and liquidation flow have no historical endpoint at all,
  only a live-forward stream.
* The **bulk archive** at ``data.binance.vision``, which publishes daily ZIP
  files for the *entire life of each contract*, including precisely the
  datasets the REST API cannot serve historically: ``bookTicker``,
  ``liquidationSnapshot``, ``bookDepth`` and ``metrics``.

Commit 2b56c8a removed five micro-structure features on the premise that
Binance "exposes no historical endpoint for either, ever - only a live snapshot
going forward".  That premise is true of the REST API and false of the archive.
This module is the missing loader; it is what lets those five columns be
backfilled across the whole training window instead of sitting at a neutral
constant.

Design notes
------------
* **Point-in-time by construction.**  Every row an archive file contains is
  stamped with the exchange's own event time, and buckets are only emitted once
  they have fully closed (see :func:`aggregate_book_ticker`).  Nothing here can
  see past the bucket it is writing.
* **A 404 is data, not an error.**  A symbol listed in 2023 has no 2022 files,
  and some datasets skip days.  Missing files are recorded as *covered =
  false* so the feature layer can carry an honest ``NaN`` rather than a
  fabricated neutral value.
* **Disk cache.**  27 symbols x ~740 days is ~20k downloads; re-running a
  backfill must not re-download.  Parsed 5m aggregates are cached as Parquet
  (a few KB/day) rather than the raw ZIPs (up to ~100 MB/day for bookTicker on
  a liquid symbol), so the cache stays small.
* **Streamed parsing.**  bookTicker files are large.  They are read in chunks
  and reduced to 288 rows before anything is held in memory.
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Sequence

import aiohttp
import numpy as np
import pandas as pd

from config.settings import Settings
from core.logger import get_logger

_LOGGER = get_logger(__name__)

#: Root of the USDT-margined ("um") futures archive.
_ARCHIVE_BASE: Final[str] = "https://data.binance.vision/data/futures/um/daily"

#: Datasets this loader knows how to parse and reduce onto the 5m grid.
DATASET_BOOK_TICKER: Final[str] = "bookTicker"
DATASET_LIQUIDATION: Final[str] = "liquidationSnapshot"

#: Rows per read chunk for the large event-level datasets.
_CHUNK_ROWS: Final[int] = 500_000

#: Column layouts Binance publishes.  Files are parsed *by header name* where a
#: header row exists, and fall back to these positional layouts where it does
#: not - older archive days ship headerless CSVs and the two generations are
#: mixed within a single symbol's history.
_HEADERS: Final[dict[str, tuple[str, ...]]] = {
    DATASET_BOOK_TICKER: (
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "transaction_time",
        "event_time",
    ),
    DATASET_LIQUIDATION: (
        "time",
        "symbol",
        "side",
        "order_type",
        "time_in_force",
        "original_quantity",
        "price",
        "average_price",
        "order_status",
        "last_fill_quantity",
        "accumulated_fill_quantity",
    ),
}


def to_archive_symbol(symbol: str) -> str:
    """Map a ccxt unified symbol onto the archive's naming.

    ``"SOL/USDT:USDT"`` -> ``"SOLUSDT"``.  Symbols that are already in exchange
    form pass through unchanged, so the panel's stored selection works either
    way.
    """
    base: str = symbol.split(":", 1)[0]
    return base.replace("/", "").replace("-", "").upper()


@dataclass(slots=True)
class ArchiveCoverage:
    """What a backfill actually managed to retrieve, per symbol."""

    symbol: str
    dataset: str
    days_requested: int = 0
    days_downloaded: int = 0
    days_absent: int = 0
    days_failed: int = 0
    buckets_written: int = 0

    @property
    def coverage_pct(self) -> float:
        """Share of requested days the archive actually had."""
        if self.days_requested <= 0:
            return 0.0
        return round(self.days_downloaded / self.days_requested, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "dataset": self.dataset,
            "days_requested": self.days_requested,
            "days_downloaded": self.days_downloaded,
            "days_absent": self.days_absent,
            "days_failed": self.days_failed,
            "buckets_written": self.buckets_written,
            "coverage_pct": self.coverage_pct,
        }


class BinanceArchiveLoader:
    """Downloads, unzips, parses and 5m-aggregates Binance archive datasets.

    The loader owns an :class:`aiohttp.ClientSession`; close it with
    :meth:`close` or use it as an async context manager.
    """

    def __init__(
        self,
        settings: Settings,
        cache_dir: Path | None = None,
        max_concurrent_downloads: int = 4,
    ) -> None:
        self._settings: Settings = settings
        self._cache_dir: Path = cache_dir or Path(settings.data.archive_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "BinanceArchiveLoader":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._settings.data.archive_timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Release the underlying HTTP session (idempotent)."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def load_5m_buckets(
        self,
        symbol: str,
        dataset: str,
        start_ms: int,
        end_ms: int,
    ) -> tuple[pd.DataFrame, ArchiveCoverage]:
        """Return 5m-bucketed aggregates for ``symbol`` over ``[start_ms, end_ms]``.

        Args:
            symbol: ccxt unified symbol (``"SOL/USDT:USDT"``) or exchange form.
            dataset: :data:`DATASET_BOOK_TICKER` or :data:`DATASET_LIQUIDATION`.
            start_ms: Inclusive window start, epoch milliseconds.
            end_ms: Inclusive window end, epoch milliseconds.

        Returns:
            ``(frame, coverage)``.  ``frame`` is indexed by nothing and carries a
            ``timestamp`` column on the exact 5-minute grid plus the dataset's
            aggregate columns; it is empty when the archive had nothing.  Days
            the archive does not publish are simply absent from ``frame`` -
            never zero-filled - so the caller can distinguish "no coverage" from
            "no activity".
        """
        if dataset not in _HEADERS:
            raise ValueError(f"unsupported archive dataset: {dataset!r}")

        archive_symbol: str = to_archive_symbol(symbol)
        days: list[date] = list(_days_between(start_ms, end_ms))
        coverage = ArchiveCoverage(symbol=symbol, dataset=dataset, days_requested=len(days))

        results: list[pd.DataFrame | None] = await asyncio.gather(
            *(self._one_day(archive_symbol, dataset, day, coverage) for day in days)
        )
        frames: list[pd.DataFrame] = [frame for frame in results if frame is not None and not frame.empty]
        if not frames:
            _LOGGER.warning(
                "No %s archive data recovered for %s across %d day(s)",
                dataset,
                symbol,
                len(days),
            )
            return pd.DataFrame(), coverage

        combined: pd.DataFrame = pd.concat(frames, axis=0, ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        combined = combined[
            (combined["timestamp"] >= start_ms) & (combined["timestamp"] <= end_ms)
        ].reset_index(drop=True)
        coverage.buckets_written = len(combined)
        return combined, coverage

    # ------------------------------------------------------------------
    # Per-day pipeline: cache -> download -> unzip -> parse -> aggregate
    # ------------------------------------------------------------------
    async def _one_day(
        self,
        archive_symbol: str,
        dataset: str,
        day: date,
        coverage: ArchiveCoverage,
    ) -> pd.DataFrame | None:
        cache_path: Path = self._cache_path(archive_symbol, dataset, day)
        absent_marker: Path = cache_path.with_suffix(".absent")

        if cache_path.exists():
            coverage.days_downloaded += 1
            try:
                return pd.read_parquet(cache_path)
            except Exception as error:  # pragma: no cover - corrupt cache entry
                _LOGGER.warning("Discarding unreadable cache entry %s: %s", cache_path, error)
                cache_path.unlink(missing_ok=True)
        if absent_marker.exists():
            # Binance does not publish this day for this symbol and never will;
            # re-checking on every run would waste an HTTP round trip per day
            # per symbol for the entire pre-listing history.
            coverage.days_absent += 1
            return None

        payload: bytes | None = await self._download(archive_symbol, dataset, day)
        if payload is None:
            coverage.days_absent += 1
            absent_marker.parent.mkdir(parents=True, exist_ok=True)
            absent_marker.touch()
            return None

        try:
            aggregated: pd.DataFrame = self._parse_and_aggregate(payload, dataset)
        except Exception as error:
            coverage.days_failed += 1
            _LOGGER.error(
                "Failed to parse %s archive for %s %s: %s", dataset, archive_symbol, day, error
            )
            return None

        coverage.days_downloaded += 1
        if not aggregated.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                aggregated.to_parquet(cache_path, index=False)
            except Exception as error:  # pragma: no cover - pyarrow absent/disk full
                _LOGGER.warning("Could not cache %s: %s", cache_path, error)
        return aggregated

    async def _download(self, archive_symbol: str, dataset: str, day: date) -> bytes | None:
        """Fetch one daily ZIP, returning ``None`` when Binance has no such file."""
        filename: str = f"{archive_symbol}-{dataset}-{day.isoformat()}.zip"
        url: str = f"{_ARCHIVE_BASE}/{dataset}/{archive_symbol}/{filename}"

        async with self._semaphore:
            session: aiohttp.ClientSession = await self._client()
            for attempt in range(self._settings.data.archive_max_retries):
                try:
                    async with session.get(url) as response:
                        if response.status == 404:
                            _LOGGER.debug("Archive has no %s", filename)
                            return None
                        if response.status == 429:
                            delay: float = 2.0 * (attempt + 1)
                            _LOGGER.warning("Archive rate-limited on %s; waiting %.1fs", filename, delay)
                            await asyncio.sleep(delay)
                            continue
                        response.raise_for_status()
                        return await response.read()
                except aiohttp.ClientError as error:
                    if attempt == self._settings.data.archive_max_retries - 1:
                        _LOGGER.error("Giving up on %s: %s", filename, error)
                        return None
                    await asyncio.sleep(2.0**attempt)
        return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_and_aggregate(self, payload: bytes, dataset: str) -> pd.DataFrame:
        """Unzip one daily archive and reduce it onto the 5-minute grid."""
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names: list[str] = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise ValueError("archive contains no CSV member")
            with archive.open(names[0]) as handle:
                raw: bytes = handle.read()

        frames: Iterable[pd.DataFrame] = _read_csv_chunks(raw, _HEADERS[dataset])
        if dataset == DATASET_BOOK_TICKER:
            return aggregate_book_ticker(frames)
        return aggregate_liquidations(frames)

    def _cache_path(self, archive_symbol: str, dataset: str, day: date) -> Path:
        return self._cache_dir / dataset / archive_symbol / f"{day.isoformat()}.parquet"


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------
def _read_csv_chunks(raw: bytes, expected_columns: tuple[str, ...]) -> Iterator[pd.DataFrame]:
    """Yield chunks of a Binance archive CSV, header row present or not.

    Binance changed these files mid-life: older days are headerless, newer ones
    carry a header.  Both generations appear inside a single symbol's history,
    so the header is detected rather than assumed, and columns are then
    addressed **by name** - a positional read would silently transpose
    ``best_bid_qty`` and ``best_ask_qty`` the day Binance adds a column, which
    inverts every downstream imbalance while leaving all shapes and ranges
    perfectly healthy.
    """
    first_line: bytes = raw.split(b"\n", 1)[0]
    has_header: bool = any(
        token.strip().decode("utf-8", "ignore").lower() in {c.lower() for c in expected_columns}
        for token in first_line.split(b",")
    )

    reader = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=None if has_header else list(expected_columns),
        chunksize=_CHUNK_ROWS,
        low_memory=False,
    )
    for chunk in reader:
        chunk.columns = [str(name).strip() for name in chunk.columns]
        yield chunk


def _bucket_ms(series: pd.Series, timeframe_ms: int = 300_000) -> pd.Series:
    """Floor an epoch-millisecond series onto the 5-minute candle grid."""
    numeric: pd.Series = pd.to_numeric(series, errors="coerce")
    # Some archive days stamp microseconds rather than milliseconds; a value
    # past year ~2286 in ms is the giveaway. Normalising here keeps every
    # downstream join on one clock.
    scaled: pd.Series = numeric.where(numeric < 1e13, numeric / 1_000.0)
    return (scaled // timeframe_ms * timeframe_ms).astype("Int64")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_book_ticker(chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Reduce event-level best bid/ask updates to per-5m-bucket averages.

    Returns a frame with ``timestamp``, ``bid_qty``, ``ask_qty`` and
    ``spread_bps``.  ``bid_qty``/``ask_qty`` are the bucket's mean top-of-book
    sizes (the feature layer forms the imbalance from them, so the raw sides
    stay available for a depth-weighted variant later), and ``spread_bps`` is
    the mean relative spread in basis points.
    """
    sums: dict[int, np.ndarray] = {}
    for chunk in chunks:
        needed: set[str] = {
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
        }
        if not needed.issubset(chunk.columns):
            raise ValueError(f"bookTicker chunk is missing columns: {sorted(needed - set(chunk.columns))}")

        time_column: str = "transaction_time" if "transaction_time" in chunk.columns else "event_time"
        bucket: pd.Series = _bucket_ms(chunk[time_column])

        bid_price = pd.to_numeric(chunk["best_bid_price"], errors="coerce")
        ask_price = pd.to_numeric(chunk["best_ask_price"], errors="coerce")
        bid_qty = pd.to_numeric(chunk["best_bid_qty"], errors="coerce")
        ask_qty = pd.to_numeric(chunk["best_ask_qty"], errors="coerce")

        mid = (bid_price + ask_price) / 2.0
        spread_bps = ((ask_price - bid_price) / mid.where(mid > 0.0)) * 10_000.0

        frame = pd.DataFrame(
            {
                "bucket": bucket,
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
                "spread_bps": spread_bps,
            }
        ).dropna(subset=["bucket"])
        # A crossed or zero-width book is a feed artifact, not a market state.
        frame = frame[frame["spread_bps"].fillna(-1.0) >= 0.0]
        if frame.empty:
            continue

        grouped = frame.groupby("bucket", sort=False).agg(
            bid_qty=("bid_qty", "sum"),
            ask_qty=("ask_qty", "sum"),
            spread_bps=("spread_bps", "sum"),
            count=("spread_bps", "size"),
        )
        for bucket_key, row in grouped.iterrows():
            accumulated = sums.get(int(bucket_key))
            values = np.array(
                [row["bid_qty"], row["ask_qty"], row["spread_bps"], row["count"]], dtype=np.float64
            )
            sums[int(bucket_key)] = values if accumulated is None else accumulated + values

    if not sums:
        return pd.DataFrame(columns=["timestamp", "bid_qty", "ask_qty", "spread_bps"])

    buckets = np.array(sorted(sums), dtype=np.int64)
    stacked = np.vstack([sums[int(key)] for key in buckets])
    counts = np.where(stacked[:, 3] > 0.0, stacked[:, 3], np.nan)
    return pd.DataFrame(
        {
            "timestamp": buckets,
            "bid_qty": stacked[:, 0] / counts,
            "ask_qty": stacked[:, 1] / counts,
            "spread_bps": stacked[:, 2] / counts,
        }
    )


def aggregate_liquidations(chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Reduce forced-order events to per-5m-bucket liquidated notional per side.

    Returns ``timestamp``, ``liquidation_buy_volume``, ``liquidation_sell_volume``.

    **Side convention.**  ``side`` is the side of the *liquidation order the
    exchange submits*, not the side of the position being closed.  A long
    position is force-closed by a SELL order and a short position by a BUY
    order.  ``liquidation_sell_volume`` therefore measures **longs being
    liquidated** and ``liquidation_buy_volume`` measures **shorts being
    liquidated**, which is the orientation the imbalance feature assumes: a
    positive ``liquidation_imbalance`` means shorts are the ones getting
    stopped out, i.e. upward forced buying pressure.  A sign error here would
    invert the feature while leaving every shape and range healthy, so it is
    asserted in ``tests/test_archive_loader.py``.

    Only *closed* buckets are emitted, and every bucket of a covered day is
    emitted - including buckets with no liquidations, which are written as a
    genuine 0.0.  That is what lets the feature layer tell "no liquidations
    happened" apart from "this day was never downloaded".
    """
    totals: dict[int, np.ndarray] = {}
    day_buckets: set[int] = set()

    for chunk in chunks:
        if "side" not in chunk.columns:
            raise ValueError("liquidationSnapshot chunk is missing the 'side' column")

        time_column: str = "time" if "time" in chunk.columns else chunk.columns[0]
        bucket: pd.Series = _bucket_ms(chunk[time_column])

        # Prefer the actually-filled quantity; fall back to the order quantity
        # when the archive day predates that column.
        quantity_column: str = next(
            (
                name
                for name in ("accumulated_fill_quantity", "last_fill_quantity", "original_quantity")
                if name in chunk.columns
            ),
            "",
        )
        if not quantity_column:
            raise ValueError("liquidationSnapshot chunk has no usable quantity column")

        price_column: str = "average_price" if "average_price" in chunk.columns else "price"
        quantity = pd.to_numeric(chunk[quantity_column], errors="coerce")
        price = pd.to_numeric(chunk[price_column], errors="coerce")
        # Notional, not contract count: 1 BTC and 1 DOGE are not comparable
        # sizes, and the feature is a ratio across a single symbol's own history.
        notional = (quantity * price).abs()
        side = chunk["side"].astype(str).str.strip().str.upper()

        frame = pd.DataFrame({"bucket": bucket, "notional": notional, "side": side}).dropna(
            subset=["bucket", "notional"]
        )
        if frame.empty:
            continue

        day_buckets.update(int(value) for value in frame["bucket"].unique())
        grouped = frame.groupby(["bucket", "side"], sort=False)["notional"].sum()
        for (bucket_key, side_value), notional_sum in grouped.items():
            key = int(bucket_key)
            accumulated = totals.get(key, np.zeros(2, dtype=np.float64))
            if side_value == "BUY":
                accumulated[0] += float(notional_sum)
            elif side_value == "SELL":
                accumulated[1] += float(notional_sum)
            totals[key] = accumulated

    if not day_buckets:
        return pd.DataFrame(
            columns=["timestamp", "liquidation_buy_volume", "liquidation_sell_volume"]
        )

    # Emit every bucket of every day that had at least one event, so a quiet
    # bucket inside a covered day reads as a real zero rather than as a gap.
    covered_days: set[int] = {value - (value % 86_400_000) for value in day_buckets}
    grid: list[int] = sorted(
        day_start + offset
        for day_start in covered_days
        for offset in range(0, 86_400_000, 300_000)
    )
    buy = np.array([totals.get(key, (0.0, 0.0))[0] for key in grid], dtype=np.float64)
    sell = np.array([totals.get(key, (0.0, 0.0))[1] for key in grid], dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": np.array(grid, dtype=np.int64),
            "liquidation_buy_volume": buy,
            "liquidation_sell_volume": sell,
        }
    )


def merge_microstructure(
    book: pd.DataFrame,
    liquidations: pd.DataFrame,
) -> pd.DataFrame:
    """Outer-join the two archive aggregates onto one 5m-keyed frame."""
    columns: list[str] = [
        "timestamp",
        "bid_qty",
        "ask_qty",
        "spread_bps",
        "liquidation_buy_volume",
        "liquidation_sell_volume",
    ]
    if book.empty and liquidations.empty:
        return pd.DataFrame(columns=columns)
    if book.empty:
        merged = liquidations.copy()
    elif liquidations.empty:
        merged = book.copy()
    else:
        merged = book.merge(liquidations, on="timestamp", how="outer")

    for column in columns:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[columns].sort_values("timestamp").reset_index(drop=True)


def _days_between(start_ms: int, end_ms: int) -> Iterator[date]:
    """Yield every UTC calendar day touched by ``[start_ms, end_ms]``."""
    if end_ms < start_ms:
        return
    current: date = datetime.fromtimestamp(start_ms / 1_000.0, tz=timezone.utc).date()
    final: date = datetime.fromtimestamp(end_ms / 1_000.0, tz=timezone.utc).date()
    # The archive publishes a day only after it has fully closed.
    today: date = datetime.now(tz=timezone.utc).date()
    final = min(final, today - timedelta(days=1))
    while current <= final:
        yield current
        current += timedelta(days=1)


def coverage_summary(reports: Sequence[ArchiveCoverage]) -> dict[str, Any]:
    """Aggregate per-symbol coverage into a diagnostic-report block."""
    if not reports:
        return {"status": "NOT_AVAILABLE", "reason": "no archive backfill has run"}
    requested: int = sum(item.days_requested for item in reports)
    downloaded: int = sum(item.days_downloaded for item in reports)
    return {
        "status": "AVAILABLE",
        "days_requested": requested,
        "days_downloaded": downloaded,
        "overall_coverage_pct": round(downloaded / requested, 4) if requested else 0.0,
        "per_symbol": [item.as_dict() for item in reports],
    }
