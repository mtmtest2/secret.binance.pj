"""End-to-end Module A orchestration: fetch -> validate -> heal -> persist.

:class:`DataPipeline` is the only object the orchestrator (``main.py``) needs to
know about from Module A.  It fans out across the symbol universe with a bounded
concurrency, and guarantees that anything it reports as *valid* has already
passed the QC gatekeeper and been committed to SQLite.

A symbol that cannot be healed is **skipped for that cycle** - never patched with
interpolated data.  Trading on invented candles is worse than not trading.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Final

from config.settings import Settings
from core.exceptions import DataFetchError, DataIntegrityError, DatabaseError
from core.logger import get_logger
from core.utils import last_closed_candle_open_ms, utc_now_ms
from module_a_data.db_handler import DatabaseHandler
from module_a_data.fetcher import BinanceDataFetcher
from module_a_data.models import (
    AggTradeFlow,
    FuturesMetrics,
    MarketDataBundle,
    OHLCVCandle,
    OrderBookSnapshot,
    QCIssue,
    QCReport,
)
from module_a_data.qc_validator import QCValidator

_LOGGER = get_logger(__name__)

#: Extra candles fetched beyond the strict minimum, so rolling windows warm up.
_LOOKBACK_SAFETY_BARS: Final[int] = 50

#: ``(symbol, completed, total) -> None`` progress reporter for long backfills.
ProgressCallback = Callable[[str, int, int], None]


class DataPipeline:
    """Coordinates the fetcher, the QC validator and the database handler."""

    def __init__(
        self,
        settings: Settings,
        fetcher: BinanceDataFetcher,
        validator: QCValidator,
        database: DatabaseHandler,
    ) -> None:
        self._settings: Settings = settings
        self._fetcher: BinanceDataFetcher = fetcher
        self._validator: QCValidator = validator
        self._db: DatabaseHandler = database
        self._symbol_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            settings.exchange.max_concurrent_requests
        )
        self.last_cycle_ms: int = 0
        self.last_cycle_symbols_ok: int = 0
        self.last_cycle_symbols_failed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def bootstrap_history(
        self,
        symbols: list[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Backfill historical candles so the feature/label stack has depth.

        Only the *missing* tail is requested: if the database already holds data
        for a symbol, the fetch starts one bar after the newest stored candle.
        That makes the call cheap to repeat and safe to run on every startup.

        Args:
            symbols: Universe to backfill (defaults to the configured fallback).
            progress: Optional ``(symbol, done, total) -> None`` callback invoked
                as each symbol completes, so the panel can render a progress bar.

        Returns:
            Mapping of symbol -> number of candles written.
        """
        universe: list[str] = symbols if symbols is not None else list(self._settings.data.symbols)
        completed: int = 0
        total: int = len(universe)
        target_bars: int = self._settings.data.history_bootstrap_candles
        timeframe_ms: int = self._settings.data.timeframe_ms
        end_ms: int = last_closed_candle_open_ms(timeframe_ms)

        async def _bootstrap_one(symbol: str) -> tuple[str, int]:
            async with self._symbol_semaphore:
                try:
                    newest: int | None = await self._db.latest_candle_timestamp(symbol)
                    default_start: int = end_ms - target_bars * timeframe_ms
                    start_ms: int = (
                        max(default_start, newest + timeframe_ms)
                        if newest is not None
                        else default_start
                    )
                    if start_ms > end_ms:
                        return symbol, 0

                    candles: list[OHLCVCandle] = await self._fetcher.fetch_ohlcv_range(
                        symbol, start_ms=start_ms, end_ms=end_ms
                    )
                    if not candles:
                        return symbol, 0

                    report: QCReport = self._validator.validate_candles(symbol, candles)
                    if not report.passed:
                        healed, _ = await self._validator.validate_and_heal(
                            symbol, candles, self._refetch
                        )
                        candles = healed

                    written: int = await self._db.upsert_candles(candles)
                    return symbol, written
                except (DataFetchError, DataIntegrityError, DatabaseError) as error:
                    # One bad symbol must not abort the backfill of the other 29.
                    _LOGGER.error("Bootstrap failed for %s: %s", symbol, error)
                    return symbol, 0
                finally:
                    nonlocal completed
                    completed += 1
                    if progress is not None:
                        try:
                            progress(symbol, completed, total)
                        except Exception as callback_error:  # pragma: no cover
                            _LOGGER.debug("Progress callback failed: %s", callback_error)

        results: list[tuple[str, int]] = await asyncio.gather(
            *(_bootstrap_one(symbol) for symbol in universe)
        )
        written_by_symbol: dict[str, int] = dict(results)
        total_written: int = sum(written_by_symbol.values())
        _LOGGER.info(
            "Bootstrap complete: %d candles across %d symbols",
            total_written,
            len(written_by_symbol),
        )
        return written_by_symbol

    async def bootstrap_agg_trade_flow(
        self,
        symbols: list[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Backfill 5-minute aggressive buy/sell volume buckets from aggTrades.

        This is the historical counterpart of the live refresh in
        :meth:`_process_symbol`, and it is what gives the order-flow feature
        block the same depth as the candle history.  As with the candle
        backfill, only the missing tail is requested, so re-running is cheap.

        The window is bounded by the candles actually stored for the symbol:
        flow buckets outside the candle range have nothing to attach to.
        """
        if not self._settings.data.collect_agg_trades:
            _LOGGER.info("Aggregated-trade collection is disabled; skipping the flow backfill")
            return {}

        universe: list[str] = symbols if symbols is not None else list(self._settings.data.symbols)
        timeframe_ms: int = self._settings.data.timeframe_ms
        completed: int = 0
        total: int = len(universe)

        async def _one(symbol: str) -> tuple[str, int]:
            async with self._symbol_semaphore:
                try:
                    oldest, newest = await self._db.candle_range(symbol)
                    if oldest is None or newest is None:
                        return symbol, 0

                    stored: int | None = await self._db.latest_agg_trade_flow_timestamp(symbol)
                    start_ms: int = oldest if stored is None else max(oldest, stored + timeframe_ms)
                    # `newest` is a candle *open* time, so its bucket closes one
                    # timeframe later; that instant is the exclusive upper bound.
                    end_ms: int = newest + timeframe_ms
                    if start_ms >= end_ms:
                        return symbol, 0

                    buckets: list[AggTradeFlow] = await self._fetcher.fetch_agg_trade_flow(
                        symbol, start_ms=start_ms, end_ms=end_ms
                    )
                    return symbol, await self._db.upsert_agg_trade_flow(buckets)
                except (DataFetchError, DataIntegrityError, DatabaseError) as error:
                    _LOGGER.error("Order-flow backfill failed for %s: %s", symbol, error)
                    return symbol, 0
                finally:
                    nonlocal completed
                    completed += 1
                    if progress is not None:
                        try:
                            progress(symbol, completed, total)
                        except Exception as callback_error:  # pragma: no cover
                            _LOGGER.debug("Progress callback failed: %s", callback_error)

        results: list[tuple[str, int]] = await asyncio.gather(*(_one(symbol) for symbol in universe))
        written: dict[str, int] = dict(results)
        _LOGGER.info(
            "Order-flow backfill complete: %d buckets across %d symbols",
            sum(written.values()),
            len(written),
        )
        return written

    async def run_cycle(self, symbols: list[str] | None = None) -> dict[str, MarketDataBundle]:
        """Run one 5-minute ingestion cycle across the universe.

        Returns:
            Mapping of symbol -> :class:`MarketDataBundle`.  Symbols that failed
            QC beyond repair are **absent** from the mapping.
        """
        universe: list[str] = symbols if symbols is not None else list(self._settings.data.symbols)
        started_ms: int = utc_now_ms()

        bundles: list[MarketDataBundle | None] = await asyncio.gather(
            *(self._process_symbol(symbol) for symbol in universe)
        )

        valid: dict[str, MarketDataBundle] = {
            bundle.symbol: bundle for bundle in bundles if bundle is not None and bundle.is_valid
        }
        self.last_cycle_ms = utc_now_ms()
        self.last_cycle_symbols_ok = len(valid)
        self.last_cycle_symbols_failed = len(universe) - len(valid)

        _LOGGER.info(
            "Ingestion cycle finished in %.2fs: %d/%d symbols valid",
            (self.last_cycle_ms - started_ms) / 1_000.0,
            len(valid),
            len(universe),
        )
        return valid

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _process_symbol(self, symbol: str) -> MarketDataBundle | None:
        """Fetch, validate, heal and persist everything for one symbol."""
        async with self._symbol_semaphore:
            try:
                candles_task = self._fetcher.fetch_ohlcv(symbol)
                book_task = self._fetcher.fetch_order_book(symbol)
                futures_task = self._fetcher.fetch_futures_metrics(symbol)

                results: list[object] = await asyncio.gather(
                    candles_task, book_task, futures_task, return_exceptions=True
                )
            except asyncio.CancelledError:
                raise

            candles_result, book_result, futures_result = results

            if isinstance(candles_result, BaseException):
                _LOGGER.error("OHLCV fetch failed for %s: %s", symbol, candles_result)
                return None
            candles: list[OHLCVCandle] = list(candles_result)  # type: ignore[arg-type]

            order_book: OrderBookSnapshot | None = (
                None if isinstance(book_result, BaseException) else book_result  # type: ignore[assignment]
            )
            if isinstance(book_result, BaseException):
                _LOGGER.warning("Order book fetch failed for %s: %s", symbol, book_result)

            futures: FuturesMetrics | None = (
                None if isinstance(futures_result, BaseException) else futures_result  # type: ignore[assignment]
            )
            if isinstance(futures_result, BaseException):
                _LOGGER.warning("Futures metrics fetch failed for %s: %s", symbol, futures_result)

            # --- QC gatekeeper (with auto-healing) ------------------------
            try:
                healed, report = await self._validator.validate_and_heal(
                    symbol, candles, self._refetch
                )
            except DataIntegrityError as error:
                _LOGGER.error("QC rejected %s and healing failed: %s", symbol, error)
                return None
            except DataFetchError as error:
                _LOGGER.error("Re-fetch during healing failed for %s: %s", symbol, error)
                return None

            side_issues: list[QCIssue] = [
                *self._validator.validate_order_book(order_book, symbol),
                *self._validator.validate_futures_metrics(futures, symbol),
            ]
            for issue in side_issues:
                _LOGGER.debug("Non-fatal QC note: %s", issue)

            enriched_report: QCReport = QCReport(
                symbol=report.symbol,
                checked_rows=report.checked_rows,
                issues=tuple([*report.issues, *side_issues]),
                missing_timestamps=report.missing_timestamps,
                first_timestamp=report.first_timestamp,
                last_timestamp=report.last_timestamp,
            )

            # --- Persist --------------------------------------------------
            try:
                await self._db.upsert_candles(healed)
                if order_book is not None:
                    await self._db.upsert_order_book(order_book)
                if futures is not None:
                    await self._db.upsert_futures_metrics(futures)
                await self._refresh_agg_trade_flow(symbol)
            except DatabaseError as error:
                _LOGGER.error("Persistence failed for %s: %s", symbol, error)
                return None

            return MarketDataBundle(
                symbol=symbol,
                candles=tuple(healed),
                order_book=order_book,
                futures=futures,
                qc_report=enriched_report,
                fetched_at_ms=utc_now_ms(),
            )

    async def _refresh_agg_trade_flow(self, symbol: str) -> None:
        """Re-ingest the last few *closed* 5m order-flow buckets for one symbol.

        Only closed buckets are written: the bucket the exchange is still filling
        would otherwise reach the feature stack as a partial observation, which
        is exactly the kind of subtle look-ahead-adjacent error that makes a
        live-vs-backtest mismatch impossible to diagnose later.
        """
        if not self._settings.data.collect_agg_trades:
            return

        timeframe_ms: int = self._settings.data.timeframe_ms
        end_ms: int = last_closed_candle_open_ms(timeframe_ms) + timeframe_ms
        start_ms: int = end_ms - self._settings.data.agg_trade_live_buckets * timeframe_ms
        try:
            buckets: list[AggTradeFlow] = await self._fetcher.fetch_agg_trade_flow(
                symbol, start_ms=start_ms, end_ms=end_ms
            )
        except DataFetchError as error:
            # Order flow is additive: losing a bucket costs one row of training
            # data, and must never take the whole ingestion cycle down with it.
            _LOGGER.warning("Order-flow refresh failed for %s: %s", symbol, error)
            return
        await self._db.upsert_agg_trade_flow(buckets)

    async def _refetch(self, symbol: str, start_ms: int, end_ms: int) -> list[OHLCVCandle]:
        """Targeted re-fetch callback handed to the QC auto-healer."""
        _LOGGER.info(
            "Healing %s: re-fetching window [%d, %d] (%d bars)",
            symbol,
            start_ms,
            end_ms,
            max(1, (end_ms - start_ms) // self._settings.data.timeframe_ms),
        )
        return await self._fetcher.fetch_ohlcv_range(symbol, start_ms=start_ms, end_ms=end_ms)

    # ------------------------------------------------------------------
    # Helpers for downstream modules
    # ------------------------------------------------------------------
    def required_lookback_bars(self) -> int:
        """Minimum candle history Module B needs before features are trustworthy."""
        features = self._settings.features
        return (
            max(
                features.garch_window,
                features.hmm_window,
                features.rank_window,
                features.fdi_window,
                features.kama_slow,
                features.atr_window,
            )
            + _LOOKBACK_SAFETY_BARS
        )
