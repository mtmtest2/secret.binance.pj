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

import pandas as pd

from config.settings import Settings
from core.exceptions import DataFetchError, DataIntegrityError, DatabaseError
from core.logger import get_logger
from core.utils import last_closed_candle_open_ms, utc_now_ms
from module_a_data.db_handler import DatabaseHandler
from module_a_data.fetcher import BinanceDataFetcher
from module_a_data.models import (
    FuturesMetrics,
    HealAttempt,
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

#: Rolling history caps for QC/healing telemetry persisted to durable state -
#: enough for a meaningful diagnostic report without the state blob growing
#: unbounded across the life of a long-running deployment.
_MAX_STORED_HEAL_ATTEMPTS: Final[int] = 500
_MAX_STORED_EXCLUSIONS: Final[int] = 500

#: Durable state keys (see ``DatabaseHandler.set_state``/``get_state``).
HEAL_TELEMETRY_STATE_KEY: Final[str] = "qc_heal_telemetry"
SYMBOL_EXCLUSION_STATE_KEY: Final[str] = "qc_symbol_exclusions"

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
        #: Every heal round attempted in the most recent cycle, and every
        #: symbol excluded from it with the reason - both also persisted to
        #: durable state (bounded history) for the ML diagnostic report.
        self.last_cycle_heal_attempts: list[dict[str, object]] = []
        self.last_cycle_exclusions: list[dict[str, object]] = []

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
                        # Quarantine rather than discard: a permanently unfetchable
                        # window (an exchange halt, a pre-listing gap) must not
                        # cost the whole symbol its otherwise-clean history on
                        # every single bootstrap run.
                        healed, _, _ = await self._validator.validate_and_heal(
                            symbol, candles, self._refetch, quarantine_unhealable=True
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

    async def backfill_futures_metrics(
        self,
        symbols: list[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Backfill funding-rate/open-interest/positioning history for training.

        ``run_cycle`` only ever captures a live "now" snapshot, once per
        5-minute cycle - a freshly-bootstrapped deployment therefore has
        almost no real ``futures_metrics`` rows under a multi-month OHLCV
        backfill window, so every feature derived from them (``funding_rate``,
        ``open_interest_change``, ``long_short_ratio``, ``taker_buy_sell_ratio``)
        sits at its neutral default for nearly every training row. This
        recovers as much real history as Binance actually retains: funding
        rate has full history since contract inception; open interest and the
        positioning ratios are capped to roughly the last 30 days on
        Binance's side (an exchange limitation, not something this code can
        work around) - see :meth:`module_a_data.fetcher.BinanceDataFetcher.
        fetch_open_interest_history` and friends for the per-source detail.

        Order-book depth (``ob_imbalance``, ``ob_imbalance_delta``,
        ``ob_spread_bps``, ``ob_spread_rank``) and liquidation flow
        (``liquidation_imbalance``) had no historical endpoint on Binance at
        all - a snapshot was only ever "now", with nothing to backfill - and
        were therefore removed from ``FEATURE_COLUMNS`` entirely rather than
        left permanently unpopulated. This method backfills only the four
        features named above, which do have a real Binance history endpoint.
        """
        universe: list[str] = symbols if symbols is not None else list(self._settings.data.symbols)
        timeframe_ms: int = self._settings.data.timeframe_ms
        target_bars: int = self._settings.data.history_bootstrap_candles
        end_ms: int = last_closed_candle_open_ms(timeframe_ms)
        completed: int = 0
        total: int = len(universe)

        async def _backfill_one(symbol: str) -> tuple[str, int]:
            nonlocal completed
            async with self._symbol_semaphore:
                try:
                    default_start: int = end_ms - target_bars * timeframe_ms
                    # Resume from the *oldest* stored row, not the newest.  The
                    # live cycle writes a "now" snapshot every 5 minutes, so the
                    # newest timestamp is always the present moment - a
                    # newest-first resume rule made ``start_ms`` exceed ``end_ms``
                    # on every run and the backfill wrote 0 rows forever, leaving
                    # funding_rate, open_interest_change, long_short_ratio and
                    # taker_buy_sell_ratio pinned to their neutral defaults across
                    # the whole training set.
                    oldest: int | None = await self._db.earliest_futures_metrics_timestamp(symbol)
                    start_ms: int = default_start
                    if oldest is not None and oldest <= default_start:
                        # History already reaches back past the requested window;
                        # only the leading edge can still be missing.
                        newest: int | None = await self._db.latest_futures_metrics_timestamp(symbol)
                        if newest is not None:
                            start_ms = max(default_start, newest + 1)
                    if start_ms > end_ms:
                        _LOGGER.debug(
                            "Futures-metrics history for %s already covers [%d, %d]",
                            symbol,
                            default_start,
                            end_ms,
                        )
                        return symbol, 0

                    funding, open_interest, long_short, taker = await asyncio.gather(
                        self._fetcher.fetch_funding_rate_history(symbol, start_ms, end_ms),
                        self._fetcher.fetch_open_interest_history(symbol, start_ms, end_ms),
                        self._fetcher.fetch_long_short_ratio_history(symbol, start_ms, end_ms),
                        self._fetcher.fetch_taker_ratio_history(symbol, start_ms, end_ms),
                    )

                    records: list[FuturesMetrics] = self._merge_futures_history(
                        symbol, funding, open_interest, long_short, taker
                    )
                    if not records:
                        _LOGGER.warning(
                            "No historical futures metrics recovered for %s in [%d, %d]",
                            symbol,
                            start_ms,
                            end_ms,
                        )
                        return symbol, 0

                    written: int = await self._db.upsert_futures_metrics_batch(records)
                    return symbol, written
                except (DataFetchError, DatabaseError) as error:
                    # One symbol's backfill failing must not abort the other 28 -
                    # same isolation policy as `bootstrap_history`.
                    _LOGGER.error("Futures-metrics backfill failed for %s: %s", symbol, error)
                    return symbol, 0
                finally:
                    completed += 1
                    if progress is not None:
                        try:
                            progress(symbol, completed, total)
                        except Exception as callback_error:  # pragma: no cover
                            _LOGGER.debug("Progress callback failed: %s", callback_error)

        results: list[tuple[str, int]] = await asyncio.gather(
            *(_backfill_one(symbol) for symbol in universe)
        )
        written_by_symbol: dict[str, int] = dict(results)
        _LOGGER.info(
            "Futures-metrics backfill complete: %d row(s) across %d symbol(s). "
            "Order-book and liquidation history cannot be backfilled (no exchange "
            "endpoint for either) and will only accumulate real data from the live "
            "5-minute cycle going forward.",
            sum(written_by_symbol.values()),
            len(written_by_symbol),
        )
        return written_by_symbol

    @staticmethod
    def _merge_futures_history(
        symbol: str,
        funding: list[tuple[int, float]],
        open_interest: list[tuple[int, float]],
        long_short: list[tuple[int, float]],
        taker: list[tuple[int, float]],
    ) -> list[FuturesMetrics]:
        """Combine independently-paced historical series onto one timeline.

        Each source updates on its own cadence (funding every 8h; open
        interest/positioning at Binance's native ~5m granularity where still
        retained). Every column is forward-filled independently across the
        union of all observed timestamps before being read off, replicating
        the "most recent value as of this bar" semantics the backward as-of
        join in :mod:`module_b_features.features` applies on the live path -
        so a sparse historical write is exactly as valid an input to that join
        as a dense live one.
        """
        sources: dict[str, list[tuple[int, float]]] = {
            "funding_rate": funding,
            "open_interest": open_interest,
            "long_short_ratio": long_short,
            "taker_buy_sell_ratio": taker,
        }
        all_timestamps: set[int] = set()
        for points in sources.values():
            all_timestamps.update(timestamp for timestamp, _ in points)
        if not all_timestamps:
            return []

        frame: pd.DataFrame = pd.DataFrame(index=pd.Index(sorted(all_timestamps), name="timestamp"))
        for column, points in sources.items():
            series: pd.Series = pd.Series(dict(points), dtype=float).sort_index()
            frame[column] = series.reindex(frame.index).ffill()

        defaults: dict[str, float] = {
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "long_short_ratio": 1.0,
            "taker_buy_sell_ratio": 1.0,
        }
        records: list[FuturesMetrics] = []
        for timestamp, row in frame.iterrows():
            try:
                records.append(
                    FuturesMetrics(
                        symbol=symbol,
                        timestamp=int(timestamp),
                        funding_rate=(
                            float(row["funding_rate"])
                            if pd.notna(row["funding_rate"])
                            else defaults["funding_rate"]
                        ),
                        open_interest=max(
                            0.0,
                            float(row["open_interest"])
                            if pd.notna(row["open_interest"])
                            else defaults["open_interest"],
                        ),
                        long_short_ratio=max(
                            0.0,
                            float(row["long_short_ratio"])
                            if pd.notna(row["long_short_ratio"])
                            else defaults["long_short_ratio"],
                        ),
                        taker_buy_sell_ratio=max(
                            0.0,
                            float(row["taker_buy_sell_ratio"])
                            if pd.notna(row["taker_buy_sell_ratio"])
                            else defaults["taker_buy_sell_ratio"],
                        ),
                    )
                )
            except ValueError as error:
                _LOGGER.warning(
                    "Skipping implausible backfilled futures row for %s at %d: %s",
                    symbol,
                    timestamp,
                    error,
                )
        return records

    async def run_cycle(self, symbols: list[str] | None = None) -> dict[str, MarketDataBundle]:
        """Run one 5-minute ingestion cycle across the universe.

        Returns:
            Mapping of symbol -> :class:`MarketDataBundle`.  Symbols that failed
            QC beyond repair are **absent** from the mapping.
        """
        universe: list[str] = symbols if symbols is not None else list(self._settings.data.symbols)
        started_ms: int = utc_now_ms()
        self.last_cycle_heal_attempts = []
        self.last_cycle_exclusions = []

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
        await self._persist_qc_telemetry()
        return valid

    async def _persist_qc_telemetry(self) -> None:
        """Append this cycle's heal telemetry and exclusions to durable state.

        Stored as a bounded rolling history (see ``_MAX_STORED_*``) so the ML
        diagnostic report can show real healing/exclusion history across many
        cycles without the state blob growing without bound. Persistence
        failures are logged, never raised - telemetry must not be able to
        break the ingestion cycle it is describing.
        """
        if not self.last_cycle_heal_attempts and not self.last_cycle_exclusions:
            return
        try:
            if self.last_cycle_heal_attempts:
                stored = await self._db.get_state(HEAL_TELEMETRY_STATE_KEY)
                history: list[object] = list((stored or {}).get("records", []))
                history.extend(self.last_cycle_heal_attempts)
                history = history[-_MAX_STORED_HEAL_ATTEMPTS:]
                await self._db.set_state(HEAL_TELEMETRY_STATE_KEY, {"records": history})
            if self.last_cycle_exclusions:
                stored = await self._db.get_state(SYMBOL_EXCLUSION_STATE_KEY)
                history = list((stored or {}).get("records", []))
                history.extend(self.last_cycle_exclusions)
                history = history[-_MAX_STORED_EXCLUSIONS:]
                await self._db.set_state(SYMBOL_EXCLUSION_STATE_KEY, {"records": history})
        except DatabaseError as error:
            _LOGGER.error("Failed to persist QC/heal telemetry: %s", error)

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
                healed: list[OHLCVCandle]
                report: QCReport
                heal_attempts: list[HealAttempt]
                healed, report, heal_attempts = await self._validator.validate_and_heal(
                    symbol, candles, self._refetch
                )
                if heal_attempts:
                    self.last_cycle_heal_attempts.extend(
                        record.model_dump(mode="json") for record in heal_attempts
                    )
            except DataIntegrityError as error:
                _LOGGER.error("QC rejected %s and healing failed: %s", symbol, error)
                self.last_cycle_heal_attempts.extend(error.context.get("heal_attempts", []))
                self.last_cycle_exclusions.append(
                    {
                        "symbol": symbol,
                        "cycle": "live",
                        "reason": error.message,
                        "codes": list(error.context.get("codes", ())),
                        "excluded_at_ms": utc_now_ms(),
                    }
                )
                return None
            except DataFetchError as error:
                _LOGGER.error("Re-fetch during healing failed for %s: %s", symbol, error)
                self.last_cycle_exclusions.append(
                    {
                        "symbol": symbol,
                        "cycle": "live",
                        "reason": f"re-fetch failed during healing: {error}",
                        "codes": [],
                        "excluded_at_ms": utc_now_ms(),
                    }
                )
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
