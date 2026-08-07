"""The QC Gatekeeper - nothing reaches the ML stack without passing through here.

The validator implements four families of checks:

1. **Structural / timestamp integrity.**  Every 5-minute candle must open on an
   exact multiple of 300_000 ms, timestamps must be strictly increasing with no
   duplicates, and consecutive open times must differ by exactly one timeframe.
   Any deviation yields the precise list of missing grid timestamps so the
   healer can re-fetch exactly the damaged block.

2. **Price logic.**  ``high >= low``, ``high >= max(open, close)``,
   ``low <= min(open, close)`` and all prices strictly positive.  These are also
   enforced by the Pydantic model, so a violation reaching this layer means the
   data arrived from a different (unvalidated) path - it is still treated as
   CRITICAL.

3. **Volume logic.**  ``volume >= 0``; a candle whose volume exceeds
   ``median * volume_spike_median_multiple`` is flagged as a probable API glitch;
   an excessive share of zero-volume candles means the feed is broken rather than
   the market being quiet.

4. **Statistical anomaly detection.**  Log returns are screened with a robust
   MAD z-score (median absolute deviation, scaled by 1.4826 so it estimates
   sigma for Gaussian data).  The MAD is used instead of the standard deviation
   precisely because a single 40 % glitch print inflates the standard deviation
   enough to hide itself.  An absolute per-candle return ceiling backs it up for
   the pathological case where *most* of the block is corrupt.

Auto-healing: :meth:`QCValidator.validate_and_heal` re-fetches the damaged
timestamp range, merges the repaired rows over the corrupt ones and re-validates,
repeating with exponential backoff until the block is pristine or the attempt
budget is exhausted (in which case :class:`DataIntegrityError` is raised and the
symbol is skipped for this cycle - never silently accepted).
"""

from __future__ import annotations

import asyncio
import math
from typing import Awaitable, Callable, Final, Iterable, Sequence

import numpy as np

from config.settings import Settings
from core.exceptions import DataIntegrityError
from core.logger import get_logger
from core.utils import backoff_delay, expected_timestamps, last_closed_candle_open_ms, utc_now_ms
from module_a_data.models import (
    FuturesMetrics,
    OHLCVCandle,
    OrderBookSnapshot,
    QCIssue,
    QCIssueCode,
    QCReport,
    QCSeverity,
)

_LOGGER = get_logger(__name__)

#: Scale factor turning a median-absolute-deviation into a sigma estimate.
_MAD_TO_SIGMA: Final[float] = 1.4826

#: Signature of the re-fetch callback used for auto-healing.
RefetchCallback = Callable[[str, int, int], Awaitable[list[OHLCVCandle]]]


class QCValidator:
    """Stateless validator for OHLCV blocks, order books and futures metrics."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._timeframe_ms: int = settings.data.timeframe_ms
        self._qc = settings.qc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate_candles(self, symbol: str, candles: Sequence[OHLCVCandle]) -> QCReport:
        """Run the full check battery over a candle block.

        Args:
            symbol: Symbol the block belongs to (used for reporting only).
            candles: Candles in any order; the validator sorts defensively.

        Returns:
            A :class:`QCReport`.  ``report.passed`` is ``False`` when at least one
            CRITICAL issue was found; ``report.missing_timestamps`` lists exactly
            which grid points need re-fetching.
        """
        issues: list[QCIssue] = []

        if not candles:
            issues.append(
                QCIssue(
                    code=QCIssueCode.EMPTY_DATASET,
                    severity=QCSeverity.CRITICAL,
                    message="no candles returned by the exchange",
                    symbol=symbol,
                    healable=True,
                )
            )
            return QCReport(symbol=symbol, checked_rows=0, issues=tuple(issues))

        ordered: list[OHLCVCandle] = sorted(candles, key=lambda item: item.timestamp)
        timestamps: list[int] = [candle.timestamp for candle in ordered]

        issues.extend(self._check_timestamp_integrity(symbol, ordered, timestamps))
        missing: list[int] = self._find_missing_timestamps(timestamps)
        if missing:
            issues.append(
                QCIssue(
                    code=QCIssueCode.MISSING_CANDLES,
                    severity=QCSeverity.CRITICAL,
                    message=(
                        f"{len(missing)} missing 5m candle(s) between "
                        f"{timestamps[0]} and {timestamps[-1]}"
                    ),
                    symbol=symbol,
                    timestamps=tuple(missing[:50]),
                    healable=True,
                )
            )

        issues.extend(self._check_price_logic(symbol, ordered))
        issues.extend(self._check_volume_logic(symbol, ordered))
        issues.extend(self._check_return_outliers(symbol, ordered))
        issues.extend(self._check_freshness(symbol, timestamps))

        if len(ordered) < self._qc.min_rows_for_statistics:
            issues.append(
                QCIssue(
                    code=QCIssueCode.TOO_FEW_ROWS,
                    severity=QCSeverity.WARNING,
                    message=(
                        f"only {len(ordered)} rows available; statistical checks are "
                        f"weak below {self._qc.min_rows_for_statistics}"
                    ),
                    symbol=symbol,
                    healable=False,
                )
            )

        return QCReport(
            symbol=symbol,
            checked_rows=len(ordered),
            issues=tuple(issues),
            missing_timestamps=tuple(missing),
            first_timestamp=timestamps[0],
            last_timestamp=timestamps[-1],
        )

    async def validate_and_heal(
        self,
        symbol: str,
        candles: Sequence[OHLCVCandle],
        refetch: RefetchCallback,
    ) -> tuple[list[OHLCVCandle], QCReport]:
        """Validate a block and recursively repair it until it is pristine.

        The heal loop is *targeted*: only the damaged timestamp window is
        re-requested, and repaired rows are merged over the corrupt ones by
        timestamp.  Candles flagged as corrupt (bad price/volume logic or return
        outliers) are dropped before the merge so the exchange's fresh copy wins.

        Args:
            symbol: Symbol under repair.
            candles: The block as first fetched.
            refetch: ``async (symbol, start_ms, end_ms) -> list[OHLCVCandle]``.

        Returns:
            ``(healed_candles, final_report)`` where the report has passed.

        Raises:
            DataIntegrityError: When the block is still invalid after
                ``qc.max_heal_attempts`` repair rounds, or when the failure is
                not the kind a re-fetch can fix.
        """
        working: list[OHLCVCandle] = sorted(candles, key=lambda item: item.timestamp)
        report: QCReport = self.validate_candles(symbol, working)

        attempt: int = 0
        while not report.passed and attempt < self._qc.max_heal_attempts:
            if not report.healable:
                raise DataIntegrityError(
                    "QC failure cannot be healed by re-fetching",
                    symbol=symbol,
                    codes=report.critical_codes,
                )

            delay: float = backoff_delay(
                attempt,
                base_seconds=self._qc.heal_backoff_seconds,
                max_seconds=self._settings.exchange.backoff_max_seconds,
                jitter=self._settings.exchange.backoff_jitter,
            )
            _LOGGER.warning(
                "QC failed for %s (%s) - heal attempt %d/%d in %.2fs",
                symbol,
                ", ".join(report.critical_codes),
                attempt + 1,
                self._qc.max_heal_attempts,
                delay,
            )
            await asyncio.sleep(delay)

            start_ms, end_ms = self._heal_window(working, report)
            repaired: list[OHLCVCandle] = await refetch(symbol, start_ms, end_ms)

            working = self._merge_blocks(
                base=working,
                patch=repaired,
                drop_range=(start_ms, end_ms),
            )
            report = self.validate_candles(symbol, working)
            attempt += 1

        if not report.passed:
            raise DataIntegrityError(
                "data still invalid after exhausting heal attempts",
                symbol=symbol,
                attempts=attempt,
                codes=report.critical_codes,
            )

        if attempt:
            _LOGGER.info("Healed %s after %d attempt(s): %d rows", symbol, attempt, len(working))
        return working, report

    def validate_order_book(self, book: OrderBookSnapshot | None, symbol: str) -> list[QCIssue]:
        """Sanity-check a reduced order-book snapshot."""
        issues: list[QCIssue] = []
        if book is None:
            issues.append(
                QCIssue(
                    code=QCIssueCode.ORDERBOOK_EMPTY,
                    severity=QCSeverity.WARNING,
                    message="order book unavailable; micro-structure features degrade to neutral",
                    symbol=symbol,
                    healable=False,
                )
            )
            return issues

        if book.best_bid >= book.best_ask:
            issues.append(
                QCIssue(
                    code=QCIssueCode.ORDERBOOK_CROSSED,
                    severity=QCSeverity.WARNING,
                    message=f"crossed book bid={book.best_bid} ask={book.best_ask}",
                    symbol=symbol,
                    healable=False,
                )
            )
        if book.bid_volume <= 0.0 and book.ask_volume <= 0.0:
            issues.append(
                QCIssue(
                    code=QCIssueCode.ORDERBOOK_EMPTY,
                    severity=QCSeverity.WARNING,
                    message="both book sides report zero depth",
                    symbol=symbol,
                    healable=False,
                )
            )
        return issues

    def validate_futures_metrics(self, metrics: FuturesMetrics | None, symbol: str) -> list[QCIssue]:
        """Sanity-check futures metrics (all issues are non-fatal by design)."""
        issues: list[QCIssue] = []
        if metrics is None:
            issues.append(
                QCIssue(
                    code=QCIssueCode.EMPTY_DATASET,
                    severity=QCSeverity.WARNING,
                    message="futures metrics unavailable; funding/OI features neutralised",
                    symbol=symbol,
                    healable=False,
                )
            )
            return issues

        if abs(metrics.funding_rate) > 0.02:
            issues.append(
                QCIssue(
                    code=QCIssueCode.RETURN_OUTLIER,
                    severity=QCSeverity.WARNING,
                    message=f"funding rate at the exchange cap: {metrics.funding_rate:.5f}",
                    symbol=symbol,
                    healable=False,
                )
            )
        if metrics.open_interest <= 0.0:
            issues.append(
                QCIssue(
                    code=QCIssueCode.EMPTY_DATASET,
                    severity=QCSeverity.WARNING,
                    message="open interest reported as zero",
                    symbol=symbol,
                    healable=False,
                )
            )
        return issues

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------
    def _check_timestamp_integrity(
        self,
        symbol: str,
        candles: Sequence[OHLCVCandle],
        timestamps: Sequence[int],
    ) -> list[QCIssue]:
        """Grid alignment, duplicates, ordering and future-dated candles."""
        issues: list[QCIssue] = []

        misaligned: list[int] = [ts for ts in timestamps if ts % self._timeframe_ms != 0]
        if misaligned:
            issues.append(
                QCIssue(
                    code=QCIssueCode.MISALIGNED_TIMESTAMP,
                    severity=QCSeverity.CRITICAL,
                    message=(
                        f"{len(misaligned)} timestamp(s) are not multiples of "
                        f"{self._timeframe_ms} ms"
                    ),
                    symbol=symbol,
                    timestamps=tuple(misaligned[:50]),
                    healable=True,
                )
            )

        duplicates: list[int] = self._find_duplicates(timestamps)
        if duplicates:
            issues.append(
                QCIssue(
                    code=QCIssueCode.DUPLICATE_TIMESTAMP,
                    severity=QCSeverity.CRITICAL,
                    message=f"{len(duplicates)} duplicated candle open time(s)",
                    symbol=symbol,
                    timestamps=tuple(duplicates[:50]),
                    healable=True,
                )
            )

        # `candles` is pre-sorted, so compare against the caller's original order.
        original_order: list[int] = [candle.timestamp for candle in candles]
        if original_order != sorted(original_order):
            issues.append(
                QCIssue(
                    code=QCIssueCode.UNSORTED_TIMESTAMPS,
                    severity=QCSeverity.WARNING,
                    message="candles arrived unsorted and were reordered",
                    symbol=symbol,
                    healable=False,
                )
            )

        horizon_ms: int = utc_now_ms() + self._timeframe_ms
        future_dated: list[int] = [ts for ts in timestamps if ts > horizon_ms]
        if future_dated:
            issues.append(
                QCIssue(
                    code=QCIssueCode.FUTURE_TIMESTAMP,
                    severity=QCSeverity.CRITICAL,
                    message=f"{len(future_dated)} candle(s) dated in the future",
                    symbol=symbol,
                    timestamps=tuple(future_dated[:50]),
                    healable=True,
                )
            )
        return issues

    def _find_missing_timestamps(self, timestamps: Sequence[int]) -> list[int]:
        """Return every grid timestamp absent between the first and last candle.

        This is the exact inverse of the "timestamp integrity" rule: consecutive
        5-minute candles must be exactly ``timeframe_ms`` apart, so the expected
        grid is fully determined by the block's endpoints.
        """
        if len(timestamps) < 2:
            return []
        present: set[int] = set(timestamps)
        grid: list[int] = expected_timestamps(
            timestamps[0], timestamps[-1], self._timeframe_ms
        )
        return [ts for ts in grid if ts not in present]

    @staticmethod
    def _find_duplicates(timestamps: Sequence[int]) -> list[int]:
        """Return the timestamps that occur more than once, in ascending order."""
        seen: set[int] = set()
        duplicates: set[int] = set()
        for ts in timestamps:
            if ts in seen:
                duplicates.add(ts)
            seen.add(ts)
        return sorted(duplicates)

    def _check_price_logic(self, symbol: str, candles: Sequence[OHLCVCandle]) -> list[QCIssue]:
        """Verify OHLC geometry and finiteness for every row."""
        issues: list[QCIssue] = []
        violations: list[int] = []
        non_positive: list[int] = []
        nan_rows: list[int] = []

        for candle in candles:
            prices: tuple[float, float, float, float] = (
                candle.open,
                candle.high,
                candle.low,
                candle.close,
            )
            if any(not math.isfinite(price) for price in prices) or not math.isfinite(
                candle.volume
            ):
                nan_rows.append(candle.timestamp)
                continue
            if any(price <= 0.0 for price in prices):
                non_positive.append(candle.timestamp)
                continue
            if candle.high < candle.low:
                violations.append(candle.timestamp)
                continue
            if candle.high < max(candle.open, candle.close):
                violations.append(candle.timestamp)
                continue
            if candle.low > min(candle.open, candle.close):
                violations.append(candle.timestamp)

        if nan_rows:
            issues.append(
                QCIssue(
                    code=QCIssueCode.NAN_VALUES,
                    severity=QCSeverity.CRITICAL,
                    message=f"{len(nan_rows)} candle(s) contain NaN/inf values",
                    symbol=symbol,
                    timestamps=tuple(nan_rows[:50]),
                    healable=True,
                )
            )
        if non_positive:
            issues.append(
                QCIssue(
                    code=QCIssueCode.NON_POSITIVE_PRICE,
                    severity=QCSeverity.CRITICAL,
                    message=f"{len(non_positive)} candle(s) contain non-positive prices",
                    symbol=symbol,
                    timestamps=tuple(non_positive[:50]),
                    healable=True,
                )
            )
        if violations:
            issues.append(
                QCIssue(
                    code=QCIssueCode.PRICE_LOGIC_VIOLATION,
                    severity=QCSeverity.CRITICAL,
                    message=(
                        f"{len(violations)} candle(s) violate high >= max(o,c) >= "
                        f"min(o,c) >= low"
                    ),
                    symbol=symbol,
                    timestamps=tuple(violations[:50]),
                    healable=True,
                )
            )
        return issues

    def _check_volume_logic(self, symbol: str, candles: Sequence[OHLCVCandle]) -> list[QCIssue]:
        """Screen for negative volume, glitch spikes and dead feeds."""
        issues: list[QCIssue] = []
        volumes: np.ndarray = np.asarray([candle.volume for candle in candles], dtype=np.float64)

        negative_mask: np.ndarray = volumes < 0.0
        if bool(negative_mask.any()):
            bad: list[int] = [
                candles[index].timestamp for index in np.flatnonzero(negative_mask).tolist()
            ]
            issues.append(
                QCIssue(
                    code=QCIssueCode.NEGATIVE_VOLUME,
                    severity=QCSeverity.CRITICAL,
                    message=f"{len(bad)} candle(s) report negative volume",
                    symbol=symbol,
                    timestamps=tuple(bad[:50]),
                    healable=True,
                )
            )

        zero_ratio: float = float((volumes <= 0.0).mean()) if volumes.size else 1.0
        if volumes.size >= self._qc.min_rows_for_statistics and (
            zero_ratio > self._qc.max_zero_volume_ratio
        ):
            issues.append(
                QCIssue(
                    code=QCIssueCode.EXCESSIVE_ZERO_VOLUME,
                    severity=QCSeverity.CRITICAL,
                    message=(
                        f"{zero_ratio:.1%} of candles have zero volume "
                        f"(limit {self._qc.max_zero_volume_ratio:.0%})"
                    ),
                    symbol=symbol,
                    healable=True,
                )
            )

        positive: np.ndarray = volumes[volumes > 0.0]
        if positive.size >= self._qc.min_rows_for_statistics:
            median_volume: float = float(np.median(positive))
            if median_volume > 0.0:
                threshold: float = median_volume * self._qc.volume_spike_median_multiple
                spike_mask: np.ndarray = volumes > threshold
                if bool(spike_mask.any()):
                    spikes: list[int] = [
                        candles[index].timestamp
                        for index in np.flatnonzero(spike_mask).tolist()
                    ]
                    issues.append(
                        QCIssue(
                            code=QCIssueCode.VOLUME_SPIKE,
                            severity=QCSeverity.WARNING,
                            message=(
                                f"{len(spikes)} candle(s) exceed {threshold:,.0f} "
                                f"({self._qc.volume_spike_median_multiple:.0f}x median volume)"
                            ),
                            symbol=symbol,
                            timestamps=tuple(spikes[:50]),
                            healable=False,
                        )
                    )
        return issues

    def _check_return_outliers(self, symbol: str, candles: Sequence[OHLCVCandle]) -> list[QCIssue]:
        """Robust outlier screen on close-to-close log returns.

        Uses the median absolute deviation instead of the standard deviation:
        a single corrupt print inflates sigma enough to mask itself, whereas the
        MAD has a 50 % breakdown point and stays anchored to the bulk of the
        distribution.
        """
        issues: list[QCIssue] = []
        if len(candles) < self._qc.min_rows_for_statistics:
            return issues

        closes: np.ndarray = np.asarray([candle.close for candle in candles], dtype=np.float64)
        if bool((closes <= 0.0).any()):
            return issues  # already reported by the price-logic check

        log_returns: np.ndarray = np.diff(np.log(closes))
        if log_returns.size == 0:
            return issues

        simple_returns: np.ndarray = np.expm1(log_returns)
        hard_mask: np.ndarray = np.abs(simple_returns) > self._qc.max_abs_candle_return
        if bool(hard_mask.any()):
            offenders: list[int] = [
                candles[index + 1].timestamp for index in np.flatnonzero(hard_mask).tolist()
            ]
            issues.append(
                QCIssue(
                    code=QCIssueCode.RETURN_OUTLIER,
                    severity=QCSeverity.CRITICAL,
                    message=(
                        f"{len(offenders)} candle(s) move more than "
                        f"{self._qc.max_abs_candle_return:.0%} in a single 5m bar"
                    ),
                    symbol=symbol,
                    timestamps=tuple(offenders[:50]),
                    healable=True,
                )
            )

        median: float = float(np.median(log_returns))
        mad: float = float(np.median(np.abs(log_returns - median)))
        if mad <= 0.0:
            return issues

        sigma: float = _MAD_TO_SIGMA * mad
        z_scores: np.ndarray = np.abs(log_returns - median) / sigma
        soft_mask: np.ndarray = z_scores > self._qc.return_mad_zscore_limit
        soft_mask &= ~hard_mask  # do not report the same bar twice
        if bool(soft_mask.any()):
            offenders = [
                candles[index + 1].timestamp for index in np.flatnonzero(soft_mask).tolist()
            ]
            issues.append(
                QCIssue(
                    code=QCIssueCode.RETURN_OUTLIER,
                    severity=QCSeverity.WARNING,
                    message=(
                        f"{len(offenders)} candle(s) exceed "
                        f"{self._qc.return_mad_zscore_limit:.0f} robust sigmas "
                        f"(sigma={sigma:.5f})"
                    ),
                    symbol=symbol,
                    timestamps=tuple(offenders[:50]),
                    healable=False,
                )
            )
        return issues

    def _check_freshness(self, symbol: str, timestamps: Sequence[int]) -> list[QCIssue]:
        """Ensure the newest closed candle is not unacceptably stale."""
        if not timestamps:
            return []

        newest_expected: int = last_closed_candle_open_ms(self._timeframe_ms)
        lag_bars: int = int((newest_expected - timestamps[-1]) // self._timeframe_ms)
        if lag_bars <= 0:
            return []
        if lag_bars > self._qc.max_lag_bars:
            return [
                QCIssue(
                    code=QCIssueCode.STALE_DATA,
                    severity=QCSeverity.CRITICAL,
                    message=(
                        f"newest candle is {lag_bars} bars behind the grid "
                        f"(limit {self._qc.max_lag_bars})"
                    ),
                    symbol=symbol,
                    timestamps=(timestamps[-1],),
                    healable=True,
                )
            ]
        return [
            QCIssue(
                code=QCIssueCode.STALE_DATA,
                severity=QCSeverity.INFO,
                message=f"newest candle lags the grid by {lag_bars} bar(s)",
                symbol=symbol,
                timestamps=(timestamps[-1],),
                healable=False,
            )
        ]

    # ------------------------------------------------------------------
    # Healing helpers
    # ------------------------------------------------------------------
    def _heal_window(
        self,
        candles: Sequence[OHLCVCandle],
        report: QCReport,
    ) -> tuple[int, int]:
        """Compute the ``[start_ms, end_ms]`` window that must be re-fetched.

        The window spans every suspicious timestamp (missing candles plus the
        timestamps attached to CRITICAL issues), padded by one bar on each side so
        boundary candles are refreshed too.  When the block is empty, the last
        ``ohlcv_limit`` bars are requested from scratch.
        """
        suspicious: set[int] = set(report.missing_timestamps)
        for issue in report.issues:
            if issue.severity is QCSeverity.CRITICAL:
                suspicious.update(issue.timestamps)

        if not suspicious:
            if candles:
                return candles[0].timestamp, candles[-1].timestamp + self._timeframe_ms
            end_ms: int = last_closed_candle_open_ms(self._timeframe_ms)
            span: int = self._settings.data.ohlcv_limit * self._timeframe_ms
            return end_ms - span, end_ms

        start_ms: int = min(suspicious) - self._timeframe_ms
        end_ms = max(suspicious) + self._timeframe_ms
        return start_ms, end_ms

    def _merge_blocks(
        self,
        base: Sequence[OHLCVCandle],
        patch: Sequence[OHLCVCandle],
        drop_range: tuple[int, int],
    ) -> list[OHLCVCandle]:
        """Merge freshly fetched candles over the damaged window.

        Every base candle inside ``drop_range`` is discarded first: the exchange's
        new copy is authoritative, and keeping the corrupt row would let a bad
        print survive the heal.  Candles outside the window are preserved so the
        history depth needed by Module B is not lost.
        """
        low, high = drop_range
        merged: dict[int, OHLCVCandle] = {
            candle.timestamp: candle
            for candle in base
            if not (low <= candle.timestamp <= high)
        }
        for candle in patch:
            merged[candle.timestamp] = candle
        return [merged[key] for key in sorted(merged)]

    @staticmethod
    def describe(issues: Iterable[QCIssue]) -> str:
        """Render a compact, log-friendly description of a set of issues."""
        rendered: list[str] = [str(issue) for issue in issues]
        return " | ".join(rendered) if rendered else "clean"
