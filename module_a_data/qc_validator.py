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
import time
from typing import Awaitable, Callable, Final, Iterable, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from core.exceptions import DataIntegrityError
from core.logger import get_logger
from core.utils import (
    backoff_delay,
    expected_timestamps,
    last_closed_candle_open_ms,
    longest_clean_trailing_run,
    utc_now_ms,
)
from module_a_data.models import (
    FuturesMetrics,
    HealAttempt,
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
        *,
        quarantine_unhealable: bool = False,
    ) -> tuple[list[OHLCVCandle], QCReport, list[HealAttempt]]:
        """Validate a block and recursively repair it until it is pristine.

        The heal loop is *targeted*: only the damaged timestamp windows are
        re-requested (see :meth:`_heal_windows`), and repaired rows are merged
        over the corrupt ones by timestamp.  Candles flagged as corrupt (bad
        price/volume logic or return outliers) are dropped before the merge so
        the exchange's fresh copy wins.

        Args:
            symbol: Symbol under repair.
            candles: The block as first fetched.
            refetch: ``async (symbol, start_ms, end_ms) -> list[OHLCVCandle]``.
            quarantine_unhealable: When heal attempts are exhausted (or the
                wall-clock budget runs out) and this is ``True``, do not discard
                the whole block.  Instead drop every candle tied to a surviving
                CRITICAL issue plus anything before the last remaining gap, and
                keep the longest clean run ending at the newest candle - the
                shape every downstream consumer expects.  This is what lets a
                historical backfill make monotonic progress in the face of a
                permanently unfetchable window (an exchange-side halt, a
                pre-listing gap) instead of re-attempting - and re-failing - the
                exact same doomed re-fetch on every single bootstrap run.  The
                live per-cycle path leaves this ``False``: a symbol whose fresh
                data cannot be fully healed is excluded from that cycle outright
                rather than traded on a trimmed subset.

        Returns:
            ``(healed_candles, final_report, heal_attempts)``.  ``final_report.passed``
            is always ``True`` for the returned candles, whether they arrived
            pristine, were healed, or - when quarantining - were trimmed down to
            their longest clean trailing run.  ``heal_attempts`` records every
            re-fetch round tried (symbol, reason, window span, bars
            requested/received/written, remaining damage, duration and
            per-round result), even when the call ultimately raises - the
            records are attached to the raised error's ``context["heal_attempts"]``
            so a caller can log or persist them either way.

        Raises:
            DataIntegrityError: When the block is still invalid after
                ``qc.max_heal_attempts`` repair rounds (or the wall-clock
                budget elapses) and either quarantining is disabled or nothing
                clean survives it, or when the failure is not the kind a
                re-fetch can fix.
        """
        working: list[OHLCVCandle] = sorted(candles, key=lambda item: item.timestamp)
        report: QCReport = self.validate_candles(symbol, working)
        heal_attempts: list[HealAttempt] = []

        def _record_quarantine(before: QCReport, kept: list[OHLCVCandle]) -> None:
            heal_attempts.append(
                HealAttempt(
                    symbol=symbol,
                    attempt_number=len(heal_attempts) + 1,
                    reason=before.critical_codes,
                    window_count=0,
                    start_timestamp=kept[0].timestamp if kept else None,
                    end_timestamp=kept[-1].timestamp if kept else None,
                    bars_requested=0,
                    bars_received=0,
                    bars_written=0,
                    bars_invalid_after_heal=0,
                    duration_seconds=0.0,
                    result="quarantined",
                )
            )

        attempt: int = 0
        deadline: float = time.monotonic() + self._qc.max_heal_duration_seconds
        while not report.passed and attempt < self._qc.max_heal_attempts:
            if not report.healable:
                quarantined = self._try_quarantine(symbol, working, report, quarantine_unhealable)
                if quarantined is not None:
                    healed_candles, final_report = quarantined
                    _record_quarantine(report, healed_candles)
                    return healed_candles, final_report, heal_attempts
                raise DataIntegrityError(
                    "QC failure cannot be healed by re-fetching",
                    symbol=symbol,
                    codes=report.critical_codes,
                    heal_attempts=[record.model_dump(mode="json") for record in heal_attempts],
                )
            if time.monotonic() >= deadline:
                # A stuck healer must not hold the shared request-rate budget
                # indefinitely and starve every other symbol's cycle.
                quarantined = self._try_quarantine(symbol, working, report, quarantine_unhealable)
                if quarantined is not None:
                    healed_candles, final_report = quarantined
                    _record_quarantine(report, healed_candles)
                    return healed_candles, final_report, heal_attempts
                raise DataIntegrityError(
                    "heal loop exceeded its wall-clock budget",
                    symbol=symbol,
                    attempts=attempt,
                    budget_seconds=self._qc.max_heal_duration_seconds,
                    codes=report.critical_codes,
                    heal_attempts=[record.model_dump(mode="json") for record in heal_attempts],
                )

            delay: float = backoff_delay(
                attempt,
                base_seconds=self._qc.heal_backoff_seconds,
                max_seconds=self._settings.exchange.backoff_max_seconds,
                jitter=self._settings.exchange.backoff_jitter,
            )
            windows: list[tuple[int, int]] = self._heal_windows(working, report)
            reason: tuple[str, ...] = report.critical_codes
            _LOGGER.warning(
                "QC failed for %s (%s) - heal attempt %d/%d: %d window(s) in %.2fs",
                symbol,
                ", ".join(reason),
                attempt + 1,
                self._qc.max_heal_attempts,
                len(windows),
                delay,
            )
            await asyncio.sleep(delay)

            attempt_started: float = time.monotonic()
            bars_requested: int = sum(
                max(1, (end_ms - start_ms) // self._timeframe_ms) for start_ms, end_ms in windows
            )
            patches: list[list[OHLCVCandle]] = await asyncio.gather(
                *(refetch(symbol, start_ms, end_ms) for start_ms, end_ms in windows)
            )
            bars_received: int = sum(len(patch) for patch in patches)

            working = self._merge_patches(working, self._suspicious_timestamps(report), patches)
            report = self.validate_candles(symbol, working)
            attempt += 1

            heal_attempts.append(
                HealAttempt(
                    symbol=symbol,
                    attempt_number=attempt,
                    reason=reason,
                    window_count=len(windows),
                    start_timestamp=windows[0][0] if windows else None,
                    end_timestamp=windows[-1][1] if windows else None,
                    bars_requested=bars_requested,
                    bars_received=bars_received,
                    bars_written=bars_received,
                    bars_invalid_after_heal=len(self._suspicious_timestamps(report)),
                    duration_seconds=time.monotonic() - attempt_started,
                    result="resolved" if report.passed else "still_invalid",
                )
            )

        if not report.passed:
            quarantined = self._try_quarantine(symbol, working, report, quarantine_unhealable)
            if quarantined is not None:
                healed_candles, final_report = quarantined
                _record_quarantine(report, healed_candles)
                return healed_candles, final_report, heal_attempts
            raise DataIntegrityError(
                "data still invalid after exhausting heal attempts",
                symbol=symbol,
                attempts=attempt,
                codes=report.critical_codes,
                heal_attempts=[record.model_dump(mode="json") for record in heal_attempts],
            )

        if attempt:
            _LOGGER.info("Healed %s after %d attempt(s): %d rows", symbol, attempt, len(working))
        return working, report, heal_attempts

    def _try_quarantine(
        self,
        symbol: str,
        working: list[OHLCVCandle],
        report: QCReport,
        enabled: bool,
    ) -> tuple[list[OHLCVCandle], QCReport] | None:
        """Attempt to salvage a clean trailing run when quarantining is enabled.

        Returns ``None`` (never salvageable) unless ``enabled`` is set and a
        non-empty, fully-passing trailing run survives trimming.
        """
        if not enabled:
            return None
        trimmed: list[OHLCVCandle] = self._longest_clean_trailing_run(working, report)
        if not trimmed:
            return None
        trimmed_report: QCReport = self.validate_candles(symbol, trimmed)
        if not trimmed_report.passed:
            return None
        _LOGGER.warning(
            "%s: %d candle(s) quarantined as unhealable (%s) - keeping the clean "
            "trailing run of %d candle(s) from %d to %d",
            symbol,
            len(working) - len(trimmed),
            ", ".join(report.critical_codes),
            len(trimmed),
            trimmed[0].timestamp,
            trimmed[-1].timestamp,
        )
        return trimmed, trimmed_report

    def _longest_clean_trailing_run(
        self,
        working: Sequence[OHLCVCandle],
        report: QCReport,
    ) -> list[OHLCVCandle]:
        """Drop every candle tied to a surviving CRITICAL issue, then keep only
        the contiguous run ending at the newest candle.

        See :func:`core.utils.longest_clean_trailing_run` for the shared rule
        this and :class:`module_b_features.processor.DatasetProcessor` both
        apply.
        """
        bad: set[int] = set()
        for issue in report.issues:
            if issue.severity is QCSeverity.CRITICAL:
                bad.update(issue.timestamps)

        kept_timestamps: set[int] = set(
            longest_clean_trailing_run(
                (candle.timestamp for candle in working), bad, self._timeframe_ms
            )
        )
        return sorted(
            (candle for candle in working if candle.timestamp in kept_timestamps),
            key=lambda item: item.timestamp,
        )

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

    def validate_stored_frame(self, symbol: str, frame: pd.DataFrame) -> list[QCIssue]:
        """Fast structural check for OHLCV already read back from storage.

        Module A's gatekeeper runs once, at ingestion.  Module B then reads an
        arbitrary trailing window straight out of SQLite - and that window can
        span rows written by different cycles (or different bootstrap runs)
        that were each individually clean but are not guaranteed to be
        *jointly* contiguous.  This is the second gate: it re-checks grid
        alignment, duplicates, ordering, gaps and basic price/finite sanity over
        whatever Module B is about to hand to feature engineering, so a symbol
        whose stored window turns out to be broken is rejected before it can
        reach the ML pipeline rather than silently producing features over a
        discontinuous series.

        Freshness and statistical-outlier checks are deliberately not repeated
        here: they are only meaningful for the newest fetched tail (a live
        cycle), not for an arbitrary historical slice used for training.
        """
        issues: list[QCIssue] = []
        if frame.empty or "timestamp" not in frame.columns:
            issues.append(
                QCIssue(
                    code=QCIssueCode.EMPTY_DATASET,
                    severity=QCSeverity.CRITICAL,
                    message="no stored candles available",
                    symbol=symbol,
                    healable=False,
                )
            )
            return issues

        timestamps: list[int] = [int(value) for value in frame["timestamp"].tolist()]

        duplicates: list[int] = self._find_duplicates(timestamps)
        if duplicates:
            issues.append(
                QCIssue(
                    code=QCIssueCode.DUPLICATE_TIMESTAMP,
                    severity=QCSeverity.CRITICAL,
                    message=f"{len(duplicates)} duplicated candle open time(s) in storage",
                    symbol=symbol,
                    timestamps=tuple(duplicates[:50]),
                    healable=False,
                )
            )

        if timestamps != sorted(timestamps):
            issues.append(
                QCIssue(
                    code=QCIssueCode.UNSORTED_TIMESTAMPS,
                    severity=QCSeverity.CRITICAL,
                    message="stored candles are not chronologically ordered",
                    symbol=symbol,
                    healable=False,
                )
            )
            timestamps = sorted(timestamps)

        misaligned: list[int] = [ts for ts in timestamps if ts % self._timeframe_ms != 0]
        if misaligned:
            issues.append(
                QCIssue(
                    code=QCIssueCode.MISALIGNED_TIMESTAMP,
                    severity=QCSeverity.CRITICAL,
                    message=f"{len(misaligned)} stored timestamp(s) off the 5m grid",
                    symbol=symbol,
                    timestamps=tuple(misaligned[:50]),
                    healable=False,
                )
            )

        missing: list[int] = self._find_missing_timestamps(timestamps)
        if missing:
            issues.append(
                QCIssue(
                    code=QCIssueCode.MISSING_CANDLES,
                    severity=QCSeverity.CRITICAL,
                    message=(
                        f"{len(missing)} missing 5m candle(s) inside the stored window "
                        f"between {timestamps[0]} and {timestamps[-1]}"
                    ),
                    symbol=symbol,
                    timestamps=tuple(missing[:50]),
                    healable=False,
                )
            )

        price_columns: list[str] = [
            column for column in ("open", "high", "low", "close") if column in frame.columns
        ]
        if len(price_columns) == 4:
            prices: np.ndarray = frame[price_columns].to_numpy(dtype=np.float64)
            volume: np.ndarray = (
                frame["volume"].to_numpy(dtype=np.float64)
                if "volume" in frame.columns
                else np.zeros(len(frame))
            )
            finite: np.ndarray = np.isfinite(prices).all(axis=1) & np.isfinite(volume)
            if not bool(finite.all()):
                issues.append(
                    QCIssue(
                        code=QCIssueCode.NAN_VALUES,
                        severity=QCSeverity.CRITICAL,
                        message=f"{int((~finite).sum())} stored row(s) contain NaN/inf values",
                        symbol=symbol,
                        healable=False,
                    )
                )
            positive: np.ndarray = finite & (prices > 0.0).all(axis=1)
            open_, high, low, close = (prices[:, index] for index in range(4))
            geometry: np.ndarray = (
                positive
                & (high >= low)
                & (high >= np.maximum(open_, close))
                & (low <= np.minimum(open_, close))
            )
            bad_geometry: np.ndarray = finite & ~geometry
            if bool(bad_geometry.any()):
                issues.append(
                    QCIssue(
                        code=QCIssueCode.PRICE_LOGIC_VIOLATION,
                        severity=QCSeverity.CRITICAL,
                        message=(
                            f"{int(bad_geometry.sum())} stored row(s) violate OHLC price "
                            "logic or contain a non-positive price"
                        ),
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
    def _suspicious_timestamps(self, report: QCReport) -> set[int]:
        """Every timestamp a heal round must treat as damaged: missing candles
        plus the timestamps attached to CRITICAL issues."""
        suspicious: set[int] = set(report.missing_timestamps)
        for issue in report.issues:
            if issue.severity is QCSeverity.CRITICAL:
                suspicious.update(issue.timestamps)
        return suspicious

    def _heal_windows(
        self,
        candles: Sequence[OHLCVCandle],
        report: QCReport,
    ) -> list[tuple[int, int]]:
        """Compute the precise ``[start_ms, end_ms]`` windows that need re-fetching.

        Suspicious timestamps (missing candles plus the timestamps attached to
        CRITICAL issues) are grouped into contiguous runs - merging runs within
        ``heal_merge_gap_bars`` of each other - and each run becomes its own
        targeted re-fetch window, padded by one bar on each side so the exchange
        is asked for a little context around the damage.  This is what keeps a
        heal *precise*: two unrelated one-candle glitches at opposite ends of a
        long history each cost one small request instead of the two being fused
        into a single re-fetch spanning the entire block.  The padding only
        widens what gets *requested* - :meth:`_merge_patches` only ever drops the
        exact suspicious timestamps from the working set, so a padded boundary
        candle that was already clean survives even when the re-fetch does not
        happen to return it again.

        A ``STALE_DATA`` verdict additionally stretches the newest window through
        to the freshest candle the grid currently expects, so a heal that is
        several bars behind catches all the way up in one round rather than
        creeping forward one bar per attempt while real time keeps moving.

        When the block is empty, the last ``ohlcv_limit`` bars are requested from
        scratch.  When grouping would produce more windows than
        ``max_heal_window_groups``, the damage is judged widespread enough that
        per-gap fragmentation buys no precision - but the runs are still
        coalesced into **bounded batches** of at most ``max_heal_window_bars``
        each, never one unbounded window spanning the whole range.  Without this
        cap, sufficiently scattered damage across a long history (a stale-data
        stretch, a multi-day exchange gap) could otherwise turn into a single
        re-fetch of tens of thousands of candles in one attempt.
        """
        suspicious: set[int] = self._suspicious_timestamps(report)

        if not suspicious:
            if candles:
                return [(candles[0].timestamp, candles[-1].timestamp + self._timeframe_ms)]
            end_ms: int = last_closed_candle_open_ms(self._timeframe_ms)
            span: int = self._settings.data.ohlcv_limit * self._timeframe_ms
            return [(end_ms - span, end_ms)]

        ordered: list[int] = sorted(suspicious)
        merge_gap_ms: int = self._qc.heal_merge_gap_bars * self._timeframe_ms
        runs: list[list[int]] = [[ordered[0]]]
        for ts in ordered[1:]:
            if ts - runs[-1][-1] <= self._timeframe_ms + merge_gap_ms:
                runs[-1].append(ts)
            else:
                runs.append([ts])

        windows: list[tuple[int, int]] = [
            (run[0] - self._timeframe_ms, run[-1] + self._timeframe_ms) for run in runs
        ]

        if QCIssueCode.STALE_DATA.value in report.critical_codes:
            newest_expected: int = last_closed_candle_open_ms(self._timeframe_ms) + self._timeframe_ms
            last_start, last_end = windows[-1]
            windows[-1] = (last_start, max(last_end, newest_expected))

        if len(windows) > self._qc.max_heal_window_groups:
            return self._batch_windows(windows)
        return windows

    def _batch_windows(self, windows: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """Coalesce many small windows into few, but bounded-size, batches.

        Greedily extends each batch until adding the next window would push its
        span past ``max_heal_window_bars``, then starts a new batch.  This keeps
        every single re-fetch request controlled in size regardless of how many
        (or how widely scattered) the original damaged windows were.
        """
        max_span_ms: int = self._qc.max_heal_window_bars * self._timeframe_ms
        batches: list[tuple[int, int]] = []
        batch_start, batch_end = windows[0]
        for start_ms, end_ms in windows[1:]:
            if end_ms - batch_start <= max_span_ms:
                batch_end = end_ms
            else:
                batches.append((batch_start, batch_end))
                batch_start, batch_end = start_ms, end_ms
        batches.append((batch_start, batch_end))
        return batches

    def _merge_patches(
        self,
        base: Sequence[OHLCVCandle],
        drop_timestamps: set[int],
        patches: Sequence[Sequence[OHLCVCandle]],
    ) -> list[OHLCVCandle]:
        """Apply one heal round's re-fetched patches over the working set.

        Only the *exact* suspicious timestamps are dropped from ``base`` - never
        the wider padded window a patch was fetched over - so a clean boundary
        candle is never discarded just because a re-fetch legitimately returned
        nothing for a neighbouring, genuinely damaged timestamp.  Every patch row
        is then applied as an upsert, which both fills the dropped timestamps and
        opportunistically refreshes any padded-but-clean candle the exchange
        happened to send back too.
        """
        merged: dict[int, OHLCVCandle] = {
            candle.timestamp: candle for candle in base if candle.timestamp not in drop_timestamps
        }
        for patch in patches:
            for candle in patch:
                merged[candle.timestamp] = candle
        return [merged[key] for key in sorted(merged)]

    @staticmethod
    def describe(issues: Iterable[QCIssue]) -> str:
        """Render a compact, log-friendly description of a set of issues."""
        rendered: list[str] = [str(issue) for issue in issues]
        return " | ".join(rendered) if rendered else "clean"
