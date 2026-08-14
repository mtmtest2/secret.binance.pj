"""Module B controller: SQLite -> features -> labels -> a model-ready dataset.

Two entry points, deliberately separated so the live path can never accidentally
touch forward-looking data:

* :meth:`DatasetProcessor.build_training_dataset` - historical mode.  Runs the
  feature engine *and* the labeler, drops warm-up rows and the un-simulatable
  tail, and returns aligned design matrices for all four models.
* :meth:`DatasetProcessor.build_inference_payload` - live mode.  Runs the feature
  engine only, and returns the **single most recent fully-formed feature row**.
  The labeler is not imported into this path at all.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from core.exceptions import FeatureEngineeringError, InsufficientDataError
from core.logger import get_logger
from core.utils import longest_clean_trailing_run
from module_a_data.db_handler import DatabaseHandler
from module_a_data.models import QCIssue, QCSeverity
from module_a_data.qc_validator import QCValidator
from module_b_features.features import (
    FEATURE_COLUMNS,
    REQUIRED_FEATURE_COLUMNS,
    FeatureService,
)
from module_b_features.labeler import LABEL_ORDER, TradeLabeler

_LOGGER = get_logger(__name__)

#: Columns carried alongside the features for bookkeeping / backtesting.
_META_COLUMNS: Final[tuple[str, ...]] = ("timestamp", "open", "high", "low", "close", "volume")

#: Average Gregorian month length - used only to translate the "N months"
#: split configuration into millisecond boundaries. Precise to well under an
#: hour over a 24-month window, which is immaterial next to the multi-hour
#: purge/embargo gap applied at every split boundary.
_DAYS_PER_MONTH: Final[float] = 30.4375
_MS_PER_DAY: Final[int] = 86_400_000


@dataclass(slots=True, frozen=True)
class SplitBoundaries:
    """Millisecond boundaries for a strict, time-ordered 3-way split.

    Computed once from a dataset's own timestamp range so every model head -
    and every report that describes the split - agrees on exactly the same
    calendar cut points.  ``validation_start_ms`` and ``test_start_ms`` are
    the *nominal* (embargo-free) boundaries; :func:`assign_split` is what
    actually removes the trailing ``embargo_ms`` of each earlier block so a
    forward-looking label can never reach across into the next block.
    """

    train_days: float
    validation_days: float
    test_days: float
    validation_start_ms: int
    test_start_ms: int
    embargo_ms: int
    #: ``True`` when the dataset's actual history was shorter than
    #: ``train_days + validation_days + test_days`` and every window was
    #: scaled down proportionally (2:1:1) to fit - the split is still
    #: strictly chronological and still embargoed, just narrower than the
    #: nominal 12/6/6 month configuration.
    scaled_down: bool


@dataclass(slots=True, frozen=True)
class ChronologicalSplit:
    """Row positions plus the realised (data-backed) date range of each block.

    Train is always the oldest block, validation the middle block and test -
    the block a live deployment would eventually have traded through - is
    always the newest.  Row positions are positional indices into
    :class:`ProcessedDataset`'s ``features``/``metadata`` frames (or into
    whatever timestamp-aligned frame :func:`assign_split` was called with).
    """

    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray
    train_start_ms: int | None
    train_end_ms: int | None
    validation_start_ms: int | None
    validation_end_ms: int | None
    test_start_ms: int | None
    test_end_ms: int | None
    boundaries: SplitBoundaries


def compute_split_boundaries(
    data_start_ms: int,
    data_end_ms: int,
    *,
    train_months: float,
    validation_months: float,
    test_months: float,
    purge_bars: int,
    timeframe_ms: int,
) -> SplitBoundaries:
    """Derive the train/validation/test cut points from a dataset's time span.

    ``test`` is anchored to the *end* of the available history (the most
    recent data) and is exactly ``test_months`` wide; ``validation`` is the
    ``validation_months``-wide block immediately before it; everything older
    is ``train``. When the dataset does not actually span
    ``train_months + validation_months + test_months``, all three windows are
    scaled down proportionally (preserving the configured ratio) rather than
    silently starving validation/test or raising - a newly-listed symbol with
    less than 2 years of history is an expected, not exceptional, case (see
    ``UniverseSettings.min_history_days``).

    The embargo (``purge_bars`` bars, converted to milliseconds) is *not*
    scaled down: it is sized to cover the label horizon
    (``LabelSettings.max_holding_bars``) regardless of how much history is
    available, since a shorter dataset does not make a forward-looking label
    resolve any faster.
    """
    day_ms: float = _MS_PER_DAY
    train_days: float = train_months * _DAYS_PER_MONTH
    validation_days: float = validation_months * _DAYS_PER_MONTH
    test_days: float = test_months * _DAYS_PER_MONTH

    train_ms: float = train_days * day_ms
    validation_ms: float = validation_days * day_ms
    test_ms: float = test_days * day_ms
    requested_span: float = train_ms + validation_ms + test_ms
    available_span: float = max(0.0, float(data_end_ms - data_start_ms))

    scaled_down: bool = False
    if requested_span > 0.0 and available_span < requested_span:
        scale: float = available_span / requested_span
        train_ms *= scale
        validation_ms *= scale
        test_ms *= scale
        train_days *= scale
        validation_days *= scale
        test_days *= scale
        scaled_down = True
        _LOGGER.warning(
            "Dataset spans only %.1f day(s) - narrower than the configured "
            "%.1f-day (train+validation+test) window. Scaling all three "
            "blocks down proportionally (2:1:1 ratio preserved) instead of "
            "starving validation/test.",
            available_span / day_ms,
            requested_span / day_ms,
        )

    test_start_ms: int = int(round(data_end_ms - test_ms))
    validation_start_ms: int = int(round(test_start_ms - validation_ms))
    embargo_ms: int = int(purge_bars) * int(timeframe_ms)

    return SplitBoundaries(
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        validation_start_ms=validation_start_ms,
        test_start_ms=test_start_ms,
        embargo_ms=embargo_ms,
        scaled_down=scaled_down,
    )


def assign_split(
    timestamps: np.ndarray, boundaries: SplitBoundaries
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bucket row positions into (train, validation, test) given ``boundaries``.

    The embargo is cut from the *end* of the earlier block at each boundary,
    never from the start of the later one: a label simulated from a row near
    the end of train looks forward at most ``max_holding_bars`` candles, so
    it is *train*'s trailing rows - not validation's leading ones - that can
    see into the next block.  Test therefore keeps its full nominal width and
    starts exactly at ``boundaries.test_start_ms``.
    """
    train_mask: np.ndarray = timestamps < (boundaries.validation_start_ms - boundaries.embargo_ms)
    validation_mask: np.ndarray = (timestamps >= boundaries.validation_start_ms) & (
        timestamps < (boundaries.test_start_ms - boundaries.embargo_ms)
    )
    test_mask: np.ndarray = timestamps >= boundaries.test_start_ms
    return (
        np.nonzero(train_mask)[0].astype(np.int64),
        np.nonzero(validation_mask)[0].astype(np.int64),
        np.nonzero(test_mask)[0].astype(np.int64),
    )


@dataclass(slots=True)
class ProcessedDataset:
    """A fully aligned, NaN-free training dataset for the four ML heads."""

    features: pd.DataFrame
    direction_target: pd.Series
    entry_target: pd.Series
    exit_targets: pd.DataFrame
    risk_target: pd.Series
    metadata: pd.DataFrame
    symbols: tuple[str, ...] = field(default=())
    feature_columns: tuple[str, ...] = field(default=FEATURE_COLUMNS)
    #: Real, measured dataset-health counters from the cleaning step that
    #: produced this dataset - never approximated - for the ML diagnostic
    #: report's Dataset Health section.
    total_candidate_rows: int = field(default=0)
    rejected_invalid_label_rows: int = field(default=0)
    dropped_missing_or_inf_rows: int = field(default=0)
    duplicate_feature_rows: int = field(default=0)
    #: Per-feature null counts *after* cleaning.  Rows are no longer destroyed
    #: for a NaN in a single feature column (the boosters split on NaN
    #: natively), so this is the only remaining visibility into which features
    #: are actually sparse - and it is what turns "the model underperforms"
    #: into "this one column is 70% empty before 2025".
    null_counts_by_feature: dict[str, int] = field(default_factory=dict)
    #: ``{symbol: {"YYYY-MM": null_rate}}`` for the sparsest features, so a
    #: coverage collapse that is concentrated in one symbol or one period is
    #: visible instead of being averaged away.
    null_rate_by_symbol_month: dict[str, dict[str, float]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.features)

    @property
    def is_empty(self) -> bool:
        """``True`` when no usable training row survived cleaning."""
        return self.features.empty

    def class_distribution(self) -> dict[str, int]:
        """Row count per direction class, for logging and sanity checks."""
        counts: dict[str, int] = self.direction_target.value_counts().to_dict()
        return {str(name): int(count) for name, count in counts.items()}

    def _timestamps(self) -> np.ndarray:
        if "timestamp" not in self.metadata.columns:
            return np.array([], dtype=np.int64)
        return self.metadata["timestamp"].to_numpy(dtype=np.int64)

    def split_boundaries(
        self,
        *,
        train_months: float,
        validation_months: float,
        test_months: float,
        purge_bars: int,
        timeframe_ms: int,
    ) -> SplitBoundaries | None:
        """The dataset's own train/validation/test cut points, or ``None`` when empty.

        Computed from the *full* dataset's timestamp range so every model
        head - even one that later filters down to a row subset (Exit,
        Risk) - can bucket its own rows against the exact same calendar
        boundaries. Callers with a filtered subset should reuse the
        :class:`SplitBoundaries` this returns (via :func:`assign_split`)
        rather than recomputing it from their narrower timestamp range.
        """
        timestamps: np.ndarray = self._timestamps()
        if timestamps.size == 0:
            return None
        return compute_split_boundaries(
            int(timestamps.min()),
            int(timestamps.max()),
            train_months=train_months,
            validation_months=validation_months,
            test_months=test_months,
            purge_bars=purge_bars,
            timeframe_ms=timeframe_ms,
        )

    def chronological_split(self, boundaries: SplitBoundaries | None) -> ChronologicalSplit:
        """Strict, time-ordered train/validation/test row positions.

        Train is always the oldest block, validation the middle block, test
        the newest (and, by construction, the only block that is ever
        genuinely out-of-sample with respect to everything done in
        ``train()`` - fitting, early stopping, calibration, threshold
        selection). Never randomly shuffles rows.
        """
        empty: np.ndarray = np.array([], dtype=np.int64)
        if boundaries is None:
            return ChronologicalSplit(
                empty, empty, empty, None, None, None, None, None, None, boundaries  # type: ignore[arg-type]
            )

        timestamps: np.ndarray = self._timestamps()
        train_index, validation_index, test_index = assign_split(timestamps, boundaries)

        def _bounds(index: np.ndarray) -> tuple[int | None, int | None]:
            if index.size == 0:
                return None, None
            subset: np.ndarray = timestamps[index]
            return int(subset.min()), int(subset.max())

        train_start, train_end = _bounds(train_index)
        validation_start, validation_end = _bounds(validation_index)
        test_start, test_end = _bounds(test_index)

        return ChronologicalSplit(
            train_index=train_index,
            validation_index=validation_index,
            test_index=test_index,
            train_start_ms=train_start,
            train_end_ms=train_end,
            validation_start_ms=validation_start,
            validation_end_ms=validation_end,
            test_start_ms=test_start,
            test_end_ms=test_end,
            boundaries=boundaries,
        )


@dataclass(slots=True)
class InferencePayload:
    """The live-inference view of a single symbol at a single candle close."""

    symbol: str
    timestamp: int
    close: float
    features: pd.DataFrame  # exactly one row, columns == FEATURE_COLUMNS
    snapshot: dict[str, float]  # human-readable feature values for the audit log

    @property
    def feature_vector(self) -> np.ndarray:
        """The single feature row as a ``(1, n_features)`` float array."""
        return self.features.to_numpy(dtype=np.float64)


class DatasetProcessor:
    """Coordinates the database, the feature service and the labeler."""

    def __init__(
        self,
        settings: Settings,
        database: DatabaseHandler,
        feature_service: FeatureService | None = None,
        labeler: TradeLabeler | None = None,
        validator: QCValidator | None = None,
    ) -> None:
        self._settings: Settings = settings
        self._db: DatabaseHandler = database
        self._features: FeatureService = feature_service or FeatureService(settings)
        self._labeler: TradeLabeler = labeler or TradeLabeler(settings)
        #: Second gate, run on whatever Module B reads back from storage - see
        #: `QCValidator.validate_stored_frame` for why ingestion-time validation
        #: alone is not sufficient.
        self._validator: QCValidator = validator or QCValidator(settings)

    @property
    def feature_service(self) -> FeatureService:
        """Expose the feature service (shared with the backtester)."""
        return self._features

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------
    async def build_training_dataset(
        self,
        symbols: Sequence[str] | None = None,
        max_candles_per_symbol: int | None = None,
    ) -> ProcessedDataset:
        """Assemble a pooled, cross-sectional training dataset.

        All symbols are concatenated into one dataset and sorted by timestamp, so
        the chronological train/validation split is a genuine out-of-time split
        rather than a per-symbol one.

        Args:
            symbols: Universe to include (defaults to the configured universe).
            max_candles_per_symbol: Cap on history depth per symbol.

        Returns:
            A :class:`ProcessedDataset` with warm-up rows and the un-simulatable
            tail removed.
        """
        universe: list[str] = list(symbols) if symbols else list(self._settings.data.symbols)
        depth: int = max_candles_per_symbol or self._settings.data.history_bootstrap_candles

        frames: list[pd.DataFrame] = []
        for symbol in universe:
            frame: pd.DataFrame | None = await self._build_labeled_symbol(symbol, depth)
            if frame is not None and not frame.empty:
                frames.append(frame)

        if not frames:
            _LOGGER.warning("No symbol produced a usable training frame")
            return self._empty_dataset()

        pooled: pd.DataFrame = pd.concat(frames, axis=0, ignore_index=True)
        pooled = pooled.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        dataset: ProcessedDataset = self._to_dataset(pooled, tuple(universe))
        _LOGGER.info(
            "Training dataset ready: %d rows, %d features, distribution=%s",
            len(dataset),
            len(dataset.feature_columns),
            dataset.class_distribution(),
        )
        return dataset

    async def _build_labeled_symbol(self, symbol: str, depth: int) -> pd.DataFrame | None:
        """Build the feature+label frame for one symbol, or ``None`` on failure."""
        try:
            ohlcv, futures, book = await self._load_symbol_inputs(symbol, depth)
            if ohlcv.empty:
                return None

            featured: pd.DataFrame = await self._features.build(ohlcv, futures, book)
            labeled: pd.DataFrame = await asyncio.to_thread(self._labeler.generate, featured)
            labeled["symbol"] = symbol
            return labeled
        except (InsufficientDataError, FeatureEngineeringError) as error:
            _LOGGER.warning("Skipping %s while building the training set: %s", symbol, error)
            return None
        except Exception as error:  # pragma: no cover - defensive per-symbol isolation
            _LOGGER.error("Unexpected failure building %s: %s", symbol, error, exc_info=True)
            return None

    def _to_dataset(self, pooled: pd.DataFrame, symbols: tuple[str, ...]) -> ProcessedDataset:
        """Clean the pooled frame and split it into per-model targets."""
        total_candidate_rows: int = len(pooled)
        usable: pd.DataFrame = pooled[pooled["label_is_valid"].fillna(False)].copy()
        rejected_invalid_label_rows: int = total_candidate_rows - len(usable)

        feature_columns: list[str] = list(FEATURE_COLUMNS)
        usable = usable.replace([np.inf, -np.inf], np.nan)

        # Only the *targets* justify destroying a row.  A NaN in any one of the
        # ~56 feature columns used to take the whole sample with it, and because
        # the sparsest features (flat-market volatility ratios, GARCH warm-up,
        # archive coverage) cluster in low-liquidity symbols and quiet periods,
        # that removed 73% of the intended training window while leaving
        # validation and test nearly intact - a systematic sample-selection bias
        # logged as harmless warm-up trim.  LightGBM and XGBoost both route NaN
        # down a learned default branch, so a sparse feature costs only its own
        # information, not the entire observation.
        before: int = len(usable)
        usable = usable.dropna(subset=["label", "target_risk_score"])
        dropped: int = before - len(usable)
        if dropped:
            _LOGGER.info("Dropped %d rows with an unusable label/target", dropped)

        if usable.empty:
            return self._empty_dataset()

        usable = usable.reset_index(drop=True)
        duplicate_feature_rows: int = int(usable.duplicated(subset=feature_columns).sum())
        null_counts, null_rates = self._null_diagnostics(usable, feature_columns)

        metadata_columns: list[str] = [
            column for column in ("symbol", *_META_COLUMNS) if column in usable.columns
        ]
        exit_columns: list[str] = ["target_tp_pct", "target_sl_pct", "target_trailing_pct"]
        exit_targets: pd.DataFrame = usable[exit_columns].astype(float)

        return ProcessedDataset(
            features=usable[feature_columns].astype(float),
            direction_target=usable["label"].astype(str),
            entry_target=usable["entry_quality"].astype(int),
            exit_targets=exit_targets,
            risk_target=usable["target_risk_score"].astype(float),
            metadata=usable[metadata_columns],
            symbols=symbols,
            feature_columns=tuple(feature_columns),
            total_candidate_rows=total_candidate_rows,
            rejected_invalid_label_rows=rejected_invalid_label_rows,
            dropped_missing_or_inf_rows=dropped,
            duplicate_feature_rows=duplicate_feature_rows,
            null_counts_by_feature=null_counts,
            null_rate_by_symbol_month=null_rates,
        )

    @staticmethod
    def _null_diagnostics(
        usable: pd.DataFrame,
        feature_columns: list[str],
        top_n: int = 5,
    ) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
        """Per-feature null counts, plus a symbol x month map for the worst ones.

        Rows survive NaN features now, so without this the sparsity that used to
        announce itself as a catastrophic row drop would instead be completely
        silent.  The per-symbol/per-month breakdown is what distinguishes
        "this feature is uniformly 3% sparse" from "this feature is 90% empty
        for eight symbols before March", which are very different problems.
        """
        null_counts: dict[str, int] = {
            column: int(usable[column].isna().sum())
            for column in feature_columns
            if column in usable.columns
        }

        sparsest: list[str] = [
            column
            for column, count in sorted(null_counts.items(), key=lambda item: -item[1])
            if count > 0
        ][:top_n]
        if not sparsest or "symbol" not in usable.columns or "timestamp" not in usable.columns:
            return null_counts, {}

        month: pd.Series = pd.to_datetime(
            usable["timestamp"], unit="ms", utc=True
        ).dt.strftime("%Y-%m")
        any_null: pd.Series = usable[sparsest].isna().any(axis=1)
        grouped: pd.Series = any_null.groupby([usable["symbol"], month]).mean()

        null_rates: dict[str, dict[str, float]] = {}
        for (symbol, period), rate in grouped.items():
            null_rates.setdefault(str(symbol), {})[str(period)] = round(float(rate), 4)
        return null_counts, null_rates

    @staticmethod
    def _empty_dataset() -> ProcessedDataset:
        """An empty but structurally valid dataset."""
        return ProcessedDataset(
            features=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
            direction_target=pd.Series(dtype=str),
            entry_target=pd.Series(dtype=int),
            exit_targets=pd.DataFrame(
                columns=["target_tp_pct", "target_sl_pct", "target_trailing_pct"]
            ),
            risk_target=pd.Series(dtype=float),
            metadata=pd.DataFrame(),
        )

    # ------------------------------------------------------------------
    # Live-inference path
    # ------------------------------------------------------------------
    async def build_inference_payload(
        self,
        symbol: str,
        lookback_candles: int | None = None,
    ) -> InferencePayload | None:
        """Produce the feature row for the most recent closed candle.

        No labeling is performed here - the labeler is a training-only tool and
        is never invoked on the live path.

        Returns:
            An :class:`InferencePayload`, or ``None`` when history is too shallow
            or every feature row is still warming up.
        """
        depth: int = lookback_candles or self._inference_depth()
        try:
            ohlcv, futures, book = await self._load_symbol_inputs(symbol, depth)
            if ohlcv.empty:
                _LOGGER.warning("No stored candles for %s - cannot infer", symbol)
                return None

            featured: pd.DataFrame = await self._features.build(ohlcv, futures, book)
        except (InsufficientDataError, FeatureEngineeringError) as error:
            _LOGGER.warning("Inference features unavailable for %s: %s", symbol, error)
            return None

        feature_columns: list[str] = list(FEATURE_COLUMNS)
        # Only the *required* block gates tradeability.  The optional
        # micro-structure columns are allowed through as NaN so a momentary gap
        # in book coverage does not silently stop the system from trading -
        # which is also exactly how those columns were encoded during training,
        # so the booster sees the same thing either way.
        candidates: pd.DataFrame = featured.replace([np.inf, -np.inf], np.nan).dropna(
            subset=list(REQUIRED_FEATURE_COLUMNS)
        )
        if candidates.empty:
            _LOGGER.warning("All feature rows for %s are still warming up", symbol)
            return None

        last_row: pd.Series = candidates.iloc[-1]
        latest_timestamp: int = int(featured["timestamp"].iloc[-1])
        row_timestamp: int = int(last_row["timestamp"])
        if row_timestamp != latest_timestamp:
            _LOGGER.warning(
                "Newest complete feature row for %s is %d bar(s) stale - skipping",
                symbol,
                (latest_timestamp - row_timestamp) // self._settings.data.timeframe_ms,
            )
            return None

        features: pd.DataFrame = candidates.iloc[[-1]][feature_columns].astype(float)
        return InferencePayload(
            symbol=symbol,
            timestamp=row_timestamp,
            close=float(last_row["close"]),
            features=features,
            snapshot=self._audit_snapshot(last_row),
        )

    async def build_inference_payloads(
        self,
        symbols: Sequence[str],
        lookback_candles: int | None = None,
    ) -> dict[str, InferencePayload]:
        """Build inference payloads for many symbols concurrently."""
        results: list[InferencePayload | None] = await asyncio.gather(
            *(self.build_inference_payload(symbol, lookback_candles) for symbol in symbols)
        )
        return {payload.symbol: payload for payload in results if payload is not None}

    def _inference_depth(self) -> int:
        """History depth needed for the slowest feature to be warm at the tail."""
        config = self._settings.features
        return (
            max(config.garch_window, config.hmm_window, config.rank_window)
            + config.hmm_refit_every
            + 120
        )

    @staticmethod
    def _audit_snapshot(row: pd.Series) -> dict[str, float]:
        """Extract the critical feature values the Audit Engine records verbatim."""
        keys: tuple[str, ...] = (
            "hmm_regime",
            "garch_volatility",
            "garch_vol_rank",
            "kama_slope",
            "kama_distance",
            "fdi",
            "atr",
            "atr_pct",
            "adx",
            "rsi",
            "funding_rate",
            "volume_zscore",
            "close",
        )
        snapshot: dict[str, float] = {}
        for key in keys:
            if key in row.index:
                value: Any = row[key]
                snapshot[key] = float(value) if pd.notna(value) else 0.0
        return snapshot

    # ------------------------------------------------------------------
    # Shared loading
    # ------------------------------------------------------------------
    async def _load_symbol_inputs(
        self,
        symbol: str,
        depth: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Load OHLCV, futures metrics and order-book history for one symbol.

        The OHLCV window is trimmed to its longest clean trailing run (see
        :meth:`_trim_to_clean_window`) before anything downstream sees it.

        Raises:
            InsufficientDataError: When the stored OHLCV window fails the
                structural integrity check and nothing usable survives
                trimming to the newest clean run.  A symbol in this state
                must not reach feature engineering or the ML pipeline,
                whether the call originates from training or from live
                inference.
        """
        ohlcv: pd.DataFrame = await self._db.load_ohlcv_dataframe(symbol, limit=depth)
        if ohlcv.empty:
            return ohlcv, None, None

        ohlcv = self._trim_to_clean_window(symbol, ohlcv)

        futures: pd.DataFrame = await self._db.load_futures_metrics_frame(symbol, limit=depth)
        book: pd.DataFrame = await self._load_order_book_frame(symbol, depth)
        return (
            ohlcv,
            futures if not futures.empty else None,
            book if not book.empty else None,
        )

    def _trim_to_clean_window(self, symbol: str, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Validate a stored OHLCV window and trim it to its clean trailing run.

        A gap or corrupt row anywhere in a symbol's stored history must never
        silently reach feature engineering - but a symbol should not lose its
        *entire* history over one old, already-superseded defect either (this
        is exactly what a symbol whose data predates a validator improvement,
        or was written by a less careful earlier ingestion run, looks like).
        The same "keep the newest clean run" rule Module A applies when
        quarantining an unhealable window at ingestion time
        (:meth:`module_a_data.qc_validator.QCValidator._longest_clean_trailing_run`)
        is applied here on read, via the shared
        :func:`core.utils.longest_clean_trailing_run`, so a symbol only loses
        the pipeline entirely when nothing usable survives trimming.
        """
        issues: list[QCIssue] = self._validator.validate_stored_frame(symbol, ohlcv)
        critical: list[QCIssue] = [
            issue for issue in issues if issue.severity is QCSeverity.CRITICAL
        ]
        if not critical:
            return ohlcv

        bad: set[int] = set()
        for issue in critical:
            bad.update(issue.timestamps)

        timestamps: list[int] = ohlcv["timestamp"].astype("int64").tolist()
        kept: set[int] = set(
            longest_clean_trailing_run(timestamps, bad, self._settings.data.timeframe_ms)
        )
        trimmed: pd.DataFrame = (
            ohlcv[ohlcv["timestamp"].isin(kept)].sort_values("timestamp").reset_index(drop=True)
        )

        codes: tuple[str, ...] = tuple(sorted({issue.code.value for issue in critical}))
        if trimmed.empty:
            raise InsufficientDataError(
                "stored candle window failed integrity validation and nothing "
                "clean survived trimming",
                symbol=symbol,
                codes=codes,
            )

        # Defensive re-check: guarantees the returned frame is truly clean
        # rather than trusting the trim logic blindly.
        residual: list[QCIssue] = self._validator.validate_stored_frame(symbol, trimmed)
        if any(issue.severity is QCSeverity.CRITICAL for issue in residual):
            raise InsufficientDataError(
                "stored candle window failed integrity validation even after "
                "trimming to the clean trailing run",
                symbol=symbol,
                codes=codes,
            )

        dropped: int = len(ohlcv) - len(trimmed)
        if dropped:
            _LOGGER.warning(
                "%s: %d stored candle(s) failed integrity validation (%s) - "
                "using the clean trailing run of %d candle(s) from %d to %d",
                symbol,
                dropped,
                ", ".join(codes),
                len(trimmed),
                int(trimmed["timestamp"].iloc[0]),
                int(trimmed["timestamp"].iloc[-1]),
            )
        return trimmed

    async def _load_order_book_frame(self, symbol: str, depth: int) -> pd.DataFrame:
        """Load the 5m micro-structure buckets backing the order-book features.

        Reads ``market_microstructure`` - the archive-backfilled, candle-grid
        aligned table - rather than ``order_book_snapshots``, which holds
        irregular live snapshots taken whenever a cycle happened to fire.  The
        feature layer joins these on the exact bucket key, so they must sit on
        the candle grid; an irregular snapshot stream cannot.
        """
        return await self._db.load_microstructure_frame(symbol, limit=depth)


#: Re-exported so consumers do not need to import the labeler directly.
DIRECTION_CLASSES: Final[tuple[str, ...]] = LABEL_ORDER
