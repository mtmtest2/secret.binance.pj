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
from module_a_data.db_handler import DatabaseHandler
from module_b_features.features import FEATURE_COLUMNS, FeatureService
from module_b_features.labeler import LABEL_ORDER, TradeLabeler

_LOGGER = get_logger(__name__)

#: Columns carried alongside the features for bookkeeping / backtesting.
_META_COLUMNS: Final[tuple[str, ...]] = ("timestamp", "open", "high", "low", "close", "volume")


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

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def timestamps(self) -> np.ndarray:
        """Candle open times (ms) aligned with the feature rows."""
        if "timestamp" in self.metadata.columns and len(self.metadata) == len(self.features):
            return self.metadata["timestamp"].to_numpy(dtype=np.int64)
        # No metadata (empty dataset): fall back to positional pseudo-time so the
        # split helpers stay total functions.
        return np.arange(len(self.features), dtype=np.int64)

    def chronological_split(
        self,
        validation_fraction: float,
        test_fraction: float,
        purge_bars: int,
        embargo_bars: int,
        timeframe_ms: int,
    ) -> "DatasetSplit":
        """Split train / validation / test in time, with a purged embargo gap.

        Two properties matter here and neither is optional:

        * **The cut points are timestamps, not row positions.**  The dataset is
          pooled across ~30 symbols, so one bar of history is ~30 rows.  Purging
          *rows* would have removed two bars where 120 were needed, and the
          forward-looking labels of the last training bars would still overlap
          the first validation bars.  Purging a *time window* removes the overlap
          for every symbol simultaneously.
        * **The gap is applied on both sides of every boundary**: ``purge_bars``
          covers the label horizon and ``embargo_bars`` covers the slow rolling
          features, so no row of one block can share information with the next.

        The test block is the chronological tail and is returned separately so it
        can be excluded everywhere except final evaluation.
        """
        rows: int = len(self.features)
        empty: np.ndarray = np.array([], dtype=np.int64)
        if rows == 0:
            return DatasetSplit(train=empty, validation=empty, test=empty)

        stamps: np.ndarray = self.timestamps()
        order: np.ndarray = np.argsort(stamps, kind="stable")
        sorted_stamps: np.ndarray = stamps[order]

        test_share: float = max(0.0, min(0.9, test_fraction))
        validation_share: float = max(0.0, min(0.9, validation_fraction))

        test_start_ts: int = int(sorted_stamps[min(rows - 1, int(rows * (1.0 - test_share)))])
        validation_start_ts: int = int(
            sorted_stamps[min(rows - 1, int(rows * (1.0 - test_share - validation_share)))]
        )
        gap_ms: int = max(0, purge_bars + embargo_bars) * max(1, timeframe_ms)

        positions: np.ndarray = np.arange(rows, dtype=np.int64)
        test_mask: np.ndarray = stamps >= test_start_ts if test_share > 0.0 else np.zeros(rows, bool)
        validation_mask: np.ndarray = (
            (stamps >= validation_start_ts)
            & (stamps < test_start_ts - gap_ms)
            if validation_share > 0.0
            else np.zeros(rows, bool)
        )
        train_mask: np.ndarray = stamps < validation_start_ts - gap_ms

        return DatasetSplit(
            train=positions[train_mask],
            validation=positions[validation_mask],
            test=positions[test_mask],
        )

    def train_validation_split(
        self,
        validation_fraction: float,
        purge_bars: int,
        test_fraction: float = 0.0,
        embargo_bars: int = 0,
        timeframe_ms: int = 300_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train/validation index positions, with the test tail already removed.

        Every head fits through this helper, so passing a non-zero
        ``test_fraction`` is what structurally guarantees that no model - not
        even by accident - is fitted or early-stopped on the held-out block.
        """
        split: DatasetSplit = self.chronological_split(
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            timeframe_ms=timeframe_ms,
        )
        return split.train, split.validation

    def subset(self, positions: np.ndarray) -> "ProcessedDataset":
        """Return a row-subset of this dataset, preserving every target."""
        index: np.ndarray = np.asarray(positions, dtype=np.int64)
        return ProcessedDataset(
            features=self.features.iloc[index].reset_index(drop=True),
            direction_target=self.direction_target.iloc[index].reset_index(drop=True),
            entry_target=self.entry_target.iloc[index].reset_index(drop=True),
            exit_targets=self.exit_targets.iloc[index].reset_index(drop=True),
            risk_target=self.risk_target.iloc[index].reset_index(drop=True),
            metadata=self.metadata.iloc[index].reset_index(drop=True),
            symbols=self.symbols,
            feature_columns=self.feature_columns,
        )


@dataclass(slots=True)
class DatasetSplit:
    """Row positions of the three chronological blocks.

    ``test`` is a *contract*: nothing outside final evaluation may read it.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def describe(self, stamps: np.ndarray) -> dict[str, Any]:
        """Row counts and time bounds per block, for the training report."""

        def _block(name: str, positions: np.ndarray) -> dict[str, Any]:
            if positions.size == 0:
                return {"block": name, "rows": 0, "start": None, "end": None}
            values: np.ndarray = stamps[positions]
            return {
                "block": name,
                "rows": int(positions.size),
                "start": int(values.min()),
                "end": int(values.max()),
            }

        return {
            "train": _block("train", self.train),
            "validation": _block("validation", self.validation),
            "test": _block("test", self.test),
        }


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
    ) -> None:
        self._settings: Settings = settings
        self._db: DatabaseHandler = database
        self._features: FeatureService = feature_service or FeatureService(settings)
        self._labeler: TradeLabeler = labeler or TradeLabeler(settings)

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
            ohlcv, futures, book, flow = await self._load_symbol_inputs(symbol, depth)
            if ohlcv.empty:
                return None

            featured: pd.DataFrame = await self._features.build(ohlcv, futures, book, flow)
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
        usable: pd.DataFrame = pooled[pooled["label_is_valid"].fillna(False)].copy()

        feature_columns: list[str] = list(FEATURE_COLUMNS)
        usable = usable.replace([np.inf, -np.inf], np.nan)

        before: int = len(usable)
        usable = usable.dropna(subset=feature_columns + ["label", "target_risk_score"])
        dropped: int = before - len(usable)
        if dropped:
            _LOGGER.info("Dropped %d rows with NaN features/labels (rolling-window warm-up)", dropped)

        if usable.empty:
            return self._empty_dataset()

        usable = usable.reset_index(drop=True)

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
        )

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
            ohlcv, futures, book, flow = await self._load_symbol_inputs(symbol, depth)
            if ohlcv.empty:
                _LOGGER.warning("No stored candles for %s - cannot infer", symbol)
                return None

            featured: pd.DataFrame = await self._features.build(ohlcv, futures, book, flow)
        except (InsufficientDataError, FeatureEngineeringError) as error:
            _LOGGER.warning("Inference features unavailable for %s: %s", symbol, error)
            return None

        feature_columns: list[str] = list(FEATURE_COLUMNS)
        candidates: pd.DataFrame = featured.replace([np.inf, -np.inf], np.nan).dropna(
            subset=feature_columns
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
            "ob_imbalance",
            "ob_spread_bps",
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
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        """Load OHLCV, futures metrics, order-book and order-flow history."""
        ohlcv: pd.DataFrame = await self._db.load_ohlcv_dataframe(symbol, limit=depth)
        if ohlcv.empty:
            return ohlcv, None, None, None

        futures: pd.DataFrame = await self._db.load_futures_metrics_frame(symbol, limit=depth)
        book: pd.DataFrame = await self._load_order_book_frame(symbol, depth)
        flow: pd.DataFrame = await self._db.load_agg_trade_flow_frame(symbol, limit=depth)
        return (
            ohlcv,
            futures if not futures.empty else None,
            book if not book.empty else None,
            flow if not flow.empty else None,
        )

    async def _load_order_book_frame(self, symbol: str, depth: int) -> pd.DataFrame:
        """Load recent order-book snapshots into a timestamp-keyed frame."""
        from sqlalchemy import desc, select  # local import keeps the ORM out of the hot path

        from module_a_data.db_models import OrderBookRow

        query = (
            select(
                OrderBookRow.timestamp,
                OrderBookRow.spread_bps,
                OrderBookRow.imbalance,
                OrderBookRow.microprice,
            )
            .where(OrderBookRow.symbol == symbol)
            .order_by(desc(OrderBookRow.timestamp))
            .limit(depth)
        )
        factory = self._db._factory()  # noqa: SLF001 - intentional internal reuse
        async with factory() as session:
            result = await session.execute(query)
            rows = result.all()

        columns: list[str] = ["timestamp", "spread_bps", "imbalance", "microprice"]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame: pd.DataFrame = pd.DataFrame(rows, columns=columns)
        return frame.sort_values("timestamp").reset_index(drop=True)


#: Re-exported so consumers do not need to import the labeler directly.
DIRECTION_CLASSES: Final[tuple[str, ...]] = LABEL_ORDER
