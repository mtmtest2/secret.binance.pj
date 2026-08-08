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
from core.exceptions import DataFetchError, FeatureEngineeringError, InsufficientDataError
from core.logger import get_logger
from module_a_data.db_handler import DatabaseHandler
from module_a_data.fetcher import BinanceDataFetcher
from module_a_data.models import OHLCVCandle
from module_b_features.features import FEATURE_COLUMNS, FeatureService, MarketContext
from module_b_features.labeler import LABEL_ORDER, IntrabarLookup, SubCandle, TradeLabeler

_LOGGER = get_logger(__name__)

#: Columns carried alongside the features for bookkeeping / backtesting.
_META_COLUMNS: Final[tuple[str, ...]] = ("timestamp", "open", "high", "low", "close", "volume")

#: Minimum training rows a purged split must leave; below this a subsampled
#: booster's effective sample size can round to zero (see
#: ``ProcessedDataset.train_validation_split``).
_MIN_TRAIN_ROWS: Final[int] = 50

#: Timeframe fetched for the labeler's intra-candle barrier-order refinement.
_INTRABAR_TIMEFRAME: Final[str] = "1m"
_INTRABAR_TIMEFRAME_MS: Final[int] = 60_000
#: Width of one 5m candle being resolved, in milliseconds.
_LABEL_CANDLE_MS: Final[int] = 5 * 60_000
#: Ambiguous 5m candles closer together than this are fetched as a single
#: contiguous 1-minute range instead of one request each - ambiguity tends to
#: cluster in volatile stretches, so this keeps the request count low without
#: pulling in large spans of 1-minute data nobody needs.
_INTRABAR_MERGE_GAP_MS: Final[int] = 2 * 60 * 60 * 1_000


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
    #: Per-row training weight combining label-overlap uniqueness and outcome
    #: decisiveness (see ``TradeLabeler._sample_weights``); mean 1.0 over the
    #: dataset. Passed to every head's ``.fit(sample_weight=...)``.
    sample_weight: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

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

    def train_validation_split(
        self,
        validation_fraction: float,
        purge_bars: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return time-ordered train/validation index positions with a purge gap.

        Financial labels are forward-looking, so the last ``purge_bars`` rows of
        the training block overlap the validation block's label horizon.  Those
        rows are dropped entirely - without the purge, validation scores are
        optimistically biased by construction.

        Raises:
            ValueError: When ``purge_bars`` consumes so much of a small
                dataset that fewer than :data:`_MIN_TRAIN_ROWS` training rows
                would remain. A 1-row training set does not fail loudly on
                its own - a subsampled booster can silently round its
                effective sample size to zero and crash deep inside the
                native library - so this is caught and reported explicitly
                instead (a real scenario for the walk-forward evaluation
                harness's early, still-short folds; the single full-dataset
                training path has never had few enough rows to hit it).
        """
        rows: int = len(self.features)
        if rows == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        split_point: int = int(rows * (1.0 - validation_fraction))
        split_point = max(1, min(rows - 1, split_point))
        train_end: int = max(1, split_point - purge_bars)
        if train_end < _MIN_TRAIN_ROWS:
            raise ValueError(
                f"purge_bars ({purge_bars}) leaves only {train_end} training row(s) "
                f"out of {rows} - need at least {_MIN_TRAIN_ROWS}"
            )

        train_index: np.ndarray = np.arange(0, train_end, dtype=np.int64)
        validation_index: np.ndarray = np.arange(split_point, rows, dtype=np.int64)
        return train_index, validation_index


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
        fetcher: BinanceDataFetcher | None = None,
    ) -> None:
        self._settings: Settings = settings
        self._db: DatabaseHandler = database
        self._features: FeatureService = feature_service or FeatureService(settings)
        self._labeler: TradeLabeler = labeler or TradeLabeler(settings)
        #: Optional: without it, ambiguous same-bar TP/SL touches simply keep
        #: the labeler's conservative stop-first fallback (see `_build_labeled_symbol`).
        self._fetcher: BinanceDataFetcher | None = fetcher

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
        end_ms: int | None = None,
    ) -> ProcessedDataset:
        """Assemble a pooled, cross-sectional training dataset.

        All symbols are concatenated into one dataset and sorted by timestamp, so
        the chronological train/validation split is a genuine out-of-time split
        rather than a per-symbol one.

        Args:
            symbols: Universe to include (defaults to the configured universe).
            max_candles_per_symbol: Cap on history depth per symbol.
            end_ms: Optional hard cutoff - only candles at or before this
                timestamp are used. Used by the walk-forward evaluation
                harness (``module_e_execution.walk_forward_eval``) to train
                each fold strictly on data from before its own validation
                window, rather than the trailing "most recent N candles"
                depth window used everywhere else.

        Returns:
            A :class:`ProcessedDataset` with warm-up rows and the un-simulatable
            tail removed.
        """
        universe: list[str] = list(symbols) if symbols else list(self._settings.data.symbols)
        depth: int = max_candles_per_symbol or self._settings.data.history_bootstrap_candles

        #: Built once for the whole universe - BTC/ETH reference series and the
        #: cross-sectional return distribution are shared inputs, not something
        #: worth recomputing (or re-fetching) once per symbol.
        market_context: MarketContext = await self._build_market_context(universe, depth, end_ms)

        frames: list[pd.DataFrame] = []
        for symbol in universe:
            frame: pd.DataFrame | None = await self._build_labeled_symbol(
                symbol, depth, market_context, end_ms
            )
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

    async def _build_labeled_symbol(
        self,
        symbol: str,
        depth: int,
        market_context: MarketContext | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame | None:
        """Build the feature+label frame for one symbol, or ``None`` on failure."""
        try:
            ohlcv, futures, book = await self._load_symbol_inputs(symbol, depth, end_ms)
            if ohlcv.empty:
                return None

            featured: pd.DataFrame = await self._features.build(
                ohlcv, futures, book, market_context=market_context, symbol=symbol
            )
            labeled: pd.DataFrame = await asyncio.to_thread(self._labeler.generate, featured)

            ambiguous_ts: list[int] = list(labeled.attrs.get("ambiguous_candle_timestamps", []))
            if ambiguous_ts and self._fetcher is not None:
                intrabar: IntrabarLookup = await self._fetch_intrabar_candles(symbol, ambiguous_ts)
                if intrabar:
                    labeled = await asyncio.to_thread(self._labeler.generate, featured, intrabar)

            labeled["symbol"] = symbol
            return labeled
        except (InsufficientDataError, FeatureEngineeringError) as error:
            _LOGGER.warning("Skipping %s while building the training set: %s", symbol, error)
            return None
        except Exception as error:  # pragma: no cover - defensive per-symbol isolation
            _LOGGER.error("Unexpected failure building %s: %s", symbol, error, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Intra-candle (1-minute) data for the labeler's barrier-order refinement
    # ------------------------------------------------------------------
    async def _fetch_intrabar_candles(
        self,
        symbol: str,
        candle_timestamps: Sequence[int],
    ) -> IntrabarLookup:
        """Read (or fetch) the 1-minute sub-candles for a set of ambiguous 5m bars.

        Nearby candles are merged into contiguous ranges (see
        ``_merge_intrabar_windows``) so this costs a handful of paginated
        requests rather than one per ambiguous candle. Each range is served
        from the local cache when a previous run already stored it; a cache
        miss falls back to a live fetch, which is itself best-effort - a
        failed window is skipped, leaving those candles on the labeler's
        conservative stop-first fallback rather than aborting training.
        """
        if self._fetcher is None or not candle_timestamps:
            return {}

        windows: list[tuple[int, int]] = self._merge_intrabar_windows(
            sorted(set(int(value) for value in candle_timestamps))
        )

        collected: list[OHLCVCandle] = []
        for start_ms, end_ms_exclusive in windows:
            expected_rows: int = (end_ms_exclusive - start_ms) // _INTRABAR_TIMEFRAME_MS
            cached: pd.DataFrame = await self._db.load_ohlcv_dataframe(
                symbol,
                start_ms=start_ms,
                end_ms=end_ms_exclusive - 1,
                timeframe=_INTRABAR_TIMEFRAME,
            )
            if len(cached) >= expected_rows > 0:
                collected.extend(
                    OHLCVCandle(
                        symbol=symbol,
                        timeframe=_INTRABAR_TIMEFRAME,
                        timestamp=int(row.timestamp),
                        open=float(row.open),
                        high=float(row.high),
                        low=float(row.low),
                        close=float(row.close),
                        volume=float(row.volume),
                    )
                    for row in cached.itertuples()
                )
                continue

            try:
                fetched: list[OHLCVCandle] = await self._fetcher.fetch_ohlcv_range(
                    symbol,
                    start_ms=start_ms,
                    end_ms=end_ms_exclusive - 1,
                    timeframe=_INTRABAR_TIMEFRAME,
                )
            except DataFetchError as error:
                _LOGGER.warning(
                    "1-minute intrabar fetch failed for %s [%d, %d): %s",
                    symbol, start_ms, end_ms_exclusive, error,
                )
                continue

            if fetched:
                await self._db.upsert_candles(fetched)
                collected.extend(fetched)

        return self._bucket_by_5m_window(collected, set(candle_timestamps))

    @staticmethod
    def _merge_intrabar_windows(candle_timestamps: list[int]) -> list[tuple[int, int]]:
        """Merge ambiguous 5m candle timestamps into contiguous fetch ranges.

        Returns ``(start_ms, end_ms_exclusive)`` pairs, each covering every
        1-minute sub-candle of the 5m bars it spans.
        """
        if not candle_timestamps:
            return []

        merged: list[tuple[int, int]] = []
        window_start: int = candle_timestamps[0]
        window_end: int = candle_timestamps[0] + _LABEL_CANDLE_MS
        for timestamp in candle_timestamps[1:]:
            if timestamp - window_end > _INTRABAR_MERGE_GAP_MS:
                merged.append((window_start, window_end))
                window_start = timestamp
            window_end = timestamp + _LABEL_CANDLE_MS
        merged.append((window_start, window_end))
        return merged

    @staticmethod
    def _bucket_by_5m_window(
        candles: list[OHLCVCandle],
        wanted: set[int],
    ) -> dict[int, list[SubCandle]]:
        """Group 1-minute candles under the 5m bar they belong to.

        Only buckets actually present in ``wanted`` are returned - a merged
        fetch range can carry extra 1-minute candles that no ambiguous bar
        asked for, and those are dropped here rather than handed to the
        labeler as if they resolved something.
        """
        buckets: dict[int, list[tuple[int, float, float, float, float]]] = {}
        for candle in candles:
            bucket_start: int = (candle.timestamp // _LABEL_CANDLE_MS) * _LABEL_CANDLE_MS
            if bucket_start not in wanted:
                continue
            buckets.setdefault(bucket_start, []).append(
                (candle.timestamp, candle.open, candle.high, candle.low, candle.close)
            )

        result: dict[int, list[SubCandle]] = {}
        for bucket_start, rows in buckets.items():
            rows.sort(key=lambda row: row[0])
            result[bucket_start] = [(o, h, l, c) for _, o, h, l, c in rows]
        return result

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
        sample_weight: pd.Series = (
            usable["sample_weight"].astype(float)
            if "sample_weight" in usable.columns
            else pd.Series(1.0, index=usable.index)
        )

        return ProcessedDataset(
            features=usable[feature_columns].astype(float),
            direction_target=usable["label"].astype(str),
            entry_target=usable["entry_quality"].astype(int),
            exit_targets=exit_targets,
            risk_target=usable["target_risk_score"].astype(float),
            metadata=usable[metadata_columns],
            symbols=symbols,
            feature_columns=tuple(feature_columns),
            sample_weight=sample_weight,
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
            sample_weight=pd.Series(dtype=float),
        )

    # ------------------------------------------------------------------
    # Live-inference path
    # ------------------------------------------------------------------
    async def build_inference_payload(
        self,
        symbol: str,
        lookback_candles: int | None = None,
        market_context: MarketContext | None = None,
    ) -> InferencePayload | None:
        """Produce the feature row for the most recent closed candle.

        No labeling is performed here - the labeler is a training-only tool and
        is never invoked on the live path.

        Args:
            symbol: Symbol to build the payload for.
            lookback_candles: History depth override.
            market_context: Shared BTC/ETH + cross-sectional context. When a
                caller invokes this directly for a single symbol without one,
                a minimal reference-only context is built on the fly (no
                cross-sectional stats - those need the whole batch, see
                :meth:`build_inference_payloads`).

        Returns:
            An :class:`InferencePayload`, or ``None`` when history is too shallow
            or every feature row is still warming up.
        """
        depth: int = lookback_candles or self._inference_depth()
        if market_context is None:
            market_context = await self._build_market_context([symbol], depth)
        try:
            ohlcv, futures, book = await self._load_symbol_inputs(symbol, depth)
            if ohlcv.empty:
                _LOGGER.warning("No stored candles for %s - cannot infer", symbol)
                return None

            featured: pd.DataFrame = await self._features.build(
                ohlcv, futures, book, market_context=market_context, symbol=symbol
            )
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
        """Build inference payloads for many symbols concurrently.

        The BTC/ETH + cross-sectional :class:`MarketContext` is built exactly
        once for the whole batch and shared across every symbol's payload -
        it is universe-wide data, not per-symbol.
        """
        depth: int = lookback_candles or self._inference_depth()
        market_context: MarketContext = await self._build_market_context(list(symbols), depth)
        results: list[InferencePayload | None] = await asyncio.gather(
            *(
                self.build_inference_payload(symbol, lookback_candles, market_context)
                for symbol in symbols
            )
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
    # Market-wide context (BTC/ETH reference + cross-sectional stats)
    # ------------------------------------------------------------------
    async def _build_market_context(
        self, symbols: Sequence[str], depth: int, end_ms: int | None = None
    ) -> MarketContext:
        """Load BTC/ETH reference OHLCV and the universe's cross-sectional returns.

        Built once per training run / per inference cycle and shared across
        every symbol - see :class:`~module_b_features.features.MarketContext`.
        A symbol whose own data is used to build the cross-sectional
        distribution is not excluded from it (a leave-one-out computation
        would be marginally more correct but, at a ~30-symbol universe, the
        difference is negligible next to the implementation risk of getting
        a leave-one-out formula subtly wrong).

        ``end_ms``, when given, restricts every load to data at or before
        that timestamp - see ``build_training_dataset``.
        """
        config = self._settings.features
        reference_ohlcv: dict[str, pd.DataFrame] = {}
        for reference_symbol in config.reference_symbols:
            frame: pd.DataFrame = await self._db.load_ohlcv_dataframe(
                reference_symbol, limit=depth, end_ms=end_ms
            )
            if not frame.empty:
                reference_ohlcv[reference_symbol] = frame.reset_index(drop=True)

        cross_sectional_mean: pd.Series | None = None
        cross_sectional_std: pd.Series | None = None
        cross_sectional_count: pd.Series | None = None

        if len(symbols) >= config.cross_sectional_min_symbols:
            returns_by_symbol: dict[str, pd.Series] = {}
            for symbol in symbols:
                frame = await self._db.load_ohlcv_dataframe(symbol, limit=depth, end_ms=end_ms)
                if frame.empty:
                    continue
                close: pd.Series = frame["close"].astype(float)
                log_return: pd.Series = np.log(close.clip(lower=1e-12)).diff(1)
                returns_by_symbol[symbol] = pd.Series(
                    log_return.to_numpy(), index=frame["timestamp"].to_numpy()
                )

            if len(returns_by_symbol) >= config.cross_sectional_min_symbols:
                wide: pd.DataFrame = pd.DataFrame(returns_by_symbol)
                cross_sectional_mean = wide.mean(axis=1, skipna=True)
                cross_sectional_std = wide.std(axis=1, skipna=True)
                cross_sectional_count = wide.count(axis=1)

        return MarketContext(
            reference_ohlcv=reference_ohlcv,
            cross_sectional_mean_return=cross_sectional_mean,
            cross_sectional_std_return=cross_sectional_std,
            cross_sectional_symbol_count=cross_sectional_count,
        )

    # ------------------------------------------------------------------
    # Shared loading
    # ------------------------------------------------------------------
    async def _load_symbol_inputs(
        self,
        symbol: str,
        depth: int,
        end_ms: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Load OHLCV, futures metrics and order-book history for one symbol.

        ``end_ms``, when given, restricts every feed to the most recent
        ``depth`` rows at or before that timestamp instead of the most recent
        ``depth`` rows overall - see ``build_training_dataset``.
        """
        ohlcv: pd.DataFrame = await self._db.load_ohlcv_dataframe(symbol, limit=depth, end_ms=end_ms)
        if ohlcv.empty:
            return ohlcv, None, None

        futures: pd.DataFrame = await self._db.load_futures_metrics_frame(
            symbol, limit=depth, end_ms=end_ms
        )
        book: pd.DataFrame = await self._load_order_book_frame(symbol, depth, end_ms)
        return (
            ohlcv,
            futures if not futures.empty else None,
            book if not book.empty else None,
        )

    async def _load_order_book_frame(
        self, symbol: str, depth: int, end_ms: int | None = None
    ) -> pd.DataFrame:
        """Load recent order-book snapshots into a timestamp-keyed frame."""
        from sqlalchemy import desc, select  # local import keeps the ORM out of the hot path

        from module_a_data.db_models import OrderBookRow

        query = select(
            OrderBookRow.timestamp,
            OrderBookRow.spread_bps,
            OrderBookRow.imbalance,
            OrderBookRow.microprice,
        ).where(OrderBookRow.symbol == symbol)
        if end_ms is not None:
            query = query.where(OrderBookRow.timestamp <= end_ms)
        query = query.order_by(desc(OrderBookRow.timestamp)).limit(depth)
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
