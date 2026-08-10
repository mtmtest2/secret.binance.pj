"""Tests for the strict, time-ordered train/validation/test split
(module_b_features/processor.py::compute_split_boundaries/assign_split/
ProcessedDataset.chronological_split).

Covers the leakage-relevant guarantees the ML pipeline's data split depends
on:

* Train is strictly older than validation, which is strictly older than
  test - never a random shuffle.
* The embargo/purge gap is a *time* duration, not a row-count one, so it is
  applied consistently regardless of how many symbols share a timestamp in
  the pooled, cross-sectional training dataset (the previous row-count-based
  purge only removed a fraction of a bar's worth of real time once more than
  one symbol was present - see module_c_ml/ml_models.py's old
  ``train_validation_split`` usage).
* No row within the embargo gap of a boundary survives into either side.
* A dataset shorter than the configured train+validation+test window scales
  all three down proportionally (2:1:1) instead of starving validation/test
  or raising.
* The test split is always anchored to the newest data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module_b_features.features import FEATURE_COLUMNS
from module_b_features.processor import (
    ProcessedDataset,
    assign_split,
    compute_split_boundaries,
)

_TIMEFRAME_MS = 5 * 60 * 1_000
_DAY_MS = 86_400_000


def _dataset_with_timestamps(timestamps: np.ndarray, symbols: np.ndarray | None = None) -> ProcessedDataset:
    n = timestamps.size
    features = pd.DataFrame(
        np.random.default_rng(0).normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS)
    )
    metadata = pd.DataFrame(
        {"timestamp": timestamps, "symbol": symbols if symbols is not None else ["BTC/USDT:USDT"] * n}
    )
    return ProcessedDataset(
        features=features,
        direction_target=pd.Series(["NO_TRADE_OR_FAIL"] * n),
        entry_target=pd.Series([0] * n),
        exit_targets=pd.DataFrame(
            {"target_tp_pct": [0.0] * n, "target_sl_pct": [0.0] * n, "target_trailing_pct": [0.0] * n}
        ),
        risk_target=pd.Series([0.0] * n),
        metadata=metadata,
        feature_columns=tuple(FEATURE_COLUMNS),
    )


def _two_year_timestamps() -> np.ndarray:
    """One bar every 5 minutes across 735 days - comfortably >= the 730.5-day
    (24 * 30.4375) nominal 12+6+6 month window, mirroring the production
    ``DataSettings.history_bootstrap_candles`` buffer over the exact
    calendar-month span so this fixture never triggers proportional
    scale-down."""
    total_bars = (735 * _DAY_MS) // _TIMEFRAME_MS
    return np.arange(total_bars, dtype=np.int64) * _TIMEFRAME_MS


class TestBoundariesOnAFullTwoYearDataset:
    def test_nominal_12_6_6_split_lands_at_the_right_calendar_offsets(self) -> None:
        timestamps = _two_year_timestamps()
        boundaries = compute_split_boundaries(
            int(timestamps.min()),
            int(timestamps.max()),
            train_months=12.0,
            validation_months=6.0,
            test_months=6.0,
            purge_bars=60,
            timeframe_ms=_TIMEFRAME_MS,
        )
        assert boundaries.scaled_down is False

        data_end = int(timestamps.max())
        expected_test_start = data_end - int(round(6.0 * 30.4375 * _DAY_MS))
        expected_validation_start = expected_test_start - int(round(6.0 * 30.4375 * _DAY_MS))

        # Within one bar of the exact calendar arithmetic.
        assert abs(boundaries.test_start_ms - expected_test_start) <= _TIMEFRAME_MS
        assert abs(boundaries.validation_start_ms - expected_validation_start) <= _TIMEFRAME_MS

    def test_split_is_strictly_chronological_train_then_validation_then_test(self) -> None:
        timestamps = _two_year_timestamps()
        dataset = _dataset_with_timestamps(timestamps)
        boundaries = dataset.split_boundaries(
            train_months=12.0, validation_months=6.0, test_months=6.0, purge_bars=60, timeframe_ms=_TIMEFRAME_MS
        )
        split = dataset.chronological_split(boundaries)

        assert len(split.train_index) > 0
        assert len(split.validation_index) > 0
        assert len(split.test_index) > 0

        assert split.train_end_ms < split.validation_start_ms
        assert split.validation_end_ms < split.test_start_ms
        # No index appears in more than one block - a true partition.
        train_set = set(split.train_index.tolist())
        val_set = set(split.validation_index.tolist())
        test_set = set(split.test_index.tolist())
        assert train_set.isdisjoint(val_set)
        assert val_set.isdisjoint(test_set)
        assert train_set.isdisjoint(test_set)

    def test_test_split_is_always_the_most_recent_data(self) -> None:
        timestamps = _two_year_timestamps()
        dataset = _dataset_with_timestamps(timestamps)
        boundaries = dataset.split_boundaries(
            train_months=12.0, validation_months=6.0, test_months=6.0, purge_bars=60, timeframe_ms=_TIMEFRAME_MS
        )
        split = dataset.chronological_split(boundaries)

        assert split.test_end_ms == int(timestamps.max())
        assert split.test_start_ms > split.validation_start_ms
        assert split.test_start_ms > split.train_end_ms

    def test_never_shuffles_rows_index_order_matches_time_order(self) -> None:
        """Row positions within each block stay in ascending time order -
        confirms no shuffling ever happens anywhere in the split."""
        timestamps = _two_year_timestamps()
        dataset = _dataset_with_timestamps(timestamps)
        boundaries = dataset.split_boundaries(
            train_months=12.0, validation_months=6.0, test_months=6.0, purge_bars=60, timeframe_ms=_TIMEFRAME_MS
        )
        split = dataset.chronological_split(boundaries)

        for index in (split.train_index, split.validation_index, split.test_index):
            assert np.all(np.diff(index) > 0), "row positions must stay ascending (no shuffling)"


class TestEmbargoIsTimeBasedNotRowCountBased:
    def test_embargo_scales_with_symbol_count_pooled_dataset(self) -> None:
        """Regression test: a pooled, multi-symbol dataset interleaves every
        symbol at each timestamp. The embargo must remove `purge_bars` worth
        of *time* regardless of how many symbols share each timestamp - the
        old row-count-based purge removed only `purge_bars / n_symbols`
        worth of real time once more than one symbol was present, which
        under-purged the label horizon and leaked validation-period
        information into training labels.
        """
        n_symbols = 30
        purge_bars = 60
        timestamps = _two_year_timestamps()
        # Pooled layout: every symbol repeats at every timestamp, exactly as
        # DatasetProcessor.build_training_dataset's pooled/sorted frame does.
        pooled_timestamps = np.repeat(timestamps, n_symbols)
        symbols = np.tile([f"SYM{i}/USDT:USDT" for i in range(n_symbols)], timestamps.size)
        dataset = _dataset_with_timestamps(pooled_timestamps, symbols)

        boundaries = dataset.split_boundaries(
            train_months=12.0,
            validation_months=6.0,
            test_months=6.0,
            purge_bars=purge_bars,
            timeframe_ms=_TIMEFRAME_MS,
        )
        split = dataset.chronological_split(boundaries)

        expected_embargo_ms = purge_bars * _TIMEFRAME_MS
        assert boundaries.embargo_ms == expected_embargo_ms

        # No train row within the embargo window of validation_start.
        train_timestamps = pooled_timestamps[split.train_index]
        assert train_timestamps.max() <= boundaries.validation_start_ms - expected_embargo_ms

        # No validation row within the embargo window of test_start.
        validation_timestamps = pooled_timestamps[split.validation_index]
        assert validation_timestamps.max() <= boundaries.test_start_ms - expected_embargo_ms

        # The embargo gap itself contains rows for every symbol (i.e. it
        # really did remove `purge_bars` bars' worth of *time*, not
        # `purge_bars` rows out of a `purge_bars / n_symbols`-bar gap).
        gap_mask = (
            pooled_timestamps >= boundaries.validation_start_ms - expected_embargo_ms
        ) & (pooled_timestamps < boundaries.validation_start_ms)
        distinct_timestamps_in_gap = np.unique(pooled_timestamps[gap_mask]).size
        assert distinct_timestamps_in_gap == purge_bars

    def test_a_label_horizon_within_the_embargo_never_crosses_the_boundary(self) -> None:
        """The embargo must be at least as wide as the label horizon
        (LabelSettings.max_holding_bars) for the purge to actually prevent a
        forward-looking label from reaching across a split boundary."""
        max_holding_bars = 48
        purge_bars = 60  # production default; must stay >= max_holding_bars
        assert purge_bars >= max_holding_bars

        timestamps = _two_year_timestamps()
        dataset = _dataset_with_timestamps(timestamps)
        boundaries = dataset.split_boundaries(
            train_months=12.0,
            validation_months=6.0,
            test_months=6.0,
            purge_bars=purge_bars,
            timeframe_ms=_TIMEFRAME_MS,
        )
        split = dataset.chronological_split(boundaries)

        # The last train row's label horizon (up to max_holding_bars candles
        # forward) must resolve strictly before validation begins.
        last_train_ts = int(timestamps[split.train_index].max())
        label_horizon_end = last_train_ts + max_holding_bars * _TIMEFRAME_MS
        assert label_horizon_end < boundaries.validation_start_ms

        last_validation_ts = int(timestamps[split.validation_index].max())
        label_horizon_end_val = last_validation_ts + max_holding_bars * _TIMEFRAME_MS
        assert label_horizon_end_val < boundaries.test_start_ms


class TestShortHistoryFallsBackToProportionalScaling:
    def test_short_dataset_scales_all_three_blocks_down_preserving_ratio(self) -> None:
        # Only 20 days of history - far short of the nominal 24-month window.
        total_bars = (20 * _DAY_MS) // _TIMEFRAME_MS
        timestamps = np.arange(total_bars, dtype=np.int64) * _TIMEFRAME_MS
        dataset = _dataset_with_timestamps(timestamps)

        boundaries = dataset.split_boundaries(
            train_months=12.0, validation_months=6.0, test_months=6.0, purge_bars=10, timeframe_ms=_TIMEFRAME_MS
        )
        split = dataset.chronological_split(boundaries)

        assert boundaries.scaled_down is True
        assert len(split.train_index) > 0
        assert len(split.validation_index) > 0
        assert len(split.test_index) > 0
        # 2:1:1 ratio roughly preserved.
        assert boundaries.train_days == pytest.approx(2 * boundaries.validation_days, rel=0.05)
        assert boundaries.validation_days == pytest.approx(boundaries.test_days, rel=0.05)
        # Still strictly chronological and still partitioned.
        assert split.train_end_ms < split.validation_start_ms
        assert split.validation_end_ms < split.test_start_ms

    def test_empty_dataset_never_raises(self) -> None:
        dataset = _dataset_with_timestamps(np.array([], dtype=np.int64))
        boundaries = dataset.split_boundaries(
            train_months=12.0, validation_months=6.0, test_months=6.0, purge_bars=60, timeframe_ms=_TIMEFRAME_MS
        )
        assert boundaries is None
        split = dataset.chronological_split(boundaries)
        assert len(split.train_index) == 0
        assert len(split.validation_index) == 0
        assert len(split.test_index) == 0


class TestAssignSplitPureFunction:
    def test_assign_split_matches_boundaries_used(self) -> None:
        boundaries = compute_split_boundaries(
            data_start_ms=0,
            data_end_ms=1_000 * _TIMEFRAME_MS,
            train_months=12.0,
            validation_months=6.0,
            test_months=6.0,
            purge_bars=5,
            timeframe_ms=_TIMEFRAME_MS,
        )
        timestamps = np.arange(1_001, dtype=np.int64) * _TIMEFRAME_MS
        train_index, validation_index, test_index = assign_split(timestamps, boundaries)

        assert train_index.size + validation_index.size + test_index.size <= timestamps.size
        assert set(train_index.tolist()).isdisjoint(validation_index.tolist())
        assert set(validation_index.tolist()).isdisjoint(test_index.tolist())
        # test always starts exactly at the nominal boundary (no embargo cut
        # from the start of test - only from the end of train/validation).
        if test_index.size:
            assert int(timestamps[test_index[0]]) == boundaries.test_start_ms
