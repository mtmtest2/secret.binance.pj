"""Point-in-time correctness of the features and integrity of the splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS, ORDER_FLOW_FEATURES, FeatureEngineer
from tests.conftest import make_market, make_pooled_dataset


def test_order_flow_features_are_point_in_time(settings: Settings) -> None:
    """Truncating the future must not change a single past feature value.

    This is the definitive look-ahead test: build the features over the whole
    history, then rebuild them over a prefix.  If any transform peeked forward,
    the overlapping rows would disagree.
    """
    ohlcv, flow = make_market(rows=900)
    cut = 700

    engineer = FeatureEngineer(settings)
    full = engineer.build(ohlcv, agg_trade_flow=flow)
    prefix = engineer.build(
        ohlcv.iloc[:cut].copy(),
        agg_trade_flow=flow.iloc[:cut].copy(),
    )

    for column in ORDER_FLOW_FEATURES:
        left = full[column].iloc[:cut].to_numpy(dtype=float)
        right = prefix[column].to_numpy(dtype=float)
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_future_flow_cannot_leak_backwards(settings: Settings) -> None:
    """Rewriting a future bucket leaves every earlier feature untouched."""
    ohlcv, flow = make_market(rows=600)
    tampered = flow.copy()
    tampered.loc[500:, "buy_volume"] *= 100.0

    engineer = FeatureEngineer(settings)
    original = engineer.build(ohlcv, agg_trade_flow=flow)
    modified = engineer.build(ohlcv, agg_trade_flow=tampered)

    for column in ORDER_FLOW_FEATURES:
        np.testing.assert_allclose(
            original[column].iloc[:500].to_numpy(dtype=float),
            modified[column].iloc[:500].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )


def test_split_blocks_are_disjoint_and_ordered(settings: Settings) -> None:
    """Train precedes validation precedes test, with no overlapping rows."""
    dataset = make_pooled_dataset(settings, symbols=2, rows=1_500)
    split = dataset.chronological_split(
        validation_fraction=settings.ml.validation_fraction,
        test_fraction=settings.ml.test_fraction,
        purge_bars=settings.ml.purge_bars,
        embargo_bars=settings.ml.embargo_bars,
        timeframe_ms=settings.data.timeframe_ms,
    )
    stamps = dataset.timestamps()

    assert split.train.size > 0
    assert split.validation.size > 0
    assert split.test.size > 0

    assert not set(split.train.tolist()) & set(split.validation.tolist())
    assert not set(split.validation.tolist()) & set(split.test.tolist())
    assert not set(split.train.tolist()) & set(split.test.tolist())

    assert stamps[split.train].max() < stamps[split.validation].min()
    assert stamps[split.validation].max() < stamps[split.test].min()


def test_purge_gap_is_measured_in_time_not_rows(settings: Settings) -> None:
    """The gap must cover the label horizon for every pooled symbol at once.

    With N symbols interleaved, one bar of history is N rows; purging rows would
    leave the forward-looking labels of the last training bars overlapping the
    first validation bars.  The gap is therefore asserted in milliseconds.
    """
    dataset = make_pooled_dataset(settings, symbols=3, rows=1_500)
    split = dataset.chronological_split(
        validation_fraction=0.2,
        test_fraction=0.2,
        purge_bars=settings.ml.purge_bars,
        embargo_bars=settings.ml.embargo_bars,
        timeframe_ms=settings.data.timeframe_ms,
    )
    stamps = dataset.timestamps()
    expected_gap = (
        (settings.ml.purge_bars + settings.ml.embargo_bars) * settings.data.timeframe_ms
    )

    train_to_validation = stamps[split.validation].min() - stamps[split.train].max()
    validation_to_test = stamps[split.test].min() - stamps[split.validation].max()
    assert train_to_validation >= expected_gap
    assert validation_to_test >= expected_gap


def test_heads_never_train_on_the_test_block(settings: Settings) -> None:
    """The shared split helper removes the test tail for every head."""
    from module_c_ml.ml_models import DirectionModel

    dataset = make_pooled_dataset(settings, symbols=2, rows=1_500)
    head = DirectionModel(settings)
    train_index, validation_index = head._split(dataset)  # noqa: SLF001 - contract under test

    split = dataset.chronological_split(
        validation_fraction=settings.ml.validation_fraction,
        test_fraction=settings.ml.test_fraction,
        purge_bars=settings.ml.purge_bars,
        embargo_bars=settings.ml.embargo_bars,
        timeframe_ms=settings.data.timeframe_ms,
    )
    test_rows = set(split.test.tolist())
    assert not set(train_index.tolist()) & test_rows
    assert not set(validation_index.tolist()) & test_rows


def test_walk_forward_folds_stay_out_of_the_test_block(settings: Settings) -> None:
    """No fold may borrow a row from the held-out tail."""
    from module_c_ml.evaluation import build_walk_forward_folds

    dataset = make_pooled_dataset(settings, symbols=3, rows=1_800)
    split = dataset.chronological_split(
        validation_fraction=settings.ml.validation_fraction,
        test_fraction=settings.ml.test_fraction,
        purge_bars=settings.ml.purge_bars,
        embargo_bars=settings.ml.embargo_bars,
        timeframe_ms=settings.data.timeframe_ms,
    )
    folds = build_walk_forward_folds(dataset, settings, split)
    assert folds

    test_rows = set(split.test.tolist())
    stamps = dataset.timestamps()
    for fold in folds:
        assert not set(fold.train.tolist()) & test_rows
        assert not set(fold.validation.tolist()) & test_rows
        assert not set(fold.train.tolist()) & set(fold.validation.tolist())
        assert stamps[fold.train].max() < stamps[fold.validation].min()
