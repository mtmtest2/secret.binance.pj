"""The three new 5-minute order-flow features, and the three removed ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import (
    BASE_FEATURE_COLUMNS,
    DIRECTION_FEATURE_COLUMNS,
    ENTRY_FEATURE_COLUMNS,
    EXIT_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    ORDER_FLOW_FEATURES,
    RISK_FEATURE_COLUMNS,
    FeatureEngineer,
)
from tests.conftest import make_market

REMOVED = ("long_short_ratio", "open_interest_change", "taker_buy_sell_ratio")


def test_removed_features_are_absent_from_every_contract() -> None:
    """The three unreliable derivatives metrics are gone, not zero-filled."""
    contracts = (
        FEATURE_COLUMNS,
        BASE_FEATURE_COLUMNS,
        DIRECTION_FEATURE_COLUMNS,
        ENTRY_FEATURE_COLUMNS,
        EXIT_FEATURE_COLUMNS,
        RISK_FEATURE_COLUMNS,
    )
    for contract in contracts:
        for column in REMOVED:
            assert column not in contract


def test_removed_features_are_not_produced_by_the_engineer(settings: Settings) -> None:
    """They must not reappear as stray columns on the frame either."""
    ohlcv, flow = make_market(rows=600)
    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)
    for column in REMOVED:
        assert column not in frame.columns


def test_feature_routing_matches_the_specification() -> None:
    """OFI and delta go to Direction+Entry; relative volume also to Risk."""
    assert "order_flow_imbalance_5m" in DIRECTION_FEATURE_COLUMNS
    assert "order_flow_imbalance_5m" in ENTRY_FEATURE_COLUMNS
    assert "order_flow_imbalance_5m" not in RISK_FEATURE_COLUMNS

    assert "volume_delta_5m" in DIRECTION_FEATURE_COLUMNS
    assert "volume_delta_5m" in ENTRY_FEATURE_COLUMNS
    assert "volume_delta_5m" not in RISK_FEATURE_COLUMNS

    for contract in (DIRECTION_FEATURE_COLUMNS, ENTRY_FEATURE_COLUMNS, RISK_FEATURE_COLUMNS):
        assert "relative_volume_5m" in contract

    # No extra order-flow features crept in, and the count grew by exactly the
    # three new columns minus the three removed ones.
    assert set(ORDER_FLOW_FEATURES) == {
        "order_flow_imbalance_5m",
        "volume_delta_5m",
        "relative_volume_5m",
    }
    assert len(FEATURE_COLUMNS) == len(BASE_FEATURE_COLUMNS) + 3


def test_order_flow_imbalance_is_bounded_and_finite(settings: Settings) -> None:
    """OFI stays in [-1, 1] and never emits NaN or Inf on warm rows."""
    ohlcv, flow = make_market(rows=800)
    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)
    values = frame["order_flow_imbalance_5m"].to_numpy(dtype=float)

    assert np.isfinite(values).all()
    assert values.min() >= -1.0
    assert values.max() <= 1.0


def test_order_flow_imbalance_matches_its_definition(settings: Settings) -> None:
    """The value equals (buy - sell) / (buy + sell) for the bar's own bucket."""
    ohlcv, flow = make_market(rows=400)
    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)

    merged = frame.merge(flow, on="timestamp", how="left", suffixes=("", "_flow"))
    expected = (merged["buy_volume"] - merged["sell_volume"]) / (
        merged["buy_volume"] + merged["sell_volume"]
    )
    np.testing.assert_allclose(
        merged["order_flow_imbalance_5m"].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        merged["volume_delta_5m"].to_numpy(dtype=float),
        (merged["buy_volume"] - merged["sell_volume"]).to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


def test_zero_volume_bucket_is_neutral_not_nan(settings: Settings) -> None:
    """A bucket in which nothing traded is flat, and never Inf or NaN."""
    ohlcv, flow = make_market(rows=400)
    flow.loc[200:210, ["buy_volume", "sell_volume"]] = 0.0

    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)
    affected = frame.loc[frame["timestamp"].isin(flow.loc[200:210, "timestamp"])]

    assert np.isfinite(affected["order_flow_imbalance_5m"]).all()
    assert (affected["order_flow_imbalance_5m"] == 0.0).all()
    assert (affected["volume_delta_5m"] == 0.0).all()


def test_missing_flow_frame_leaves_features_neutral(settings: Settings) -> None:
    """No aggTrade data at all degrades to zero flow rather than crashing."""
    ohlcv, _ = make_market(rows=400)
    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=None)

    assert (frame["order_flow_imbalance_5m"] == 0.0).all()
    assert (frame["volume_delta_5m"] == 0.0).all()
    # Relative volume comes from OHLCV, so it is still a real number.
    assert frame["relative_volume_5m"].notna().sum() > 0


def test_relative_volume_excludes_the_current_candle(settings: Settings) -> None:
    """The bar's own volume must not appear in its own baseline.

    Checked two ways: against the explicit shifted-rolling formula, and by the
    behavioural consequence - inflating one bar's volume must not change that
    bar's *baseline*, only its ratio.
    """
    ohlcv, flow = make_market(rows=400)
    lookback = settings.features.relative_volume_lookback
    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)

    volume = ohlcv["volume"].astype(float).reset_index(drop=True)
    baseline = volume.shift(1).rolling(lookback, min_periods=lookback).mean()
    expected = (volume / baseline).clip(upper=settings.features.relative_volume_cap)

    actual = frame["relative_volume_5m"].reset_index(drop=True)
    warm = expected.notna()
    np.testing.assert_allclose(
        actual[warm].to_numpy(dtype=float),
        expected[warm].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )
    # Warm-up rows carry NaN and are dropped downstream, never silently filled.
    assert actual[:lookback].isna().all()

    spiked = ohlcv.copy()
    target = 300
    spiked.iloc[target, spiked.columns.get_loc("volume")] *= 50.0
    spiked_frame = FeatureEngineer(settings).build(spiked, agg_trade_flow=flow)

    # The spiked bar's own ratio jumps ...
    assert spiked_frame["relative_volume_5m"].iloc[target] > frame["relative_volume_5m"].iloc[target]
    # ... and every earlier bar is untouched, because the baseline is trailing.
    np.testing.assert_allclose(
        spiked_frame["relative_volume_5m"].iloc[lookback:target].to_numpy(dtype=float),
        frame["relative_volume_5m"].iloc[lookback:target].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


def test_relative_volume_survives_a_zero_baseline(settings: Settings) -> None:
    """A flat-zero volume history yields the neutral 1.0, not a division blow-up."""
    ohlcv, flow = make_market(rows=400)
    lookback = settings.features.relative_volume_lookback
    ohlcv.iloc[100 : 100 + lookback, ohlcv.columns.get_loc("volume")] = 0.0

    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)
    affected = frame["relative_volume_5m"].iloc[100 + lookback]
    assert np.isfinite(affected)
    assert affected == pytest.approx(1.0)


def test_order_flow_is_joined_on_the_exact_bucket(settings: Settings) -> None:
    """A missing bucket reads as zero flow, never as the previous bar's flow."""
    ohlcv, flow = make_market(rows=400)
    dropped_timestamp = int(flow.loc[250, "timestamp"])
    flow = flow.drop(index=250).reset_index(drop=True)

    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)
    row = frame.loc[frame["timestamp"] == dropped_timestamp]

    assert float(row["order_flow_imbalance_5m"].iloc[0]) == 0.0
    assert float(row["volume_delta_5m"].iloc[0]) == 0.0
