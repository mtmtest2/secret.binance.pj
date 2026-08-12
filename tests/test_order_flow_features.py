"""The three 5-minute order-flow features built from Binance aggTrades.

These replace what ``taker_buy_sell_ratio`` only ever approximated. The
distinction that matters is coverage: the ``futures/data`` ratio endpoints are
capped at roughly 30 days of Binance-side retention, while ``aggTrades`` is
served from full contract history - so this block can actually be reconstructed
across a multi-month training window, and the removed column could not.
"""

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

_BUCKET_MS = 5 * 60 * 1_000
_START_MS = 1_700_000_000_000 // _BUCKET_MS * _BUCKET_MS


def make_market(rows: int = 800, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic 5m market with matching aggTrade flow buckets."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.0008, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.0008, rows)) * close
    volume = np.abs(rng.lognormal(6.0, 0.45, rows))
    timestamps = _START_MS + np.arange(rows, dtype=np.int64) * _BUCKET_MS

    ohlcv = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": volume,
        }
    )
    ohlcv.index = pd.to_datetime(ohlcv["timestamp"], unit="ms", utc=True)
    ohlcv.index.name = "open_time"

    buy_share = np.clip(rng.normal(0.5, 0.12, rows), 0.02, 0.98)
    flow = pd.DataFrame(
        {
            "timestamp": timestamps,
            "buy_volume": volume * buy_share,
            "sell_volume": volume * (1.0 - buy_share),
            "trades": np.maximum(1, (volume / 10.0).astype(int)),
        }
    )
    return ohlcv, flow


@pytest.fixture()
def settings() -> Settings:
    """Settings with the slow windows shrunk so tests run in seconds."""
    config = Settings()
    config.features.garch_window = 120
    config.features.hmm_window = 200
    config.features.rank_window = 60
    return config


# ---------------------------------------------------------------------------
# Feature contract and routing
# ---------------------------------------------------------------------------
def test_order_flow_block_is_exactly_three_features() -> None:
    """No extra order-flow features, and the count moved by -3/+3 overall."""
    assert set(ORDER_FLOW_FEATURES) == {
        "order_flow_imbalance_5m",
        "volume_delta_5m",
        "relative_volume_5m",
    }
    assert len(FEATURE_COLUMNS) == len(BASE_FEATURE_COLUMNS) + 3
    for name in ORDER_FLOW_FEATURES:
        assert name in FEATURE_COLUMNS


def test_features_are_routed_to_the_heads_that_use_them() -> None:
    """Imbalance and delta -> Direction + Entry; relative volume also -> Risk."""
    for name in ("order_flow_imbalance_5m", "volume_delta_5m"):
        assert name in DIRECTION_FEATURE_COLUMNS
        assert name in ENTRY_FEATURE_COLUMNS
        assert name not in RISK_FEATURE_COLUMNS
        assert name not in EXIT_FEATURE_COLUMNS

    for contract in (DIRECTION_FEATURE_COLUMNS, ENTRY_FEATURE_COLUMNS, RISK_FEATURE_COLUMNS):
        assert "relative_volume_5m" in contract
    assert "relative_volume_5m" not in EXIT_FEATURE_COLUMNS


def test_heads_declare_the_expected_contracts() -> None:
    """Each head's declared column set matches the routing table."""
    from module_c_ml.ml_models import DirectionModel, EntryModel, ExitModel, RiskModel

    assert DirectionModel._declared_columns() == DIRECTION_FEATURE_COLUMNS
    assert EntryModel._declared_columns() == ENTRY_FEATURE_COLUMNS
    assert ExitModel._declared_columns() == EXIT_FEATURE_COLUMNS
    assert RiskModel._declared_columns() == RISK_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
def test_imbalance_and_delta_match_their_definitions(settings: Settings) -> None:
    """Both are read straight off the bar's own aggTrades bucket."""
    ohlcv, flow = make_market(rows=400)
    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)

    merged = frame.merge(flow, on="timestamp", how="left", suffixes=("", "_flow"))
    total = merged["buy_volume"] + merged["sell_volume"]
    np.testing.assert_allclose(
        merged["order_flow_imbalance_5m"].to_numpy(dtype=float),
        ((merged["buy_volume"] - merged["sell_volume"]) / total).to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        merged["volume_delta_5m"].to_numpy(dtype=float),
        (merged["buy_volume"] - merged["sell_volume"]).to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


def test_imbalance_is_bounded_and_finite(settings: Settings) -> None:
    """Expected range is [-1, +1], with no Inf or NaN anywhere."""
    ohlcv, flow = make_market(rows=600)
    values = (
        FeatureEngineer(settings)
        .build(ohlcv, agg_trade_flow=flow)["order_flow_imbalance_5m"]
        .to_numpy(dtype=float)
    )
    assert np.isfinite(values).all()
    assert values.min() >= -1.0
    assert values.max() <= 1.0


def test_zero_volume_bucket_is_neutral_not_nan(settings: Settings) -> None:
    """A bucket in which nothing traded is flat - never Inf, never NaN."""
    ohlcv, flow = make_market(rows=400)
    flow.loc[200:210, ["buy_volume", "sell_volume"]] = 0.0

    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)
    affected = frame.loc[frame["timestamp"].isin(flow.loc[200:210, "timestamp"])]

    assert np.isfinite(affected["order_flow_imbalance_5m"]).all()
    assert (affected["order_flow_imbalance_5m"] == 0.0).all()
    assert (affected["volume_delta_5m"] == 0.0).all()


def test_missing_flow_degrades_to_neutral(settings: Settings) -> None:
    """No aggTrade data at all must not raise, and must not invent flow."""
    ohlcv, _ = make_market(rows=400)
    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=None)

    assert (frame["order_flow_imbalance_5m"] == 0.0).all()
    assert (frame["volume_delta_5m"] == 0.0).all()
    # relative_volume_5m comes from OHLCV, so it is still a real number.
    assert frame["relative_volume_5m"].notna().sum() > 0


def test_flow_is_joined_on_the_exact_bucket(settings: Settings) -> None:
    """A missing bucket reads as zero flow, never as the previous bar's flow.

    This is why the join is an exact-key merge rather than the backward as-of
    join used for the funding/OI snapshots: an as-of match would silently carry
    the previous bar's aggression forward into a bar that had none.
    """
    ohlcv, flow = make_market(rows=400)
    dropped = int(flow.loc[250, "timestamp"])
    previous_imbalance = (flow.loc[249, "buy_volume"] - flow.loc[249, "sell_volume"]) / (
        flow.loc[249, "buy_volume"] + flow.loc[249, "sell_volume"]
    )
    flow = flow.drop(index=250).reset_index(drop=True)

    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)
    row = frame.loc[frame["timestamp"] == dropped]

    assert float(row["order_flow_imbalance_5m"].iloc[0]) == 0.0
    assert float(row["volume_delta_5m"].iloc[0]) == 0.0
    assert float(row["order_flow_imbalance_5m"].iloc[0]) != pytest.approx(previous_imbalance)


# ---------------------------------------------------------------------------
# relative_volume_5m - the current candle must be excluded
# ---------------------------------------------------------------------------
def test_relative_volume_excludes_the_current_candle(settings: Settings) -> None:
    """Checked against the explicit formula and behaviourally.

    No existing lookback was reusable: volume_trend's 12/96 means and the HMM's
    96-bar relative volume all include the current bar, which is exactly what
    this feature must not do.
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


def test_a_volume_spike_does_not_contaminate_its_own_baseline(settings: Settings) -> None:
    """Inflating one bar raises only that bar's ratio, never an earlier one."""
    ohlcv, flow = make_market(rows=400)
    lookback = settings.features.relative_volume_lookback
    engineer = FeatureEngineer(settings)

    base = engineer.build(ohlcv, agg_trade_flow=flow)
    spiked_ohlcv = ohlcv.copy()
    target = 300
    spiked_ohlcv.iloc[target, spiked_ohlcv.columns.get_loc("volume")] *= 50.0
    spiked = engineer.build(spiked_ohlcv, agg_trade_flow=flow)

    assert spiked["relative_volume_5m"].iloc[target] > base["relative_volume_5m"].iloc[target]
    np.testing.assert_allclose(
        spiked["relative_volume_5m"].iloc[lookback:target].to_numpy(dtype=float),
        base["relative_volume_5m"].iloc[lookback:target].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


def test_relative_volume_survives_a_zero_baseline(settings: Settings) -> None:
    """A flat-zero volume history gives the neutral 1.0, not a division blow-up."""
    ohlcv, flow = make_market(rows=400)
    lookback = settings.features.relative_volume_lookback
    ohlcv.iloc[100 : 100 + lookback, ohlcv.columns.get_loc("volume")] = 0.0

    frame = FeatureEngineer(settings).build(ohlcv, agg_trade_flow=flow)
    value = frame["relative_volume_5m"].iloc[100 + lookback]
    assert np.isfinite(value)
    assert value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Point-in-time correctness
# ---------------------------------------------------------------------------
def test_order_flow_features_are_point_in_time(settings: Settings) -> None:
    """Truncating the future must not change a single past value.

    The definitive look-ahead test: build over the whole history, rebuild over a
    prefix, and require the overlapping rows to agree exactly.
    """
    ohlcv, flow = make_market(rows=700)
    cut = 550
    engineer = FeatureEngineer(settings)

    full = engineer.build(ohlcv, agg_trade_flow=flow)
    prefix = engineer.build(ohlcv.iloc[:cut].copy(), agg_trade_flow=flow.iloc[:cut].copy())

    for column in ORDER_FLOW_FEATURES:
        np.testing.assert_allclose(
            full[column].iloc[:cut].to_numpy(dtype=float),
            prefix[column].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )


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
