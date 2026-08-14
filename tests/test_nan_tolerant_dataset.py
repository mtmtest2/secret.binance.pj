"""Rows must survive a NaN in a feature column.

The previous behaviour dropped any row with a NaN in *any* of the 50 feature
columns.  That is only harmless if missingness is uniform; it is not.  The
sparsest features (flat-market volatility ratios, GARCH warm-up, archive
coverage) cluster in low-liquidity symbols and quiet periods, so the drop
removed the oldest, quietest, smallest-cap part of the training window and left
validation and test nearly intact.

These tests pin the new contract: only an unusable *label* costs a row, the
boosters receive NaN and split on it, and the sparsity is measured rather than
silently converted into a smaller dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import (
    FEATURE_COLUMNS,
    OPTIONAL_FEATURE_COLUMNS,
    FeatureEngineer,
)
from module_b_features.processor import DatasetProcessor


def _pooled(rows: int = 400, symbols: tuple[str, ...] = ("A/USDT:USDT", "B/USDT:USDT")) -> pd.DataFrame:
    """A minimal pooled frame in the shape `_to_dataset` expects."""
    frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(7)
    for offset, symbol in enumerate(symbols):
        frame = pd.DataFrame(
            {column: rng.normal(size=rows) for column in FEATURE_COLUMNS}
        )
        frame["timestamp"] = np.arange(rows, dtype=np.int64) * 300_000 + offset
        frame["symbol"] = symbol
        frame["label"] = "NO_TRADE_OR_FAIL"
        frame["label_is_valid"] = True
        frame["entry_quality"] = 0
        frame["target_tp_pct"] = 0.02
        frame["target_sl_pct"] = 0.01
        frame["target_trailing_pct"] = 0.01
        frame["target_risk_score"] = 0.5
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = 100.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _processor() -> DatasetProcessor:
    return DatasetProcessor.__new__(DatasetProcessor)


def test_rows_with_nan_features_are_kept() -> None:
    pooled = _pooled()
    # Blank an entire feature for the first half of the history, exactly the
    # shape of a sparse archive column or an un-warmed rolling window.
    pooled.loc[: len(pooled) // 2, "ob_imbalance"] = np.nan

    dataset = _processor()._to_dataset(pooled, ("A/USDT:USDT", "B/USDT:USDT"))

    assert len(dataset) == len(pooled), "no row may be lost to a NaN feature"
    assert dataset.features["ob_imbalance"].isna().sum() > 0, "the NaN must reach the model"


def test_rows_with_unusable_labels_are_still_dropped() -> None:
    pooled = _pooled()
    pooled.loc[:9, "target_risk_score"] = np.nan

    dataset = _processor()._to_dataset(pooled, ("A/USDT:USDT", "B/USDT:USDT"))

    assert len(dataset) == len(pooled) - 10
    assert dataset.dropped_missing_or_inf_rows == 10


def test_infinities_become_nan_rather_than_dropping_the_row() -> None:
    pooled = _pooled()
    pooled.loc[0, "garch_vol_ratio"] = np.inf
    pooled.loc[1, "vol_of_vol"] = -np.inf

    dataset = _processor()._to_dataset(pooled, ("A/USDT:USDT", "B/USDT:USDT"))

    assert len(dataset) == len(pooled)
    assert np.isnan(dataset.features.loc[0, "garch_vol_ratio"])
    assert np.isnan(dataset.features.loc[1, "vol_of_vol"])


def test_null_counts_are_reported_per_feature() -> None:
    pooled = _pooled()
    pooled.loc[:24, "ob_spread_bps"] = np.nan

    dataset = _processor()._to_dataset(pooled, ("A/USDT:USDT", "B/USDT:USDT"))

    assert dataset.null_counts_by_feature["ob_spread_bps"] == 25
    assert dataset.null_counts_by_feature["adx"] == 0


def test_null_rate_is_broken_down_by_symbol_and_month() -> None:
    """Uniform 3% sparsity and 90% sparsity in one symbol are different problems."""
    pooled = _pooled()
    only_a = pooled["symbol"] == "A/USDT:USDT"
    pooled.loc[only_a, "liquidation_imbalance"] = np.nan

    dataset = _processor()._to_dataset(pooled, ("A/USDT:USDT", "B/USDT:USDT"))

    rates = dataset.null_rate_by_symbol_month
    assert "A/USDT:USDT" in rates
    assert max(rates["A/USDT:USDT"].values()) == pytest.approx(1.0)
    if "B/USDT:USDT" in rates:
        assert max(rates["B/USDT:USDT"].values()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Feature-level NaN sources that were pure arithmetic accident
# ---------------------------------------------------------------------------
def _flat_market(rows: int = 700) -> pd.DataFrame:
    """A dead market: every candle closes at the same price."""
    index = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 10.0,
            "timestamp": (index.view("int64") // 1_000_000).astype("int64"),
        },
        index=index,
    )
    frame.index.name = "open_time"
    return frame


def test_flat_market_does_not_produce_nan_volatility_ratios() -> None:
    """`realized_vol_12 == 0` used to make vol_of_vol and garch_vol_ratio NaN.

    Since a NaN anywhere killed the whole row, and flat candles are routine for
    a coarse-tick alt in a quiet hour, this alone deleted a large, systematically
    biased share of the training window.
    """
    engineer = FeatureEngineer(Settings())
    built = engineer.build(_flat_market())

    assert built["realized_vol_12_is_zero"].max() == pytest.approx(1.0), (
        "the dead-market state must be carried explicitly"
    )
    assert built["vol_of_vol"].notna().any()
    assert built["garch_vol_ratio"].notna().any()


def test_microstructure_columns_are_nan_without_archive_data() -> None:
    """No book data must read as 'unobserved', never as a balanced book."""
    engineer = FeatureEngineer(Settings())
    built = engineer.build(_flat_market())

    for column in OPTIONAL_FEATURE_COLUMNS:
        assert built[column].isna().all(), f"{column} must be NaN without archive coverage"
    assert built["microstructure_is_missing"].min() == pytest.approx(1.0)


def test_microstructure_columns_populate_from_joined_buckets() -> None:
    """And with coverage, they must carry the real reading on the exact bucket."""
    engineer = FeatureEngineer(Settings())
    ohlcv = _flat_market()
    buckets = pd.DataFrame(
        {
            "timestamp": ohlcv["timestamp"].to_numpy(),
            "bid_qty": 75.0,
            "ask_qty": 25.0,
            "spread_bps": 2.0,
            "liquidation_buy_volume": 0.0,
            "liquidation_sell_volume": 100.0,
        }
    )
    built = engineer.build(ohlcv, order_book=buckets)

    assert built["ob_imbalance"].iloc[-1] == pytest.approx(0.5)
    assert built["ob_spread_bps"].iloc[-1] == pytest.approx(2.0)
    # Only longs liquidated -> fully negative imbalance.
    assert built["liquidation_imbalance"].iloc[-1] == pytest.approx(-1.0)
    assert built["microstructure_is_missing"].iloc[-1] == pytest.approx(0.0)


def test_microstructure_join_never_carries_a_stale_bucket_forward() -> None:
    """A gap must stay a gap: exact-key join, not as-of."""
    engineer = FeatureEngineer(Settings())
    ohlcv = _flat_market()
    timestamps = ohlcv["timestamp"].to_numpy()
    # Only the first 10 buckets are covered.
    buckets = pd.DataFrame(
        {
            "timestamp": timestamps[:10],
            "bid_qty": 60.0,
            "ask_qty": 40.0,
            "spread_bps": 1.0,
        }
    )
    built = engineer.build(ohlcv, order_book=buckets)

    assert built["ob_imbalance"].iloc[0] == pytest.approx(0.2)
    assert np.isnan(built["ob_imbalance"].iloc[-1]), (
        "an uncovered bucket must not inherit the last observed book"
    )
    assert built["microstructure_is_missing"].iloc[-1] == pytest.approx(1.0)
