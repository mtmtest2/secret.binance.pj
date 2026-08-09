"""Round 2: real backtest wiring, Risk robustness, Direction cascade, Entry auto-threshold.

Covers the four follow-up improvements requested after the first diagnostic
comparison:

* module_c_ml/ml_models.py - RiskModel now fits with the same L1 (robust)
  objective as Exit, since target_risk_score is right-skewed too.
* module_b_features/features.py - two new causal path-heat features
  (wick_ratio, whipsaw_rate) feed the Risk model specifically.
* module_c_ml/ml_models.py - DirectionModel is now a two-stage cascade
  (trade gate, then long-vs-short) instead of one 3-way softmax.
* module_c_ml/ml_models.py - EntryModel auto-selects its decision threshold
  from its own validation sweep (F-beta=0.5) instead of a fixed config value.
* module_f_panel/diagnostics.py - a new microstructure/derivatives data
  coverage check, so a feed that is silently defaulting to neutral values
  is measured and flagged rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS, FeatureEngineer
from module_b_features.processor import ProcessedDataset
from module_c_ml.ml_models import DirectionModel, EntryModel, RiskModel
from module_f_panel import diagnostics


def _ohlcv(rng: np.random.Generator, n: int = 1500) -> pd.DataFrame:
    returns = rng.normal(0.0, 0.004, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, size=n)))
    volume = np.abs(rng.normal(1000, 200, size=n))
    index = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )
    frame.index.name = "open_time"
    return frame


def test_wick_ratio_and_whipsaw_rate_are_causal_and_bounded() -> None:
    settings = Settings()
    engineer = FeatureEngineer(settings)
    rng = np.random.default_rng(9)
    ohlcv = _ohlcv(rng)

    assert "wick_ratio" in FEATURE_COLUMNS
    assert "whipsaw_rate" in FEATURE_COLUMNS

    full = engineer.build(ohlcv)
    truncated = engineer.build(ohlcv.iloc[:-30])

    for column in ("wick_ratio", "whipsaw_rate"):
        values = full[column].dropna()
        assert values.between(0.0, 1.0).all()

        common = full.index.intersection(truncated.index)
        common = common[common < ohlcv.index[-60]]
        a = full.loc[common, column].to_numpy()
        b = truncated.loc[common, column].to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"{column} depends on future data"


def _dataset_with_risk_target(rng: np.random.Generator, n: int = 600) -> ProcessedDataset:
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    direction_target = pd.Series(rng.choice(["LONG_SUCCESS", "SHORT_SUCCESS", "NO_TRADE_OR_FAIL"], size=n))
    entry_target = pd.Series(rng.integers(0, 2, size=n))
    exit_targets = pd.DataFrame(
        {
            "target_tp_pct": np.abs(rng.normal(0.02, 0.005, size=n)),
            "target_sl_pct": np.abs(rng.normal(0.01, 0.002, size=n)),
            "target_trailing_pct": np.abs(rng.normal(0.01, 0.002, size=n)),
        }
    )
    # Right-skewed, like the real target_risk_score in production.
    risk_target = pd.Series(rng.beta(1.5, 4.0, size=n))
    metadata = pd.DataFrame({"symbol": ["BTC/USDT:USDT"] * n, "timestamp": np.arange(n) * 300_000})
    return ProcessedDataset(
        features=features,
        direction_target=direction_target,
        entry_target=entry_target,
        exit_targets=exit_targets,
        risk_target=risk_target,
        metadata=metadata,
        symbols=("BTC/USDT:USDT",),
        feature_columns=tuple(FEATURE_COLUMNS),
    )


def test_risk_model_uses_robust_l1_objective() -> None:
    settings = Settings(ml={"n_estimators": 15, "early_stopping_rounds": 5})
    dataset = _dataset_with_risk_target(np.random.default_rng(5))

    model = RiskModel(settings)
    model.train(dataset)

    assert model._model.get_params()["objective"] == "regression_l1"


def test_direction_model_is_a_two_stage_cascade() -> None:
    settings = Settings(ml={"n_estimators": 15, "early_stopping_rounds": 5})
    dataset = _dataset_with_risk_target(np.random.default_rng(6))

    model = DirectionModel(settings)
    model.train(dataset)

    assert model.metadata["architecture"] == "two_stage_cascade"
    assert set(model._model) == {"gate", "direction"}
    assert set(model.metadata["feature_importance"]) == {"gate", "direction"}


def test_entry_select_recommended_threshold_prefers_precision() -> None:
    # Threshold A: high recall, mediocre precision. Threshold B: strong on
    # both. F-beta=0.5 (precision-weighted) should prefer B over the
    # loosest/highest-recall candidate.
    sweep = [
        {"threshold": 0.5, "precision": 0.35, "recall": 0.9, "f1": 0.50, "meets_min_sample_size": True},
        {"threshold": 0.6, "precision": 0.70, "recall": 0.55, "f1": 0.61, "meets_min_sample_size": True},
        {"threshold": 0.9, "precision": 0.95, "recall": 0.02, "f1": 0.04, "meets_min_sample_size": True},
    ]
    chosen = EntryModel._select_recommended_threshold(sweep, floor=0.55)
    assert chosen == 0.6


def test_entry_select_recommended_threshold_falls_back_when_nothing_qualifies() -> None:
    sweep = [{"threshold": 0.5, "precision": 0.9, "recall": 0.9, "f1": 0.9, "meets_min_sample_size": False}]
    assert EntryModel._select_recommended_threshold(sweep, floor=0.55) == 0.55


def test_microstructure_coverage_flags_unpopulated_features() -> None:
    n = 500
    features = pd.DataFrame(0.0, index=range(n), columns=list(FEATURE_COLUMNS))
    # ob_imbalance genuinely populated; funding_rate left at its neutral default.
    features["ob_imbalance"] = np.random.default_rng(1).uniform(-0.5, 0.5, size=n)

    dataset = ProcessedDataset(
        features=features,
        direction_target=pd.Series(["NO_TRADE_OR_FAIL"] * n),
        entry_target=pd.Series([0] * n),
        exit_targets=pd.DataFrame(
            {"target_tp_pct": [0.01] * n, "target_sl_pct": [0.005] * n, "target_trailing_pct": [0.005] * n}
        ),
        risk_target=pd.Series([0.5] * n),
        metadata=pd.DataFrame({"symbol": ["BTC/USDT:USDT"] * n}),
        symbols=("BTC/USDT:USDT",),
        feature_columns=tuple(FEATURE_COLUMNS),
    )

    result = diagnostics._microstructure_coverage(dataset)
    assert result["status"] == "AVAILABLE"
    assert result["features"]["ob_imbalance"]["likely_populated"] is True
    assert result["features"]["funding_rate"]["likely_populated"] is False


def test_microstructure_coverage_empty_dataset_is_not_available() -> None:
    dataset = ProcessedDataset(
        features=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
        direction_target=pd.Series(dtype=str),
        entry_target=pd.Series(dtype=int),
        exit_targets=pd.DataFrame(columns=["target_tp_pct", "target_sl_pct", "target_trailing_pct"]),
        risk_target=pd.Series(dtype=float),
        metadata=pd.DataFrame(),
    )
    result = diagnostics._microstructure_coverage(dataset)
    assert result["status"] == diagnostics.NOT_AVAILABLE
