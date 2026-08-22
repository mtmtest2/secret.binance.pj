"""The reported metrics must describe the artifact that ships (audit P1, P28).

The audited run reported a Direction cascade that never once predicted
LONG across 1.4M validation rows, while the artifact actually saved - and
replayed in the backtest - went long on most of its trades.  Both statements
were true of *different* models: `train()` scored the raw estimators, then
`_calibrate_cascade` replaced `self._model` with isotonic-calibrated wrappers
before anything was persisted.

Nothing caught it because no test ever compared the two paths.  These tests are
that comparison, at three levels:

* the metrics recorded in metadata reproduce when the *saved* model is re-scored;
* `save()` refuses an artifact whose metrics were measured against something else;
* regression heads are measured through the same clamping their `predict()`
  applies, so a documented output domain means what it says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS
from module_b_features.labeler import LABEL_ORDER, LabelClass
from module_b_features.processor import ProcessedDataset
from module_c_ml import metrics as ml_metrics
from module_c_ml.ml_models import DirectionModel, RiskModel, _underlying_estimator
from core.exceptions import ModelTrainingError

_TIMEFRAME_MS = 5 * 60 * 1_000


def _dataset(rng: np.random.Generator, rows: int = 3_000) -> ProcessedDataset:
    """A learnable synthetic dataset: the label depends on two features.

    Signal matters here - a pure-noise dataset trains a degenerate cascade whose
    calibrated and raw forms agree trivially, which would let a broken
    implementation pass.
    """
    features = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(rows, len(FEATURE_COLUMNS))),
        columns=list(FEATURE_COLUMNS),
    )
    score = features["kama_slope"] * 1.5 + features["rsi"] * 0.8 + rng.normal(0, 0.5, size=rows)
    labels = np.where(
        score > 0.8,
        LabelClass.LONG_SUCCESS.value,
        np.where(score < -0.8, LabelClass.SHORT_SUCCESS.value, LabelClass.NO_TRADE_OR_FAIL.value),
    )
    timestamps = np.arange(rows, dtype=np.int64) * _TIMEFRAME_MS + 1_700_000_000_000
    metadata = pd.DataFrame({"timestamp": timestamps, "symbol": "BTC/USDT:USDT"})
    return ProcessedDataset(
        features=features,
        direction_target=pd.Series(labels, name="label"),
        entry_target=pd.Series((score.abs() > 1.2).astype(int)),
        exit_targets=pd.DataFrame(
            {
                "target_tp_pct": np.abs(rng.normal(0.02, 0.005, rows)),
                "target_sl_pct": np.abs(rng.normal(0.01, 0.002, rows)),
                "target_trailing_pct": np.abs(rng.normal(0.01, 0.0025, rows)),
            }
        ),
        risk_target=pd.Series(rng.uniform(0.05, 0.95, rows)),
        metadata=metadata,
        feature_columns=tuple(FEATURE_COLUMNS),
        symbols=("BTC/USDT:USDT",),
        total_candidate_rows=rows,
    )


def _settings(tmp_path) -> Settings:
    return Settings(
        ml={
            "model_dir": tmp_path / "models",
            "n_estimators": 25,
            "early_stopping_rounds": 5,
            "purge_bars": 10,
        }
    )


def test_direction_metrics_reproduce_from_the_saved_cascade(tmp_path) -> None:
    """Re-scoring the shipped model must reproduce the reported metrics.

    This is the check that would have failed loudly on the audited run.
    """
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(5))
    head = DirectionModel(settings)
    head.train(dataset)

    split = head._chronological_split(dataset)
    validation_features = dataset.features.iloc[split.validation_index]
    validation_target = (
        dataset.direction_target.map({name: i for i, name in enumerate(LABEL_ORDER)})
        .iloc[split.validation_index]
        .astype(int)
    )

    # Score through the model the head is holding - the one save() writes.
    probabilities = head._combined_probabilities(
        head._model["gate"], head._model.get("direction"), validation_features
    )
    rescored = ml_metrics.direction_metrics(validation_target, probabilities, LABEL_ORDER)
    reported = head.metadata["metrics"]

    assert rescored["accuracy"] == pytest.approx(reported["accuracy"], abs=1e-9)
    assert rescored["log_loss"] == pytest.approx(reported["log_loss"], abs=1e-9)
    assert rescored["predicted_class_distribution"] == reported["predicted_class_distribution"]


def test_save_refuses_metrics_measured_against_a_different_estimator(tmp_path) -> None:
    """The identity guard turns the P1 class of bug into a loud failure."""
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(6))
    head = DirectionModel(settings)
    head.train(dataset)

    # Saving as trained is fine.
    head.save()

    # Simulate the exact defect: swap the model out after metrics were computed.
    head._model = {"gate": head._model["gate"], "direction": None}
    with pytest.raises(ModelTrainingError) as excinfo:
        head.save()
    assert "different estimator" in str(excinfo.value)


def test_load_then_save_round_trip_is_allowed(tmp_path) -> None:
    """Object identity does not survive pickling, so the guard re-stamps on load.

    Without this the guard would make every reload-and-persist cycle illegal,
    which would be a bug of its own.
    """
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(7))
    head = DirectionModel(settings)
    head.train(dataset)
    path = head.save()

    reloaded = DirectionModel(settings)
    assert reloaded.load(path) is True
    reloaded.save()  # must not raise


def test_direction_reports_raw_and_production_metrics_separately(tmp_path) -> None:
    """When calibration is adopted, both readings stay visible.

    Reporting only one of two materially different models is what made the
    audited run unreadable; reporting both makes the adoption decision
    inspectable.
    """
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(8))
    head = DirectionModel(settings)
    head.train(dataset)

    metadata = head.metadata
    assert "metrics_raw_uncalibrated" in metadata
    production = metadata["production_calibration"]
    if production != {"gate": "raw", "direction": "raw"}:
        raw_block = metadata["metrics_raw_uncalibrated"]
        assert "accuracy" in raw_block, "calibration was applied but the raw reading was not kept"


def test_feature_importance_survives_a_calibration_wrapper(tmp_path) -> None:
    """Importance is read off the booster, not the wrapper that hides it."""
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(9))
    head = DirectionModel(settings)
    head.train(dataset)

    gate_importance = head.metadata["feature_importance"]["gate"]
    assert gate_importance.get("status") == "AVAILABLE", gate_importance
    assert gate_importance.get("top_features"), "importance came back empty through the wrapper"


def test_underlying_estimator_unwraps_and_passes_through_plain_estimators() -> None:
    class Booster:
        feature_importances_ = np.array([1.0, 2.0])

    class Wrapper:
        def __init__(self, inner):
            self.estimator = inner

    booster = Booster()
    assert _underlying_estimator(Wrapper(booster)) is booster
    assert _underlying_estimator(booster) is booster
    assert _underlying_estimator(None) is None


def test_risk_metrics_are_measured_through_predicts_clamping(tmp_path) -> None:
    """A head documented as bounded must not report values outside its domain."""
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(10))
    head = RiskModel(settings)
    head.train(dataset)

    stats = head.metadata["metrics"]["prediction_stats"]
    assert stats["max"] <= 1.0 + 1e-12, f"reported max {stats['max']} exceeds the documented domain"
    assert stats["min"] >= -1e-12
    assert "clipped_prediction_rate" in head.metadata["metrics"]
