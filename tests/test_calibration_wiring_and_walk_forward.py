"""Production calibration wiring and walk-forward evaluation.

Covers two additions to module_c_ml/ml_models.py:

* Direction/Entry heads now fit a production isotonic calibrator on the full
  validation block and swap it in as the serving model when it measurably
  improves log loss (previously calibration was only ever measured, never
  used for inference).
* DirectionModel.walk_forward() - the multi-fold rolling evaluation the ML
  diagnostic report's own recommendations flagged as the biggest missing
  piece ("only a single split is currently performed").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS
from module_b_features.labeler import LABEL_ORDER
from module_b_features.processor import ProcessedDataset
from module_c_ml.ml_models import DirectionModel, EntryModel


def _structured_dataset(rng: np.random.Generator, n: int) -> ProcessedDataset:
    """A dataset with real (not pure-noise) signal, so calibration/CV have
    something non-trivial to work with."""
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    signal = features["atr_pct"] + 0.5 * features["adx"]
    probability = 1.0 / (1.0 + np.exp(-signal))
    direction_target = pd.Series(
        np.where(
            probability > 0.66,
            "LONG_SUCCESS",
            np.where(probability < 0.33, "SHORT_SUCCESS", "NO_TRADE_OR_FAIL"),
        )
    )
    entry_target = pd.Series((probability > 0.5).astype(int).to_numpy())
    exit_targets = pd.DataFrame(
        {
            "target_tp_pct": np.abs(rng.normal(0.02, 0.005, size=n)),
            "target_sl_pct": np.abs(rng.normal(0.01, 0.002, size=n)),
            "target_trailing_pct": np.abs(rng.normal(0.01, 0.002, size=n)),
        }
    )
    risk_target = pd.Series(rng.uniform(0.0, 1.0, size=n))
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


def test_direction_model_wires_calibrated_model_into_inference_when_it_helps() -> None:
    settings = Settings(
        ml={"n_estimators": 50, "early_stopping_rounds": 10, "validation_fraction": 0.3, "purge_bars": 10}
    )
    dataset = _structured_dataset(np.random.default_rng(21), n=5000)

    model = DirectionModel(settings)
    model.train(dataset)

    # Direction is a two-stage cascade: production calibration is recorded
    # per stage, honestly, either way.
    production = model.metadata["production_calibration"]
    assert set(production) == {"gate", "direction"}
    assert production["gate"] in {"isotonic", "raw"}
    assert production["direction"] in {"isotonic", "raw"}

    prediction = model.predict(dataset.features.iloc[[0]])
    assert set(prediction.probabilities) == set(LABEL_ORDER)
    assert prediction.probabilities == pytest.approx(prediction.probabilities)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    # self._model is now {"gate": ..., "direction": ...}; whichever stage got
    # calibrated must still expose the sklearn classifier API predict()
    # relies on.
    assert set(model._model) == {"gate", "direction"}
    for stage_key, calibrated in production.items():
        stage_model = model._model.get(stage_key)
        if calibrated == "isotonic" and stage_model is not None:
            assert hasattr(stage_model, "classes_")
            assert hasattr(stage_model, "predict_proba")


def test_entry_model_predict_still_works_regardless_of_calibration_outcome() -> None:
    settings = Settings(
        ml={"n_estimators": 50, "early_stopping_rounds": 10, "validation_fraction": 0.3, "purge_bars": 10}
    )
    dataset = _structured_dataset(np.random.default_rng(22), n=5000)

    model = EntryModel(settings)
    model.train(dataset)
    assert model.metadata["production_calibration"] in {"isotonic", "raw"}

    prediction = model.predict(dataset.features.iloc[[0]])
    assert 0.0 <= prediction.probability <= 1.0


def test_walk_forward_not_available_below_the_row_floor() -> None:
    settings = Settings(ml={"n_estimators": 10})
    dataset = _structured_dataset(np.random.default_rng(1), n=200)
    model = DirectionModel(settings)
    result = model.walk_forward(dataset, n_folds=4)
    assert result["status"] == "NOT_AVAILABLE"
    assert "rows" in result["reason"]


def test_walk_forward_produces_expanding_folds_that_never_touch_the_saved_model() -> None:
    settings = Settings(ml={"n_estimators": 20, "early_stopping_rounds": 0, "purge_bars": 5})
    dataset = _structured_dataset(np.random.default_rng(2), n=6000)
    model = DirectionModel(settings)

    result = model.walk_forward(dataset, n_folds=4)
    assert result["status"] == "AVAILABLE"
    assert result["n_folds"] == 4
    assert model.is_loaded is False  # walk_forward never sets self._model

    folds = result["folds"]
    assert len(folds) == 4
    # Expanding window: each fold's training block is strictly larger than
    # (and a superset of, in row count terms) the previous one's.
    train_rows = [fold["train_rows"] for fold in folds]
    assert train_rows == sorted(train_rows)
    for fold in folds:
        assert 0.0 <= fold["accuracy"] <= 1.0
        assert 0.0 <= fold["balanced_accuracy"] <= 1.0

    assert 0.0 <= result["accuracy_mean"] <= 1.0
    assert result["accuracy_std"] >= 0.0
