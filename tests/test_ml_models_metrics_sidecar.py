"""BaseModelHead.save() writes a full JSON metrics sidecar (module_c_ml/ml_models.py).

The joblib artifact is opaque to anything that isn't unpickling an estimator;
the diagnostic report is built from this sidecar instead, so its presence and
content are worth testing directly rather than only indirectly through the
full report-builder tests.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS
from module_b_features.labeler import LABEL_ORDER
from module_b_features.processor import ProcessedDataset
from module_c_ml.ml_models import DirectionModel, EntryModel


def _dataset(rng: np.random.Generator, n: int = 600) -> ProcessedDataset:
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    direction_target = pd.Series(rng.choice(LABEL_ORDER, size=n))
    entry_target = pd.Series(rng.integers(0, 2, size=n))
    exit_targets = pd.DataFrame(
        {
            "target_tp_pct": np.abs(rng.normal(0.02, 0.005, size=n)),
            "target_sl_pct": np.abs(rng.normal(0.01, 0.002, size=n)),
            "target_trailing_pct": np.abs(rng.normal(0.01, 0.002, size=n)),
        }
    )
    risk_target = pd.Series(rng.uniform(0, 1, size=n))
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


def test_direction_model_save_writes_metrics_sidecar(tmp_path) -> None:
    settings = Settings(ml={"model_dir": tmp_path, "n_estimators": 10, "early_stopping_rounds": 5})
    dataset = _dataset(np.random.default_rng(11))

    head = DirectionModel(settings)
    metrics = head.train(dataset)
    assert "confusion_matrix" in metrics  # the full metric set, not just accuracy/log_loss

    saved_path = head.save()
    sidecar = saved_path.with_suffix(".metrics.json")
    assert sidecar.exists()

    payload = json.loads(sidecar.read_text())
    assert payload["head"] == "direction_model"
    assert payload["feature_columns"] == list(FEATURE_COLUMNS)
    assert "git_commit" in payload
    assert payload["hyperparameters"]["n_estimators"] == 10
    assert "feature_importance" in payload
    assert "calibration" in payload
    # The sidecar must be valid, self-contained JSON.
    assert isinstance(payload["metrics"]["accuracy"], float)


def test_entry_model_metadata_includes_threshold_sweep_and_decision_threshold(tmp_path) -> None:
    settings = Settings(ml={"model_dir": tmp_path, "n_estimators": 10, "early_stopping_rounds": 5})
    dataset = _dataset(np.random.default_rng(12))

    head = EntryModel(settings)
    head.train(dataset)

    # decision_threshold is now auto-selected from the validation sweep
    # (F-beta=0.5, precision-weighted); configured_floor_threshold is the
    # untouched config default and is what a degenerate sweep falls back to.
    assert head.metadata["configured_floor_threshold"] == pytest.approx(
        settings.decision.min_entry_probability
    )
    sweep = head.metadata["threshold_sweep"]
    valid_thresholds = {row["threshold"] for row in sweep} | {settings.decision.min_entry_probability}
    assert head.metadata["decision_threshold"] in valid_thresholds
    assert len(sweep) == len(head.metadata["threshold_sweep"])  # sanity: non-empty, self-consistent
    # ENTRY_THRESHOLDS was extended downward in task 7a (grid used to bottom
    # out at 0.50) - assert against the live constant rather than a frozen
    # literal, so this test does not silently drift out of sync again.
    from module_c_ml.metrics import ENTRY_THRESHOLDS

    assert {row["threshold"] for row in sweep} == set(ENTRY_THRESHOLDS)
    for row in sweep:
        assert row["average_r"] == "NOT_AVAILABLE"
