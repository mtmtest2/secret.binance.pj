"""Unit tests for the pure metric-computation helpers (module_c_ml/metrics.py).

These are the functions the ML diagnostic report's Direction/Entry/Exit/Risk
sections are built from, so correctness here is what keeps the report honest.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from module_c_ml import metrics as m

LABELS = ("A", "B", "C")


def test_direction_metrics_empty_returns_empty_dict() -> None:
    assert m.direction_metrics(pd.Series(dtype=int), np.zeros((0, 3)), LABELS) == {}


def test_direction_metrics_perfect_predictions() -> None:
    target = pd.Series([0, 1, 2, 0, 1, 2])
    # One-hot "probabilities" that always pick the true class with high confidence.
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]
    )
    result = m.direction_metrics(target, probabilities, LABELS)
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["confusion_matrix"]["raw"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert result["class_distribution"] == {"A": 2, "B": 2, "C": 2}
    assert result["predicted_class_distribution"] == {"A": 2, "B": 2, "C": 2}
    assert len(result["confidence_threshold_analysis"]) == len(m.CONFIDENCE_THRESHOLDS)
    # At threshold 0.85 every prediction (confidence 0.9) still qualifies.
    row_85 = next(r for r in result["confidence_threshold_analysis"] if r["confidence_threshold"] == 0.85)
    assert row_85["n_predictions"] == 6
    assert row_85["accuracy"] == 1.0


def test_direction_metrics_confidence_threshold_excludes_low_confidence_rows() -> None:
    target = pd.Series([0, 1])
    probabilities = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
    result = m.direction_metrics(target, probabilities, LABELS)
    row_80 = next(r for r in result["confidence_threshold_analysis"] if r["confidence_threshold"] == 0.80)
    assert row_80["n_predictions"] == 1  # only the 0.9-confidence row qualifies
    row_none = {**result["confidence_threshold_analysis"][0]}
    assert row_none["confidence_threshold"] == 0.50


def test_entry_metrics_at_threshold() -> None:
    target = pd.Series([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.4, 0.3, 0.8])  # 1 true positive, 1 false negative, 1 false positive
    result = m.entry_metrics(target, probabilities, threshold=0.5)
    assert result["threshold"] == 0.5
    assert result["class_distribution"] == {"positive": 2, "negative": 2}
    # predictions at 0.5: [1, 0, 0, 1] vs target [1, 1, 0, 0]
    assert result["predicted_positive_rate"] == 0.5
    assert 0.0 <= result["precision"] <= 1.0
    assert "confusion_matrix" in result


def test_entry_threshold_sweep_marks_trading_fields_not_available() -> None:
    target = pd.Series([1, 0, 1, 0, 1])
    probabilities = np.array([0.9, 0.2, 0.6, 0.1, 0.95])
    rows = m.entry_threshold_sweep(target, probabilities, thresholds=(0.5, 0.9), min_signal_sample_size=10)
    assert len(rows) == 2
    for row in rows:
        assert row["average_r"] == m.NOT_AVAILABLE
        assert row["win_rate"] == m.NOT_AVAILABLE
        assert row["profit_factor"] == m.NOT_AVAILABLE
        assert row["meets_min_sample_size"] is False  # only 5 rows total, cap is 10


def test_regression_metrics_known_values() -> None:
    target = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.array([1.0, 2.0, 3.0, 5.0])  # one off-by-one error
    result = m.regression_metrics(target, predictions)
    assert result["mae"] == pytest.approx(0.25)
    assert result["target_stats"]["mean"] == pytest.approx(2.5)
    assert result["prediction_stats"]["max"] == pytest.approx(5.0)


def test_regression_metrics_constant_target_yields_nan_r2() -> None:
    target = np.array([5.0, 5.0, 5.0])
    predictions = np.array([5.0, 5.1, 4.9])
    result = m.regression_metrics(target, predictions)
    assert math.isnan(result["r2"])  # R^2 is undefined for a zero-variance target


def test_regression_metrics_empty_returns_empty_dict() -> None:
    assert m.regression_metrics(np.array([]), np.array([])) == {}


def test_calibrate_classifier_not_available_when_too_few_rows() -> None:
    result = m.calibrate_classifier(
        estimator=object(),
        x_calibration=pd.DataFrame({"a": [1, 2]}),
        y_calibration=pd.Series([0, 1]),
        x_eval=pd.DataFrame({"a": [1, 2]}),
        y_eval=pd.Series([0, 1]),
        n_classes=2,
    )
    assert result["status"] == "NOT_AVAILABLE"
    assert "insufficient rows" in result["reason"]


def test_feature_importance_missing_attribute_reports_not_available() -> None:
    class NoImportance:
        pass

    result = m.feature_importance(NoImportance(), ["f1", "f2"])
    assert result["status"] == "NOT_AVAILABLE"


def test_feature_importance_ranks_and_reports_shap_unavailable() -> None:
    class FakeEstimator:
        feature_importances_ = np.array([10.0, 30.0, 60.0])

    result = m.feature_importance(FakeEstimator(), ["low", "mid", "high"], top_n=2)
    assert result["status"] == "AVAILABLE"
    assert [row["feature"] for row in result["top_features"]] == ["high", "mid"]
    assert result["top_features"][0]["importance_pct"] == pytest.approx(0.6)
    assert result["shap"]["status"] == "NOT_AVAILABLE"


def test_per_symbol_direction_accuracy_groups_correctly() -> None:
    target = pd.Series([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])  # BTC: 1/2 correct, ETH: 2/2 correct
    symbols = pd.Series(["BTC", "BTC", "ETH", "ETH"])
    result = m.per_symbol_direction_accuracy(target, predictions, symbols)
    assert result["BTC"]["samples"] == 2
    assert result["BTC"]["accuracy"] == pytest.approx(0.5)
    assert result["ETH"]["accuracy"] == pytest.approx(1.0)
