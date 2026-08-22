"""The diagnostic must report the problems its own data already describes.

Audit findings P3, P17, P18, P24, P27 and P29 were all the same failure: the
report computed rich evidence - per-feature null counts, drift in train-sigma
units, per-fold walk-forward metrics, duplicate counts, split coverage - and
then judged the run's health from three scalars that could not see any of it.
A run with four all-NaN feature columns, 72% degenerate training rows and a
walk-forward accuracy that fell monotonically from 0.80 to 0.46 summarised as
"Overall status: GOOD, biggest data problem: none measured".

These tests drive the check registry with reports shaped like that one.
"""

from __future__ import annotations

import numpy as np
import pytest

from module_c_ml import metrics as ml_metrics
from module_f_panel import diagnostics


def _report(**overrides):
    """A minimally valid report that a naive checker would call healthy."""
    base = {
        "direction": {"status": "TRAINED", "metrics": {"balanced_accuracy": 0.60}},
        "entry": {"status": "TRAINED", "metrics": {"roc_auc": 0.70}},
        "exit": {"status": "TRAINED", "metrics": {}},
        "risk": {"status": "TRAINED", "metrics": {"r2": 0.40}},
        "data_quality": {"symbol_exclusions_total": 0},
        "backtest": {"status": "NOT_AVAILABLE"},
        "dataset": {"valid_samples": 1000, "null_counts_by_feature": {}},
        "features": {},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# P3 - a feature that is NaN on every row is not a feature
# --------------------------------------------------------------------------
def test_all_nan_features_force_critical_and_are_named() -> None:
    report = _report(
        dataset={
            "valid_samples": 1000,
            "null_counts_by_feature": {"ob_imbalance": 1000, "ob_spread_bps": 1000, "rsi": 3},
        }
    )
    summary = diagnostics._ai_summary(report, comparison=[])
    assert summary["overall_status"] == "CRITICAL"
    assert "ob_imbalance" in summary["biggest_data_problem"]
    assert "ob_spread_bps" in summary["biggest_data_problem"]
    # A merely sparse column is not swept up with the dead ones.
    assert "rsi" not in summary["biggest_data_problem"]

    recommendations = diagnostics._recommendations(report, comparison=[])
    assert any("effectively empty" in line for line in recommendations["CRITICAL"])


def test_a_healthy_dataset_reports_no_data_problem() -> None:
    summary = diagnostics._ai_summary(_report(), comparison=[])
    assert summary["biggest_data_problem"] == "none measured"
    assert summary["overall_status"] == "GOOD"


# --------------------------------------------------------------------------
# P2 detector - drift the report measured but never read back
# --------------------------------------------------------------------------
def test_severe_feature_drift_is_escalated() -> None:
    report = _report(
        features={
            "drift": {
                "most_drifted_features": [
                    {"feature": "wick_ratio", "mean_shift_in_train_std": 2.35},
                    {"feature": "adx", "mean_shift_in_train_std": 1.34},
                ]
            }
        }
    )
    summary = diagnostics._ai_summary(report, comparison=[])
    assert summary["overall_status"] == "CRITICAL"
    assert "wick_ratio" in summary["biggest_data_problem"]
    assert "2.35" in summary["biggest_data_problem"]


def test_mild_drift_warns_rather_than_failing() -> None:
    report = _report(
        features={"drift": {"most_drifted_features": [{"feature": "adx", "mean_shift_in_train_std": 1.1}]}}
    )
    summary = diagnostics._ai_summary(report, comparison=[])
    assert summary["overall_status"] == "WARNING"


def test_drift_below_the_threshold_is_not_reported() -> None:
    report = _report(
        features={"drift": {"most_drifted_features": [{"feature": "adx", "mean_shift_in_train_std": 0.4}]}}
    )
    assert diagnostics._ai_summary(report, comparison=[])["overall_status"] == "GOOD"


# --------------------------------------------------------------------------
# P18 - a monotone walk-forward collapse must not read as "none measured"
# --------------------------------------------------------------------------
def test_monotone_walk_forward_decay_is_named_as_a_trend() -> None:
    walk_forward = {
        "status": "AVAILABLE",
        "n_folds": 4,
        "accuracy_std": 0.131,
        "folds": [
            {"fold": 1, "accuracy": 0.8034},
            {"fold": 2, "accuracy": 0.6680},
            {"fold": 3, "accuracy": 0.5322},
            {"fold": 4, "accuracy": 0.4623},
        ],
    }
    line = diagnostics._walk_forward_problem_summary(walk_forward)
    assert not line.startswith("none measured")
    assert "monoton" in line
    assert "0.462" in line  # the latest fold, i.e. the honest read

    report = _report(walk_forward=walk_forward)
    recommendations = diagnostics._recommendations(report, comparison=[])
    assert any("Walk-forward" in line for line in recommendations["CRITICAL"])


def test_wide_but_unordered_walk_forward_spread_is_still_reported() -> None:
    walk_forward = {
        "status": "AVAILABLE",
        "n_folds": 4,
        "accuracy_std": 0.12,
        "folds": [
            {"fold": 1, "accuracy": 0.50},
            {"fold": 2, "accuracy": 0.75},
            {"fold": 3, "accuracy": 0.52},
            {"fold": 4, "accuracy": 0.70},
        ],
    }
    line = diagnostics._walk_forward_problem_summary(walk_forward)
    assert "varies widely" in line


def test_stable_walk_forward_stays_quiet() -> None:
    walk_forward = {
        "status": "AVAILABLE",
        "n_folds": 4,
        "accuracy_std": 0.01,
        "folds": [{"fold": i, "accuracy": 0.60 + 0.005 * (i % 2)} for i in range(1, 5)],
    }
    assert diagnostics._walk_forward_problem_summary(walk_forward).startswith("none measured")


def test_spearman_rho_detects_a_perfect_decline() -> None:
    assert diagnostics._spearman_rho([0.8, 0.67, 0.53, 0.46]) == pytest.approx(-1.0)
    assert diagnostics._spearman_rho([0.4, 0.5, 0.6, 0.7]) == pytest.approx(1.0)
    assert np.isnan(diagnostics._spearman_rho([0.5, 0.6]))


# --------------------------------------------------------------------------
# P24 - a sweep whose accuracy rises only by going silent
# --------------------------------------------------------------------------
def test_degenerate_confidence_sweep_is_flagged() -> None:
    report = _report(
        direction={
            "status": "TRAINED",
            "metrics": {
                "balanced_accuracy": 0.36,
                "confidence_threshold_analysis": [
                    {"confidence_threshold": 0.30, "accuracy": 0.46, "is_degenerate": False},
                    {"confidence_threshold": 0.70, "accuracy": 0.9835, "is_degenerate": True},
                ],
            },
        }
    )
    summary = diagnostics._ai_summary(report, comparison=[])
    assert summary["overall_status"] == "WARNING"
    assert any("single class" in check["message"] for check in summary["data_health_checks"])


def test_direction_metrics_marks_single_class_slices_as_degenerate() -> None:
    """The flag is produced where the sweep is computed, not inferred later."""
    rows = 400
    rng = np.random.default_rng(3)
    target = rng.integers(0, 3, size=rows)
    # Very high mass on class 2 -> the high-confidence slice contains only class 2.
    probabilities = np.tile(np.array([0.05, 0.05, 0.90]), (rows, 1))
    result = ml_metrics.direction_metrics(
        __import__("pandas").Series(target), probabilities, ["LONG", "SHORT", "NO_TRADE"]
    )
    high_rows = [r for r in result["confidence_threshold_analysis"] if r["confidence_threshold"] >= 0.85]
    assert high_rows and all(r["is_degenerate"] for r in high_rows)
    assert all(r["distinct_predicted_classes"] == 1 for r in high_rows)


# --------------------------------------------------------------------------
# P27/P29 - label shift and duplicates the report already counted
# --------------------------------------------------------------------------
def test_label_distribution_shift_is_surfaced() -> None:
    report = _report(
        direction={
            "status": "TRAINED",
            "metrics": {"balanced_accuracy": 0.6},
            "distribution": {
                "train": {"LONG_SUCCESS": 700, "SHORT_SUCCESS": 300},
                "validation": {"LONG_SUCCESS": 400, "SHORT_SUCCESS": 600},
                "label_distribution_shift": 0.30,
            },
        }
    )
    summary = diagnostics._ai_summary(report, comparison=[])
    assert any(check["id"] == "label_shift" for check in summary["data_health_checks"])


def test_duplicate_rows_and_over_capacity_split_are_reported() -> None:
    report = _report(
        dataset={
            "valid_samples": 1000,
            "null_counts_by_feature": {},
            "duplicate_feature_rows": 1141,
            "split_coverage_pct": {"validation": {"rows": 1418472, "capacity_rows": 1418445}},
        }
    )
    summary = diagnostics._ai_summary(report, comparison=[])
    ids = {check["id"] for check in summary["data_health_checks"]}
    assert "duplicate_rows" in ids
    assert "split_over_capacity_validation" in ids


# --------------------------------------------------------------------------
# P16 - calibration must clear a material effect size
# --------------------------------------------------------------------------
def test_calibration_gain_threshold_rejects_noise_and_accepts_a_real_gain() -> None:
    assert ml_metrics.MIN_CALIBRATION_RELATIVE_GAIN > 0.0

    # The audited gate "improvement": 0.6387282 -> 0.6380239, i.e. 0.11%.
    noise_gain = (0.6387282315683127 - 0.6380239320070472) / 0.6387282315683127
    assert noise_gain < ml_metrics.MIN_CALIBRATION_RELATIVE_GAIN

    real_gain = (0.70 - 0.60) / 0.70
    assert real_gain > ml_metrics.MIN_CALIBRATION_RELATIVE_GAIN


# --------------------------------------------------------------------------
# Split capacity: bars, not intervals
# --------------------------------------------------------------------------
def _dataset_with(timestamps: np.ndarray, symbols: tuple[str, ...]):
    import pandas as pd

    from module_b_features.features import FEATURE_COLUMNS
    from module_b_features.processor import ProcessedDataset

    rows = len(timestamps)
    return ProcessedDataset(
        features=pd.DataFrame(0.0, index=range(rows), columns=list(FEATURE_COLUMNS)),
        direction_target=pd.Series(["NO_TRADE_OR_FAIL"] * rows),
        entry_target=pd.Series([0] * rows),
        exit_targets=pd.DataFrame({"target_tp_pct": [0.02] * rows, "target_sl_pct": [0.01] * rows}),
        risk_target=pd.Series([0.5] * rows),
        metadata=pd.DataFrame({"timestamp": timestamps, "symbol": "X/USDT:USDT"}),
        feature_columns=tuple(FEATURE_COLUMNS),
        symbols=symbols,
        total_candidate_rows=rows,
    )


def test_split_capacity_counts_bars_inclusively() -> None:
    """A gap-free block must report exactly 100% coverage, not slightly over.

    Capacity was `span / bar_ms`, which counts the gaps between bars rather than
    the bars themselves - under-counting by one row per symbol. On the audited
    27-symbol run that produced a 27-row "overflow" which read as duplicated
    timestamps when it was really this arithmetic.
    """
    bar_ms, bars, symbol_count = 300_000, 50, 3
    timestamps = np.tile(np.arange(bars, dtype=np.int64) * bar_ms, symbol_count)
    dataset = _dataset_with(timestamps, tuple(f"S{i}/USDT:USDT" for i in range(symbol_count)))

    coverage = diagnostics._split_coverage(
        dataset, np.arange(len(timestamps)), np.array([], dtype=int), np.array([], dtype=int)
    )
    assert coverage["train"]["capacity_rows"] == bars * symbol_count
    assert coverage["train"]["coverage_pct"] == 1.0


def test_split_capacity_reports_over_one_when_timestamps_repeat() -> None:
    """And the ratio is not clamped, because >1.0 is the duplicate detector."""
    bar_ms, bars = 300_000, 20
    timestamps = np.concatenate(
        [np.arange(bars, dtype=np.int64) * bar_ms, np.array([0, bar_ms], dtype=np.int64)]
    )
    dataset = _dataset_with(timestamps, ("X/USDT:USDT",))

    coverage = diagnostics._split_coverage(
        dataset, np.arange(len(timestamps)), np.array([], dtype=int), np.array([], dtype=int)
    )
    assert coverage["train"]["coverage_pct"] > 1.0
