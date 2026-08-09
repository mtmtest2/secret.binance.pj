"""Tests for the ML diagnostic report generator (module_f_panel/diagnostics.py).

Covers the parts that are cheap to test directly (sanitisation, comparison,
AI summary) plus one full end-to-end run through real (small/fast) model
training, so the JSON/Markdown export path is exercised against real
artifacts rather than only mocks.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_a_data.db_handler import DatabaseHandler
from module_b_features.features import FEATURE_COLUMNS
from module_b_features.labeler import LABEL_ORDER
from module_b_features.processor import ProcessedDataset
from module_c_ml.ml_models import MLSubsystem
from module_f_panel import diagnostics


def test_sanitize_non_finite_replaces_nan_and_inf_only() -> None:
    dirty = {
        "a": float("nan"),
        "b": float("inf"),
        "c": float("-inf"),
        "d": 1.5,
        "e": [1.0, float("nan"), {"f": float("inf")}],
        "g": "NOT_AVAILABLE",
        "h": None,
        "i": 3,
    }
    clean = diagnostics._sanitize_non_finite(dirty)
    assert clean["a"] is None
    assert clean["b"] is None
    assert clean["c"] is None
    assert clean["d"] == 1.5
    assert clean["e"] == [1.0, None, {"f": None}]
    assert clean["g"] == "NOT_AVAILABLE"
    assert clean["h"] is None
    assert clean["i"] == 3
    # The sanitized blob must always be strictly valid JSON (no NaN/Infinity tokens).
    blob = json.dumps(clean)
    assert "NaN" not in blob
    assert "Infinity" not in blob
    json.loads(blob)  # re-parses without error


def test_compare_to_baseline_classifies_improved_and_regressed() -> None:
    baseline = {
        "direction": {"metrics": {"accuracy": 0.50, "log_loss": 1.0}},
        "entry": {"metrics": {"precision": 0.40, "roc_auc": 0.55}},
    }
    current = {
        "direction": {"metrics": {"accuracy": 0.60, "log_loss": 1.2}},  # accuracy up, log_loss up (worse)
        "entry": {"metrics": {"precision": 0.40, "roc_auc": 0.55}},  # unchanged
    }
    rows = diagnostics._compare_to_baseline(current, baseline)
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["direction.metrics.accuracy"]["verdict"] == "improved"
    assert by_metric["direction.metrics.log_loss"]["verdict"] == "regressed"  # lower is better
    assert by_metric["entry.metrics.precision"]["verdict"] == "unchanged"


def test_compare_to_baseline_returns_empty_without_a_prior_report() -> None:
    assert diagnostics._compare_to_baseline({"direction": {}}, None) == []


def test_ai_summary_is_critical_when_a_head_never_trained() -> None:
    report = {
        "direction": {"status": "NOT_TRAINED"},
        "entry": {"status": "TRAINED", "metrics": {"roc_auc": 0.7}},
        "exit": {"status": "TRAINED"},
        "risk": {"status": "TRAINED", "metrics": {"r2": 0.3}},
        "data_quality": {"symbol_exclusions_total": 0},
        "backtest": {"status": "NOT_AVAILABLE"},
    }
    summary = diagnostics._ai_summary(report, comparison=[])
    assert summary["overall_status"] == "CRITICAL"
    assert "direction" in summary["recommended_next_action"]


def test_ai_summary_is_good_with_no_issues_and_no_baseline() -> None:
    report = {
        "direction": {"status": "TRAINED", "metrics": {"balanced_accuracy": 0.7}},
        "entry": {"status": "TRAINED", "metrics": {"roc_auc": 0.65}},
        "exit": {"status": "TRAINED"},
        "risk": {"status": "TRAINED", "metrics": {"r2": 0.6}},
        "data_quality": {"symbol_exclusions_total": 0},
        "backtest": {"status": "NOT_AVAILABLE"},
    }
    summary = diagnostics._ai_summary(report, comparison=[])
    assert summary["overall_status"] == "GOOD"
    assert summary["weakest_component"] == "risk"  # lowest of 0.7/0.65/0.6
    assert summary["strongest_component"] == "direction"


def _synthetic_dataset(rng: np.random.Generator, n: int = 900) -> ProcessedDataset:
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
    symbols_col = rng.choice(["BTC/USDT:USDT", "ETH/USDT:USDT"], size=n)
    metadata = pd.DataFrame({"symbol": symbols_col, "timestamp": np.arange(n) * 300_000})
    return ProcessedDataset(
        features=features,
        direction_target=direction_target,
        entry_target=entry_target,
        exit_targets=exit_targets,
        risk_target=risk_target,
        metadata=metadata,
        symbols=("BTC/USDT:USDT", "ETH/USDT:USDT"),
        feature_columns=tuple(FEATURE_COLUMNS),
        total_candidate_rows=n + 40,
        rejected_invalid_label_rows=25,
        dropped_missing_or_inf_rows=15,
        duplicate_feature_rows=2,
    )


@pytest.mark.asyncio
async def test_build_report_end_to_end_is_valid_json_and_markdown(tmp_path) -> None:
    settings = Settings(
        ml={"model_dir": tmp_path / "models", "n_estimators": 10, "early_stopping_rounds": 5},
        db={"path": tmp_path / "test.db"},
    )
    database = DatabaseHandler(settings)
    await database.initialize()

    dataset = _synthetic_dataset(np.random.default_rng(3))
    ml = MLSubsystem(settings)
    await ml.train_all(dataset)

    report = await diagnostics.build_report(
        settings=settings, database=database, ml=ml, dataset=dataset, run_id="unit-test-run"
    )

    # Nothing fabricated: sections with no source data are explicitly marked.
    assert report["walk_forward"]["status"] == "NOT_AVAILABLE"
    assert report["backtest"]["status"] == "NOT_AVAILABLE"
    assert report["run"]["run_id"] == "unit-test-run"
    assert report["dataset"]["valid_samples"] == len(dataset)
    assert report["dataset"]["total_candidate_rows"] == dataset.total_candidate_rows
    assert report["direction"]["status"] == "TRAINED"
    assert "confusion_matrix" in report["direction"]["metrics"]

    # Strict JSON round-trip (would fail if any NaN/Infinity token leaked through).
    blob = json.dumps(report)
    assert "NaN" not in blob and "Infinity" not in blob
    reloaded = json.loads(blob)
    assert reloaded["run"]["run_id"] == "unit-test-run"

    markdown = diagnostics.render_markdown(report)
    assert "# AI Diagnostic Summary" in markdown
    assert "# ML Diagnostic Report" in markdown

    # The report was persisted to disk and recorded as the new baseline pointer.
    pointer = await database.get_state(diagnostics.LATEST_REPORT_STATE_KEY)
    assert pointer is not None
    assert pointer["run_id"] == "unit-test-run"

    await database.close()


@pytest.mark.asyncio
async def test_build_report_second_run_compares_against_first(tmp_path) -> None:
    settings = Settings(
        ml={"model_dir": tmp_path / "models", "n_estimators": 10, "early_stopping_rounds": 5},
        db={"path": tmp_path / "test.db"},
    )
    database = DatabaseHandler(settings)
    await database.initialize()

    dataset = _synthetic_dataset(np.random.default_rng(9))
    ml = MLSubsystem(settings)
    await ml.train_all(dataset)

    first = await diagnostics.build_report(
        settings=settings, database=database, ml=ml, dataset=dataset, run_id="run-1"
    )
    assert first["comparison_to_previous_baseline"] == []  # nothing to compare against yet

    second = await diagnostics.build_report(
        settings=settings, database=database, ml=ml, dataset=dataset, run_id="run-2"
    )
    # Same model/dataset retrained -> comparison rows should exist and be self-consistent.
    assert len(second["comparison_to_previous_baseline"]) > 0
    for row in second["comparison_to_previous_baseline"]:
        assert row["verdict"] in ("improved", "regressed", "unchanged")
        assert not math.isnan(row["before"])
        assert not math.isnan(row["after"])

    await database.close()
