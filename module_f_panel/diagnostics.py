"""ML Diagnostic Report generator (spec Parts 27-30).

Assembles one structured report per training run from real, already-computed
artifacts:

* the per-head metrics JSON sidecars written by
  :mod:`module_c_ml.ml_models` (``BaseModelHead.save``),
* QC/healing telemetry and per-cycle pipeline timings persisted to durable
  state by :mod:`module_a_data.pipeline` and ``main.TradingSystem``,
* dataset-health counters captured directly on the
  :class:`~module_b_features.processor.ProcessedDataset` used for training.

Nothing here re-computes a metric, estimates a number, or fills in a plausible
guess. Any section whose source data does not exist is set to the literal
string ``"NOT_AVAILABLE"`` - see the project's diagnostic reporting rules.

The report is both human-readable (:func:`render_markdown`) and
machine-readable (the JSON returned by :func:`build_report` itself), with an
AI-ready summary generated purely from measured thresholds at the top.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from config.settings import Settings
from core.logger import get_logger
from core.utils import git_commit_hash, utc_now_ms
from module_a_data.db_handler import DatabaseHandler
from module_a_data.pipeline import HEAL_TELEMETRY_STATE_KEY, SYMBOL_EXCLUSION_STATE_KEY
from module_b_features.processor import ProcessedDataset
from module_c_ml.ml_models import MLSubsystem

_LOGGER = get_logger(__name__)

NOT_AVAILABLE: Final[str] = "NOT_AVAILABLE"
LATEST_REPORT_STATE_KEY: Final[str] = "ml_diagnostic_latest_report"
REPORTS_DIR_NAME: Final[str] = "reports"

#: Feature -> group, matching the audited 53-feature inventory
#: (module_b_features/features.py::FEATURE_COLUMNS). Kept here rather than in
#: the feature module itself since grouping is purely a reporting concern.
FEATURE_GROUPS: Final[dict[str, str]] = {
    "kama_distance": "Trend",
    "kama_slope": "Trend",
    "kama_slope_fast": "Trend",
    "ema_fast_slow_spread": "Trend",
    "close_ema_slow_ratio": "Trend",
    "adx": "Trend",
    "di_spread": "Trend",
    "fdi": "Regime",
    "fdi_trending": "Regime",
    "fdi_delta": "Regime",
    "hmm_regime": "Regime",
    "hmm_prob_bull": "Regime",
    "hmm_prob_bear": "Regime",
    "hmm_prob_high_vol": "Regime",
    "hmm_prob_sideways": "Regime",
    "hmm_regime_age": "Regime",
    "bb_width": "Volatility",
    "bb_position": "Volatility",
    "atr_pct": "Volatility",
    "atr_rank": "Volatility",
    "realized_vol_12": "Volatility",
    "realized_vol_48": "Volatility",
    "garch_volatility": "Volatility",
    "garch_vol_rank": "Volatility",
    "garch_vol_ratio": "Volatility",
    "vol_of_vol": "Volatility",
    "rsi": "Momentum",
    "rsi_delta": "Momentum",
    "log_return_1": "Momentum",
    "log_return_3": "Momentum",
    "log_return_12": "Momentum",
    "log_return_48": "Momentum",
    "momentum_rank": "Momentum",
    "volume_zscore": "Volume",
    "volume_rank": "Volume",
    "volume_trend": "Volume",
    "dollar_volume_rank": "Volume",
    "ob_imbalance": "Microstructure",
    "ob_imbalance_delta": "Microstructure",
    "ob_spread_bps": "Microstructure",
    "ob_spread_rank": "Microstructure",
    "funding_rate": "Derivatives",
    "funding_rate_delta": "Derivatives",
    "funding_rate_rank": "Derivatives",
    "open_interest_change": "Derivatives",
    "open_interest_rank": "Derivatives",
    "long_short_ratio": "Derivatives",
    "taker_buy_sell_ratio": "Derivatives",
    "liquidation_imbalance": "Derivatives",
    "hour_sin": "Time/Seasonality",
    "hour_cos": "Time/Seasonality",
    "dow_sin": "Time/Seasonality",
    "dow_cos": "Time/Seasonality",
}

#: (report path, higher_is_better) for the before/after comparison table.
_COMPARISON_METRICS: Final[tuple[tuple[str, bool], ...]] = (
    ("direction.metrics.accuracy", True),
    ("direction.metrics.balanced_accuracy", True),
    ("direction.metrics.log_loss", False),
    ("entry.metrics.precision", True),
    ("entry.metrics.roc_auc", True),
    ("risk.metrics.r2", True),
    ("backtest.metrics.profit_factor", True),
    ("backtest.metrics.max_drawdown_pct", False),
    ("backtest.metrics.net_profit", True),
    ("backtest.metrics.win_rate", True),
    ("backtest.metrics.expectancy", True),
)


def _get_path(report: dict[str, Any], dotted_path: str) -> Any:
    """Walk a dotted path through nested dicts; ``None`` on any miss."""
    node: Any = report
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return node
    return None


def _feature_statistics(dataset: ProcessedDataset) -> dict[str, Any]:
    """Per-feature descriptive statistics computed directly from the training frame."""
    features: pd.DataFrame = dataset.features
    if features.empty:
        return {"status": NOT_AVAILABLE, "reason": "empty dataset"}

    described: pd.DataFrame = features.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    per_feature: dict[str, Any] = {}
    for column in dataset.feature_columns:
        series: pd.Series = features[column]
        stats_row = described.loc[column] if column in described.index else None
        per_feature[column] = {
            "group": FEATURE_GROUPS.get(column, "Other"),
            "missing_rate": float(series.isna().mean()),
            "infinite_rate": float(np.isinf(series.to_numpy(dtype=np.float64)).mean()),
            "mean": float(stats_row["mean"]) if stats_row is not None else None,
            "std": float(stats_row["std"]) if stats_row is not None else None,
            "min": float(stats_row["min"]) if stats_row is not None else None,
            "max": float(stats_row["max"]) if stats_row is not None else None,
            "p1": float(stats_row["1%"]) if stats_row is not None else None,
            "p5": float(stats_row["5%"]) if stats_row is not None else None,
            "median": float(stats_row["50%"]) if stats_row is not None else None,
            "p95": float(stats_row["95%"]) if stats_row is not None else None,
            "p99": float(stats_row["99%"]) if stats_row is not None else None,
            "unique_count": int(series.nunique()),
        }
    return {"status": "AVAILABLE", "per_feature": per_feature}


def _feature_correlation(dataset: ProcessedDataset, top_n: int = 25) -> dict[str, Any]:
    """The strongest pairwise feature correlations - full N x N matrices are
    omitted deliberately: for 53 features that is 1,378 pairs, most of them
    uninformative noise that would bloat the report without helping diagnosis.
    """
    if dataset.features.empty or len(dataset.features) < 5:
        return {"status": NOT_AVAILABLE, "reason": "not enough rows to correlate"}
    correlation: pd.DataFrame = dataset.features.corr(numeric_only=True)
    pairs: list[dict[str, Any]] = []
    columns: list[str] = list(correlation.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value):
                pairs.append({"feature_a": left, "feature_b": right, "correlation": float(value)})
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return {"status": "AVAILABLE", "top_correlated_pairs": pairs[:top_n]}


def _feature_drift(dataset: ProcessedDataset, validation_index: np.ndarray) -> dict[str, Any]:
    """Train vs. validation distribution drift per feature (mean/std/quantile).

    A model can look worse purely because the validation period's market
    regime differs from training, not because the model got worse - this is
    the check that tells the two apart (spec Part 24).
    """
    features: pd.DataFrame = dataset.features
    if features.empty or len(validation_index) == 0:
        return {"status": NOT_AVAILABLE, "reason": "empty dataset or validation slice"}

    train_mask: np.ndarray = np.ones(len(features), dtype=bool)
    train_mask[validation_index] = False
    train_frame: pd.DataFrame = features[train_mask]
    validation_frame: pd.DataFrame = features.iloc[validation_index]
    if train_frame.empty or validation_frame.empty:
        return {"status": NOT_AVAILABLE, "reason": "empty train or validation slice"}

    drifted: list[dict[str, Any]] = []
    for column in dataset.feature_columns:
        train_series, val_series = train_frame[column], validation_frame[column]
        train_std, val_std = float(train_series.std()), float(val_series.std())
        train_mean, val_mean = float(train_series.mean()), float(val_series.mean())
        mean_shift_in_std: float = (
            abs(val_mean - train_mean) / train_std if train_std > 1e-12 else float("nan")
        )
        variance_ratio: float = (val_std / train_std) if train_std > 1e-12 else float("nan")
        median_shift: float = float(val_series.median() - train_series.median())
        drifted.append(
            {
                "feature": column,
                "train_mean": train_mean,
                "validation_mean": val_mean,
                "mean_shift_in_train_std": mean_shift_in_std,
                "train_std": train_std,
                "validation_std": val_std,
                "variance_ratio": variance_ratio,
                "median_shift": median_shift,
            }
        )
    drifted.sort(
        key=lambda item: item["mean_shift_in_train_std"]
        if item["mean_shift_in_train_std"] == item["mean_shift_in_train_std"]
        else -1.0,
        reverse=True,
    )
    return {
        "status": "AVAILABLE",
        "method": "train-vs-validation mean/variance shift, in train-std units",
        "most_drifted_features": drifted[:20],
    }


async def _qc_telemetry(database: DatabaseHandler) -> dict[str, Any]:
    """Real, persisted QC/healing history - see module_a_data.pipeline."""
    heal_state: dict[str, Any] | None = await database.get_state(HEAL_TELEMETRY_STATE_KEY)
    exclusion_state: dict[str, Any] | None = await database.get_state(SYMBOL_EXCLUSION_STATE_KEY)
    heal_records: list[dict[str, Any]] = list((heal_state or {}).get("records", []))
    exclusion_records: list[dict[str, Any]] = list((exclusion_state or {}).get("records", []))

    result_counts: dict[str, int] = {}
    for record in heal_records:
        result_counts[record.get("result", "unknown")] = (
            result_counts.get(record.get("result", "unknown"), 0) + 1
        )

    return {
        "status": "AVAILABLE" if (heal_records or exclusion_records) else "NO_HEALING_NEEDED",
        "heal_attempts_total": len(heal_records),
        "heal_attempts_by_result": result_counts,
        "recent_heal_attempts": heal_records[-100:],
        "symbol_exclusions_total": len(exclusion_records),
        "recent_symbol_exclusions": exclusion_records[-100:],
    }


async def _cycle_timings(database: DatabaseHandler) -> dict[str, Any]:
    """Real, persisted per-stage cycle timing history - see main.TradingSystem."""
    stored: dict[str, Any] | None = await database.get_state("cycle_timings_history")
    records: list[dict[str, Any]] = list((stored or {}).get("records", []))
    if not records:
        return {"status": NOT_AVAILABLE, "reason": "no completed trading cycle yet"}

    frame: pd.DataFrame = pd.DataFrame(records)
    stage_columns: list[str] = [
        column for column in frame.columns if column not in ("cycle_id", "at")
    ]
    per_stage: dict[str, Any] = {}
    for column in stage_columns:
        series: pd.Series = frame[column].dropna()
        if series.empty:
            continue
        per_stage[column] = {
            "mean_seconds": float(series.mean()),
            "median_seconds": float(series.median()),
            "max_seconds": float(series.max()),
            "min_seconds": float(series.min()),
        }
    return {"status": "AVAILABLE", "cycles_measured": len(records), "per_stage": per_stage}


def _label_configuration(settings: Settings) -> dict[str, Any]:
    labels = settings.labels
    return {
        "tp_atr_multiple": labels.tp_atr_multiple,
        "sl_atr_multiple": labels.sl_atr_multiple,
        "max_holding_bars": labels.max_holding_bars,
        "low_risk_mae_ratio": labels.low_risk_mae_ratio,
        "medium_risk_mae_ratio": getattr(labels, "medium_risk_mae_ratio", None),
        "high_risk_mae_ratio": getattr(labels, "high_risk_mae_ratio", None),
        "discard_very_high_risk": getattr(labels, "discard_very_high_risk", None),
    }


def _walk_forward_problem_summary(walk_forward: dict[str, Any]) -> str:
    """One-line, measurement-derived read on walk-forward validation health."""
    if not isinstance(walk_forward, dict) or walk_forward.get("status") != "AVAILABLE":
        return "walk-forward evaluation not available (single train/validation split only)"
    std = walk_forward.get("accuracy_std")
    folds = walk_forward.get("n_folds", "?")
    if isinstance(std, (int, float)):
        return f"none measured (walk-forward across {folds} folds, accuracy std={std:.3f})"
    return f"none measured (walk-forward across {folds} folds)"


def _ai_summary(report: dict[str, Any], comparison: list[dict[str, Any]]) -> dict[str, Any]:
    """Rule-based AI-ready summary - every field is derived from measured
    values already present in ``report``, never invented (spec Part 29)."""
    warnings: list[str] = []
    critical: list[str] = []

    heads: dict[str, dict[str, Any]] = {
        "direction": report.get("direction", {}),
        "entry": report.get("entry", {}),
        "exit": report.get("exit", {}),
        "risk": report.get("risk", {}),
    }
    for name, head in heads.items():
        if head.get("status") == "NOT_TRAINED":
            critical.append(f"{name} model has no trained artifact")

    dq = report.get("data_quality", {})
    if isinstance(dq.get("symbol_exclusions_total"), int) and dq["symbol_exclusions_total"] > 0:
        warnings.append(f"{dq['symbol_exclusions_total']} symbol-cycle exclusion(s) recorded")

    direction_metrics: dict[str, Any] = heads["direction"].get("metrics", {}) or {}
    entry_metrics: dict[str, Any] = heads["entry"].get("metrics", {}) or {}
    scored: dict[str, float] = {}
    if isinstance(direction_metrics.get("balanced_accuracy"), (int, float)):
        scored["direction"] = float(direction_metrics["balanced_accuracy"])
    if isinstance(entry_metrics.get("roc_auc"), (int, float)) and entry_metrics["roc_auc"] == entry_metrics["roc_auc"]:
        scored["entry"] = float(entry_metrics["roc_auc"])
    risk_metrics: dict[str, Any] = heads["risk"].get("metrics", {}) or {}
    if isinstance(risk_metrics.get("r2"), (int, float)) and risk_metrics["r2"] == risk_metrics["r2"]:
        scored["risk"] = float(risk_metrics["r2"])

    strongest: str = max(scored, key=scored.get) if scored else NOT_AVAILABLE
    weakest: str = min(scored, key=scored.get) if scored else NOT_AVAILABLE

    improvements: list[str] = [row["metric"] for row in comparison if row.get("verdict") == "improved"]
    regressions: list[str] = [row["metric"] for row in comparison if row.get("verdict") == "regressed"]

    if critical:
        status = "CRITICAL"
    elif warnings or regressions:
        status = "WARNING"
    else:
        status = "GOOD"

    next_action: str
    if critical:
        next_action = f"Fix before anything else: {critical[0]}"
    elif regressions:
        next_action = f"Investigate the regression in {regressions[0]} before promoting this run"
    elif weakest != NOT_AVAILABLE and scored.get(weakest, 1.0) < 0.55:
        next_action = f"Improve the {weakest} model - it is the weakest measured component"
    else:
        next_action = "No critical issues measured; continue with the walk-forward/backtest checklist"

    return {
        "overall_status": status,
        "strongest_component": strongest,
        "weakest_component": weakest,
        "biggest_data_problem": (
            f"{dq['symbol_exclusions_total']} symbol exclusion(s) this run"
            if isinstance(dq.get("symbol_exclusions_total"), int) and dq["symbol_exclusions_total"] > 0
            else "none measured"
        ),
        "biggest_ml_problem": f"{weakest} is the weakest scored component" if scored else NOT_AVAILABLE,
        "biggest_validation_problem": _walk_forward_problem_summary(report.get("walk_forward", {})),
        "biggest_trading_problem": (
            "backtest not available for this run" if report.get("backtest", {}).get("status") == NOT_AVAILABLE
            else NOT_AVAILABLE
        ),
        "most_important_metric_improvement": improvements[0] if improvements else NOT_AVAILABLE,
        "most_important_metric_degradation": regressions[0] if regressions else NOT_AVAILABLE,
        "recommended_next_action": next_action,
    }


def _compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Metric-by-metric before/after table against the previous stored report."""
    if baseline is None:
        return []
    rows: list[dict[str, Any]] = []
    for path, higher_is_better in _COMPARISON_METRICS:
        before = _get_path(baseline, path)
        after = _get_path(current, path)
        if before is None or after is None:
            continue
        change: float = after - before
        if abs(change) < 1e-12:
            verdict = "unchanged"
        elif (change > 0) == higher_is_better:
            verdict = "improved"
        else:
            verdict = "regressed"
        rows.append(
            {
                "metric": path,
                "higher_is_better": higher_is_better,
                "before": before,
                "after": after,
                "change": change,
                "verdict": verdict,
            }
        )
    return rows


def _head_section(ml: MLSubsystem, name: str) -> dict[str, Any]:
    head = ml.heads[name]
    if not head.is_loaded:
        return {"status": "NOT_TRAINED"}
    metadata: dict[str, Any] = head.metadata
    return {"status": "TRAINED", **metadata}


async def build_report(
    *,
    settings: Settings,
    database: DatabaseHandler,
    ml: MLSubsystem,
    dataset: ProcessedDataset,
    run_id: str,
    backtest: Any | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the full ML diagnostic report for a just-completed training run."""
    train_index, validation_index = dataset.train_validation_split(
        settings.ml.validation_fraction, settings.ml.purge_bars
    )
    timestamps: pd.Series = (
        dataset.metadata["timestamp"] if "timestamp" in dataset.metadata.columns else pd.Series(dtype="int64")
    )

    def _period(index: np.ndarray) -> dict[str, Any]:
        if len(index) == 0 or timestamps.empty:
            return {"start": None, "end": None, "rows": 0}
        subset = timestamps.iloc[index]
        return {
            "start": int(subset.min()),
            "end": int(subset.max()),
            "rows": int(len(index)),
        }

    per_symbol_rows: dict[str, int] = (
        dataset.metadata["symbol"].value_counts().to_dict() if "symbol" in dataset.metadata.columns else {}
    )

    report: dict[str, Any] = {
        "run": {
            "run_id": run_id,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit_hash(),
            "timeframe": settings.data.timeframe,
            "symbols": list(dataset.symbols),
            "model_versions": ml.versions(),
            "training_period": _period(train_index),
            "validation_period": _period(validation_index),
            "test_period": {"start": None, "end": None, "rows": 0, "note": "no held-out test split configured"},
            "purge_bars": settings.ml.purge_bars,
            "embargo_bars": 0,
        },
        "dataset": {
            "total_candidate_rows": dataset.total_candidate_rows,
            "valid_samples": len(dataset),
            "rejected_invalid_label_rows": dataset.rejected_invalid_label_rows,
            "dropped_missing_or_inf_rows": dataset.dropped_missing_or_inf_rows,
            "duplicate_feature_rows": dataset.duplicate_feature_rows,
            "training_samples": int(len(train_index)),
            "validation_samples": int(len(validation_index)),
            "test_samples": 0,
            "feature_count": len(dataset.feature_columns),
            "feature_names": list(dataset.feature_columns),
            "per_symbol_rows": {str(k): int(v) for k, v in per_symbol_rows.items()},
        },
        "data_quality": await _qc_telemetry(database),
        "features": {
            "statistics": _feature_statistics(dataset),
            "correlation": _feature_correlation(dataset),
            "drift": _feature_drift(dataset, validation_index),
        },
        "labels": {
            "configuration": _label_configuration(settings),
            "class_distribution": dataset.class_distribution(),
        },
        "direction": _head_section(ml, "direction"),
        "entry": _head_section(ml, "entry"),
        "exit": _head_section(ml, "exit"),
        "risk": _head_section(ml, "risk"),
        "walk_forward": (
            ml.direction.walk_forward(dataset)
            if ml.direction.is_loaded
            else {"status": NOT_AVAILABLE, "reason": "direction model is not trained"}
        ),
        "backtest": backtest.to_dict() if backtest is not None else {"status": NOT_AVAILABLE},
        "regimes": {
            "status": NOT_AVAILABLE,
            "reason": "per-regime performance breakdown is not computed by this run",
        },
        "timings": await _cycle_timings(database),
        "warnings": warnings or [],
        "errors": errors or [],
    }
    report["calibration"] = {
        "direction": report["direction"].get("calibration", {"status": NOT_AVAILABLE}),
        "entry": report["entry"].get("calibration", {"status": NOT_AVAILABLE}),
    }
    report["symbols"] = {
        "per_symbol_rows": report["dataset"]["per_symbol_rows"],
        "per_symbol_direction_accuracy": report["direction"].get("per_symbol", {}),
        "note": (
            "Entry/Exit/Risk per-symbol performance is not broken out separately "
            "in this run; only Direction is scored per-symbol."
        ),
    }

    baseline: dict[str, Any] | None = await load_latest_report(database)
    comparison: list[dict[str, Any]] = _compare_to_baseline(report, baseline)
    report["comparison_to_previous_baseline"] = comparison
    report["ai_summary"] = _ai_summary(report, comparison)
    report["recommendations"] = _recommendations(report, comparison)

    report = _sanitize_non_finite(report)
    await _persist_report(settings, database, report)
    return report


def _sanitize_non_finite(node: Any) -> Any:
    """Recursively replace NaN/+-Inf floats with ``None``.

    A degenerate validation slice can genuinely produce a NaN metric (e.g.
    log loss on a single-class block, R^2 on a near-constant target) - that
    is a real computed result, not a bug, and is kept as ``None`` (JSON
    ``null``) rather than dropped or replaced with a fabricated number.
    Plain ``json.dumps``/``JSONResponse`` emit the non-standard ``NaN`` /
    ``Infinity`` tokens for these, which Python's own parser accepts but a
    browser's native ``JSON.parse`` does not - so this keeps every export a
    strictly valid JSON document.
    """
    if isinstance(node, float):
        return node if np.isfinite(node) else None
    if isinstance(node, dict):
        return {key: _sanitize_non_finite(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_sanitize_non_finite(item) for item in node]
    if isinstance(node, tuple):
        return tuple(_sanitize_non_finite(item) for item in node)
    return node


def _recommendations(report: dict[str, Any], comparison: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Prioritised, measurement-derived recommendations - never invented."""
    critical: list[str] = []
    high: list[str] = []
    medium: list[str] = []
    low: list[str] = []

    for name in ("direction", "entry", "exit", "risk"):
        if report.get(name, {}).get("status") == "NOT_TRAINED":
            critical.append(f"Train the {name} model - no artifact is currently available")

    if report.get("walk_forward", {}).get("status") == NOT_AVAILABLE:
        high.append("Add walk-forward evaluation across multiple rolling folds before trusting a single split")
    if report.get("backtest", {}).get("status") == NOT_AVAILABLE:
        high.append("Run a backtest for this artifact set before considering it for paper/live trading")

    for row in comparison:
        if row.get("verdict") == "regressed":
            medium.append(f"{row['metric']} regressed from {row['before']:.4f} to {row['after']:.4f}")

    exit_section = report.get("exit", {})
    exit_metrics = exit_section.get("metrics", {}) if isinstance(exit_section, dict) else {}
    for target, target_metrics in (exit_metrics or {}).items():
        if isinstance(target_metrics, dict) and target_metrics.get("beats_rule_based_baseline") is False:
            medium.append(
                f"Exit target {target} does not beat the rule-based ATR baseline - "
                "consider not shipping this regressor for that target"
            )

    for name in ("direction", "entry"):
        calib = report.get("calibration", {}).get(name, {})
        if calib.get("status") == "AVAILABLE" and calib.get("improved"):
            if report.get(name, {}).get("production_calibration") == "isotonic":
                low.append(f"{name} isotonic calibration measurably improves log loss and is wired into inference")
            else:
                low.append(
                    f"{name} isotonic calibration measurably improves log loss but is not yet "
                    "wired into inference (not enough held-out rows to fit a production calibrator)"
                )

    if not (critical or high or medium or low):
        low.append("No issues detected from measured results this run")

    return {"CRITICAL": critical, "HIGH": high, "MEDIUM": medium, "LOW": low}


async def load_latest_report(database: DatabaseHandler) -> dict[str, Any] | None:
    pointer: dict[str, Any] | None = await database.get_state(LATEST_REPORT_STATE_KEY)
    if not pointer or "path" not in pointer:
        return None
    path = Path(str(pointer["path"]))
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:  # pragma: no cover - corrupt/missing file
        _LOGGER.warning("Could not load the previous ML diagnostic report: %s", error)
        return None


async def _persist_report(settings: Settings, database: DatabaseHandler, report: dict[str, Any]) -> None:
    """Write the report to disk and remember it as the new comparison baseline."""
    reports_dir: Path = settings.ml.model_dir.parent / REPORTS_DIR_NAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_id: str = report["run"]["run_id"]
    json_path: Path = reports_dir / f"ml_diagnostic_{run_id}.json"
    markdown_path: Path = reports_dir / f"ml_diagnostic_{run_id}.md"
    try:
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        await database.set_state(
            LATEST_REPORT_STATE_KEY,
            {"path": str(json_path), "markdown_path": str(markdown_path), "run_id": run_id, "at": utc_now_ms()},
        )
    except OSError as error:  # pragma: no cover - disk full / permissions
        _LOGGER.error("Could not persist the ML diagnostic report: %s", error)


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable Markdown rendering of the full report (spec Part 27/28)."""
    run = report.get("run", {})
    summary = report.get("ai_summary", {})
    lines: list[str] = [
        "# AI Diagnostic Summary",
        "",
        f"- Overall status: **{summary.get('overall_status', NOT_AVAILABLE)}**",
        f"- Strongest component: {summary.get('strongest_component', NOT_AVAILABLE)}",
        f"- Weakest component: {summary.get('weakest_component', NOT_AVAILABLE)}",
        f"- Biggest data problem: {summary.get('biggest_data_problem', NOT_AVAILABLE)}",
        f"- Biggest ML problem: {summary.get('biggest_ml_problem', NOT_AVAILABLE)}",
        f"- Biggest validation problem: {summary.get('biggest_validation_problem', NOT_AVAILABLE)}",
        f"- Biggest trading problem: {summary.get('biggest_trading_problem', NOT_AVAILABLE)}",
        f"- Most important improvement: {summary.get('most_important_metric_improvement', NOT_AVAILABLE)}",
        f"- Most important degradation: {summary.get('most_important_metric_degradation', NOT_AVAILABLE)}",
        f"- Recommended next action: {summary.get('recommended_next_action', NOT_AVAILABLE)}",
        "",
        "# ML Diagnostic Report",
        "",
        f"Run `{run.get('run_id')}` generated {run.get('generated_at')} "
        f"(git `{run.get('git_commit')}`)",
        "",
        "## Training Overview",
        f"- Timeframe: {run.get('timeframe')}",
        f"- Symbols: {', '.join(run.get('symbols', [])) or NOT_AVAILABLE}",
        f"- Training period: {run.get('training_period')}",
        f"- Validation period: {run.get('validation_period')}",
        f"- Model versions: {run.get('model_versions')}",
        "",
        "## Dataset Health",
        f"```json\n{json.dumps(report.get('dataset', {}), indent=2, default=str)}\n```",
        "",
        "## Data Quality / Healing",
        f"```json\n{json.dumps(_trim(report.get('data_quality', {})), indent=2, default=str)}\n```",
        "",
        "## Direction",
        f"```json\n{json.dumps(_trim(report.get('direction', {})), indent=2, default=str)}\n```",
        "",
        "## Entry",
        f"```json\n{json.dumps(_trim(report.get('entry', {})), indent=2, default=str)}\n```",
        "",
        "## Exit",
        f"```json\n{json.dumps(_trim(report.get('exit', {})), indent=2, default=str)}\n```",
        "",
        "## Risk",
        f"```json\n{json.dumps(_trim(report.get('risk', {})), indent=2, default=str)}\n```",
        "",
        "## Labels",
        f"```json\n{json.dumps(report.get('labels', {}), indent=2, default=str)}\n```",
        "",
        "## Walk-Forward",
        f"{report.get('walk_forward', {})}",
        "",
        "## Backtest",
        f"```json\n{json.dumps(_trim(report.get('backtest', {})), indent=2, default=str)}\n```",
        "",
        "## Symbols",
        f"```json\n{json.dumps(report.get('symbols', {}), indent=2, default=str)}\n```",
        "",
        "## Pipeline Timing",
        f"```json\n{json.dumps(report.get('timings', {}), indent=2, default=str)}\n```",
        "",
        "## Before vs After (previous accepted baseline)",
    ]
    comparison = report.get("comparison_to_previous_baseline", [])
    if not comparison:
        lines.append("No previous baseline report to compare against.")
    else:
        lines.append("| Metric | Before | After | Change | Verdict |")
        lines.append("|---|---|---|---|---|")
        for row in comparison:
            lines.append(
                f"| {row['metric']} | {row['before']:.4f} | {row['after']:.4f} | "
                f"{row['change']:+.4f} | {row['verdict']} |"
            )
    lines += [
        "",
        "## Warnings / Errors",
        f"Warnings: {report.get('warnings', [])}",
        f"Errors: {report.get('errors', [])}",
        "",
        "## Recommendations",
        f"```json\n{json.dumps(report.get('recommendations', {}), indent=2)}\n```",
    ]
    return "\n".join(lines)


def _trim(section: dict[str, Any]) -> dict[str, Any]:
    """Drop the bulkiest nested fields from a section for the Markdown view.

    The JSON export keeps everything; the Markdown view exists for a human
    (or another AI) to read quickly, so per-row trade logs, full equity
    curves and 500-entry telemetry histories are summarised by length
    instead of dumped verbatim.
    """
    trimmed: dict[str, Any] = {}
    for key, value in section.items():
        if isinstance(value, list) and len(value) > 20:
            trimmed[key] = f"[{len(value)} entries omitted from Markdown - see JSON export]"
        elif key in ("trades", "equity_curve", "recent_heal_attempts", "recent_symbol_exclusions"):
            trimmed[key] = f"[{len(value)} entries omitted from Markdown - see JSON export]"
        else:
            trimmed[key] = value
    return trimmed
