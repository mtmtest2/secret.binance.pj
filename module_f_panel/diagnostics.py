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
from typing import Any, Final, Sequence

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

#: Fold-to-fold accuracy spread above which a single train/validation split
#: cannot be trusted on its own.
_WALK_FORWARD_STD_LIMIT: Final[float] = 0.05
#: Null rate at or above which a feature is not a feature - it is a column of
#: NaN being handed to a booster on every row.
_DEAD_FEATURE_NULL_RATE: Final[float] = 0.98
#: Train-vs-validation mean shift (in train sigma) that makes a feature's
#: learned split thresholds meaningless outside the training window.
_DRIFT_CRITICAL_SIGMA: Final[float] = 1.5
_DRIFT_WARNING_SIGMA: Final[float] = 1.0
#: Total-variation distance between train and validation label distributions
#: worth surfacing as an explanation for a head behaving differently.
_LABEL_SHIFT_LIMIT: Final[float] = 0.05
LATEST_REPORT_STATE_KEY: Final[str] = "ml_diagnostic_latest_report"
REPORTS_DIR_NAME: Final[str] = "reports"

#: Below this trade count, ratio-based backtest metrics (win rate, profit
#: factor, Sharpe/Sortino/Calmar, expectancy) are dominated by sampling noise
#: rather than measuring anything about the strategy - a handful of trades can
#: swing "profit factor" between 0 and infinity. 30 is a conventional rule-of-
#: thumb floor for trusting a win-rate/ratio estimate at all, not a rigorous
#: statistical bound; it exists so a report never presents an n=3 result as if
#: it were a reliable performance read.
_MIN_RELIABLE_BACKTEST_TRADES: Final[int] = 30

#: Feature -> group, matching the feature inventory
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
    "wick_ratio": "Path Heat",
    "whipsaw_rate": "Path Heat",
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
    # Microstructure category removed entirely - ob_imbalance,
    # ob_imbalance_delta, ob_spread_bps, ob_spread_rank and liquidation_imbalance
    # (Derivatives) no longer exist as features: Binance has no historical
    # endpoint for order-book depth or liquidation flow, only a live snapshot
    # going forward, so they could never be backfilled for training.
    "funding_rate": "Derivatives",
    "funding_rate_delta": "Derivatives",
    "funding_rate_rank": "Derivatives",
    "open_interest_change": "Derivatives",
    "open_interest_rank": "Derivatives",
    "long_short_ratio": "Derivatives",
    "taker_buy_sell_ratio": "Derivatives",
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


def _split_coverage(
    dataset: ProcessedDataset,
    train_index: Any,
    validation_index: Any,
    test_index: Any,
) -> dict[str, Any]:
    """Rows per split against the theoretical capacity of its own time span.

    The single number that would have made the previous run's biggest problem
    obvious at a glance: train sat at 26.9% of capacity while validation and
    test were at 96.9% and 99.8%, so the model was fitted on a fundamentally
    different population from the one it was scored against.  A raw row count
    cannot show that - only the ratio against the window's own capacity can.
    """
    if "timestamp" not in dataset.metadata.columns:
        return {"status": "NOT_AVAILABLE", "reason": "dataset carries no timestamps"}

    timestamps = dataset.metadata["timestamp"].to_numpy(dtype="int64")
    symbol_count: int = max(1, len(set(dataset.symbols)) or 1)
    bar_ms: int = 300_000
    report: dict[str, Any] = {}

    for name, index in (
        ("train", train_index),
        ("validation", validation_index),
        ("test", test_index),
    ):
        if len(index) == 0:
            report[name] = {"rows": 0, "coverage_pct": 0.0}
            continue
        block = timestamps[index]
        span_ms: int = int(block.max() - block.min())
        # Bars, not intervals: a block running from the first bar to the Nth
        # holds N+1 bars but only N gaps between them, so the count is inclusive
        # of both ends. Dropping the +1 under-counted capacity by exactly one row
        # per symbol, which is what produced the "rows exceed capacity" signature
        # the audit read as duplication - on a 27-symbol run the overflow was 27
        # rows, one per symbol, and it was this arithmetic rather than repeated
        # timestamps.
        capacity: int = max(1, ((span_ms // bar_ms) + 1) * symbol_count)
        # Deliberately not clamped: a ratio above 1.0 is the only automatic
        # detector of genuinely repeated timestamps, and clamping it hides the
        # thing the field exists to reveal.
        report[name] = {
            "rows": int(len(index)),
            "span_days": round(span_ms / 86_400_000, 1),
            "capacity_rows": int(capacity),
            "coverage_pct": round(len(index) / capacity, 4),
        }
    return report


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


def _feature_drift(
    dataset: ProcessedDataset, train_index: np.ndarray, validation_index: np.ndarray
) -> dict[str, Any]:
    """Train vs. validation distribution drift per feature (mean/std/quantile).

    A model can look worse purely because the validation period's market
    regime differs from training, not because the model got worse - this is
    the check that tells the two apart (spec Part 24). Deliberately compares
    only ``train_index`` vs ``validation_index`` (never the held-out test
    split) - this report exists to explain validation behaviour, and must
    not require looking at test data to do it.
    """
    features: pd.DataFrame = dataset.features
    if features.empty or len(validation_index) == 0 or len(train_index) == 0:
        return {"status": NOT_AVAILABLE, "reason": "empty dataset or train/validation slice"}

    train_frame: pd.DataFrame = features.iloc[train_index]
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


#: Neutral placeholder value each microstructure/derivatives feature takes
#: when its underlying data source was empty at feature-build time (see
#: module_b_features/features.py::_add_microstructure_features). A feature
#: parked at this value for nearly every row is a strong signal that source
#: is not actually being collected for this run, not that the market was
#: genuinely neutral on every single bar.
#: ob_imbalance, ob_spread_bps and liquidation_imbalance removed - they no
#: longer exist as features (see FEATURE_COLUMNS). funding_rate,
#: open_interest_change, long_short_ratio and taker_buy_sell_ratio remain:
#: all four have a real, working Binance history endpoint - funding_rate
#: full-history, the other three Binance-side ~30-day-retention-limited but
#: genuinely real where present - so it is still meaningful to measure what
#: fraction of rows carry live data versus this neutral default.
_NEUTRAL_MICROSTRUCTURE_DEFAULTS: Final[dict[str, float]] = {
    "funding_rate": 0.0,
    "open_interest_change": 0.0,
    "long_short_ratio": 0.0,  # log(1.0)
    "taker_buy_sell_ratio": 0.0,  # log(1.0)
}


def _microstructure_coverage(dataset: ProcessedDataset) -> dict[str, Any]:
    """Real fraction of training rows where each microstructure/derivatives
    feature carries live data rather than sitting at its neutral default.

    The Entry model's own feature-importance ranking has repeatedly shown
    order-book features absent from its top 20 despite being exactly what
    "is this a clean entry" should lean on - this makes it possible to tell,
    from measured data rather than a guess, whether that is because the
    signal genuinely is not very informative or because the underlying feed
    (order-book snapshots, funding, open interest, liquidations) is not
    actually populated for this run.
    """
    if dataset.features.empty:
        return {"status": NOT_AVAILABLE, "reason": "empty dataset"}

    coverage: dict[str, Any] = {}
    for feature, neutral in _NEUTRAL_MICROSTRUCTURE_DEFAULTS.items():
        if feature not in dataset.features.columns:
            continue
        values: np.ndarray = dataset.features[feature].to_numpy(dtype=float)
        non_neutral_fraction: float = float(np.mean(~np.isclose(values, neutral, atol=1e-9)))
        coverage[feature] = {
            "non_neutral_row_fraction": non_neutral_fraction,
            "likely_populated": non_neutral_fraction > 0.05,
        }
    return {"status": "AVAILABLE", "features": coverage}


def _backtest_reliability(backtest: dict[str, Any] | None) -> dict[str, Any]:
    """Flag whether a backtest's trade count is large enough to trust its
    ratio-based metrics (win rate, profit factor, expectancy, Sharpe, Sortino,
    Calmar) at all - see :data:`_MIN_RELIABLE_BACKTEST_TRADES`.

    A backtest that rejects nearly every candidate signal (a very selective
    decision cascade, a short validation window, or both) can produce a
    headline profit factor of 20+ from two winning trades; without this check
    that reads identically to a genuinely robust result.
    """
    if not backtest or backtest.get("status") == NOT_AVAILABLE:
        return {"status": NOT_AVAILABLE}
    total_trades = backtest.get("metrics", {}).get("total_trades")
    if not isinstance(total_trades, (int, float)):
        return {"status": NOT_AVAILABLE}
    total_trades = int(total_trades)
    reliable: bool = total_trades >= _MIN_RELIABLE_BACKTEST_TRADES
    return {
        "status": "AVAILABLE",
        "total_trades": total_trades,
        "minimum_trades_for_reliability": _MIN_RELIABLE_BACKTEST_TRADES,
        "statistically_reliable": reliable,
        "signals_generated": backtest.get("signals_generated"),
        "signals_rejected": backtest.get("signals_rejected"),
        "rejection_breakdown": backtest.get("rejection_breakdown", {}),
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


def _spearman_rho(values: Sequence[float]) -> float:
    """Rank correlation of ``values`` against their position.

    Written out rather than imported so the diagnostic does not grow a scipy
    dependency for eight lines of arithmetic.  Values are already a fold
    sequence, so ties are rare and average-rank handling is enough.
    """
    n = len(values)
    if n < 3:
        return float("nan")
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank_position, original_index in enumerate(order):
        ranks[original_index] = float(rank_position + 1)
    positions = [float(i + 1) for i in range(n)]
    mean_rank = sum(ranks) / n
    mean_position = sum(positions) / n
    numerator = sum((r - mean_rank) * (p - mean_position) for r, p in zip(ranks, positions))
    denominator = (
        sum((r - mean_rank) ** 2 for r in ranks) * sum((p - mean_position) ** 2 for p in positions)
    ) ** 0.5
    return float(numerator / denominator) if denominator else float("nan")


def _walk_forward_problem_summary(walk_forward: dict[str, Any]) -> str:
    """One-line, measurement-derived read on walk-forward validation health.

    This used to begin every AVAILABLE-path answer with "none measured" and
    summarise the folds by their standard deviation alone.  A spread is the
    wrong statistic for a fold sequence: it takes the same value however the
    folds are ordered, so a run whose accuracy fell monotonically from 0.80 to
    0.46 - the single most important measurement in that report - was
    indistinguishable from benign noise.
    """
    if not isinstance(walk_forward, dict) or walk_forward.get("status") != "AVAILABLE":
        return "walk-forward evaluation not available (single train/validation split only)"

    folds = walk_forward.get("folds") or []
    accuracies = [f.get("accuracy") for f in folds if isinstance(f.get("accuracy"), (int, float))]
    std = walk_forward.get("accuracy_std")
    n_folds = walk_forward.get("n_folds", len(accuracies) or "?")

    if len(accuracies) >= 3:
        rho = _spearman_rho(accuracies)
        if rho <= -0.9:
            return (
                f"accuracy degrades monotonically across all {n_folds} folds "
                f"({accuracies[0]:.3f} -> {accuracies[-1]:.3f}, Spearman rho={rho:.2f}); the "
                "earliest fold is not representative of the period the model will trade, so the "
                f"mean overstates it - the latest fold ({accuracies[-1]:.3f}) is the honest read"
            )
        if isinstance(std, (int, float)) and std > _WALK_FORWARD_STD_LIMIT:
            return (
                f"accuracy varies widely across {n_folds} folds (std={std:.3f}, "
                f"range {min(accuracies):.3f}-{max(accuracies):.3f}); a single split cannot be "
                "trusted at this spread"
            )
    if isinstance(std, (int, float)):
        return f"none measured (walk-forward across {n_folds} folds, accuracy std={std:.3f})"
    return f"none measured (walk-forward across {n_folds} folds)"


def _dataset_health_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Predicates over the report that decide whether the *data* is usable.

    Everything these read - per-feature null counts, train/validation drift in
    sigma units, label distribution per split, duplicate counts, split coverage -
    was already being computed and written into the report.  None of it was ever
    read back, so a run with four all-NaN feature columns, 72% degenerate
    training rows and four features past one train-sigma of drift still
    summarised as "biggest data problem: none measured".

    Collecting the checks in one list rather than scattering them through
    ``_ai_summary`` and ``_recommendations`` means a check cannot exist in the
    data without also reaching the verdict.  Each entry carries a ``weight`` so
    the worst finding can be picked deterministically for the one-line summary.
    """
    checks: list[dict[str, Any]] = []
    dataset = report.get("dataset", {}) or {}
    total = dataset.get("valid_samples") or 0
    nulls = dataset.get("null_counts_by_feature") or {}

    dead = sorted(
        name
        for name, count in nulls.items()
        if total and isinstance(count, (int, float)) and count / total >= _DEAD_FEATURE_NULL_RATE
    )
    if dead:
        checks.append(
            {
                "id": "dead_features",
                "severity": "CRITICAL",
                "weight": 100.0,
                "message": (
                    f"{len(dead)} feature(s) are effectively empty on every row "
                    f"({', '.join(dead)}) - they occupy the feature contract and force a "
                    "retrain on every change without contributing anything; remove them or "
                    "fix the backfill that should be populating them"
                ),
            }
        )

    drift = ((report.get("features", {}) or {}).get("drift", {}) or {}).get(
        "most_drifted_features", []
    ) or []
    worst = drift[0] if drift else None
    if isinstance(worst, dict):
        shift = worst.get("mean_shift_in_train_std")
        if isinstance(shift, (int, float)) and shift >= _DRIFT_WARNING_SIGMA:
            severe = shift >= _DRIFT_CRITICAL_SIGMA
            drifted = [
                f["feature"]
                for f in drift
                if isinstance(f, dict)
                and isinstance(f.get("mean_shift_in_train_std"), (int, float))
                and f["mean_shift_in_train_std"] >= _DRIFT_WARNING_SIGMA
            ]
            checks.append(
                {
                    "id": "feature_drift",
                    "severity": "CRITICAL" if severe else "HIGH",
                    "weight": 50.0 + float(shift),
                    "message": (
                        f"{len(drifted)} feature(s) shift by more than {_DRIFT_WARNING_SIGMA:.1f} "
                        f"train-sigma between train and validation (worst: {worst.get('feature')} at "
                        f"{shift:.2f} sigma) - split thresholds learned on the training window do "
                        "not transfer, so validation metrics understate what live will see"
                    ),
                }
            )

    distribution = (report.get("direction", {}) or {}).get("distribution", {}) or {}
    shift_value = distribution.get("label_distribution_shift")
    if isinstance(shift_value, (int, float)) and shift_value >= _LABEL_SHIFT_LIMIT:
        checks.append(
            {
                "id": "label_shift",
                "severity": "HIGH",
                "weight": 40.0 + float(shift_value),
                "message": (
                    f"the label distribution moves by {shift_value:.3f} (total-variation) between "
                    "train and validation - a directional head trained on one balance and scored "
                    "on another will look worse than it is, or better"
                ),
            }
        )

    coverage = dataset.get("split_coverage_pct", {}) or {}
    for name, block in coverage.items():
        if not isinstance(block, dict):
            continue
        rows, capacity = block.get("rows"), block.get("capacity_rows")
        if isinstance(rows, int) and isinstance(capacity, int) and capacity and rows > capacity:
            checks.append(
                {
                    "id": f"split_over_capacity_{name}",
                    "severity": "HIGH",
                    "weight": 30.0,
                    "message": (
                        f"the {name} split holds {rows} rows against a theoretical capacity of "
                        f"{capacity} - a split cannot exceed its own time span unless timestamps "
                        "repeat, so the dataset carries duplicates"
                    ),
                }
            )

    duplicates = dataset.get("duplicate_feature_rows")
    if isinstance(duplicates, int) and duplicates > 0:
        checks.append(
            {
                "id": "duplicate_rows",
                "severity": "MEDIUM",
                "weight": 20.0,
                "message": (
                    f"{duplicates} duplicate feature row(s) survived into the dataset - identical "
                    "rows split across train and test are exact-match leakage"
                ),
            }
        )

    dq = report.get("data_quality", {}) or {}
    exclusions = dq.get("symbol_exclusions_total")
    if isinstance(exclusions, int) and exclusions > 0:
        checks.append(
            {
                "id": "symbol_exclusions",
                "severity": "MEDIUM",
                "weight": 10.0,
                "message": f"{exclusions} symbol-cycle exclusion(s) recorded this run",
            }
        )

    coverage_block = (report.get("features", {}) or {}).get("microstructure_coverage", {}) or {}
    if coverage_block.get("status") == "AVAILABLE":
        unpopulated = sorted(
            name
            for name, info in (coverage_block.get("features", {}) or {}).items()
            if isinstance(info, dict) and not info.get("likely_populated", True)
        )
        if unpopulated:
            checks.append(
                {
                    "id": "unpopulated_feeds",
                    "severity": "HIGH",
                    "weight": 35.0,
                    "message": (
                        "Microstructure/derivatives feed(s) sit at their neutral default for "
                        f"nearly every row (likely not being collected): {', '.join(unpopulated)} "
                        "- check data collection before trusting feature importance involving them"
                    ),
                }
            )

    return sorted(checks, key=lambda c: -float(c["weight"]))


def _degenerate_sweep_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flag a confidence sweep whose accuracy rises only by going silent.

    Accuracy climbing while balanced accuracy sinks toward the chance floor is
    the generic signature of a classifier collapsing onto its majority class.
    Detected once here rather than per-table, because the same shape can appear
    in any head's sweep.
    """
    checks: list[dict[str, Any]] = []
    analysis = ((report.get("direction", {}) or {}).get("metrics", {}) or {}).get(
        "confidence_threshold_analysis", []
    ) or []
    degenerate = [row for row in analysis if isinstance(row, dict) and row.get("is_degenerate")]
    if degenerate:
        best = max(degenerate, key=lambda r: float(r.get("accuracy", 0.0) or 0.0))
        checks.append(
            {
                "id": "degenerate_confidence_sweep",
                "severity": "HIGH",
                "weight": 45.0,
                "message": (
                    f"the confidence sweep reaches {float(best.get('accuracy', 0.0)):.1%} accuracy at "
                    f"threshold {best.get('confidence_threshold')} only by predicting a single class "
                    "- that number is the surviving subset's base rate, not skill, and raising the "
                    "live threshold toward it would produce a system that never trades"
                ),
            }
        )
    return checks


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

    # Data health is decided by the check registry, not by one hand-picked
    # scalar. Everything it reads was already in the report and simply was not
    # being consulted.
    data_checks: list[dict[str, Any]] = _dataset_health_checks(report) + _degenerate_sweep_checks(report)
    for check in data_checks:
        if check["severity"] == "CRITICAL":
            critical.append(check["message"])
        elif check["severity"] == "HIGH":
            warnings.append(check["message"])

    reliability: dict[str, Any] = report.get("backtest_reliability", {})
    backtest_unreliable: bool = (
        reliability.get("status") == "AVAILABLE" and not reliability.get("statistically_reliable", True)
    )
    if backtest_unreliable:
        warnings.append(
            f"backtest ran only {reliability.get('total_trades')} trade(s) - below the "
            f"{reliability.get('minimum_trades_for_reliability')}-trade floor for trusting "
            "win rate/profit factor/Sharpe as a performance estimate"
        )

    # Heads are scored on lift over their own chance baseline, so the four are
    # comparable.  Ranking balanced accuracy (chance 1/3), ROC-AUC (chance 0.5)
    # and R^2 (chance 0) against each other on raw magnitude made the verdict an
    # artifact of which metric happens to live nearest zero: it named "risk" the
    # weakest head at R^2 0.079 (+0.079 over chance) while "direction" sat at
    # balanced accuracy 0.356 - just +0.023 over chance, three times worse, and
    # the head the entire strategy depends on.
    direction_metrics: dict[str, Any] = heads["direction"].get("metrics", {}) or {}
    entry_metrics: dict[str, Any] = heads["entry"].get("metrics", {}) or {}
    exit_metrics: dict[str, Any] = heads["exit"].get("metrics", {}) or {}
    risk_metrics: dict[str, Any] = heads["risk"].get("metrics", {}) or {}

    def _finite(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) and value == value else None

    scored: dict[str, float] = {}
    n_classes: int = max(2, len(direction_metrics.get("class_distribution", {}) or {}) or 3)
    chance: float = 1.0 / n_classes
    balanced = _finite(direction_metrics.get("balanced_accuracy"))
    if balanced is not None:
        scored["direction"] = (balanced - chance) / (1.0 - chance)
    roc_auc = _finite(entry_metrics.get("roc_auc"))
    if roc_auc is not None:
        scored["entry"] = (roc_auc - 0.5) / 0.5
    # Exit was collected into `heads` but never scored, so its problems could
    # never surface here however bad they were.
    exit_r2s = [
        value
        for target in (exit_metrics or {}).values()
        if isinstance(target, dict)
        for value in (_finite(target.get("r2")),)
        if value is not None
    ]
    if exit_r2s:
        scored["exit"] = max(0.0, sum(exit_r2s) / len(exit_r2s))
    risk_r2 = _finite(risk_metrics.get("r2"))
    if risk_r2 is not None:
        scored["risk"] = max(0.0, risk_r2)

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
    elif warnings:
        next_action = f"Address before trusting this run's model metrics: {warnings[0]}"
    elif weakest != NOT_AVAILABLE and scored.get(weakest, 1.0) < 0.10:
        # Lift over chance, so the bar is "captures less than a tenth of the
        # headroom available to it", not "some metric happens to be below 0.55".
        next_action = (
            f"Improve the {weakest} model - it captures {scored[weakest]:.1%} of the lift "
            "available over its own chance baseline"
        )
    else:
        next_action = "No critical issues measured; continue with the walk-forward/backtest checklist"

    return {
        "overall_status": status,
        "strongest_component": strongest,
        "weakest_component": weakest,
        "biggest_data_problem": (data_checks[0]["message"] if data_checks else "none measured"),
        "data_health_checks": data_checks,
        "component_lift_over_chance": {name: round(value, 4) for name, value in scored.items()},
        "biggest_ml_problem": (
            f"{weakest} captures the least of its available headroom "
            f"({scored[weakest]:.1%} lift over chance)"
            if scored and weakest in scored
            else NOT_AVAILABLE
        ),
        "biggest_validation_problem": _walk_forward_problem_summary(report.get("walk_forward", {})),
        "biggest_trading_problem": (
            "backtest not available for this run"
            if report.get("backtest", {}).get("status") == NOT_AVAILABLE
            else (
                f"backtest metrics are not statistically reliable: only "
                f"{reliability.get('total_trades')} trade(s) generated from "
                f"{reliability.get('signals_generated')} candidate signal(s) "
                f"({reliability.get('signals_rejected')} rejected) - treat win rate/profit "
                "factor/Sharpe as noise, not a performance estimate"
                if backtest_unreliable
                else NOT_AVAILABLE
            )
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
    relaxed_backtest: Any | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the full ML diagnostic report for a just-completed training run."""
    boundaries = dataset.split_boundaries(
        train_months=settings.ml.train_months,
        validation_months=settings.ml.validation_months,
        test_months=settings.ml.test_months,
        purge_bars=settings.ml.purge_bars,
        timeframe_ms=settings.data.timeframe_ms,
    )
    split = dataset.chronological_split(boundaries)
    train_index, validation_index, test_index = split.train_index, split.validation_index, split.test_index

    def _period(start_ms: int | None, end_ms: int | None, rows: int) -> dict[str, Any]:
        return {"start": start_ms, "end": end_ms, "rows": rows}

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
            "training_period": _period(split.train_start_ms, split.train_end_ms, int(len(train_index))),
            "validation_period": _period(
                split.validation_start_ms, split.validation_end_ms, int(len(validation_index))
            ),
            "test_period": {
                **_period(split.test_start_ms, split.test_end_ms, int(len(test_index))),
                "note": (
                    "Held-out final backtest window - never used for training, early "
                    "stopping, calibration, threshold selection or model selection."
                ),
            },
            "purge_bars": settings.ml.purge_bars,
            "embargo_ms": boundaries.embargo_ms if boundaries is not None else 0,
            "scaled_down_from_nominal_months": boundaries.scaled_down if boundaries is not None else False,
        },
        "dataset": {
            "total_candidate_rows": dataset.total_candidate_rows,
            "valid_samples": len(dataset),
            "rejected_invalid_label_rows": dataset.rejected_invalid_label_rows,
            "dropped_missing_or_inf_rows": dataset.dropped_missing_or_inf_rows,
            "duplicate_feature_rows": dataset.duplicate_feature_rows,
            "training_samples": int(len(train_index)),
            "validation_samples": int(len(validation_index)),
            "test_samples": int(len(test_index)),
            "feature_count": len(dataset.feature_columns),
            "feature_names": list(dataset.feature_columns),
            "per_symbol_rows": {str(k): int(v) for k, v in per_symbol_rows.items()},
            # Rows are no longer destroyed for a NaN in a single feature, so
            # sparsity no longer announces itself as a collapsed row count.
            # These two blocks are the replacement signal: which columns are
            # actually empty, and whether the emptiness is uniform or
            # concentrated in particular symbols and periods.
            "null_counts_by_feature": dict(dataset.null_counts_by_feature),
            "null_rate_by_symbol_month": dict(dataset.null_rate_by_symbol_month),
            "split_coverage_pct": _split_coverage(dataset, train_index, validation_index, test_index),
        },
        "data_quality": await _qc_telemetry(database),
        "features": {
            "statistics": _feature_statistics(dataset),
            "correlation": _feature_correlation(dataset),
            "drift": _feature_drift(dataset, train_index, validation_index),
            "microstructure_coverage": _microstructure_coverage(dataset),
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
        "backtest_reliability": _backtest_reliability(
            backtest.to_dict() if backtest is not None else None
        ),
        "backtest_diagnostic_relaxed": (
            {
                "disclaimer": (
                    "Same out-of-sample window as `backtest`, replayed with the Decision "
                    "Engine's confidence/probability/reward-risk thresholds loosened "
                    "(see main._RELAXED_DECISION_SETTINGS) to gather a larger trade count. "
                    "Diagnostic only - never reflects live-configured thresholds and must "
                    "not be read as a performance estimate."
                ),
                **relaxed_backtest.to_dict(),
            }
            if relaxed_backtest is not None
            else {"status": NOT_AVAILABLE}
        ),
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

    reliability = report.get("backtest_reliability", {})
    if reliability.get("status") == "AVAILABLE" and not reliability.get("statistically_reliable", True):
        breakdown: dict[str, Any] = reliability.get("rejection_breakdown", {}) or {}
        top_rule: str = (
            max(breakdown, key=breakdown.get) if breakdown else "not recorded for this run"
        )
        high.append(
            f"Backtest produced only {reliability.get('total_trades')} trade(s) from "
            f"{reliability.get('signals_generated')} candidate signal(s) "
            f"({reliability.get('signals_rejected')} rejected, top rejection reason: {top_rule}) - "
            f"win rate/profit factor/Sharpe/expectancy are not statistically meaningful below "
            f"{reliability.get('minimum_trades_for_reliability')} trades; widen the validation "
            "window or run a walk-forward-style backtest across multiple periods before trusting "
            "these numbers"
        )
        relaxed = report.get("backtest_diagnostic_relaxed", {})
        relaxed_trades = relaxed.get("metrics", {}).get("total_trades") if isinstance(relaxed, dict) else None
        if isinstance(relaxed_trades, (int, float)):
            strict_trades = reliability.get("total_trades", 0) or 0
            if relaxed_trades > strict_trades:
                low.append(
                    f"The same window with loosened decision thresholds (diagnostic only, see "
                    f"backtest_diagnostic_relaxed) produced {int(relaxed_trades)} trade(s) vs "
                    f"{strict_trades} under the live thresholds - the low live trade count looks "
                    "like the cascade's intended selectivity rather than a data/signal-generation "
                    "bug, though the relaxed run is too loosened to read as a performance estimate"
                )
            else:
                medium.append(
                    "The same window with loosened decision thresholds (diagnostic only) still "
                    f"produced only {int(relaxed_trades)} trade(s) - worth checking for a genuine "
                    "over-rejection bug rather than assuming intended selectivity"
                )

    # Same registry the AI summary reads, so a check can never appear in one
    # output and be missing from the other.
    buckets: dict[str, list[str]] = {"CRITICAL": critical, "HIGH": high, "MEDIUM": medium, "LOW": low}
    for check in _dataset_health_checks(report) + _degenerate_sweep_checks(report):
        buckets.get(str(check["severity"]), medium).append(check["message"])

    walk_forward = report.get("walk_forward", {}) or {}
    if walk_forward.get("status") == "AVAILABLE":
        summary_line = _walk_forward_problem_summary(walk_forward)
        if not summary_line.startswith("none measured"):
            folds = walk_forward.get("folds") or []
            accuracies = [
                f.get("accuracy") for f in folds if isinstance(f.get("accuracy"), (int, float))
            ]
            severity = high
            if accuracies and _spearman_rho(accuracies) <= -0.9:
                severity = critical
            severity.append(
                f"Walk-forward: {summary_line} - prefer the latest fold over the mean when "
                "deciding whether this artifact is fit to trade"
            )

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

    entry_calib = report.get("calibration", {}).get("entry", {})
    if entry_calib.get("status") == "AVAILABLE" and entry_calib.get("improved"):
        if report.get("entry", {}).get("production_calibration") == "isotonic":
            low.append("entry isotonic calibration measurably improves log loss and is wired into inference")
        else:
            low.append(
                "entry isotonic calibration measurably improves log loss but is not yet "
                "wired into inference (not enough held-out rows to fit a production calibrator)"
            )

    # Direction is a two-stage cascade (gate + long/short); each stage is
    # calibrated independently rather than as one multiclass estimator.
    direction_calib = report.get("calibration", {}).get("direction", {})
    direction_production = report.get("direction", {}).get("production_calibration", {})
    if not isinstance(direction_production, dict):
        direction_production = {}
    for stage_key, stage_label in (("gate", "direction gate (trade vs no-trade)"), ("direction", "direction long/short")):
        stage_calib = direction_calib.get(stage_key, {}) if isinstance(direction_calib, dict) else {}
        if stage_calib.get("status") == "AVAILABLE" and stage_calib.get("improved"):
            if direction_production.get(stage_key) == "isotonic":
                low.append(f"{stage_label} isotonic calibration measurably improves log loss and is wired into inference")
            else:
                low.append(
                    f"{stage_label} isotonic calibration measurably improves log loss but is not yet "
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
        "## Backtest Reliability",
        f"```json\n{json.dumps(report.get('backtest_reliability', {}), indent=2, default=str)}\n```",
        "",
        "## Backtest (Diagnostic, Relaxed Thresholds)",
        f"```json\n{json.dumps(_trim(report.get('backtest_diagnostic_relaxed', {})), indent=2, default=str)}\n```",
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
