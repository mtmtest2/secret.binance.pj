"""Pure metric-computation helpers for the ML diagnostic report.

Deliberately separate from :mod:`module_c_ml.ml_models`: nothing here fits an
estimator, touches the filesystem or the network - every function takes
already-computed predictions/probabilities and targets and returns a plain
JSON-serialisable ``dict``. That keeps the training code in ``ml_models.py``
readable and makes the metric logic independently unit-testable.

Every function computes only what the inputs actually support. Where the
underlying data does not exist (e.g. per-threshold trading P&L without a
backtest), the field is set to the literal string ``"NOT_AVAILABLE"`` rather
than a fabricated or approximated number - see the project's diagnostic
reporting rules: values must never be invented.
"""

from __future__ import annotations

from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

#: Sentinel used throughout instead of a fabricated or approximated value.
NOT_AVAILABLE: Final[str] = "NOT_AVAILABLE"

#: Minimum *relative* log-loss improvement before isotonic calibration is
#: wired into inference. A strict inequality alone lets noise in the fourth
#: decimal place decide which estimator ships, on the strength of a single
#: temporal half-split with no error bar.
MIN_CALIBRATION_RELATIVE_GAIN: Final[float] = 0.01

#: Confidence/threshold grids requested by the diagnostic reporting spec.
#: Extended downward from a 0.50 floor (task 7a): a production diagnostic
#: report's auto-tuner recommended the F-beta-optimal gate AND direction
#: thresholds exactly at the grid's lowest value (0.50) for both sweeps,
#: which means the true optimum could sit below 0.50 and was structurally
#: unknowable on the old grid - the sweep can never recommend a value it
#: was never offered. 0.30/0.35/0.40/0.45 let the next real-data retrain
#: actually find out, rather than silently capping the search at a value
#: that already looked like a boundary optimum, not an interior one.
CONFIDENCE_THRESHOLDS: Final[tuple[float, ...]] = (
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
)
ENTRY_THRESHOLDS: Final[tuple[float, ...]] = (
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
)


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    """Mean/median/std/min/max, honestly reporting ``nan`` on empty input."""
    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ---------------------------------------------------------------------------
# Direction (multi-class)
# ---------------------------------------------------------------------------
def direction_metrics(
    target: pd.Series,
    probabilities: np.ndarray,
    class_order: Sequence[str],
) -> dict[str, Any]:
    """Full Direction-model metric set: accuracy family, per-class, confusion
    matrix, class distributions, probability stats and confidence-threshold
    analysis.

    Args:
        target: Encoded (integer class index) validation targets.
        probabilities: ``(n_rows, n_classes)`` predicted probability matrix,
            columns ordered exactly as ``class_order``.
        class_order: Class names in the same order as ``probabilities`` columns.
    """
    if len(target) == 0 or probabilities.size == 0:
        return {}

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        log_loss,
        precision_recall_fscore_support,
    )

    y: np.ndarray = target.to_numpy()
    labels: list[int] = list(range(len(class_order)))
    predictions: np.ndarray = probabilities.argmax(axis=1)

    try:
        loss: float = float(log_loss(y, probabilities, labels=labels))
    except ValueError:  # pragma: no cover - degenerate validation block
        loss = float("nan")

    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y, predictions, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y, predictions, labels=labels, average="weighted", zero_division=0
    )

    raw_cm: np.ndarray = confusion_matrix(y, predictions, labels=labels)
    row_sums: np.ndarray = raw_cm.sum(axis=1, keepdims=True).astype(float)
    normalized_cm: np.ndarray = np.divide(
        raw_cm.astype(float), row_sums, out=np.zeros_like(raw_cm, dtype=float), where=row_sums > 0
    )

    class_distribution: dict[str, int] = {
        class_order[index]: int((y == index).sum()) for index in labels
    }
    predicted_class_distribution: dict[str, int] = {
        class_order[index]: int((predictions == index).sum()) for index in labels
    }
    probability_stats: dict[str, dict[str, float]] = {
        class_order[index]: _distribution_stats(probabilities[:, index]) for index in labels
    }

    confidence: np.ndarray = probabilities.max(axis=1)
    confidence_analysis: list[dict[str, Any]] = []
    for cutoff in CONFIDENCE_THRESHOLDS:
        mask: np.ndarray = confidence >= cutoff
        n_predictions: int = int(mask.sum())
        row: dict[str, Any] = {
            "confidence_threshold": cutoff,
            "n_predictions": n_predictions,
            "pct_of_samples": float(n_predictions / len(y)) if len(y) else 0.0,
        }
        if n_predictions == 0:
            row.update(accuracy=None, balanced_accuracy=None, precision=None, recall=None, f1=None)
        else:
            y_sub, pred_sub = y[mask], predictions[mask]
            p_sub, r_sub, f_sub, _ = precision_recall_fscore_support(
                y_sub, pred_sub, labels=labels, average="macro", zero_division=0
            )
            row.update(
                accuracy=float(accuracy_score(y_sub, pred_sub)),
                balanced_accuracy=(
                    float(balanced_accuracy_score(y_sub, pred_sub))
                    if len(set(y_sub.tolist())) > 1
                    else float(accuracy_score(y_sub, pred_sub))
                ),
                precision=float(p_sub),
                recall=float(r_sub),
                f1=float(f_sub),
            )
            # Accuracy climbing while balanced accuracy sinks toward 1/n_classes
            # is the signature of the classifier collapsing onto one class:
            # everything that survives the threshold is the majority label, so
            # "accuracy" is just that label's base rate.  Without this flag the
            # sweep reads as "raise the threshold and accuracy improves", and
            # acting on it produces a system that never trades.
            distinct: int = int(len(np.unique(pred_sub)))
            row["distinct_predicted_classes"] = distinct
            row["is_degenerate"] = bool(distinct < 2)
            row["per_class_recall"] = {
                class_order[index]: float(value)
                for index, value in enumerate(
                    precision_recall_fscore_support(
                        y_sub, pred_sub, labels=labels, average=None, zero_division=0
                    )[1]
                )
            }
        confidence_analysis.append(row)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "log_loss": loss,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            class_order[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
        "confusion_matrix": {
            "labels": list(class_order),
            "raw": raw_cm.tolist(),
            "normalized": normalized_cm.tolist(),
        },
        "class_distribution": class_distribution,
        "predicted_class_distribution": predicted_class_distribution,
        "probability_stats": probability_stats,
        "confidence_threshold_analysis": confidence_analysis,
    }


# ---------------------------------------------------------------------------
# Entry (binary)
# ---------------------------------------------------------------------------
def entry_metrics(target: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Full Entry-model metric set at the production decision threshold."""
    if len(target) == 0:
        return {}

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y: np.ndarray = target.to_numpy()
    predictions: np.ndarray = (probabilities >= threshold).astype(int)
    has_both_classes: bool = len(set(y.tolist())) > 1

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if has_both_classes else float("nan"),
        "pr_auc": (
            float(average_precision_score(y, probabilities)) if has_both_classes else float("nan")
        ),
        "confusion_matrix": {
            "labels": [0, 1],
            "raw": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        },
        "class_distribution": {"positive": int(y.sum()), "negative": int(len(y) - int(y.sum()))},
        "predicted_positive_rate": float(predictions.mean()) if len(predictions) else 0.0,
        "probability_stats": _distribution_stats(probabilities),
    }


def entry_threshold_sweep(
    target: pd.Series,
    probabilities: np.ndarray,
    thresholds: Sequence[float] = ENTRY_THRESHOLDS,
    min_signal_sample_size: int = 20,
) -> list[dict[str, Any]]:
    """Precision/recall/F1/signal-count at each candidate entry threshold.

    Trading-level fields (average R, win rate, profit factor, expectancy, net
    PnL, max drawdown) require simulating trades with realistic fills, fees
    and slippage - that is the backtester's job, not this validation-split
    sweep - so they are reported as :data:`NOT_AVAILABLE` here rather than
    approximated from label geometry.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y: np.ndarray = target.to_numpy()
    rows: list[dict[str, Any]] = []
    for cutoff in thresholds:
        predictions: np.ndarray = (probabilities >= cutoff).astype(int)
        n_signals: int = int(predictions.sum())
        rows.append(
            {
                "threshold": cutoff,
                "signals": n_signals,
                "precision": float(precision_score(y, predictions, zero_division=0)),
                "recall": float(recall_score(y, predictions, zero_division=0)),
                "f1": float(f1_score(y, predictions, zero_division=0)),
                "meets_min_sample_size": n_signals >= min_signal_sample_size,
                "average_r": NOT_AVAILABLE,
                "win_rate": NOT_AVAILABLE,
                "profit_factor": NOT_AVAILABLE,
                "expectancy": NOT_AVAILABLE,
                "net_pnl": NOT_AVAILABLE,
                "max_drawdown": NOT_AVAILABLE,
            }
        )
    return rows


def gate_threshold_sweep(
    target: pd.Series,
    probabilities: np.ndarray,
    thresholds: Sequence[float] = CONFIDENCE_THRESHOLDS,
    min_signal_sample_size: int = 20,
) -> list[dict[str, Any]]:
    """Precision/recall/F1/signal-count at each candidate gate threshold.

    Same metric computation as :func:`entry_threshold_sweep` (see there for
    why the trading-level fields are :data:`NOT_AVAILABLE`), applied to the
    Direction model's trade-vs-no-trade gate stage instead of the Entry
    model: ``target`` is ground truth (1 = trade, 0 = no-trade, i.e. whether
    the row's true label is anything other than NO_TRADE_OR_FAIL) and
    ``probabilities`` is the gate's own raw ``trade_probability`` per row -
    never the multiplied joint probability.
    """
    return entry_threshold_sweep(target, probabilities, thresholds, min_signal_sample_size)


def direction_threshold_sweep(
    target: pd.Series,
    probabilities: np.ndarray,
    thresholds: Sequence[float] = CONFIDENCE_THRESHOLDS,
    min_signal_sample_size: int = 20,
) -> list[dict[str, Any]]:
    """Precision/recall/F1/signal-count at each candidate direction-given-trade
    threshold.

    Same metric computation as :func:`entry_threshold_sweep`, applied to the
    long-vs-short stage conditional on the gate having already said "trade":
    ``target`` is ground truth restricted to true-trade rows (1 = LONG,
    0 = SHORT) and ``probabilities`` is ``direction_given_trade_probability``
    per row, already restricted to those same true-trade rows.
    """
    return entry_threshold_sweep(target, probabilities, thresholds, min_signal_sample_size)


#: Grid for a two-sided confidence gate.  A rule of the form
#: ``max(p, 1 - p) >= t`` cannot be satisfied below 0.5, so sweeping there
#: describes a rule the decision engine has no way to express.
TWO_SIDED_CONFIDENCE_THRESHOLDS: Final[tuple[float, ...]] = (
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
)


def direction_confidence_sweep(
    target: pd.Series,
    long_probabilities: np.ndarray,
    thresholds: Sequence[float] = TWO_SIDED_CONFIDENCE_THRESHOLDS,
    min_signal_sample_size: int = 20,
) -> list[dict[str, Any]]:
    """Sweep the quantity R1b actually gates on: ``max(p, 1 - p)``.

    :func:`direction_threshold_sweep` sweeps ``p >= t`` one-sided, which does not
    correspond to the live rule: the engine picks a side from ``p >= 0.5`` and
    then gates on how far the probability sits from the coin flip.  Tuning
    against the one-sided form produced recommendations below 0.5 - thresholds
    that cannot reject anything.

    Each row reports, for signals clearing the confidence bar, how often the
    chosen side was the correct one.  ``side_accuracy`` is directly comparable to
    0.5, so a model with no directional edge is obvious rather than inferred.
    """
    y: np.ndarray = np.asarray(target, dtype=int)
    probabilities: np.ndarray = np.asarray(long_probabilities, dtype=float)
    confidence: np.ndarray = np.maximum(probabilities, 1.0 - probabilities)
    chosen_long: np.ndarray = probabilities >= 0.5
    correct: np.ndarray = chosen_long == (y == 1)
    total: int = len(y)

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected: np.ndarray = confidence >= threshold
        n_signals: int = int(selected.sum())
        accuracy: float = float(correct[selected].mean()) if n_signals else 0.0
        coverage: float = float(n_signals / total) if total else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "signals": n_signals,
                "signal_rate": coverage,
                # "precision"/"recall"/"f1" keep the row shape compatible with the
                # other sweeps and with _select_recommended_threshold's F-beta.
                "precision": accuracy,
                "recall": coverage,
                "f1": (
                    2 * accuracy * coverage / (accuracy + coverage)
                    if n_signals and (accuracy + coverage) > 0
                    else 0.0
                ),
                "side_accuracy": accuracy,
                "long_share": float(chosen_long[selected].mean()) if n_signals else 0.0,
                "meets_min_sample_size": n_signals >= min_signal_sample_size,
                "average_r": NOT_AVAILABLE,
                "win_rate": NOT_AVAILABLE,
                "profit_factor": NOT_AVAILABLE,
                "expectancy": NOT_AVAILABLE,
                "net_pnl": NOT_AVAILABLE,
                "max_drawdown": NOT_AVAILABLE,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Regression (Exit targets, Risk)
# ---------------------------------------------------------------------------
def regression_metrics(target: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """MAE/RMSE/R²/median-AE plus target and prediction distributions."""
    if len(target) == 0:
        return {}

    from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score

    target = np.asarray(target, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    r_squared: float
    if len(target) > 1 and float(np.var(target)) > 0.0:
        r_squared = float(r2_score(target, predictions))
    else:  # pragma: no cover - degenerate (near-constant) validation block
        r_squared = float("nan")

    return {
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(target, predictions))),
        "r2": r_squared,
        "median_absolute_error": float(median_absolute_error(target, predictions)),
        "target_stats": _distribution_stats(target),
        "prediction_stats": _distribution_stats(predictions),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate_classifier(
    estimator: Any,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    n_classes: int,
    *,
    min_calibration_rows: int = 50,
    min_eval_rows: int = 20,
) -> dict[str, Any]:
    """Fit isotonic calibration on a held-out slice and score reliability.

    ``x_calibration``/``y_calibration`` and ``x_eval``/``y_eval`` must already
    be two temporally-ordered, non-overlapping slices of the validation set
    (calibration fit strictly precedes the evaluation slice) - this function
    does not itself enforce temporal order, it only fits and scores.

    Returns a dict with ``status`` of ``"AVAILABLE"`` (with Brier score and
    log loss before/after) or ``"NOT_AVAILABLE"`` (with a ``reason``) - never
    a fabricated calibration outcome.
    """
    if len(x_calibration) < min_calibration_rows or len(x_eval) < min_eval_rows:
        return {
            "status": "NOT_AVAILABLE",
            "reason": (
                f"insufficient rows for a temporally safe calibration split "
                f"(calibration={len(x_calibration)}, eval={len(x_eval)})"
            ),
        }

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import log_loss

    try:
        # scikit-learn >= 1.6 removed `cv="prefit"` in favour of wrapping an
        # already-fitted estimator with `FrozenEstimator`; older versions
        # (down to the project's pinned >=1.4.0 floor) only understand
        # `cv="prefit"` and have no `sklearn.frozen` module at all.
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="isotonic")
        except ImportError:  # pragma: no cover - exercised only on sklearn < 1.6
            calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv="prefit")
        calibrated.fit(x_calibration, y_calibration)
    except Exception as error:  # noqa: BLE001 - calibration is best-effort, never fatal
        return {"status": "NOT_AVAILABLE", "reason": f"calibration failed: {error}"}

    raw_probabilities: np.ndarray = np.asarray(estimator.predict_proba(x_eval))
    calibrated_probabilities: np.ndarray = np.asarray(calibrated.predict_proba(x_eval))
    y: np.ndarray = y_eval.to_numpy()
    one_hot: np.ndarray = np.eye(n_classes)[y]

    raw_brier: float = float(np.mean(np.sum((raw_probabilities - one_hot) ** 2, axis=1)))
    calibrated_brier: float = float(np.mean(np.sum((calibrated_probabilities - one_hot) ** 2, axis=1)))

    try:
        labels = list(range(n_classes))
        raw_logloss: float = float(log_loss(y, raw_probabilities, labels=labels))
        calibrated_logloss: float = float(log_loss(y, calibrated_probabilities, labels=labels))
    except ValueError:  # pragma: no cover - degenerate eval slice
        raw_logloss = calibrated_logloss = float("nan")

    # A bare `calibrated < raw` lets an improvement in the fourth decimal place
    # decide which estimator ships. On the audited run the gate "improved" by
    # 0.0007 nats (0.11%) on a single temporal half-split, and that coin flip
    # was what separated a cascade that never predicts LONG from one that goes
    # long on most bars. Requiring a material effect size makes the decision
    # mean something; the measured delta is reported either way.
    relative_gain: float = (
        (raw_logloss - calibrated_logloss) / raw_logloss
        if raw_logloss and raw_logloss == raw_logloss and raw_logloss > 0.0
        else float("nan")
    )
    improved: bool = bool(
        relative_gain == relative_gain and relative_gain > MIN_CALIBRATION_RELATIVE_GAIN
    )
    return {
        "status": "AVAILABLE",
        "method": "isotonic",
        "calibration_rows": int(len(x_calibration)),
        "eval_rows": int(len(x_eval)),
        "brier_score_raw": raw_brier,
        "brier_score_calibrated": calibrated_brier,
        "log_loss_raw": raw_logloss,
        "log_loss_calibrated": calibrated_logloss,
        "log_loss_relative_gain": relative_gain,
        "minimum_relative_gain": MIN_CALIBRATION_RELATIVE_GAIN,
        "improved": bool(improved),
        "recommended_for_production": bool(improved),
        "note": (
            "Measured here; the caller (DirectionModel/EntryModel) swaps this "
            "stage onto the isotonic-calibrated estimator for live inference "
            "whenever `improved` is True, and leaves it on the raw estimator "
            "otherwise - see the model artifact's own `production_calibration` "
            "field for what was actually applied to this run."
        ),
    }


# ---------------------------------------------------------------------------
# Per-symbol breakdown
# ---------------------------------------------------------------------------
def per_symbol_direction_accuracy(
    target: pd.Series, predictions: np.ndarray, symbols: pd.Series
) -> dict[str, dict[str, Any]]:
    """Direction accuracy/balanced-accuracy per symbol on the validation slice.

    Prevents a handful of highly-accurate symbols from hiding poor
    performance across the rest of the universe (Part 23 of the reporting
    spec) - computed directly from real validation predictions, one group-by
    away from what :func:`direction_metrics` already does in aggregate.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    frame = pd.DataFrame(
        {"symbol": symbols.to_numpy(), "y": target.to_numpy(), "pred": predictions}
    )
    result: dict[str, dict[str, Any]] = {}
    for symbol, group in frame.groupby("symbol"):
        y_sub, pred_sub = group["y"].to_numpy(), group["pred"].to_numpy()
        result[str(symbol)] = {
            "samples": int(len(group)),
            "accuracy": float(accuracy_score(y_sub, pred_sub)),
            "balanced_accuracy": (
                float(balanced_accuracy_score(y_sub, pred_sub))
                if len(set(y_sub.tolist())) > 1
                else float(accuracy_score(y_sub, pred_sub))
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def feature_importance(estimator: Any, feature_columns: Sequence[str], top_n: int = 20) -> dict[str, Any]:
    """Gain/split-style native feature importance from a fitted booster.

    SHAP is deliberately not attempted here: the ``shap`` package is not a
    project dependency (see requirements.txt), so computing SHAP values would
    require adding a new dependency the operator has not opted into. That is
    reported explicitly rather than silently skipped.
    """
    raw_importance: Any = getattr(estimator, "feature_importances_", None)
    if raw_importance is None:
        return {"status": "NOT_AVAILABLE", "reason": "estimator exposes no feature_importances_"}

    values: np.ndarray = np.asarray(raw_importance, dtype=float)
    if values.size != len(feature_columns):  # pragma: no cover - defensive
        return {"status": "NOT_AVAILABLE", "reason": "feature_importances_ length mismatch"}

    total: float = float(values.sum())
    ranked: list[tuple[str, float]] = sorted(
        zip(feature_columns, values.tolist()), key=lambda item: item[1], reverse=True
    )
    return {
        "status": "AVAILABLE",
        "method": "native_gain_or_split",
        "top_features": [
            {
                "feature": name,
                "importance": float(value),
                "importance_pct": float(value / total) if total > 0 else 0.0,
            }
            for name, value in ranked[:top_n]
        ],
        "shap": {"status": "NOT_AVAILABLE", "reason": "shap is not an installed project dependency"},
    }
