"""Direction-model evaluation, walk-forward validation and baseline comparison.

Everything in this module exists to answer one question honestly: *is the new
Direction head better than the old one at the very same decision threshold?*

Three rules are enforced structurally rather than by discipline:

1. **The threshold never moves.**  Every metric is computed by replaying the
   Decision Engine's own R1/R2/R3 gates (``min_direction_confidence``,
   ``min_direction_margin``, ``max_no_trade_probability``) exactly as configured.
   A model cannot "improve" here by trading less - trade count is reported
   alongside every quality metric precisely so that a shrinking denominator is
   visible instead of flattering.
2. **Both models are scored by the same function on the same rows.**  There is
   one evaluator; baseline and candidate are two arguments to it.
3. **The test block is untouched until the very end.**  Walk-forward folds are
   carved out of the train+validation region only; the test evaluation is a
   single, final, read-once pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from core.logger import get_logger
from module_b_features.labeler import LABEL_ORDER, LONG_LABELS, SHORT_LABELS
from module_b_features.processor import DatasetSplit, ProcessedDataset

_LOGGER = get_logger(__name__)

#: The three aggregated actions every metric in this module is expressed over.
ACTION_ORDER: Final[tuple[str, ...]] = ("LONG", "SHORT", "NO_TRADE")

_LONG_POSITIONS: Final[tuple[int, ...]] = tuple(
    index for index, name in enumerate(LABEL_ORDER) if name in LONG_LABELS
)
_SHORT_POSITIONS: Final[tuple[int, ...]] = tuple(
    index for index, name in enumerate(LABEL_ORDER) if name in SHORT_LABELS
)


# ---------------------------------------------------------------------------
# Threshold replay
# ---------------------------------------------------------------------------
def aggregate_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse the 5-class distribution into LONG / SHORT / NO_TRADE mass.

    This is the same aggregation :class:`DirectionPrediction` performs, lifted to
    whole arrays so a million validation rows can be scored at once.
    """
    matrix: np.ndarray = np.asarray(probabilities, dtype=np.float64)
    long_mass: np.ndarray = matrix[:, list(_LONG_POSITIONS)].sum(axis=1)
    short_mass: np.ndarray = matrix[:, list(_SHORT_POSITIONS)].sum(axis=1)
    no_trade_mass: np.ndarray = np.clip(1.0 - long_mass - short_mass, 0.0, 1.0)
    return long_mass, short_mass, no_trade_mass


def predicted_actions(probabilities: np.ndarray, settings: Settings) -> np.ndarray:
    """Replay the Decision Engine's direction gates over a batch of rows.

    A row is only labelled LONG or SHORT when it would genuinely have produced a
    directional signal:

    * the winning aggregated mass is directional (R1),
    * that mass clears ``min_direction_confidence`` (R1),
    * the edge over the opposing side clears ``min_direction_margin`` (R2), and
    * the NO_TRADE mass is under ``max_no_trade_probability`` (R3).

    Everything else is NO_TRADE.  **These thresholds are read from the live
    configuration and are never overridden here** - that is what makes a
    comparison run at "the same threshold" a fact rather than a claim.
    """
    long_mass, short_mass, no_trade_mass = aggregate_probabilities(probabilities)
    decision = settings.decision

    winner_is_long: np.ndarray = long_mass >= short_mass
    directional_mass: np.ndarray = np.where(winner_is_long, long_mass, short_mass)
    margin: np.ndarray = np.abs(long_mass - short_mass)

    accepted: np.ndarray = (
        (directional_mass >= no_trade_mass)
        & (directional_mass >= decision.min_direction_confidence)
        & (margin >= decision.min_direction_margin)
        & (no_trade_mass <= decision.max_no_trade_probability)
    )
    actions: np.ndarray = np.full(len(long_mass), "NO_TRADE", dtype=object)
    actions[accepted & winner_is_long] = "LONG"
    actions[accepted & ~winner_is_long] = "SHORT"
    return actions


def true_actions(labels: Sequence[str] | np.ndarray) -> np.ndarray:
    """Collapse the five ground-truth classes into the same three actions."""
    values: np.ndarray = np.asarray(labels, dtype=object)
    actions: np.ndarray = np.full(values.size, "NO_TRADE", dtype=object)
    actions[np.isin(values, list(LONG_LABELS))] = "LONG"
    actions[np.isin(values, list(SHORT_LABELS))] = "SHORT"
    return actions


def direction_metrics(
    probabilities: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    settings: Settings,
) -> dict[str, Any]:
    """Full classification report for one model on one block of rows.

    Returns accuracy, balanced accuracy, macro F1, per-action
    precision/recall/F1, the confusion matrix, and the *counts* that make the
    quality numbers interpretable - most importantly how many directional calls
    were actually made.  A model that fires on 200 rows instead of 20 000 will
    show it here even if its precision looks spectacular.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    predicted: np.ndarray = predicted_actions(probabilities, settings)
    actual: np.ndarray = true_actions(labels)
    if predicted.size == 0:
        return {"rows": 0}

    order: list[str] = list(ACTION_ORDER)
    per_class: dict[str, dict[str, float]] = {}
    precision: np.ndarray = precision_score(
        actual, predicted, labels=order, average=None, zero_division=0
    )
    recall: np.ndarray = recall_score(
        actual, predicted, labels=order, average=None, zero_division=0
    )
    f1: np.ndarray = f1_score(actual, predicted, labels=order, average=None, zero_division=0)
    for position, name in enumerate(order):
        per_class[name] = {
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
            "support": int(np.sum(actual == name)),
            "predicted": int(np.sum(predicted == name)),
        }

    long_mass, short_mass, no_trade_mass = aggregate_probabilities(probabilities)
    return {
        "rows": int(predicted.size),
        "threshold": {
            "min_direction_confidence": settings.decision.min_direction_confidence,
            "min_direction_margin": settings.decision.min_direction_margin,
            "max_no_trade_probability": settings.decision.max_no_trade_probability,
        },
        "accuracy": float(np.mean(predicted == actual)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=order, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(actual, predicted, labels=order, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": order,
            "rows_are_actual": True,
            "matrix": confusion_matrix(actual, predicted, labels=order).tolist(),
        },
        "signals": {
            "directional": int(np.sum(predicted != "NO_TRADE")),
            "long": int(np.sum(predicted == "LONG")),
            "short": int(np.sum(predicted == "SHORT")),
            "signal_rate": float(np.mean(predicted != "NO_TRADE")),
        },
        "probability_stats": {
            "long_mean": float(long_mass.mean()),
            "short_mean": float(short_mass.mean()),
            "no_trade_mean": float(no_trade_mass.mean()),
        },
    }


def long_short_discrimination(
    probabilities: np.ndarray,
    labels: Sequence[str] | np.ndarray,
) -> dict[str, float]:
    """LONG-vs-SHORT separability on the genuinely directional rows only.

    Reported separately from the headline metrics because it is the quantity the
    change is actually trying to move: among bars where a directional trade *was*
    available, how well does the model tell which way?  It is measured with a
    threshold-free AUC so it cannot be gamed by trading less.
    """
    from sklearn.metrics import roc_auc_score

    actual: np.ndarray = true_actions(labels)
    directional: np.ndarray = actual != "NO_TRADE"
    if int(directional.sum()) < 50:
        return {"rows": int(directional.sum()), "long_vs_short_auc": float("nan")}

    long_mass, short_mass, _ = aggregate_probabilities(probabilities)
    total: np.ndarray = long_mass + short_mass
    score: np.ndarray = np.divide(
        long_mass, total, out=np.full_like(long_mass, 0.5), where=total > 1e-12
    )[directional]
    target: np.ndarray = (actual[directional] == "LONG").astype(int)
    if len(set(target.tolist())) < 2:
        return {"rows": int(directional.sum()), "long_vs_short_auc": float("nan")}
    return {
        "rows": int(directional.sum()),
        "long_vs_short_auc": float(roc_auc_score(target, score)),
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class WalkForwardFold:
    """One expanding-window fold, purged and embargoed on both boundaries."""

    index: int
    train: np.ndarray
    validation: np.ndarray

    def describe(self, stamps: np.ndarray) -> dict[str, Any]:
        """Row counts and time bounds, for the fold table in the report."""

        def _bounds(positions: np.ndarray) -> tuple[int | None, int | None]:
            if positions.size == 0:
                return None, None
            values: np.ndarray = stamps[positions]
            return int(values.min()), int(values.max())

        train_start, train_end = _bounds(self.train)
        validation_start, validation_end = _bounds(self.validation)
        return {
            "fold": self.index,
            "train_rows": int(self.train.size),
            "validation_rows": int(self.validation.size),
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
        }


def build_walk_forward_folds(
    dataset: ProcessedDataset,
    settings: Settings,
    split: DatasetSplit,
) -> list[WalkForwardFold]:
    """Carve expanding-window folds out of the train+validation region.

    The test block is excluded outright, so a walk-forward number can never be a
    disguised test-set number.  Each fold trains on everything before its
    validation window and is separated from it by the same purge+embargo gap the
    main split uses, which is what makes the fold scores an honest estimate of
    degradation over time rather than of memorisation.
    """
    stamps: np.ndarray = dataset.timestamps()
    in_sample: np.ndarray = np.sort(np.concatenate([split.train, split.validation]))
    if in_sample.size == 0:
        return []

    ordered: np.ndarray = in_sample[np.argsort(stamps[in_sample], kind="stable")]
    ordered_stamps: np.ndarray = stamps[ordered]
    folds: list[WalkForwardFold] = []
    fold_count: int = max(2, settings.ml.walk_forward_folds)
    gap_ms: int = (settings.ml.purge_bars + settings.ml.embargo_bars) * settings.data.timeframe_ms

    # The first block is always training-only, so N folds need N+1 blocks.
    edges: np.ndarray = np.linspace(0, ordered.size, fold_count + 2, dtype=np.int64)
    for fold_index in range(1, fold_count + 1):
        start, end = int(edges[fold_index]), int(edges[fold_index + 1])
        if end - start < 50:
            continue
        validation_start_ts: int = int(ordered_stamps[start])
        validation_end_ts: int = int(ordered_stamps[end - 1])

        train_mask: np.ndarray = ordered_stamps < validation_start_ts - gap_ms
        validation_mask: np.ndarray = (ordered_stamps >= validation_start_ts) & (
            ordered_stamps <= validation_end_ts
        )
        if int(train_mask.sum()) < 500 or int(validation_mask.sum()) < 50:
            continue
        folds.append(
            WalkForwardFold(
                index=len(folds) + 1,
                train=ordered[train_mask],
                validation=ordered[validation_mask],
            )
        )
    return folds


# ---------------------------------------------------------------------------
# Baseline vs candidate
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ComparisonReport:
    """Everything the before/after comparison produced."""

    generated_at: str
    settings_snapshot: dict[str, Any]
    split: dict[str, Any]
    direction: dict[str, Any] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    trading: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "generated_at": self.generated_at,
            "settings": self.settings_snapshot,
            "split": self.split,
            "direction": self.direction,
            "walk_forward": self.walk_forward,
            "trading": self.trading,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        """Human-readable report - the artefact a person actually reads."""
        return render_markdown(self)


class DirectionComparison:
    """Trains baseline and candidate on identical rows and scores them identically.

    Both models are fitted here rather than loaded, because a comparison against
    an artifact trained on a different split, a different symbol set or a
    different label version is not a comparison at all.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings

    def run(self, dataset: ProcessedDataset) -> ComparisonReport:
        """Fit both heads, evaluate on validation, walk-forward, then test once."""
        from module_c_ml.ml_models import DirectionBaselineModel, DirectionModel

        if dataset.is_empty:
            raise ValueError("comparison needs a non-empty dataset")

        ml = self._settings.ml
        split: DatasetSplit = dataset.chronological_split(
            validation_fraction=ml.validation_fraction,
            test_fraction=ml.test_fraction,
            purge_bars=ml.purge_bars,
            embargo_bars=ml.embargo_bars,
            timeframe_ms=self._settings.data.timeframe_ms,
        )
        stamps: np.ndarray = dataset.timestamps()

        report = ComparisonReport(
            generated_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            settings_snapshot=self._settings_snapshot(),
            split=split.describe(stamps),
        )

        baseline = DirectionBaselineModel(self._settings)
        candidate = DirectionModel(self._settings)
        baseline.train(dataset)
        candidate.train(dataset)
        baseline.save()
        candidate.save()

        report.direction = {
            "validation": {
                "baseline": self._evaluate(baseline, dataset, split.validation),
                "new_idea": self._evaluate(candidate, dataset, split.validation),
            }
        }
        if split.test.size > 0:
            # First and only read of the test block.  Nothing above this line
            # has seen it, and nothing below it is tuned on what it says.
            report.direction["test"] = {
                "baseline": self._evaluate(baseline, dataset, split.test),
                "new_idea": self._evaluate(candidate, dataset, split.test),
            }
        report.direction["delta"] = self._delta(
            report.direction["validation"]["baseline"],
            report.direction["validation"]["new_idea"],
        )

        report.walk_forward = self._walk_forward(dataset, split)
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _evaluate(
        self,
        model: Any,
        dataset: ProcessedDataset,
        positions: np.ndarray,
    ) -> dict[str, Any]:
        """Score one fitted head on one block of rows."""
        if positions.size == 0:
            return {"rows": 0}
        features: pd.DataFrame = dataset.features.iloc[positions]
        labels: np.ndarray = dataset.direction_target.iloc[positions].to_numpy()
        probabilities: np.ndarray = model.predict_proba_frame(features)
        metrics: dict[str, Any] = direction_metrics(probabilities, labels, self._settings)
        metrics["discrimination"] = long_short_discrimination(probabilities, labels)
        metrics["architecture"] = model.architecture
        metrics["features"] = len(model._feature_columns)  # noqa: SLF001 - report detail
        return metrics

    def _walk_forward(self, dataset: ProcessedDataset, split: DatasetSplit) -> dict[str, Any]:
        """Refit both heads per fold and report per-fold plus dispersion."""
        from module_c_ml.ml_models import DirectionBaselineModel, DirectionModel

        folds: list[WalkForwardFold] = build_walk_forward_folds(dataset, self._settings, split)
        if not folds:
            return {"status": "NOT_AVAILABLE", "reason": "not enough in-sample rows to fold"}

        stamps: np.ndarray = dataset.timestamps()
        results: list[dict[str, Any]] = []
        for fold in folds:
            fold_rows: np.ndarray = np.sort(np.concatenate([fold.train, fold.validation]))
            fold_dataset: ProcessedDataset = dataset.subset(fold_rows)
            # Inside a fold the "validation" block is the fold's own out-of-sample
            # window and there is no further test tail to protect.
            fold_settings: Settings = self._settings.model_copy(deep=True)
            fold_settings.ml.test_fraction = 0.0
            fold_settings.ml.validation_fraction = float(
                min(0.45, max(0.05, fold.validation.size / max(1, fold_rows.size)))
            )

            entry: dict[str, Any] = fold.describe(stamps)
            try:
                baseline = DirectionBaselineModel(fold_settings)
                candidate = DirectionModel(fold_settings)
                baseline.train(fold_dataset)
                candidate.train(fold_dataset)

                out_of_sample: np.ndarray = np.searchsorted(fold_rows, fold.validation)
                entry["baseline"] = self._evaluate(baseline, fold_dataset, out_of_sample)
                entry["new_idea"] = self._evaluate(candidate, fold_dataset, out_of_sample)
            except Exception as error:  # pragma: no cover - a fold may be degenerate
                _LOGGER.error("Walk-forward fold %d failed: %s", fold.index, error)
                entry["error"] = str(error)
            results.append(entry)

        return {
            "status": "AVAILABLE",
            "folds": results,
            "stability": self._fold_stability(results),
        }

    @staticmethod
    def _fold_stability(folds: list[dict[str, Any]]) -> dict[str, Any]:
        """Mean and spread of the headline metrics across folds.

        Degradation across walk-forward folds is the failure mode a single
        validation score hides, so the spread is reported next to the mean.
        """
        summary: dict[str, Any] = {}
        for variant in ("baseline", "new_idea"):
            for metric in ("balanced_accuracy", "macro_f1"):
                values: list[float] = [
                    float(fold[variant][metric])
                    for fold in folds
                    if variant in fold and metric in fold.get(variant, {})
                ]
                if values:
                    summary[f"{variant}_{metric}_mean"] = float(np.mean(values))
                    summary[f"{variant}_{metric}_std"] = float(np.std(values))
        return summary

    @staticmethod
    def _delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
        """Candidate minus baseline for the metrics that decide the verdict."""
        deltas: dict[str, float] = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            if metric in baseline and metric in candidate:
                deltas[metric] = float(candidate[metric]) - float(baseline[metric])
        for action in ("LONG", "SHORT", "NO_TRADE"):
            base_class: dict[str, float] = baseline.get("per_class", {}).get(action, {})
            new_class: dict[str, float] = candidate.get("per_class", {}).get(action, {})
            for metric in ("precision", "recall", "f1"):
                if metric in base_class and metric in new_class:
                    deltas[f"{action}_{metric}"] = float(new_class[metric]) - float(
                        base_class[metric]
                    )
        base_signals: dict[str, Any] = baseline.get("signals", {})
        new_signals: dict[str, Any] = candidate.get("signals", {})
        if base_signals and new_signals:
            deltas["directional_signals"] = float(
                new_signals["directional"] - base_signals["directional"]
            )
        return deltas

    def _settings_snapshot(self) -> dict[str, Any]:
        """The knobs a reader needs to reproduce the run."""
        decision = self._settings.decision
        ml = self._settings.ml
        return {
            "min_direction_confidence": decision.min_direction_confidence,
            "min_direction_margin": decision.min_direction_margin,
            "max_no_trade_probability": decision.max_no_trade_probability,
            "min_entry_probability": decision.min_entry_probability,
            "validation_fraction": ml.validation_fraction,
            "test_fraction": ml.test_fraction,
            "purge_bars": ml.purge_bars,
            "embargo_bars": ml.embargo_bars,
            "walk_forward_folds": ml.walk_forward_folds,
            "direction_architecture": ml.direction_architecture,
        }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _format(value: Any, digits: int = 4) -> str:
    """Render a metric, tolerating missing values without exploding."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def _metric_table(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Two-column before/after table with an explicit delta."""
    lines: list[str] = [
        "| Metric | Baseline | New idea | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]

    def _row(label: str, base: Any, new: Any) -> None:
        delta: str = "n/a"
        if isinstance(base, (int, float)) and isinstance(new, (int, float)):
            delta = f"{float(new) - float(base):+.4f}"
        lines.append(f"| {label} | {_format(base)} | {_format(new)} | {delta} |")

    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        _row(metric.replace("_", " ").title(), baseline.get(metric), candidate.get(metric))
    for action in ("LONG", "SHORT", "NO_TRADE"):
        base_class: dict[str, Any] = baseline.get("per_class", {}).get(action, {})
        new_class: dict[str, Any] = candidate.get("per_class", {}).get(action, {})
        for metric in ("precision", "recall", "f1"):
            _row(f"{action} {metric}", base_class.get(metric), new_class.get(metric))
    _row(
        "LONG-vs-SHORT AUC",
        baseline.get("discrimination", {}).get("long_vs_short_auc"),
        candidate.get("discrimination", {}).get("long_vs_short_auc"),
    )
    _row(
        "Directional signals",
        baseline.get("signals", {}).get("directional"),
        candidate.get("signals", {}).get("directional"),
    )
    return lines


def _confusion_block(title: str, metrics: dict[str, Any]) -> list[str]:
    """Render one confusion matrix with actual rows and predicted columns."""
    matrix: dict[str, Any] = metrics.get("confusion_matrix", {})
    if not matrix:
        return []
    labels: list[str] = list(matrix.get("labels", ACTION_ORDER))
    lines: list[str] = [f"**{title}** (rows = actual, columns = predicted)", ""]
    lines.append("| actual \\ predicted | " + " | ".join(labels) + " |")
    lines.append("| --- |" + " ---: |" * len(labels))
    for row_label, row in zip(labels, matrix.get("matrix", []), strict=False):
        lines.append(f"| {row_label} | " + " | ".join(str(int(value)) for value in row) + " |")
    lines.append("")
    return lines


def render_markdown(report: ComparisonReport) -> str:
    """Render the full baseline-vs-new-idea report as Markdown."""
    settings_snapshot: dict[str, Any] = report.settings_snapshot
    lines: list[str] = [
        "# Baseline vs New Idea",
        "",
        f"Generated {report.generated_at}",
        "",
        "## Evaluation contract",
        "",
        "Both models are trained on the same rows and scored by the same evaluator.",
        "The direction decision threshold is **unchanged** and is applied identically",
        "to both:",
        "",
        f"- `min_direction_confidence` = {settings_snapshot.get('min_direction_confidence')}",
        f"- `min_direction_margin` = {settings_snapshot.get('min_direction_margin')}",
        f"- `max_no_trade_probability` = {settings_snapshot.get('max_no_trade_probability')}",
        "",
        "## Split",
        "",
        "| Block | Rows | Start | End |",
        "| --- | ---: | --- | --- |",
    ]
    for name in ("train", "validation", "test"):
        block: dict[str, Any] = report.split.get(name, {})
        lines.append(
            f"| {name} | {block.get('rows', 0)} | {block.get('start')} | {block.get('end')} |"
        )
    lines.append("")

    validation: dict[str, Any] = report.direction.get("validation", {})
    if validation:
        lines += ["## Direction - validation block", ""]
        lines += _metric_table(validation.get("baseline", {}), validation.get("new_idea", {}))
        lines += [""]
        lines += _confusion_block("Baseline", validation.get("baseline", {}))
        lines += _confusion_block("New idea", validation.get("new_idea", {}))

    test: dict[str, Any] = report.direction.get("test", {})
    if test:
        lines += [
            "## Direction - test block (out of sample, read once)",
            "",
        ]
        lines += _metric_table(test.get("baseline", {}), test.get("new_idea", {}))
        lines += [""]
        lines += _confusion_block("Baseline", test.get("baseline", {}))
        lines += _confusion_block("New idea", test.get("new_idea", {}))

    walk_forward: dict[str, Any] = report.walk_forward
    if walk_forward.get("status") == "AVAILABLE":
        lines += [
            "## Walk-forward folds",
            "",
            "| Fold | Rows | Baseline balanced acc | New balanced acc | "
            "Baseline macro F1 | New macro F1 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for fold in walk_forward.get("folds", []):
            baseline_fold: dict[str, Any] = fold.get("baseline", {})
            new_fold: dict[str, Any] = fold.get("new_idea", {})
            lines.append(
                f"| {fold.get('fold')} | {fold.get('validation_rows')} | "
                f"{_format(baseline_fold.get('balanced_accuracy'))} | "
                f"{_format(new_fold.get('balanced_accuracy'))} | "
                f"{_format(baseline_fold.get('macro_f1'))} | "
                f"{_format(new_fold.get('macro_f1'))} |"
            )
        stability: dict[str, Any] = walk_forward.get("stability", {})
        if stability:
            lines += ["", "Fold dispersion:", ""]
            for key, value in sorted(stability.items()):
                lines.append(f"- `{key}` = {_format(value)}")
        lines.append("")

    trading: dict[str, Any] = report.trading
    if trading:
        lines += ["## Trading (backtest, identical execution assumptions)", ""]
        lines += _trading_table(trading.get("baseline", {}), trading.get("new_idea", {}))
        lines.append("")
        lines += _ladder_table(trading.get("new_idea", {}))

    if report.notes:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in report.notes]
    return "\n".join(lines) + "\n"


def _trading_table(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Before/after table of the trading metrics."""
    rows: tuple[tuple[str, str], ...] = (
        ("Net PnL", "net_profit"),
        ("Win rate", "win_rate"),
        ("Profit factor", "profit_factor"),
        ("Expectancy", "expectancy"),
        ("Max drawdown", "max_drawdown_pct"),
        ("Trades", "total_trades"),
        ("Average realised R", "average_r"),
        ("Average trade return", "average_trade_return"),
        ("LONG trades", "long_trades"),
        ("LONG win rate", "long_win_rate"),
        ("LONG net PnL", "long_net_profit"),
        ("SHORT trades", "short_trades"),
        ("SHORT win rate", "short_win_rate"),
        ("SHORT net PnL", "short_net_profit"),
    )
    lines: list[str] = [
        "| Metric | Baseline | New idea | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in rows:
        base: Any = baseline.get(key)
        new: Any = candidate.get(key)
        delta: str = "n/a"
        if isinstance(base, (int, float)) and isinstance(new, (int, float)):
            delta = f"{float(new) - float(base):+.4f}"
        lines.append(f"| {label} | {_format(base)} | {_format(new)} | {delta} |")
    return lines


def _ladder_table(metrics: dict[str, Any]) -> list[str]:
    """Take-profit ladder statistics for the candidate run."""
    if not metrics:
        return []
    rows: tuple[tuple[str, str], ...] = (
        ("Reached TP1", "pct_reached_tp1"),
        ("Reached TP2", "pct_reached_tp2"),
        ("Reached TP3", "pct_reached_tp3"),
        ("Stopped at breakeven", "pct_stopped_breakeven"),
        ("Stopped at TP1-protected profit", "pct_stopped_tp1_locked"),
        ("Stopped at initial SL", "pct_stopped_initial"),
        ("Average realised R", "average_r"),
        ("Average trade return", "average_trade_return"),
    )
    lines: list[str] = ["### Take-profit ladder", "", "| Metric | Value |", "| --- | ---: |"]
    for label, key in rows:
        lines.append(f"| {label} | {_format(metrics.get(key))} |")
    return lines
