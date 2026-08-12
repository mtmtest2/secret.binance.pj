"""Direction head, and the evaluator that compares it against the baseline."""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import Settings
from module_b_features.labeler import LABEL_ORDER, LabelClass
from module_c_ml.decision_engine import DecisionContext, DecisionEngine
from module_c_ml.evaluation import (
    DirectionComparison,
    direction_metrics,
    predicted_actions,
    true_actions,
)
from module_c_ml.ml_models import DirectionBaselineModel, DirectionModel
from module_c_ml.schemas import (
    DirectionPrediction,
    EntryPrediction,
    ExitParameters,
    ModelInferenceResult,
    RiskAllocation,
    TradeAction,
)
from tests.conftest import make_pooled_dataset


def _distribution(long: float, short: float) -> np.ndarray:
    """A 5-class row carrying ``long``/``short`` mass, the rest on NO_TRADE."""
    row = np.zeros(len(LABEL_ORDER), dtype=float)
    index = {name: position for position, name in enumerate(LABEL_ORDER)}
    row[index[LabelClass.LONG_SUCCESS_LOW_RISK.value]] = long
    row[index[LabelClass.SHORT_SUCCESS_LOW_RISK.value]] = short
    row[index[LabelClass.NO_TRADE_OR_FAIL.value]] = max(0.0, 1.0 - long - short)
    return row


# ---------------------------------------------------------------------------
# Threshold parity - the evaluator must agree with the live Decision Engine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "long_mass,short_mass",
    [
        (0.80, 0.10),  # comfortably accepted
        (0.72, 0.20),  # over the confidence bar, thin margin
        (0.65, 0.10),  # under the confidence bar
        (0.40, 0.38),  # nearly indifferent
        (0.10, 0.85),  # a clean short
        (0.50, 0.30),  # NO_TRADE mass too high
    ],
)
def test_evaluator_reproduces_the_decision_engine_exactly(
    settings: Settings,
    long_mass: float,
    short_mass: float,
) -> None:
    """The reported metrics must be measured at the *live* threshold.

    If the evaluator and the Decision Engine disagreed about what counts as a
    signal, every number in the comparison report would describe a system that
    is not the one being run.  This pins them together.
    """
    probabilities = _distribution(long_mass, short_mass)
    evaluator_action = predicted_actions(probabilities[np.newaxis, :], settings)[0]

    prediction = DirectionPrediction(
        probabilities={name: float(probabilities[i]) for i, name in enumerate(LABEL_ORDER)}
    )
    inference = ModelInferenceResult(
        symbol="TEST/USDT:USDT",
        timestamp=1_700_000_000_000,
        close_price=100.0,
        direction=prediction,
        entry=EntryPrediction(probability=0.99, should_enter=True),
        exit_params=ExitParameters(
            take_profit_pct=0.03,
            stop_loss_pct=0.01,
            trailing_activation_pct=0.015,
            trailing_distance_pct=0.006,
        ),
        risk=RiskAllocation(leverage=2, capital_allocation_pct=0.1, risk_score=0.6, risk_tier="LOW"),
    )
    result = DecisionEngine(settings).evaluate(inference, DecisionContext())

    engine_action = (
        result.signal.action.value if result.is_executable else TradeAction.NO_TRADE.value
    )
    assert evaluator_action == engine_action


def test_metrics_report_the_signal_count_alongside_quality(settings: Settings) -> None:
    """Trading less must be visible, so precision cannot masquerade as skill."""
    probabilities = np.vstack(
        [_distribution(0.9, 0.05)] * 5 + [_distribution(0.3, 0.3)] * 95
    )
    labels = np.array([LabelClass.LONG_SUCCESS_LOW_RISK.value] * 5 + [
        LabelClass.NO_TRADE_OR_FAIL.value
    ] * 95)

    metrics = direction_metrics(probabilities, labels, settings)
    assert metrics["signals"]["directional"] == 5
    assert metrics["signals"]["long"] == 5
    assert metrics["per_class"]["LONG"]["precision"] == pytest.approx(1.0)
    assert metrics["rows"] == 100
    assert metrics["threshold"]["min_direction_confidence"] == (
        settings.decision.min_direction_confidence
    )


def test_true_actions_collapse_the_five_classes(settings: Settings) -> None:
    """Both LONG tiers are LONG; both SHORT tiers are SHORT."""
    labels = np.array(
        [
            LabelClass.LONG_SUCCESS_LOW_RISK.value,
            LabelClass.LONG_SUCCESS_HIGH_RISK.value,
            LabelClass.SHORT_SUCCESS_LOW_RISK.value,
            LabelClass.SHORT_SUCCESS_HIGH_RISK.value,
            LabelClass.NO_TRADE_OR_FAIL.value,
        ]
    )
    assert list(true_actions(labels)) == ["LONG", "LONG", "SHORT", "SHORT", "NO_TRADE"]


# ---------------------------------------------------------------------------
# The head itself
# ---------------------------------------------------------------------------
def test_cascade_emits_a_valid_five_class_distribution(settings: Settings) -> None:
    """The new architecture keeps the exact output contract of the old one."""
    dataset = make_pooled_dataset(settings, symbols=3, rows=1_800)
    model = DirectionModel(settings)
    model.train(dataset)

    assert model.architecture == "two_stage_cascade"
    probabilities = model.predict_proba_frame(dataset.features.iloc[:200])
    assert probabilities.shape == (200, len(LABEL_ORDER))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9)

    prediction = model.predict(dataset.features.iloc[[0]])
    assert set(prediction.probabilities) == set(LABEL_ORDER)
    assert prediction.confidence <= 1.0


def test_baseline_is_pinned_to_the_old_architecture_and_features(settings: Settings) -> None:
    """The comparison point must not drift when the production head changes."""
    dataset = make_pooled_dataset(settings, symbols=2, rows=1_500)
    baseline = DirectionBaselineModel(settings)
    baseline.train(dataset)

    assert baseline.architecture == "single_stage"
    assert baseline.artifact_path.name == "direction_model_baseline.joblib"
    # No order-flow columns: the baseline is what ran before this change.
    assert not any("5m" in column for column in baseline._feature_columns)  # noqa: SLF001

    # Flipping the global architecture setting must not move the baseline.
    settings.ml.direction_architecture = "single_stage"
    assert DirectionBaselineModel(settings).architecture == "single_stage"
    settings.ml.direction_architecture = "two_stage_cascade"
    assert DirectionBaselineModel(settings).architecture == "single_stage"


def test_cascade_separates_long_from_short_better_than_the_flat_head(
    settings: Settings,
) -> None:
    """The point of the change, measured threshold-free.

    On a market whose order flow genuinely leads price, the cascade - which
    spends a whole stage on long-vs-short over tradeable rows only, and gets the
    order-flow block the baseline does not have - should discriminate direction
    at least as well as the flat five-class head.
    """
    from module_c_ml.evaluation import long_short_discrimination

    dataset = make_pooled_dataset(settings, symbols=4, rows=2_400)
    split = dataset.chronological_split(
        validation_fraction=settings.ml.validation_fraction,
        test_fraction=settings.ml.test_fraction,
        purge_bars=settings.ml.purge_bars,
        embargo_bars=settings.ml.embargo_bars,
        timeframe_ms=settings.data.timeframe_ms,
    )

    baseline = DirectionBaselineModel(settings)
    candidate = DirectionModel(settings)
    baseline.train(dataset)
    candidate.train(dataset)

    features = dataset.features.iloc[split.validation]
    labels = dataset.direction_target.iloc[split.validation].to_numpy()

    baseline_auc = long_short_discrimination(
        baseline.predict_proba_frame(features), labels
    )["long_vs_short_auc"]
    candidate_auc = long_short_discrimination(
        candidate.predict_proba_frame(features), labels
    )["long_vs_short_auc"]

    assert not np.isnan(candidate_auc)
    # A tolerance, not an equality: this is a statistical claim on 4 synthetic
    # symbols, and the real verdict comes from `python main.py compare`.
    assert candidate_auc >= baseline_auc - 0.02


def test_comparison_runs_end_to_end_and_holds_the_threshold(settings: Settings) -> None:
    """The report is produced, and both variants are scored at the same cut-off."""
    dataset = make_pooled_dataset(settings, symbols=3, rows=2_000)
    report = DirectionComparison(settings).run(dataset)

    validation = report.direction["validation"]
    assert validation["baseline"]["threshold"] == validation["new_idea"]["threshold"]
    assert validation["baseline"]["rows"] == validation["new_idea"]["rows"]
    assert validation["baseline"]["architecture"] == "single_stage"
    assert validation["new_idea"]["architecture"] == "two_stage_cascade"

    for variant in ("baseline", "new_idea"):
        metrics = validation[variant]
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "per_class", "confusion_matrix"):
            assert key in metrics
        for action in ("LONG", "SHORT", "NO_TRADE"):
            assert set(metrics["per_class"][action]) >= {"precision", "recall", "f1", "support"}

    assert "test" in report.direction
    assert report.walk_forward["status"] in {"AVAILABLE", "NOT_AVAILABLE"}

    markdown = report.to_markdown()
    assert "Baseline vs New Idea" in markdown
    assert "min_direction_confidence" in markdown
