"""Independent gate/direction confidence architecture (Direction model + Decision Engine).

Covers the fix for the root cause diagnosed in the ML report: DirectionModel's
old ``predict()`` only ever exposed the *joint* probability
(``trade_probability * direction_given_trade_probability``), so the Decision
Engine's R1 gated on one product - which silently discarded a confident
direction call (e.g. 95% sure of LONG given a trade) whenever the gate alone
read under 0.5. The fix threads the two raw per-stage probabilities through
``DirectionPrediction`` and gates on them independently (R1A/R1B in
``module_c_ml.decision_engine``).

Also covers the two new threshold-sweep helpers in ``module_c_ml.metrics``,
the joint-probability calibrator added to ``DirectionModel``, and the
experimental focal-loss gate objective's off-by-default guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS
from module_b_features.processor import ProcessedDataset
from module_c_ml import metrics as ml_metrics
from module_c_ml.decision_engine import DecisionContext, DecisionEngine, Rule
from module_c_ml.ml_models import DirectionModel
from module_c_ml.schemas import (
    DecisionVerdict,
    DirectionPrediction,
    EntryPrediction,
    ExitParameters,
    ModelInferenceResult,
    ModelSource,
    RiskAllocation,
    TradeAction,
)


# ---------------------------------------------------------------------------
# (a) DirectionPrediction exposes the two new raw fields
# ---------------------------------------------------------------------------
def test_direction_prediction_exposes_raw_gate_and_direction_probabilities() -> None:
    pred = DirectionPrediction(
        probabilities={"LONG_SUCCESS": 0.4275, "SHORT_SUCCESS": 0.0225, "NO_TRADE_OR_FAIL": 0.55},
        source=ModelSource.TRAINED,
        trade_probability=0.45,
        direction_given_trade_probability=0.95,
    )
    assert pred.trade_probability == pytest.approx(0.45)
    assert pred.direction_given_trade_probability == pytest.approx(0.95)
    # The old joint-probability action/confidence still exist, unchanged, and
    # disagree with the raw per-stage read - that disagreement is the whole point.
    assert pred.action is TradeAction.NO_TRADE
    assert pred.confidence == pytest.approx(0.55)


def test_direction_prediction_raw_fields_default_to_even_odds() -> None:
    # A caller that does not pass the new fields (e.g. old test fixtures) must
    # not break - and must not silently imply a confident read either.
    pred = DirectionPrediction(
        probabilities={"LONG_SUCCESS": 0.5, "SHORT_SUCCESS": 0.3, "NO_TRADE_OR_FAIL": 0.2},
        source=ModelSource.TRAINED,
    )
    assert pred.trade_probability == pytest.approx(0.5)
    assert pred.direction_given_trade_probability == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# (b) Decision Engine gates the gate and direction stages independently
# ---------------------------------------------------------------------------
def _inference(direction: DirectionPrediction) -> ModelInferenceResult:
    return ModelInferenceResult(
        symbol="BTC/USDT:USDT",
        timestamp=0,
        close_price=100.0,
        direction=direction,
        entry=EntryPrediction(probability=0.9, should_enter=True, source=ModelSource.TRAINED),
        exit_params=ExitParameters(
            take_profit_pct=0.02,
            stop_loss_pct=0.01,
            trailing_activation_pct=0.005,
            trailing_distance_pct=0.005,
            source=ModelSource.TRAINED,
        ),
        risk=RiskAllocation(
            leverage=3, capital_allocation_pct=0.05, risk_score=0.5, risk_tier="LOW",
            source=ModelSource.TRAINED,
        ),
    )


def test_low_gate_confidence_rejects_on_r1a_even_when_direction_is_confident() -> None:
    """The exact scenario the old architecture got wrong: gate reads <0.5 but
    direction-given-trade is 95% sure. Under the old joint-probability R1,
    this fell through to a NO_TRADE verdict framed as "Direction model favours
    NO_TRADE" - a misleading reason, since the model was never unsure about
    direction, only about whether to trade at all. It must now be rejected
    honestly by R1A (the gate), not by a rule that misattributes the cause.
    """
    engine = DecisionEngine(Settings())
    direction = DirectionPrediction(
        probabilities={"LONG_SUCCESS": 0.4275, "SHORT_SUCCESS": 0.0225, "NO_TRADE_OR_FAIL": 0.55},
        source=ModelSource.TRAINED,
        trade_probability=0.45,
        direction_given_trade_probability=0.95,
    )
    result = engine.evaluate(_inference(direction), DecisionContext(trading_enabled=True))

    assert result.verdict is DecisionVerdict.NO_TRADE
    assert result.rule_triggered == Rule.GATE_CONFIDENCE
    assert result.rule_triggered == "R1A_GATE_CONFIDENCE_TOO_LOW"
    checks_by_rule = {check["rule"]: check for check in result.checks}
    assert checks_by_rule[Rule.GATE_CONFIDENCE]["passed"] is False


def test_high_gate_and_direction_confidence_executes() -> None:
    """The counterpart: once the gate itself clears its own floor and the
    direction-given-trade read is confident, both R1A and R1B pass and the
    engine proceeds (subject to the remaining rules) rather than being
    silently blocked by an unrelated joint-probability threshold.
    """
    settings = Settings()
    engine = DecisionEngine(settings)
    trade_probability, long_given_trade = 0.75, 0.95
    long_p = trade_probability * long_given_trade
    short_p = trade_probability * (1.0 - long_given_trade)
    no_trade_p = 1.0 - long_p - short_p
    direction = DirectionPrediction(
        probabilities={"LONG_SUCCESS": long_p, "SHORT_SUCCESS": short_p, "NO_TRADE_OR_FAIL": no_trade_p},
        source=ModelSource.TRAINED,
        trade_probability=trade_probability,
        direction_given_trade_probability=long_given_trade,
    )
    result = engine.evaluate(_inference(direction), DecisionContext(trading_enabled=True))

    assert result.verdict is DecisionVerdict.EXECUTE
    assert result.rule_triggered == Rule.EXECUTE
    assert result.signal is not None
    assert result.signal.action is TradeAction.LONG


def test_direction_margin_rule_and_string_are_retired() -> None:
    assert not hasattr(Rule, "DIRECTION_MARGIN")
    assert not hasattr(Rule, "DIRECTION_NO_TRADE")
    assert Rule.GATE_CONFIDENCE == "R1A_GATE_CONFIDENCE_TOO_LOW"
    assert Rule.DIRECTION_CONFIDENCE == "R1B_DIRECTION_CONFIDENCE_TOO_LOW"


# ---------------------------------------------------------------------------
# (c) gate_threshold_sweep / direction_threshold_sweep
# ---------------------------------------------------------------------------
def test_gate_threshold_sweep_precision_recall_at_known_thresholds() -> None:
    # 10 rows: is_trade ground truth vs raw gate probability.
    target = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.65, 0.3, 0.2, 0.1, 0.05])

    sweep = ml_metrics.gate_threshold_sweep(target, probabilities, thresholds=(0.5, 0.7))
    by_threshold = {row["threshold"]: row for row in sweep}

    # At 0.5: predicted positive = {0.9,0.8,0.7,0.6,0.65} = 5 rows, 4 true positives
    # (0.4 missed) + 1 false positive (0.65) -> precision=4/5, recall=4/5.
    row_50 = by_threshold[0.5]
    assert row_50["signals"] == 5
    assert row_50["precision"] == pytest.approx(0.8)
    assert row_50["recall"] == pytest.approx(0.8)

    # At 0.7: predicted positive = {0.9,0.8,0.7} = 3 rows, all true positives.
    row_70 = by_threshold[0.7]
    assert row_70["signals"] == 3
    assert row_70["precision"] == pytest.approx(1.0)
    assert row_70["recall"] == pytest.approx(0.6)
    assert row_70["average_r"] == "NOT_AVAILABLE"


def test_direction_threshold_sweep_precision_recall_at_known_thresholds() -> None:
    # Ground truth restricted to true-trade rows: 1 = LONG, 0 = SHORT.
    target = pd.Series([1, 1, 1, 0, 0, 0])
    probabilities = np.array([0.95, 0.85, 0.55, 0.6, 0.4, 0.2])

    sweep = ml_metrics.direction_threshold_sweep(target, probabilities, thresholds=(0.5, 0.9))
    by_threshold = {row["threshold"]: row for row in sweep}

    # At 0.5: predicted LONG = {0.95,0.85,0.55,0.6} -> 3 true positives, 1 false positive.
    row_50 = by_threshold[0.5]
    assert row_50["signals"] == 4
    assert row_50["precision"] == pytest.approx(0.75)
    assert row_50["recall"] == pytest.approx(1.0)

    # At 0.9: predicted LONG = {0.95} -> 1 true positive only.
    row_90 = by_threshold[0.9]
    assert row_90["signals"] == 1
    assert row_90["precision"] == pytest.approx(1.0)
    assert row_90["recall"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# (d) Joint-probability calibrator round-trips and renormalises to ~1
# ---------------------------------------------------------------------------
def test_joint_calibrators_fit_and_renormalise_to_one() -> None:
    rng = np.random.default_rng(0)
    n = 400
    raw = rng.dirichlet(alpha=(1.0, 1.0, 1.0), size=n)  # already-normalised 3-class rows
    target = np.array([row.argmax() for row in raw])  # perfectly separable-ish ground truth

    calibrators = DirectionModel._calibrate_joint_probabilities(raw, target, n_classes=3)
    assert len(calibrators) == 3

    calibrated = DirectionModel._apply_joint_calibrators(calibrators, raw)
    assert calibrated.shape == raw.shape
    row_sums = calibrated.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(n), atol=1e-9)
    assert (calibrated >= 0.0).all()


def test_direction_model_trains_with_measurable_joint_calibration_report() -> None:
    rng = np.random.default_rng(21)
    n = 5000
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    signal = features["atr_pct"] + 0.5 * features["adx"]
    probability = 1.0 / (1.0 + np.exp(-signal))
    direction_target = pd.Series(
        np.where(
            probability > 0.66, "LONG_SUCCESS",
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
    dataset = ProcessedDataset(
        features=features, direction_target=direction_target, entry_target=entry_target,
        exit_targets=exit_targets, risk_target=risk_target, metadata=metadata,
        symbols=("BTC/USDT:USDT",), feature_columns=tuple(FEATURE_COLUMNS),
    )

    settings = Settings(
        ml={"n_estimators": 50, "early_stopping_rounds": 10, "purge_bars": 10}
    )
    model = DirectionModel(settings)
    model.train(dataset)

    joint = model.metadata["calibration"]["joint"]
    assert joint["status"] in {"AVAILABLE", "NOT_AVAILABLE"}
    if joint["status"] == "AVAILABLE":
        assert "improved" in joint and "recommended_for_production" in joint

    prediction = model.predict(dataset.features.iloc[[0]])
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    # trade_probability/direction_given_trade_probability are never touched by
    # joint calibration - they stay the raw per-stage values R1a/R1b gate on.
    assert 0.0 <= prediction.trade_probability <= 1.0
    assert 0.0 <= prediction.direction_given_trade_probability <= 1.0


# ---------------------------------------------------------------------------
# Focal loss gate objective: off by default, byte-identical when disabled
# ---------------------------------------------------------------------------
def _small_dataset(rng: np.random.Generator, n: int = 600) -> ProcessedDataset:
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    direction_target = pd.Series(rng.choice(["LONG_SUCCESS", "SHORT_SUCCESS", "NO_TRADE_OR_FAIL"], size=n))
    entry_target = pd.Series(rng.integers(0, 2, size=n))
    exit_targets = pd.DataFrame(
        {
            "target_tp_pct": np.abs(rng.normal(0.02, 0.005, size=n)),
            "target_sl_pct": np.abs(rng.normal(0.01, 0.002, size=n)),
            "target_trailing_pct": np.abs(rng.normal(0.01, 0.002, size=n)),
        }
    )
    risk_target = pd.Series(rng.beta(1.5, 4.0, size=n))
    metadata = pd.DataFrame({"symbol": ["BTC/USDT:USDT"] * n, "timestamp": np.arange(n) * 300_000})
    return ProcessedDataset(
        features=features, direction_target=direction_target, entry_target=entry_target,
        exit_targets=exit_targets, risk_target=risk_target, metadata=metadata,
        symbols=("BTC/USDT:USDT",), feature_columns=tuple(FEATURE_COLUMNS),
    )


def test_focal_loss_flag_off_keeps_the_gate_objective_untouched() -> None:
    settings = Settings(ml={"n_estimators": 15, "early_stopping_rounds": 5})
    assert settings.ml.use_focal_loss_for_gate is False
    dataset = _small_dataset(np.random.default_rng(50))

    model = DirectionModel(settings)
    model.train(dataset)

    assert model._model["gate"].get_params()["objective"] == "binary"


def test_focal_loss_flag_on_produces_valid_probabilities_and_round_trips_through_save_load(
    tmp_path,
) -> None:
    settings = Settings(
        ml={
            "model_dir": tmp_path,
            "n_estimators": 15,
            "early_stopping_rounds": 5,
            "use_focal_loss_for_gate": True,
        }
    )
    dataset = _small_dataset(np.random.default_rng(51))

    model = DirectionModel(settings)
    model.train(dataset)

    prediction = model.predict(dataset.features.iloc[[0]])
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= prediction.trade_probability <= 1.0

    path = model.save()
    reloaded = DirectionModel(settings)
    assert reloaded.load(path) is True
    reloaded_prediction = reloaded.predict(dataset.features.iloc[[0]])
    assert reloaded_prediction.trade_probability == pytest.approx(prediction.trade_probability)


def test_focal_loss_binary_gradient_and_hessian_shapes_and_signs() -> None:
    from module_c_ml.ml_models import focal_loss_binary

    # Row 0: true=1, raw=+2 -> confidently RIGHT.  Row 1: true=0, raw=+2 -> confidently WRONG.
    # Row 2: true=1, raw=-2 -> confidently WRONG.  Row 3: true=0, raw=-2 -> confidently RIGHT.
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred_raw = np.array([2.0, 2.0, -2.0, -2.0])

    grad, hess = focal_loss_binary(y_true, y_pred_raw, gamma=2.0)
    assert grad.shape == y_true.shape
    assert hess.shape == y_true.shape
    assert (hess >= 0.0).all()
    # Confidently-wrong rows (1, 2) should carry a larger-magnitude gradient
    # than confidently-right rows (0, 3) - that is the whole point of
    # down-weighting the easy region.
    assert abs(grad[1]) > abs(grad[0])
    assert abs(grad[2]) > abs(grad[3])
