"""Task 5 - RiskModel and evaluate_many must read the *independent*
direction-given-trade conditional confidence, not the stale *joint*
long/short/no_trade confidence.

Covers:
* ``MLSubsystem.infer_sync`` now passes
  ``max(direction_given_trade_probability, 1 - direction_given_trade_probability)``
  into ``RiskModel.predict``'s ``direction_confidence`` kwarg, instead of the
  old ``DirectionPrediction.confidence`` (joint) property.
* ``DecisionEngine.evaluate_many`` ranks candidates by the same conditional
  confidence, not the joint one, when the portfolio cap forces a choice.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS
from module_b_features.processor import InferencePayload
from module_c_ml.decision_engine import DecisionContext, DecisionEngine
from module_c_ml.ml_models import MLSubsystem
from module_c_ml.schemas import DirectionPrediction, ModelInferenceResult, ModelSource


def _direction_prediction(trade_probability: float, direction_given_trade: float) -> DirectionPrediction:
    """A crafted prediction where the joint `.confidence` and the new
    conditional confidence are deliberately far apart, so a test that reads
    the wrong one fails loudly.
    """
    long_mass = trade_probability * direction_given_trade
    short_mass = trade_probability * (1.0 - direction_given_trade)
    no_trade_mass = max(1e-9, 1.0 - trade_probability)
    return DirectionPrediction(
        probabilities={
            "LONG_SUCCESS": long_mass,
            "SHORT_SUCCESS": short_mass,
            "NO_TRADE_OR_FAIL": no_trade_mass,
        },
        source=ModelSource.HEURISTIC,
        trade_probability=trade_probability,
        direction_given_trade_probability=direction_given_trade,
    )


def test_infer_sync_passes_conditional_not_joint_confidence_to_risk_model(monkeypatch) -> None:
    settings = Settings()
    ml = MLSubsystem(settings)

    # trade_probability=0.7, direction_given_trade=0.65 -> long_mass=0.455,
    # short_mass=0.245, no_trade_mass=0.30. Joint `.confidence` (max of the
    # three) is 0.455 - well under the old 0.70 floor this head used to read.
    # The new conditional confidence is max(0.65, 0.35) = 0.65 - a
    # meaningfully different number that clears the 0.60 default floor.
    crafted = _direction_prediction(trade_probability=0.7, direction_given_trade=0.65)
    assert crafted.confidence == pytest.approx(0.455, abs=1e-9)
    expected_conditional = max(0.65, 1.0 - 0.65)
    assert expected_conditional == pytest.approx(0.65)
    assert crafted.confidence != pytest.approx(expected_conditional)

    monkeypatch.setattr(ml.direction, "predict", lambda features: crafted)

    captured: dict[str, float] = {}
    original_risk_predict = ml.risk.predict

    def _spy_predict(features: pd.DataFrame, direction_confidence: float) -> object:
        captured["direction_confidence"] = direction_confidence
        return original_risk_predict(features, direction_confidence=direction_confidence)

    monkeypatch.setattr(ml.risk, "predict", _spy_predict)

    features = pd.DataFrame([{column: 0.0 for column in FEATURE_COLUMNS}])
    features["garch_vol_rank"] = 0.2
    payload = InferencePayload(
        symbol="BTC/USDT:USDT", timestamp=0, close=100.0, features=features, snapshot={}
    )

    result: ModelInferenceResult = ml.infer_sync(payload)

    assert captured["direction_confidence"] == pytest.approx(expected_conditional)
    assert captured["direction_confidence"] != pytest.approx(crafted.confidence)
    assert result.risk.risk_score >= 0.0


def test_evaluate_many_ranks_by_conditional_not_joint_confidence() -> None:
    """Symbol A has a lower joint confidence but a higher conditional
    confidence than Symbol B. With a portfolio cap of 1, only the symbol
    ranked first gets to compete for the slot - evaluate_many must pick A.
    """
    settings = Settings(decision={"max_concurrent_positions": 1})
    engine = DecisionEngine(settings)

    # A: trade_probability=0.66, direction_given_trade=0.90 -> long_mass=0.594,
    # no_trade_mass=0.34 (clears every other rule's default thresholds too -
    # R1A gate, R1B direction, R3 no-trade cap). B: trade_probability=0.95,
    # direction_given_trade=0.65 -> long_mass=0.6175. B's *joint* confidence
    # (0.6175) is higher than A's (0.594), but A's *conditional* confidence
    # (0.90) is higher than B's (0.65) - a joint-ranked sort puts B first, a
    # conditional-ranked sort puts A first.
    direction_a = _direction_prediction(trade_probability=0.66, direction_given_trade=0.90)
    direction_b = _direction_prediction(trade_probability=0.95, direction_given_trade=0.65)

    conditional_a = max(direction_a.direction_given_trade_probability, 1.0 - direction_a.direction_given_trade_probability)
    conditional_b = max(direction_b.direction_given_trade_probability, 1.0 - direction_b.direction_given_trade_probability)
    assert conditional_a > conditional_b, "test fixture must make A's conditional confidence the larger one"
    assert direction_a.confidence < direction_b.confidence, "test fixture must make A's joint confidence the smaller one"

    def _inference(symbol: str, direction: DirectionPrediction) -> ModelInferenceResult:
        from module_c_ml.schemas import EntryPrediction, ExitParameters, RiskAllocation

        return ModelInferenceResult(
            symbol=symbol,
            timestamp=0,
            close_price=100.0,
            direction=direction,
            entry=EntryPrediction(probability=0.9, should_enter=True, source=ModelSource.HEURISTIC),
            exit_params=ExitParameters(
                take_profit_pct=0.02,
                stop_loss_pct=0.01,
                trailing_activation_pct=0.01,
                trailing_distance_pct=0.005,
                source=ModelSource.HEURISTIC,
            ),
            risk=RiskAllocation(
                leverage=2, capital_allocation_pct=0.05, risk_score=0.5, risk_tier="LOW",
                source=ModelSource.HEURISTIC,
            ),
            feature_snapshot={},
        )

    inference_a = _inference("AAA/USDT:USDT", direction_a)
    inference_b = _inference("BBB/USDT:USDT", direction_b)

    # Feed B first in the input order - if ranking used insertion order or
    # the joint metric, B would win the single portfolio slot.
    results = engine.evaluate_many(
        [inference_b, inference_a], context=DecisionContext(trading_enabled=True)
    )

    executed = {result.symbol: result for result in results if result.is_executable}
    assert "AAA/USDT:USDT" in executed, (
        "A should win the single portfolio slot once ranking uses the conditional "
        "confidence - if this fails, evaluate_many is still ranking by the joint metric"
    )
