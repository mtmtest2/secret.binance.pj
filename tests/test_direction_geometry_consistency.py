"""The two reasons a high-confidence signal can still close at its stop.

Both are structural rather than statistical - neither needs the model to be
wrong, and neither shows up in any accuracy metric.

1. **Geometry mismatch.**  The Direction model predicts the labeler's question:
   will price reach ``tp_atr_multiple`` x ATR before ``sl_atr_multiple`` x ATR
   against us?  The Exit model used to be allowed to place the actual stop at
   half that distance, turning every signal into a bet the model was never
   asked about - and the labeler's own tiering says MEDIUM/HIGH-risk winners
   have 0.35-0.85 ATR of path heat, so those labelled *winners* would be
   stopped out by construction.

2. **Conditional confidence read as unconditional.**  R1B gates on
   ``P(side | worth trading)``.  An 88% reading with a 55% gate is a 48% trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_c_ml.ml_models import ExitModel
from module_c_ml.schemas import DirectionPrediction, ModelSource, TradeAction


def _row(atr_pct: float, **overrides: float) -> pd.Series:
    base: dict[str, float] = {
        "atr_pct": atr_pct,
        "garch_volatility": atr_pct / 2.0,
        "garch_vol_rank": 0.5,
        "hmm_regime": -1.0,
    }
    base.update(overrides)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# 1. Exit geometry must match the geometry the Direction model was priced on
# ---------------------------------------------------------------------------
def test_stop_is_never_tighter_than_the_labelled_barrier() -> None:
    settings = Settings()
    model = ExitModel(settings)
    atr_pct = 0.006  # a volatile alt: 1 ATR == 0.6%

    # A collapsed stop prediction, far tighter than the labelled barrier.
    params = model._assemble(
        take_profit=0.02,
        stop_loss=0.0005,
        trailing=0.01,
        row=_row(atr_pct),
        source=ModelSource.TRAINED,
    )

    labelled_stop = atr_pct * settings.labels.sl_atr_multiple
    assert params.stop_loss_pct >= labelled_stop - 1e-12, (
        "the executed stop must be at least the distance the Direction model's "
        "probability refers to"
    )


def test_labelled_winners_survive_the_stop_that_is_now_placed() -> None:
    """The concrete failure the old 0.5x floor produced.

    The labeler tiers a winner by its maximum adverse excursion as a fraction of
    the labelled stop: <=0.35 LOW, <=0.60 MEDIUM, <=0.85 HIGH.  All three tiers
    are accepted by default.  Under the old floor the placed stop sat at 0.5
    ATR, so every HIGH-tier winner - and most MEDIUM ones - were stopped out
    despite the label calling them successes and the model calling them
    correctly.
    """
    settings = Settings()
    model = ExitModel(settings)
    atr_pct = 0.004
    params = model._assemble(
        take_profit=0.02,
        stop_loss=0.0001,
        trailing=0.01,
        row=_row(atr_pct),
        source=ModelSource.TRAINED,
    )

    placed_stop_in_atr = params.stop_loss_pct / atr_pct
    worst_accepted_tier_heat = settings.labels.high_risk_mae_ratio  # 0.85 ATR

    assert placed_stop_in_atr >= worst_accepted_tier_heat, (
        f"a HIGH-tier winner suffers {worst_accepted_tier_heat} ATR of heat but "
        f"the placed stop is only {placed_stop_in_atr:.2f} ATR away - it would be "
        "stopped out despite being a correctly-predicted winner"
    )
    # The old behaviour, for contrast: 0.5 ATR would not have cleared this.
    assert 0.5 < worst_accepted_tier_heat


def test_widening_the_stop_preserves_reward_risk() -> None:
    """Widening the floor must not quietly destroy the reward/risk ratio.

    The take-profit rails are expressed as multiples of the stop, so a wider
    stop scales the target with it rather than shrinking the payoff.
    """
    settings = Settings()
    model = ExitModel(settings)
    params = model._assemble(
        take_profit=0.001,
        stop_loss=0.0001,
        trailing=0.0005,
        row=_row(0.005),
        source=ModelSource.TRAINED,
    )
    assert params.reward_risk_ratio >= 1.1


def test_assemble_tolerates_a_missing_or_nan_atr() -> None:
    """`atr_pct` is NaN for an un-warmed bar now that rows are not dropped."""
    settings = Settings()
    model = ExitModel(settings)
    for atr in (0.0, float("nan")):
        params = model._assemble(
            take_profit=0.02,
            stop_loss=0.003,
            trailing=0.01,
            row=_row(atr),
            source=ModelSource.TRAINED,
        )
        assert params.stop_loss_pct > 0.0
        assert np.isfinite(params.stop_loss_pct)
        assert params.take_profit_pct > params.stop_loss_pct


def test_heuristic_geometry_also_respects_the_labelled_stop() -> None:
    settings = Settings()
    model = ExitModel(settings)
    features = pd.DataFrame([_row(0.004).to_dict()])
    params = model._heuristic(features)
    assert params.stop_loss_pct >= 0.004 * settings.labels.sl_atr_multiple - 1e-12


# ---------------------------------------------------------------------------
# 2. Conditional vs unconditional confidence
# ---------------------------------------------------------------------------
def _prediction(trade_probability: float, long_given_trade: float) -> DirectionPrediction:
    long_probability = trade_probability * long_given_trade
    short_probability = trade_probability * (1.0 - long_given_trade)
    return DirectionPrediction(
        probabilities={
            "LONG_SUCCESS": long_probability,
            "SHORT_SUCCESS": short_probability,
            "NO_TRADE_OR_FAIL": max(1e-9, 1.0 - trade_probability),
        },
        source=ModelSource.TRAINED,
        trade_probability=trade_probability,
        direction_given_trade_probability=long_given_trade,
    )


def test_an_88_percent_signal_is_a_48_percent_trade_at_the_default_gate() -> None:
    """The exact scenario behind "88% confident and it still stopped out".

    Nothing is malfunctioning here: 0.88 conditional on a 0.55 gate is a
    sub-coin-flip trade.  The point is that only the 88% was ever surfaced.
    """
    prediction = _prediction(trade_probability=0.55, long_given_trade=0.88)

    assert prediction.directional_confidence == pytest.approx(0.88)
    assert prediction.joint_success_probability == pytest.approx(0.484)
    assert prediction.joint_success_probability < 0.5


def test_joint_success_probability_is_the_product_of_the_two_stages() -> None:
    for gate, conditional in ((0.6, 0.7), (0.9, 0.95), (0.55, 0.6)):
        prediction = _prediction(gate, conditional)
        assert prediction.joint_success_probability == pytest.approx(gate * conditional)


def test_directional_confidence_is_symmetric_for_shorts() -> None:
    """A 12% long probability is an 88% *short* confidence, not 12%."""
    prediction = _prediction(trade_probability=0.7, long_given_trade=0.12)
    assert prediction.directional_confidence == pytest.approx(0.88)
    assert prediction.joint_success_probability == pytest.approx(0.616)
    assert prediction.action is TradeAction.SHORT


def test_decision_engine_records_both_confidences() -> None:
    """The audit trail must carry the joint probability, not just the gate one."""
    from module_c_ml.decision_engine import DecisionEngine, Rule
    from module_c_ml.schemas import (
        EntryPrediction,
        ExitParameters,
        ModelInferenceResult,
        RiskAllocation,
    )

    settings = Settings()
    engine = DecisionEngine(settings)
    inference = ModelInferenceResult(
        symbol="SOL/USDT:USDT",
        timestamp=1_767_571_200_000,
        close_price=100.0,
        direction=_prediction(trade_probability=0.9, long_given_trade=0.88),
        entry=EntryPrediction(probability=0.9, should_enter=True, source=ModelSource.TRAINED),
        exit_params=ExitParameters(
            take_profit_pct=0.02,
            stop_loss_pct=0.006,
            trailing_activation_pct=0.01,
            trailing_distance_pct=0.004,
            source=ModelSource.TRAINED,
        ),
        risk=RiskAllocation(
            leverage=3,
            capital_allocation_pct=0.02,
            risk_score=0.7,
            risk_tier="LOW",
            abort_reason="",
            source=ModelSource.TRAINED,
        ),
        feature_snapshot={"atr_pct": 0.006, "hmm_regime": 0.0},
        latency_ms=1.0,
        model_versions={},
    )

    result = engine.evaluate(inference)
    detail = {check["rule"]: check["detail"] for check in result.checks}

    assert "joint_success_probability" in detail[Rule.DIRECTION_CONFIDENCE]
    assert "gate=" in detail[Rule.DIRECTION_CONFIDENCE]
    # And the exit-geometry check reports the stop against the labelled barrier.
    assert "stop_vs_labelled_atr" in detail[Rule.REWARD_RISK]
    assert "stop_vs_labelled_atr=1.000" in detail[Rule.REWARD_RISK]
