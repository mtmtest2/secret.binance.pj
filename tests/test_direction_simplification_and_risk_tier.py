"""Direction label now has 3 classes, and risk tiering moved to the Risk model.

Covers the refactor that dropped the LOW_RISK/HIGH_RISK split out of the
Direction target (module_b_features/labeler.py) and made the Risk model
derive its tier from its own continuous score instead of reading it off the
Direction model's winning class name (module_c_ml/ml_models.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import LabelSettings, Settings
from module_b_features.labeler import (
    LABEL_ORDER,
    LONG_LABELS,
    SHORT_LABELS,
    RiskTier,
    TradeLabeler,
    risk_tier_from_score,
)
from module_c_ml.ml_models import RiskModel
from module_c_ml.schemas import DirectionPrediction, ModelSource


def _synthetic_ohlc(rng: np.random.Generator, n: int = 4000) -> pd.DataFrame:
    returns = rng.normal(0.0, 0.004, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.002, size=n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.002, size=n)))
    atr = pd.Series(close).diff().abs().rolling(14, min_periods=1).mean().to_numpy()
    atr = np.nan_to_num(atr, nan=0.5)
    return pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "atr": atr,
            "atr_rank": rng.uniform(0.0, 1.0, size=n),
        }
    )


def test_label_order_has_exactly_three_classes() -> None:
    assert LABEL_ORDER == ("LONG_SUCCESS", "SHORT_SUCCESS", "NO_TRADE_OR_FAIL")
    assert LONG_LABELS == {"LONG_SUCCESS"}
    assert SHORT_LABELS == {"SHORT_SUCCESS"}


def test_labeler_produces_only_known_classes_and_clipped_exit_targets() -> None:
    settings = Settings()
    labeler = TradeLabeler(settings)
    frame = _synthetic_ohlc(np.random.default_rng(11))

    labeled = labeler.generate(frame)
    valid = labeled[labeled["label_is_valid"]]

    assert not valid.empty
    assert set(valid["label"].unique()) <= set(LABEL_ORDER)
    # The old fused classes must never appear.
    assert "LOW_RISK" not in "".join(valid["label"].unique())
    assert "HIGH_RISK" not in "".join(valid["label"].unique())

    tp = valid["target_tp_pct"].dropna()
    sl = valid["target_sl_pct"].dropna()
    trailing = valid["target_trailing_pct"].dropna()
    assert not tp.empty and not sl.empty
    # Same hard rails module_c_ml.ml_models applies at inference time.
    assert tp.max() <= 0.1500 + 1e-9
    assert tp.min() >= 0.0
    assert sl.max() <= 0.0800 + 1e-9
    assert sl.min() >= 0.0015 - 1e-9
    assert trailing.max() <= tp.max() + 1e-9

    # A VERY_HIGH-tier trade is folded into NO_TRADE_OR_FAIL by default config.
    very_high_rows = valid[valid["risk_tier"] == RiskTier.VERY_HIGH.value]
    assert (very_high_rows["label"] == "NO_TRADE_OR_FAIL").all()


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.95, "LOW"),
        (0.40, "MEDIUM"),
        (0.20, "HIGH"),
        (0.01, "VERY_HIGH"),
    ],
)
def test_risk_tier_from_score_buckets_monotonically(score: float, expected: str) -> None:
    config = LabelSettings()
    assert risk_tier_from_score(score, config) == expected


def test_risk_tier_from_score_moves_with_config_thresholds() -> None:
    lenient = LabelSettings(low_risk_mae_ratio=0.2, medium_risk_mae_ratio=0.4, high_risk_mae_ratio=0.6)
    strict = LabelSettings(low_risk_mae_ratio=0.6, medium_risk_mae_ratio=0.75, high_risk_mae_ratio=0.9)
    # The same score should read as a higher (better) tier under the stricter
    # config, since a stricter config accepts more heat into "LOW".
    assert risk_tier_from_score(0.35, strict) == "LOW"
    assert risk_tier_from_score(0.35, lenient) != "LOW"


def test_risk_model_predict_derives_its_own_tier_without_a_direction_kwarg() -> None:
    settings = Settings()
    model = RiskModel(settings)
    features = pd.DataFrame([{column: 0.0 for column in model._feature_columns}])
    features["garch_vol_rank"] = 0.2

    # No `risk_tier` kwarg exists any more - the model derives it internally.
    allocation = model.predict(features, direction_confidence=0.85)
    assert allocation.risk_tier in {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}
    assert allocation.source == ModelSource.HEURISTIC


def test_direction_prediction_has_no_implied_risk_tier_field() -> None:
    pred = DirectionPrediction(
        probabilities={"LONG_SUCCESS": 0.6, "SHORT_SUCCESS": 0.1, "NO_TRADE_OR_FAIL": 0.3},
        source=ModelSource.TRAINED,
    )
    assert not hasattr(pred, "implied_risk_tier")
    assert pred.long_probability == pytest.approx(0.6)
    assert pred.short_probability == pytest.approx(0.1)
    assert pred.action.value == "LONG"
