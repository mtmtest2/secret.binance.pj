"""Heads must be trained on the population they are scored against (P8, P9, P10, P15).

The Risk and Exit heads were fitted only on rows where a trade had actually
*won* - a filter on the realised outcome, which is unknowable at decision time.
The consequence was measurable in the audited run: the Risk head's predictions
floored at 0.42 against a target floor of 0.09, carried a +0.094 upward bias
that over-sized every position, and vetoed 281 of 1,381,357 candidates.

Also covered here: the trailing target being a fixed multiple of the
take-profit target (so its regressor learned y = 0.5x), and the recency
weighting that turned a 372-day window into an effective ~65 days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS
from module_b_features.labeler import LabelClass
from module_b_features.processor import ProcessedDataset
from module_c_ml.ml_models import ExitModel, RiskModel

_TIMEFRAME_MS = 5 * 60 * 1_000


def _dataset(rng: np.random.Generator, rows: int = 2_500) -> ProcessedDataset:
    """A dataset with a realistic mix of winners and non-trades."""
    features = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(rows, len(FEATURE_COLUMNS))),
        columns=list(FEATURE_COLUMNS),
    )
    score = features["kama_slope"] * 1.2 + rng.normal(0, 0.6, size=rows)
    labels = np.where(
        score > 0.9,
        LabelClass.LONG_SUCCESS.value,
        np.where(score < -0.9, LabelClass.SHORT_SUCCESS.value, LabelClass.NO_TRADE_OR_FAIL.value),
    )
    # Non-trades carry a genuinely low - but not zero - opportunity score, which
    # is what the labeler now produces from the better side's path heat.
    is_trade = labels != LabelClass.NO_TRADE_OR_FAIL.value
    risk = np.where(is_trade, rng.uniform(0.45, 0.95, rows), rng.uniform(0.05, 0.35, rows))

    return ProcessedDataset(
        features=features,
        direction_target=pd.Series(labels, name="label"),
        entry_target=pd.Series(is_trade.astype(int)),
        exit_targets=pd.DataFrame(
            {
                "target_tp_pct": np.abs(rng.normal(0.02, 0.006, rows)),
                "target_sl_pct": np.abs(rng.normal(0.01, 0.003, rows)),
            }
        ),
        risk_target=pd.Series(risk),
        metadata=pd.DataFrame(
            {
                "timestamp": np.arange(rows, dtype=np.int64) * _TIMEFRAME_MS + 1_700_000_000_000,
                "symbol": "BTC/USDT:USDT",
                "atr_pct": np.abs(rng.normal(0.004, 0.001, rows)),
            }
        ),
        feature_columns=tuple(FEATURE_COLUMNS),
        symbols=("BTC/USDT:USDT",),
        total_candidate_rows=rows,
    )


def _settings(tmp_path) -> Settings:
    return Settings(
        ml={
            "model_dir": tmp_path / "models",
            "n_estimators": 25,
            "early_stopping_rounds": 5,
            "purge_bars": 10,
        }
    )


# --------------------------------------------------------------------------
# P8 - population, not outcome
# --------------------------------------------------------------------------
def test_risk_head_trains_on_every_bar(tmp_path) -> None:
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(2))
    head = RiskModel(settings)
    head.train(dataset)

    split = head._chronological_split(dataset)
    # Every training row is used - no outcome-conditioned subset.
    assert head.metadata["rows"] == len(split.train_index)
    assert "none" in head.metadata["training_row_filter"]


def test_risk_head_can_express_a_bad_trade(tmp_path) -> None:
    """The failure the outcome filter caused, stated as an invariant.

    A head that has only seen winners cannot predict a low score, so its
    prediction floor sits far above the target's. Trained on the real
    population, its output range should cover the lower half of the target's.
    """
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(3))
    head = RiskModel(settings)
    head.train(dataset)

    stats = head.metadata["metrics"]["prediction_stats"]
    target_stats = head.metadata["metrics"]["target_stats"]
    target_span = target_stats["max"] - target_stats["min"]
    assert stats["min"] < target_stats["min"] + 0.4 * target_span, (
        f"prediction floor {stats['min']:.3f} sits in the upper part of the target range "
        f"[{target_stats['min']:.3f}, {target_stats['max']:.3f}] - the head cannot say "
        "'this is a bad trade'"
    )


def test_labeler_scores_non_trade_rows_from_real_path_heat() -> None:
    """`target_risk_score` must not be a hard 0.0 on non-selected rows.

    That placeholder is what made the unfiltered population look unlearnable and
    motivated the outcome filter in the first place.
    """
    from module_b_features.labeler import TradeLabeler

    settings = Settings()
    rng = np.random.default_rng(11)
    rows = 900
    index = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, rows)))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1_000.0,
            "timestamp": (np.arange(rows, dtype=np.int64) * _TIMEFRAME_MS) + 1_700_000_000_000,
            "atr_pct": 0.004,
            "garch_vol_rank": 0.5,
        },
        index=index,
    )
    labeled = TradeLabeler(settings).generate(frame)

    no_trade = labeled["label"] == LabelClass.NO_TRADE_OR_FAIL.value
    if no_trade.any():
        scores = labeled.loc[no_trade, "target_risk_score"]
        assert (scores > 0.0).any(), "non-trade rows still carry a placeholder zero"


# --------------------------------------------------------------------------
# P15 - the trailing head learned y = 0.5x
# --------------------------------------------------------------------------
def test_exit_head_no_longer_trains_a_trailing_regressor(tmp_path) -> None:
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(4))
    head = ExitModel(settings)
    head.train(dataset)

    assert "target_trailing_pct" not in head._model
    assert set(head._model) == {"target_tp_pct", "target_sl_pct"}


def test_exit_geometry_is_unchanged_by_deriving_the_trail(tmp_path) -> None:
    """Retiring the regressor must not move the executed geometry."""
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(5))
    head = ExitModel(settings)
    head.train(dataset)

    row = dataset.features.iloc[[0]].copy()
    row["atr_pct"] = 0.004
    params = head.predict(row)

    assert params.take_profit_pct > 0.0
    assert params.stop_loss_pct > 0.0
    # The trail arms inside the take-profit distance, per _assemble's rails.
    assert 0.25 * params.take_profit_pct <= params.trailing_activation_pct <= 0.95 * params.take_profit_pct


def test_exit_head_reports_how_often_its_prediction_is_overridden(tmp_path) -> None:
    """A head whose output a rail discards on most bars is not in the loop.

    Since the stop floor was widened to the labelled barrier, the stop
    regressor is out-competed by that floor on most rows - which makes its R^2
    a statement about an output that rarely reaches the exchange.
    """
    settings = _settings(tmp_path)
    dataset = _dataset(np.random.default_rng(6))
    head = ExitModel(settings)
    head.train(dataset)

    rates = head.metadata["rail_override_rates"]
    assert "target_sl_pct" in rates
    assert 0.0 <= rates["target_sl_pct"] <= 1.0


# --------------------------------------------------------------------------
# P9 - the effective sample the model actually saw
# --------------------------------------------------------------------------
def test_recency_weighting_is_off_by_default() -> None:
    """The chronological split already carries the recency argument.

    At the previous 45-day half-life a 372-day window integrated to 64.7
    effective days, so the reported `train_months` and row count both described
    a training set that did not exist.
    """
    assert Settings().ml.recency_half_life_days == 0.0


def test_recency_weights_decay_when_enabled(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.ml.recency_half_life_days = 30.0
    head = RiskModel(settings)

    day_ms = 86_400_000
    timestamps = np.array([0, 30 * day_ms, 60 * day_ms], dtype=np.int64)
    weights = head._recency_weights(timestamps)

    assert weights is not None
    assert weights[-1] == pytest.approx(1.0)
    assert weights[-2] == pytest.approx(0.5, abs=1e-9)
    assert weights[-3] == pytest.approx(0.25, abs=1e-9)


def test_zero_half_life_disables_weighting(tmp_path) -> None:
    head = RiskModel(_settings(tmp_path))
    assert head._recency_weights(np.array([0, 1, 2], dtype=np.int64)) is None
