"""Task 6 - remove the 5 permanently-unobtainable microstructure features.

Covers:
* FEATURE_COLUMNS no longer contains ob_imbalance, ob_imbalance_delta,
  ob_spread_bps, ob_spread_rank or liquidation_imbalance, and still contains
  the 7 real-but-retention-limited derivatives/microstructure columns.
* EntryModel._heuristic's redesigned fallback (taker_buy_sell_ratio +
  atr_rank substituting for the removed ob_imbalance/ob_spread_rank) runs
  end-to-end and responds to its new inputs in the expected direction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from module_b_features.features import FEATURE_COLUMNS
from module_c_ml.ml_models import EntryModel
from module_c_ml.schemas import TradeAction

_REMOVED_FEATURES = (
    "ob_imbalance",
    "ob_imbalance_delta",
    "ob_spread_bps",
    "ob_spread_rank",
    "liquidation_imbalance",
)
_RETAINED_DERIVATIVES_FEATURES = (
    "funding_rate",
    "funding_rate_delta",
    "funding_rate_rank",
    "open_interest_change",
    "open_interest_rank",
    "long_short_ratio",
    "taker_buy_sell_ratio",
)


def test_feature_columns_drop_permanently_unobtainable_features() -> None:
    for name in _REMOVED_FEATURES:
        assert name not in FEATURE_COLUMNS, f"{name} should have been removed from FEATURE_COLUMNS"


def test_feature_columns_retain_real_but_retention_limited_features() -> None:
    for name in _RETAINED_DERIVATIVES_FEATURES:
        assert name in FEATURE_COLUMNS, f"{name} should still be in FEATURE_COLUMNS"


def _row(**overrides: float) -> pd.DataFrame:
    base = {column: 0.0 for column in FEATURE_COLUMNS}
    base["fdi_trending"] = 0.5
    base["atr_rank"] = 0.5
    base.update(overrides)
    return pd.DataFrame([base])


def test_entry_heuristic_runs_without_the_removed_features() -> None:
    """The heuristic must not silently degrade to a constant default just
    because ob_imbalance/ob_spread_rank no longer exist as columns.
    """
    features = _row(taker_buy_sell_ratio=0.0)
    prediction = EntryModel._heuristic(features, TradeAction.LONG, cutoff=0.5)
    assert 0.0 <= prediction.probability <= 1.0


def test_entry_heuristic_taker_flow_favours_long_when_positive() -> None:
    """Positive taker_buy_sell_ratio (net taker buying) should raise the LONG
    score relative to the neutral case - the substitute for the old
    ob_imbalance directional-flow term.
    """
    neutral = EntryModel._heuristic(_row(taker_buy_sell_ratio=0.0), TradeAction.LONG, cutoff=0.5)
    bullish = EntryModel._heuristic(_row(taker_buy_sell_ratio=0.8), TradeAction.LONG, cutoff=0.5)
    bearish = EntryModel._heuristic(_row(taker_buy_sell_ratio=-0.8), TradeAction.LONG, cutoff=0.5)

    assert bullish.probability > neutral.probability > bearish.probability


def test_entry_heuristic_taker_flow_direction_flips_for_short() -> None:
    """The same positive taker flow that favours LONG must disfavour SHORT -
    directional_flow is negated for the opposite action."""
    long_score = EntryModel._heuristic(_row(taker_buy_sell_ratio=0.8), TradeAction.LONG, cutoff=0.5)
    short_score = EntryModel._heuristic(_row(taker_buy_sell_ratio=0.8), TradeAction.SHORT, cutoff=0.5)
    assert long_score.probability > short_score.probability


def test_entry_heuristic_penalises_high_volatility_rank() -> None:
    """atr_rank substitutes for the removed ob_spread_rank: a high volatility
    percentile (proxy for adverse/wide conditions) should lower the score.
    """
    calm = EntryModel._heuristic(_row(atr_rank=0.1), TradeAction.LONG, cutoff=0.5)
    turbulent = EntryModel._heuristic(_row(atr_rank=0.95), TradeAction.LONG, cutoff=0.5)
    assert calm.probability > turbulent.probability
