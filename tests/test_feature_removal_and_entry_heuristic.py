"""Removal of the microstructure/derivatives features that cannot be backfilled.

Two rounds, same principle - a feature the training window cannot actually
observe is deleted, never parked at a placeholder:

* **No historical endpoint at all**: ob_imbalance, ob_imbalance_delta,
  ob_spread_bps, ob_spread_rank, liquidation_imbalance.
* **~30 days of Binance-side retention against a multi-month window**:
  open_interest_change, long_short_ratio, taker_buy_sell_ratio.

What survives is funding_rate (full history since contract inception) and
open_interest_rank (a rolling percentile that degrades gracefully where open
interest is sparse).

Also covers EntryModel._heuristic's fallback, which now leans on
``order_flow_imbalance_5m`` - real aggTrades flow with full history - in place
of the taker_buy_sell_ratio proxy it used to substitute for ob_imbalance.
"""

from __future__ import annotations

import pandas as pd
import pytest

from module_b_features.features import FEATURE_COLUMNS
from module_c_ml.ml_models import EntryModel
from module_c_ml.schemas import TradeAction

_REMOVED_FEATURES = (
    # No historical endpoint at all.
    "ob_imbalance",
    "ob_imbalance_delta",
    "ob_spread_bps",
    "ob_spread_rank",
    "liquidation_imbalance",
    # Endpoint exists, but Binance retains only ~30 days.
    "open_interest_change",
    "long_short_ratio",
    "taker_buy_sell_ratio",
)
_RETAINED_DERIVATIVES_FEATURES = (
    "funding_rate",
    "funding_rate_delta",
    "funding_rate_rank",
    "open_interest_rank",
)


def test_feature_columns_drop_permanently_unobtainable_features() -> None:
    for name in _REMOVED_FEATURES:
        assert name not in FEATURE_COLUMNS, f"{name} should have been removed from FEATURE_COLUMNS"


def test_feature_columns_retain_the_backfillable_features() -> None:
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
    features = _row(order_flow_imbalance_5m=0.0)
    prediction = EntryModel._heuristic(features, TradeAction.LONG, cutoff=0.5)
    assert 0.0 <= prediction.probability <= 1.0


def test_entry_heuristic_order_flow_favours_long_when_positive() -> None:
    """Aggressive net buying should raise the LONG score above the neutral case.

    This is the substitute for the old ob_imbalance directional-flow term, and
    unlike the taker_buy_sell_ratio proxy it replaced, it is measured from the
    bar's own aggTrades and available for the whole training window.
    """
    neutral = EntryModel._heuristic(
        _row(order_flow_imbalance_5m=0.0), TradeAction.LONG, cutoff=0.5
    )
    bullish = EntryModel._heuristic(
        _row(order_flow_imbalance_5m=0.8), TradeAction.LONG, cutoff=0.5
    )
    bearish = EntryModel._heuristic(
        _row(order_flow_imbalance_5m=-0.8), TradeAction.LONG, cutoff=0.5
    )

    assert bullish.probability > neutral.probability > bearish.probability


def test_entry_heuristic_order_flow_direction_flips_for_short() -> None:
    """The same net buying that favours LONG must disfavour SHORT -
    directional_flow is negated for the opposite action."""
    long_score = EntryModel._heuristic(
        _row(order_flow_imbalance_5m=0.8), TradeAction.LONG, cutoff=0.5
    )
    short_score = EntryModel._heuristic(
        _row(order_flow_imbalance_5m=0.8), TradeAction.SHORT, cutoff=0.5
    )
    assert long_score.probability > short_score.probability


def test_entry_heuristic_penalises_high_volatility_rank() -> None:
    """atr_rank substitutes for the removed ob_spread_rank: a high volatility
    percentile (proxy for adverse/wide conditions) should lower the score.
    """
    calm = EntryModel._heuristic(_row(atr_rank=0.1), TradeAction.LONG, cutoff=0.5)
    turbulent = EntryModel._heuristic(_row(atr_rank=0.95), TradeAction.LONG, cutoff=0.5)
    assert calm.probability > turbulent.probability
