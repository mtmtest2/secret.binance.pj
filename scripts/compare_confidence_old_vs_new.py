"""Compare the OLD joint-probability confidence gate against the NEW
independent gate/direction cascade, on a synthetic validation-like dataset.

SYNTHETIC DEMONSTRATION ONLY - no live database access is available in this
environment, so this script builds a small synthetic dataset (the same
pattern already used by `tests/test_round2_improvements.py::
_dataset_with_risk_target`) rather than reading real market history. The row
counts and pass-rates it prints are illustrative of the *mechanism* the fix
changes, not a forecast of what the live system will do. Getting real numbers
requires the operator to run an actual training cycle (`python main.py
train`) against live data and read the resulting ML diagnostic report's
`recommended_gate_threshold` / `recommended_direction_threshold` fields
instead.

Root cause recap
-----------------
The OLD `DirectionModel.predict()` only ever exposed the *joint* probability
long_probability = trade_probability * direction_given_trade_probability
short_probability = trade_probability * (1 - direction_given_trade_probability)
and the Decision Engine's old R1 gated a single confidence read off of
max(long_probability, short_probability) against one threshold (0.70 by
default). Because that product can never exceed trade_probability itself, a
row where the gate reads under 0.70 is *rejected outright regardless of how
confident the direction call is* - a 95%-confident LONG call given a 45%
gate read only ever produces a joint long_probability of 0.4275, which never
clears 0.70 no matter how sure the model is about direction.

The NEW architecture (module_c_ml.schemas.DirectionPrediction.trade_probability
/ direction_given_trade_probability, gated independently by
module_c_ml.decision_engine's R1A/R1B) asks the two questions separately:
R1A - is this bar worth trading at all (trade_probability >= min_gate_confidence)?
R1B - given that, which way and how sure (direction_given_trade_probability
       vs 1 - direction_given_trade_probability >= min_direction_given_trade_confidence)?

Usage::

    python scripts/compare_confidence_old_vs_new.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS
from module_b_features.processor import ProcessedDataset
from module_c_ml.ml_models import DirectionModel

#: The OLD architecture's single product-probability threshold (matched the
#: pre-fix DecisionSettings.min_direction_confidence default - that field has
#: since been removed entirely; see the "fix RiskModel's stale joint-
#: probability confidence input" commit).
OLD_JOINT_THRESHOLD = 0.70


def _synthetic_dataset(rng: np.random.Generator, n: int = 4000) -> ProcessedDataset:
    """Same synthetic-fixture pattern as tests/test_round2_improvements.py::
    _dataset_with_risk_target, reused here for consistency rather than
    reinvented - not real market data.
    """
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    direction_target = pd.Series(
        rng.choice(["LONG_SUCCESS", "SHORT_SUCCESS", "NO_TRADE_OR_FAIL"], size=n)
    )
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
        features=features,
        direction_target=direction_target,
        entry_target=entry_target,
        exit_targets=exit_targets,
        risk_target=risk_target,
        metadata=metadata,
        symbols=("BTC/USDT:USDT",),
        feature_columns=tuple(FEATURE_COLUMNS),
    )


def main() -> None:
    rng = np.random.default_rng(7)
    dataset = _synthetic_dataset(rng)

    settings = Settings(ml={"n_estimators": 60, "early_stopping_rounds": 10, "purge_bars": 10})
    min_gate = settings.decision.min_gate_confidence
    min_direction = settings.decision.min_direction_given_trade_confidence

    print("Training a DirectionModel on a synthetic dataset (this is NOT real market data)...")
    model = DirectionModel(settings)
    model.train(dataset)

    boundaries = dataset.split_boundaries(
        train_months=settings.ml.train_months,
        validation_months=settings.ml.validation_months,
        test_months=settings.ml.test_months,
        purge_bars=settings.ml.purge_bars,
        timeframe_ms=settings.data.timeframe_ms,
    )
    split = dataset.chronological_split(boundaries)
    validation_index = split.validation_index
    validation_features = dataset.features.iloc[validation_index]
    if validation_features.empty:
        print("No validation rows produced by this synthetic split - nothing to compare.")
        return

    print(
        f"\n{'row':>5}  {'trade_p':>8}  {'long|trade':>11}  {'OLD conf':>9}  "
        f"{'OLD pass@0.70':>13}  {'R1A pass':>9}  {'R1B conf':>9}  {'R1B pass':>9}"
    )
    print("-" * 92)

    old_pass_count = 0
    new_pass_count = 0
    sample_rows = min(25, len(validation_features))
    for position in range(len(validation_features)):
        row = validation_features.iloc[[position]]
        prediction = model.predict(row)

        trade_probability = prediction.trade_probability
        long_given_trade = prediction.direction_given_trade_probability
        old_confidence = trade_probability * max(long_given_trade, 1.0 - long_given_trade)
        old_pass = old_confidence >= OLD_JOINT_THRESHOLD

        gate_pass = trade_probability >= min_gate
        direction_confidence = max(long_given_trade, 1.0 - long_given_trade)
        direction_pass = direction_confidence >= min_direction
        new_pass = gate_pass and direction_pass

        old_pass_count += int(old_pass)
        new_pass_count += int(new_pass)

        if position < sample_rows:
            print(
                f"{position:>5}  {trade_probability:>8.4f}  {long_given_trade:>11.4f}  "
                f"{old_confidence:>9.4f}  {str(old_pass):>13}  {str(gate_pass):>9}  "
                f"{direction_confidence:>9.4f}  {str(direction_pass):>9}"
            )

    total = len(validation_features)
    print("-" * 92)
    print(f"\n{total} validation rows (synthetic).")
    print(
        f"OLD single product-threshold @ {OLD_JOINT_THRESHOLD:.2f}: "
        f"{old_pass_count}/{total} rows pass ({old_pass_count / total:.1%})"
    )
    print(
        f"NEW independent gates @ gate>={min_gate:.2f}, direction>={min_direction:.2f}: "
        f"{new_pass_count}/{total} rows pass ({new_pass_count / total:.1%})"
    )
    print(
        "\nNote: on synthetic (pure-noise-ish) data these two counts are not expected to "
        "differ dramatically - the architectural difference this script demonstrates is the "
        "PER-ROW disagreement (see the table above for rows where OLD and NEW reach different "
        "pass/fail verdicts), not necessarily the aggregate rate. On real market data - where "
        "the two stages genuinely decompose 'is this worth trading' from 'which way, how sure' - "
        "the diagnosed production symptom was the opposite skew: the OLD gate rejected nearly "
        "every row (4 of 458,940 candidate signals converted to trades) specifically because "
        "confident direction calls were being discarded whenever the gate alone read under 0.50. "
        "Run `python main.py train` against real data and read the ML diagnostic report's "
        "`recommended_gate_threshold` / `recommended_direction_threshold` fields for real numbers."
    )


if __name__ == "__main__":
    main()
