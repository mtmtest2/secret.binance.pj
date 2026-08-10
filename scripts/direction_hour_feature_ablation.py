"""Task 7c - is hour_sin/hour_cos dominance real signal or overfitting?

SYNTHETIC DEMONSTRATION ONLY - no live database access is available in this
environment. This script builds two synthetic datasets (same pattern as
`tests/test_round2_improvements.py::_dataset_with_risk_target` and
`scripts/compare_confidence_old_vs_new.py`): one with the full feature set,
one with hour_sin/hour_cos/dow_sin/dow_cos zeroed out (a constant column
carries zero information to a tree-based model, so this is equivalent to
excluding them without changing the feature contract's shape). It then
trains a DirectionModel on each and reports the balanced_accuracy delta.

THIS IS INHERENTLY INCONCLUSIVE ON SYNTHETIC DATA. The direction_target and
every feature (including the "hour"/"dow" ones) are pure random noise here -
there is no real time-of-day structure for the model to detect either way,
so a near-zero delta is the EXPECTED, uninformative result on this input,
not evidence that the real hour_sin/hour_cos dominance seen in production is
spurious. The real value of this script is as a ready-made tool: point it at
a real, trained dataset (swap `_synthetic_dataset` for a real
`ProcessedDataset` loaded from the database) once the operator retrains
against live data, and a *real* delta becomes informative:

* A large accuracy drop when hour/dow features are removed would support the
  "real structural effect" reading (funding-rate settlement windows at
  00:00/08:00/16:00 UTC are a documented real phenomenon in perpetual
  futures - a time-conditioned edge is plausible).
* A negligible drop would support the "overfitting to time-of-day" reading -
  the model may be latching onto a spurious correlation in a specific
  historical window that will not generalize.

Do NOT remove hour_sin/hour_cos/dow_sin/dow_cos from FEATURE_COLUMNS based on
this script's synthetic-data output alone - see the task instructions this
script was written to satisfy.

Usage::

    python scripts/direction_hour_feature_ablation.py
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

#: The four session/seasonality features under investigation.
_HOUR_DOW_FEATURES: tuple[str, ...] = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")


def _synthetic_dataset(
    rng: np.random.Generator, n: int = 6000, zero_hour_dow_features: bool = False
) -> ProcessedDataset:
    """Same synthetic-fixture pattern as tests/test_round2_improvements.py::
    _dataset_with_risk_target, reused here for consistency - not real market
    data. When ``zero_hour_dow_features`` is set, hour_sin/hour_cos/dow_sin/
    dow_cos are held at a constant 0.0, which carries no information to a
    tree-based model - equivalent to excluding them without reshaping the
    feature matrix.
    """
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    if zero_hour_dow_features:
        for column in _HOUR_DOW_FEATURES:
            features[column] = 0.0
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


def _train_and_score(dataset: ProcessedDataset, settings: Settings) -> dict[str, float]:
    model = DirectionModel(settings)
    metrics = model.train(dataset)
    return {
        "balanced_accuracy": float(metrics.get("balanced_accuracy", float("nan"))),
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "log_loss": float(metrics.get("log_loss", float("nan"))),
    }


def main() -> None:
    settings = Settings(
        ml={"n_estimators": 60, "early_stopping_rounds": 10, "validation_fraction": 0.3, "purge_bars": 10}
    )

    # Same rng SEED for both datasets so every feature except hour/dow is
    # byte-for-byte identical between the two runs - isolates the ablation to
    # exactly the four columns under test.
    seed = 11
    full_dataset = _synthetic_dataset(np.random.default_rng(seed), zero_hour_dow_features=False)
    ablated_dataset = _synthetic_dataset(np.random.default_rng(seed), zero_hour_dow_features=True)

    print("Training DirectionModel WITH hour_sin/hour_cos/dow_sin/dow_cos (synthetic data)...")
    with_hour = _train_and_score(full_dataset, settings)

    print("Training DirectionModel WITHOUT hour_sin/hour_cos/dow_sin/dow_cos (synthetic data)...")
    without_hour = _train_and_score(ablated_dataset, settings)

    delta_balanced_accuracy = with_hour["balanced_accuracy"] - without_hour["balanced_accuracy"]

    print("\n" + "=" * 72)
    print("HOUR/DOW FEATURE ABLATION (SYNTHETIC DATA - SEE CAVEATS BELOW)")
    print("=" * 72)
    print(f"{'metric':>20}  {'WITH hour/dow':>15}  {'WITHOUT hour/dow':>18}  {'delta':>8}")
    for key in ("balanced_accuracy", "accuracy", "log_loss"):
        delta = with_hour[key] - without_hour[key]
        print(f"{key:>20}  {with_hour[key]:>15.4f}  {without_hour[key]:>18.4f}  {delta:>+8.4f}")
    print("=" * 72)
    print(
        f"\nbalanced_accuracy delta (WITH - WITHOUT) = {delta_balanced_accuracy:+.4f}\n"
        "\nCAVEAT (read before drawing any conclusion): direction_target and every\n"
        "feature in this script are pure random noise - there is NO real time-of-day\n"
        "signal for the model to find either way, so a small delta here is the\n"
        "EXPECTED result on synthetic data, not evidence that production's real\n"
        "hour_sin/hour_cos dominance is spurious. This script is a ready-made tool for\n"
        "the operator to re-run against a REAL trained ProcessedDataset after the next\n"
        "retrain (swap _synthetic_dataset for real data loaded from the database) - only\n"
        "then does the delta mean anything about real vs overfit time-of-day signal."
    )


if __name__ == "__main__":
    main()
