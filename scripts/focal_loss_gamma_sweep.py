"""Task 7d - sweep the experimental focal-loss gate objective's gamma.

SYNTHETIC DEMONSTRATION ONLY - no live database access is available in this
environment, so this trains DirectionModel's gate stage with
`use_focal_loss_for_gate=True` (module_c_ml.ml_models.focal_loss_binary /
MLSettings.use_focal_loss_for_gate, added as an off-by-default experiment in
feature/improve-direction-confidence) across a few gamma values on a
synthetic dataset (same pattern as
tests/test_round2_improvements.py::_dataset_with_risk_target), and reports
the gate's own log-loss/precision/recall at each.

`use_focal_loss_for_gate` is left at its `False` default in
config/settings.py regardless of what this synthetic sweep shows - real
validation requires real data; see the docstring on `focal_loss_binary`
itself for why (not yet validated against production log-loss, may
destabilize early stopping if gamma is too aggressive).

Usage::

    python scripts/focal_loss_gamma_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score, recall_score

from config.settings import Settings
from module_b_features.features import FEATURE_COLUMNS
from module_b_features.labeler import LabelClass
from module_b_features.processor import ProcessedDataset
from module_c_ml.ml_models import DirectionModel

#: Gamma values to sweep, per the task's explicit list.
_GAMMAS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)


def _synthetic_dataset(rng: np.random.Generator, n: int = 6000) -> ProcessedDataset:
    """Same synthetic-fixture pattern as compare_confidence_old_vs_new.py -
    not real market data.
    """
    features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    direction_target = pd.Series(
        rng.choice(["LONG_SUCCESS", "SHORT_SUCCESS", "NO_TRADE_OR_FAIL"], size=n, p=[0.27, 0.27, 0.46])
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


def _gate_metrics_at(dataset: ProcessedDataset, settings: Settings) -> dict[str, float]:
    """Fit DirectionModel and score just the gate stage (is-this-a-trade)
    against its own validation labels - not the full 3-way joint metric,
    since the focal-loss objective only ever touches the gate.
    """
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
    no_trade_index_label = LabelClass.NO_TRADE_OR_FAIL.value
    is_trade_true = (dataset.direction_target.iloc[validation_index] != no_trade_index_label).astype(int)

    gate_estimator = model._model["gate"]  # noqa: SLF001 - script-only introspection
    gate_probabilities = np.asarray(gate_estimator.predict_proba(validation_features))[:, -1]
    predicted = (gate_probabilities >= 0.5).astype(int)

    return {
        "log_loss": float(log_loss(is_trade_true, gate_probabilities, labels=[0, 1])),
        "precision": float(precision_score(is_trade_true, predicted, zero_division=0)),
        "recall": float(recall_score(is_trade_true, predicted, zero_division=0)),
    }


def main() -> None:
    print(
        "Sweeping focal_loss_gamma for the experimental gate objective on SYNTHETIC "
        "data - use_focal_loss_for_gate stays False in config/settings.py regardless "
        "of these results (real validation requires real data).\n"
    )

    seed = 23
    baseline_settings = Settings(
        ml={
            "n_estimators": 60,
            "early_stopping_rounds": 10,
            "purge_bars": 10,
            "use_focal_loss_for_gate": False,
        }
    )
    baseline_dataset = _synthetic_dataset(np.random.default_rng(seed))
    baseline_metrics = _gate_metrics_at(baseline_dataset, baseline_settings)

    print(f"{'gamma':>10}  {'log_loss':>10}  {'precision':>10}  {'recall':>10}")
    print("-" * 46)
    print(
        f"{'baseline':>10}  {baseline_metrics['log_loss']:>10.4f}  "
        f"{baseline_metrics['precision']:>10.4f}  {baseline_metrics['recall']:>10.4f}"
    )

    results: list[tuple[float, dict[str, float]]] = []
    for gamma in _GAMMAS:
        dataset = _synthetic_dataset(np.random.default_rng(seed))
        settings = Settings(
            ml={
                "n_estimators": 60,
                "early_stopping_rounds": 10,
                "purge_bars": 10,
                "use_focal_loss_for_gate": True,
                "focal_loss_gamma": gamma,
            }
        )
        metrics = _gate_metrics_at(dataset, settings)
        results.append((gamma, metrics))
        print(
            f"{gamma:>10.1f}  {metrics['log_loss']:>10.4f}  "
            f"{metrics['precision']:>10.4f}  {metrics['recall']:>10.4f}"
        )

    print("-" * 46)
    best_gamma, best_metrics = min(results, key=lambda item: item[1]["log_loss"])
    print(
        f"\nLowest log-loss among swept gammas: {best_gamma} "
        f"(log_loss={best_metrics['log_loss']:.4f} vs baseline "
        f"{baseline_metrics['log_loss']:.4f})."
    )
    print(
        "\nCAVEAT (read before drawing any conclusion): direction_target here is\n"
        "independent random noise, so the gate has nothing real to learn either way -\n"
        "these numbers describe how the focal-loss objective's gradient reshaping\n"
        "behaves mechanically on a noisy binary target, not whether it will help (or\n"
        "hurt) on real, learnable market data. use_focal_loss_for_gate remains False\n"
        "by default in config/settings.py; enabling it requires a real A/B comparison\n"
        "against the standard log-loss objective on a live retrain, which is the "
        "operator's to run outside this sandbox."
    )


if __name__ == "__main__":
    main()
