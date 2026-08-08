"""Out-of-sample feature pruning via null (shuffled-noise) importance.

With a weak overall signal (5m crypto direction is, by design, mostly noise),
throwing every correlated technical feature at a boosted tree risks the model
fitting noise in *some* of them rather than signal in the rest - more
features is not free.  This module implements a standard, robust technique
(a lightweight relative of Boruta) for deciding which features are worth
keeping:

1. For each purged walk-forward fold (see
   :mod:`module_c_ml.cross_validation`), fit a small classifier on the
   training block with one extra column of *pure random noise* appended.
2. A real feature only "wins" that fold if its gain importance exceeds the
   noise column's importance - beating noise is the lowest possible bar a
   feature must clear to be worth a split at all.
3. Keep the features that win in at least ``ml.prune_min_fold_win_rate`` of
   folds; drop the rest.

Because the noise column is refreshed (reseeded) every fold, a feature that
only "wins" by luck in one fold is unlikely to keep winning across several -
the fold-win-rate threshold is what turns a noisy single-shot importance
ranking into a stable decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from config.settings import Settings
from core.logger import get_logger
from module_c_ml.cross_validation import CVFold, purged_walk_forward_splits

_LOGGER = get_logger(__name__)

_MIN_FOLD_TRAIN_ROWS: Final[int] = 200
#: However aggressive the pruning threshold, never cut below this many
#: features (or half the candidate set, whichever is larger) - a degenerate
#: config or a tiny dataset must not be able to prune the feature set to
#: (near) nothing.
_MIN_KEPT_FEATURES: Final[int] = 5


@dataclass(slots=True, frozen=True)
class FeaturePruningReport:
    """Result of one null-importance pruning pass."""

    kept_features: tuple[str, ...]
    dropped_features: tuple[str, ...]
    win_rate: dict[str, float]
    folds_used: int
    fell_back: bool = False


def select_features_by_null_importance(
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series | None,
    feature_columns: list[str],
    settings: Settings,
    horizon: int,
) -> FeaturePruningReport:
    """Rank ``feature_columns`` against a per-fold noise benchmark; return survivors.

    Args:
        features: Full feature matrix, row-aligned with ``target``.
        target: Integer-encoded classification target (works for any number
            of classes - Direction's 3-class encoding or Entry's binary one).
        sample_weight: Optional per-row training weight (see
            ``TradeLabeler._sample_weights``); applied inside each fold's fit
            so pruning judges features under the same weighting the real
            heads train with.
        feature_columns: Candidate columns to evaluate (typically
            :data:`~module_b_features.features.FEATURE_COLUMNS`, optionally
            extended with stacked Direction-OOF columns).
        settings: Root settings (``ml.cv_folds``, ``ml.cv_embargo_bars``,
            ``ml.prune_min_fold_win_rate``, ``ml.booster``,
            ``ml.random_state``).
        horizon: Label look-ahead in bars, the CV purge width.

    Returns:
        A :class:`FeaturePruningReport`. Falls back to keeping every feature
        (``fell_back=True``) when there are too few purged folds to trust a
        pruning decision, or when the threshold would cut below the safety
        floor - silently over-pruning is worse than not pruning at all.
    """
    config = settings.ml
    rows: int = len(features)
    folds: list[CVFold] = purged_walk_forward_splits(
        rows, config.cv_folds, horizon, config.cv_embargo_bars
    )
    if not folds:
        _LOGGER.info(
            "Feature pruning skipped: dataset too small for %d purged folds", config.cv_folds
        )
        return FeaturePruningReport(
            kept_features=tuple(feature_columns), dropped_features=(), win_rate={},
            folds_used=0, fell_back=True,
        )

    rng: np.random.Generator = np.random.default_rng(config.random_state)
    wins: dict[str, int] = {name: 0 for name in feature_columns}
    folds_used: int = 0

    for fold in folds:
        if len(fold.train_index) < _MIN_FOLD_TRAIN_ROWS:
            continue

        x_train: pd.DataFrame = features.iloc[fold.train_index][feature_columns].copy()
        noise_column: str = "__null_importance_noise__"
        x_train[noise_column] = rng.standard_normal(len(x_train))
        y_train: pd.Series = target.iloc[fold.train_index]
        if y_train.nunique() < 2:
            continue
        w_train: np.ndarray | None = (
            sample_weight.iloc[fold.train_index].to_numpy() if sample_weight is not None else None
        )

        estimator: Any = _build_pruning_classifier(settings, num_class=int(y_train.nunique()))
        estimator.fit(x_train, y_train, sample_weight=w_train)

        importances: pd.Series = pd.Series(
            getattr(estimator, "feature_importances_", np.zeros(x_train.shape[1])),
            index=x_train.columns,
        )
        noise_importance: float = float(importances.get(noise_column, 0.0))
        for name in feature_columns:
            if float(importances.get(name, 0.0)) > noise_importance:
                wins[name] += 1
        folds_used += 1

    if folds_used == 0:
        _LOGGER.warning("Feature pruning: no fold had enough training rows - keeping all features")
        return FeaturePruningReport(
            kept_features=tuple(feature_columns), dropped_features=(), win_rate={},
            folds_used=0, fell_back=True,
        )

    win_rate: dict[str, float] = {name: wins[name] / folds_used for name in feature_columns}
    kept: tuple[str, ...] = tuple(
        name for name in feature_columns if win_rate[name] >= config.prune_min_fold_win_rate
    )
    dropped: tuple[str, ...] = tuple(name for name in feature_columns if name not in kept)

    floor: int = max(_MIN_KEPT_FEATURES, len(feature_columns) // 2)
    if len(kept) < floor:
        _LOGGER.warning(
            "Feature pruning would keep only %d/%d features (floor %d) - "
            "keeping the full set instead of over-pruning",
            len(kept), len(feature_columns), floor,
        )
        return FeaturePruningReport(
            kept_features=tuple(feature_columns), dropped_features=(), win_rate=win_rate,
            folds_used=folds_used, fell_back=True,
        )

    _LOGGER.info(
        "Feature pruning: kept %d/%d features across %d folds (dropped: %s)",
        len(kept), len(feature_columns), folds_used, ", ".join(dropped) or "none",
    )
    return FeaturePruningReport(
        kept_features=kept, dropped_features=dropped, win_rate=win_rate, folds_used=folds_used,
    )


def _build_pruning_classifier(settings: Settings, num_class: int) -> Any:
    """A deliberately small, fast classifier - this runs once per CV fold
    purely to rank feature importance, not to be a production model."""
    config = settings.ml
    if config.booster == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob" if num_class > 2 else "binary:logistic",
            num_class=num_class if num_class > 2 else None,
            random_state=config.random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
            importance_type="gain",
        )

    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        objective="multiclass" if num_class > 2 else "binary",
        num_class=num_class if num_class > 2 else 1,
        random_state=config.random_state,
        n_jobs=1,
        verbose=-1,
        importance_type="gain",
    )
