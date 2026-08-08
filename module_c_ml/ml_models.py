"""Module C - four gradient-boosted heads, cascaded and calibrated.

Architecture
------------
Each head answers one question, but - unlike the original independent-heads
design - Entry/Exit/Risk now *condition on Direction's own output* (a form of
meta-labeling / stacking, see below):

============  ==========================================================
Model 1       *Where* is the market going?      -> probability distribution
Model 2       *Now* or wait one more candle?    -> boolean + probability
Model 3       *How* should the trade be managed? -> TP / SL / trailing geometry
Model 4       *How much* capital and leverage?   -> 0-10x + allocation %
============  ==========================================================

Cascading / meta-labeling
--------------------------
Direction proposes a side; Entry/Exit/Risk are all downstream questions about
*that specific proposed trade* ("is this a clean entry", "how should THIS
trade be managed", "how much size does THIS trade deserve") - so, when
``ml.use_direction_stacking`` is on, they are trained and scored with
Direction's predicted probabilities appended as extra input features
(:data:`STACK_COLUMNS`).  This is textbook meta-labeling: Direction is the
primary signal, Entry/Exit/Risk are meta-models conditioned on it.

To do this without leakage, Direction's stacked features are **out-of-fold**
during training - generated via :meth:`DirectionModel.generate_oof_predictions`
across purged walk-forward folds (:mod:`module_c_ml.cross_validation`), so no
row's stacked feature ever came from a Direction model that was trained on
that row.  At live inference the same columns are filled from the *final*,
fully-trained Direction model's real prediction on a genuinely new row - no
leakage concern there, since that model has never seen the row either way.

Calibration
-----------
Raw gradient-boosted ``predict_proba`` output is usually not
decision-theoretically meaningful, especially under class imbalance - and the
Decision Engine thresholds these probabilities directly
(``min_direction_confidence``, the Entry model's own tuned cutoff, ...).  When
``ml.calibrate_probabilities`` is on, an isotonic (one-vs-rest) calibrator is
fit on out-of-fold / held-out predictions and applied after every
``predict_proba`` call.

Ensembling
----------
The Direction and Entry heads (the two probability-estimating classifiers)
bag ``ml.ensemble_size`` independently-seeded boosters and average their
output - cheap variance reduction that matters more on a low-signal problem
than on an easy one.

Inference contract
------------------
* **Stateless.**  ``predict`` reads the fitted booster/calibrator and its
  arguments, and nothing else.  No instance attribute is mutated, so
  concurrent calls from several event-loop tasks are safe.
* **Non-blocking.**  Gradient-boosted inference is CPU work; the async wrappers
  push it onto a worker thread with :func:`asyncio.to_thread`.  LightGBM and
  XGBoost release the GIL inside ``predict``, so this is real concurrency.
* **Degradation is explicit.**  When an artifact is missing, a head falls back to
  a documented, deterministic heuristic and stamps ``source=HEURISTIC`` on its
  output.  The Decision Engine can then refuse to trade on fallbacks (which it
  always does in live mode).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

import joblib
import numpy as np
import pandas as pd

from config.settings import MLSettings, Settings
from core.exceptions import ModelNotLoadedError, ModelTrainingError
from core.logger import get_logger
from core.utils import clamp
from module_b_features.features import FEATURE_COLUMNS, HMMRegime
from module_b_features.labeler import LABEL_ORDER, LabelClass
from module_b_features.processor import InferencePayload, ProcessedDataset
from module_c_ml.cross_validation import CVFold, purged_walk_forward_splits
from module_c_ml.feature_selection import FeaturePruningReport, select_features_by_null_importance
from module_c_ml.schemas import (
    DirectionPrediction,
    EntryPrediction,
    ExitParameters,
    ModelInferenceResult,
    ModelSource,
    RiskAllocation,
    TradeAction,
)

_LOGGER = get_logger(__name__)

_ARTIFACT_VERSION: Final[str] = "2.0.0"

# Hard sanity rails applied to every exit geometry, trained or heuristic.
_MIN_TP_PCT: Final[float] = 0.0020
_MAX_TP_PCT: Final[float] = 0.1500
_MIN_SL_PCT: Final[float] = 0.0015
_MAX_SL_PCT: Final[float] = 0.0800
#: A stop tighter than this fraction of ATR is inside 5m noise and will be
#: taken out by ordinary chop regardless of whether the direction call was right.
_MIN_SL_ATR_FRACTION: Final[float] = 0.5
#: Reward/risk beyond this is almost always an artefact of a degenerate stop
#: prediction rather than a real edge, so the target is trimmed back to it.
_MAX_REWARD_RISK: Final[float] = 6.0

# --- Direction -> Entry/Exit/Risk stacking columns --------------------------
STACK_DIRECTION_LONG: Final[str] = "stack_direction_long"
STACK_DIRECTION_SHORT: Final[str] = "stack_direction_short"
STACK_DIRECTION_NO_TRADE: Final[str] = "stack_direction_no_trade"
STACK_DIRECTION_CONFIDENCE: Final[str] = "stack_direction_confidence"
STACK_DIRECTION_MARGIN: Final[str] = "stack_direction_margin"
STACK_COLUMNS: Final[tuple[str, ...]] = (
    STACK_DIRECTION_LONG,
    STACK_DIRECTION_SHORT,
    STACK_DIRECTION_NO_TRADE,
    STACK_DIRECTION_CONFIDENCE,
    STACK_DIRECTION_MARGIN,
)

_MIN_INNER_TRAIN_ROWS: Final[int] = 200
_MIN_OOF_FOLD_TRAIN_ROWS: Final[int] = 200
#: Mirrors `module_b_features.processor._MIN_TRAIN_ROWS` - below this a
#: subsampled booster's effective sample size can round to zero and crash
#: deep inside the native library rather than failing with a clear message.
_MIN_REGRESSION_TRAIN_ROWS: Final[int] = 50


def load_artifact(path: Path) -> dict[str, Any] | None:
    """Safely load a joblib artifact.

    Returns ``None`` - never raises - when the file is absent, unreadable, or
    does not carry the expected payload shape.  A missing model must degrade the
    system into "do not trade", not crash the trading loop.

    Args:
        path: Filesystem path of the ``.joblib`` artifact.

    Returns:
        The artifact dictionary, or ``None`` when it could not be loaded.
    """
    if not path.exists():
        _LOGGER.warning("Model artifact not found: %s", path)
        return None
    try:
        payload: Any = joblib.load(path)
    except Exception as error:  # pragma: no cover - corrupt/partial artifact
        _LOGGER.error("Failed to load model artifact %s: %s", path, error)
        return None

    if not isinstance(payload, dict) or "model" not in payload:
        _LOGGER.error("Model artifact %s has an unexpected layout", path)
        return None
    return payload


@dataclass(slots=True)
class _EnsembleClassifier:
    """Averages ``predict_proba`` across independently-seeded boosters.

    Cheap variance reduction for the two probability-estimating heads
    (Direction, Entry).  Duck-types the ``predict_proba``/``predict``/
    ``classes_`` surface the rest of this module already relies on, so no
    other code needs to know whether ``self._model`` is one estimator or a
    bag of them.
    """

    estimators: list[Any]

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.estimators[0].classes_)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average gain importance across the bag (used by feature pruning callers)."""
        return np.mean([e.feature_importances_ for e in self.estimators], axis=0)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([estimator.predict_proba(x) for estimator in self.estimators], axis=0)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba: np.ndarray = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


@dataclass(slots=True)
class _IsotonicCalibrator:
    """Per-column isotonic (one-vs-rest) probability calibrator.

    Column-position generic - works identically for Direction's 3-column
    probability matrix and Entry's 2-column one. Fit on genuinely
    out-of-fold / held-out predictions (never on rows the calibrated model
    itself trained on), then renormalised so calibrated probabilities still
    sum to 1.
    """

    regressors: list[Any]

    @classmethod
    def fit(cls, held_out_probabilities: np.ndarray, held_out_labels: np.ndarray) -> "_IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        regressors: list[Any] = []
        for column in range(held_out_probabilities.shape[1]):
            binary_target: np.ndarray = (held_out_labels == column).astype(float)
            regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            regressor.fit(held_out_probabilities[:, column], binary_target)
            regressors.append(regressor)
        return cls(regressors=regressors)

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated: np.ndarray = np.column_stack(
            [
                regressor.predict(probabilities[:, index])
                for index, regressor in enumerate(self.regressors)
            ]
        )
        totals: np.ndarray = calibrated.sum(axis=1, keepdims=True)
        totals = np.where(totals <= 1e-9, 1.0, totals)
        return calibrated / totals


def _inner_purged_split(
    train_index: np.ndarray,
    purge_bars: int,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve an early-stopping validation slice *inside* one CV fold's training block.

    Using the fold's own ``validation_index`` for early stopping would leak -
    the training process would adapt to data we are about to score as
    "out-of-fold". This purges a small inner tail from ``train_index`` itself
    instead, so the fold's real validation block stays untouched until
    scoring.
    """
    rows: int = len(train_index)
    if rows < 50:
        return train_index, np.array([], dtype=np.int64)
    split: int = max(1, int(rows * (1.0 - validation_fraction)))
    inner_train_end: int = max(1, split - purge_bars)
    return train_index[:inner_train_end], train_index[split:]


def _direction_stack_values(probabilities: dict[str, Any]) -> dict[str, Any]:
    """Derive the 5 stacking values from a label-name -> probability mapping.

    Works uniformly whether ``probabilities`` holds plain floats (a single
    live :class:`~module_c_ml.schemas.DirectionPrediction`) or numpy arrays
    (a full out-of-fold probability matrix) - numpy broadcasts either way,
    which is what keeps the training-time (OOF) and inference-time (live)
    stacked features built from exactly the same formula.
    """
    long_p: Any = probabilities.get(LabelClass.LONG_SUCCESS.value, 0.0)
    short_p: Any = probabilities.get(LabelClass.SHORT_SUCCESS.value, 0.0)
    no_trade_p: Any = np.maximum(0.0, 1.0 - long_p - short_p)
    confidence: Any = np.maximum(np.maximum(long_p, short_p), no_trade_p)
    margin: Any = np.abs(np.asarray(long_p) - np.asarray(short_p))
    return {
        STACK_DIRECTION_LONG: long_p,
        STACK_DIRECTION_SHORT: short_p,
        STACK_DIRECTION_NO_TRADE: no_trade_p,
        STACK_DIRECTION_CONFIDENCE: confidence,
        STACK_DIRECTION_MARGIN: margin,
    }


def _augment_with_direction_stack(
    dataset: ProcessedDataset,
    stack_values: dict[str, np.ndarray],
    covered: np.ndarray,
    active_features: list[str],
) -> ProcessedDataset:
    """Attach Direction's OOF stack onto the dataset, keeping only covered rows.

    Only out-of-fold-covered rows are usable, leakage-free stacking inputs
    (see the module docstring): the dataset's leading block - which no
    walk-forward fold could score without training on its own future - is
    dropped from this *augmented copy* used by Entry/Exit/Risk.  The original
    ``dataset`` (which Direction itself trains on) is untouched.
    """
    stack_frame: pd.DataFrame = pd.DataFrame(stack_values, index=dataset.features.index)
    combined_features: pd.DataFrame = pd.concat(
        [dataset.features[active_features], stack_frame], axis=1
    )
    combined_features = combined_features.loc[covered].reset_index(drop=True)
    feature_columns: tuple[str, ...] = tuple(active_features) + STACK_COLUMNS

    return ProcessedDataset(
        features=combined_features,
        direction_target=dataset.direction_target.loc[covered].reset_index(drop=True),
        entry_target=dataset.entry_target.loc[covered].reset_index(drop=True),
        exit_targets=dataset.exit_targets.loc[covered].reset_index(drop=True),
        risk_target=dataset.risk_target.loc[covered].reset_index(drop=True),
        metadata=dataset.metadata.loc[covered].reset_index(drop=True),
        symbols=dataset.symbols,
        feature_columns=feature_columns,
        sample_weight=dataset.sample_weight.loc[covered].reset_index(drop=True),
    )


class BaseModelHead(ABC):
    """Shared training / persistence plumbing for the four heads."""

    #: Filename stem of the artifact, unique per head.
    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._config: MLSettings = settings.ml
        self._model: Any = None
        self._feature_columns: tuple[str, ...] = FEATURE_COLUMNS
        self._metadata: dict[str, Any] = {}
        self._calibrator: _IsotonicCalibrator | None = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """``True`` once a fitted estimator is available for inference."""
        return self._model is not None

    @property
    def metadata(self) -> dict[str, Any]:
        """Training metadata (row counts, metrics, timestamp)."""
        return dict(self._metadata)

    @property
    def version(self) -> str:
        """Artifact version string used in the audit log."""
        return str(self._metadata.get("trained_at", "untrained"))

    @property
    def artifact_path(self) -> Path:
        """Canonical on-disk location of this head's artifact."""
        return self._config.model_dir / f"{self.name}.joblib"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        """Persist the fitted estimator plus its feature contract."""
        if self._model is None:
            raise ModelNotLoadedError(f"{self.name}: nothing to save", head=self.name)

        destination: Path = path or self.artifact_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_version": _ARTIFACT_VERSION,
                "head": self.name,
                "model": self._model,
                "feature_columns": list(self._feature_columns),
                "metadata": self._metadata,
                "calibrator": self._calibrator,
            },
            destination,
            compress=3,
        )
        _LOGGER.info("Saved %s model -> %s", self.name, destination)
        return destination

    def load(self, path: Path | None = None) -> bool:
        """Load a previously saved artifact.

        Returns:
            ``True`` when the head is ready for inference afterwards.
        """
        source: Path = path or self.artifact_path
        payload: dict[str, Any] | None = load_artifact(source)
        if payload is None:
            return False

        self._model = payload["model"]
        columns: Sequence[str] = payload.get("feature_columns") or FEATURE_COLUMNS
        self._feature_columns = tuple(str(column) for column in columns)
        self._metadata = dict(payload.get("metadata") or {})
        self._calibrator = payload.get("calibrator")
        _LOGGER.info(
            "Loaded %s model (trained_at=%s, rows=%s)",
            self.name,
            self._metadata.get("trained_at", "?"),
            self._metadata.get("rows", "?"),
        )
        return True

    # ------------------------------------------------------------------
    # Estimator factories
    # ------------------------------------------------------------------
    def _n_jobs(self) -> int:
        """Thread budget for the booster (kept small: we run many in parallel)."""
        return max(1, self._config.inference_workers)

    def _weight_array(self, weights: pd.Series) -> np.ndarray | None:
        """``weights`` as a plain array, or ``None`` to fit uniformly-weighted.

        Gated on ``ml.use_sample_weighting`` mainly so a walk-forward
        evaluation run can A/B this specific change against the pre-redesign
        uniform-weight behaviour.
        """
        return weights.to_numpy() if self._config.use_sample_weighting else None

    def _make_classifier(
        self,
        num_class: int,
        class_weight: str | None,
        seed_offset: int = 0,
    ) -> Any:
        """Build an untrained gradient-boosted classifier.

        ``class_weight`` is always supplied explicitly by the caller (rather
        than read from a single shared setting) - Direction and Entry answer
        different questions and have deliberately independent weighting
        policies (see ``MLSettings.class_weight`` / ``entry_class_weight``).
        """
        config: MLSettings = self._config
        random_state: int = config.random_state + seed_offset
        if config.booster == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                reg_lambda=config.reg_lambda,
                min_child_weight=config.min_child_samples,
                objective="multi:softprob" if num_class > 2 else "binary:logistic",
                num_class=num_class if num_class > 2 else None,
                random_state=random_state,
                n_jobs=self._n_jobs(),
                tree_method="hist",
                verbosity=0,
            )

        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            num_leaves=config.num_leaves,
            subsample=config.subsample,
            subsample_freq=1,
            colsample_bytree=config.colsample_bytree,
            reg_lambda=config.reg_lambda,
            min_child_samples=config.min_child_samples,
            objective="multiclass" if num_class > 2 else "binary",
            num_class=num_class if num_class > 2 else 1,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=self._n_jobs(),
            verbose=-1,
        )

    def _make_regressor(self) -> Any:
        """Build an untrained gradient-boosted regressor."""
        config: MLSettings = self._config
        if config.booster == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                reg_lambda=config.reg_lambda,
                min_child_weight=config.min_child_samples,
                objective="reg:squarederror",
                random_state=config.random_state,
                n_jobs=self._n_jobs(),
                tree_method="hist",
                verbosity=0,
            )

        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            num_leaves=config.num_leaves,
            subsample=config.subsample,
            subsample_freq=1,
            colsample_bytree=config.colsample_bytree,
            reg_lambda=config.reg_lambda,
            min_child_samples=config.min_child_samples,
            objective="regression",
            random_state=config.random_state,
            n_jobs=self._n_jobs(),
            verbose=-1,
        )

    def _fit_estimator(
        self,
        estimator: Any,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame,
        y_validation: pd.Series,
        eval_metric: str,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        """Fit with early stopping when a non-empty validation block exists."""
        rounds: int = self._config.early_stopping_rounds
        if x_validation.empty or rounds <= 0:
            estimator.fit(x_train, y_train, sample_weight=sample_weight)
            return estimator

        if self._config.booster == "lightgbm":
            import inspect

            import lightgbm as lgb

            callbacks: list[Any] = [
                lgb.early_stopping(rounds, verbose=False),
                lgb.log_evaluation(0),
            ]
            # LightGBM 4.7 deprecated `eval_set` in favour of `eval_X`/`eval_y`,
            # which take the matrices directly rather than a list of pairs.
            # Probe the signature so the same code works across both generations.
            parameters = inspect.signature(estimator.fit).parameters
            if "eval_X" in parameters:
                estimator.fit(
                    x_train,
                    y_train,
                    sample_weight=sample_weight,
                    eval_X=x_validation,
                    eval_y=y_validation,
                    eval_metric=eval_metric,
                    callbacks=callbacks,
                )
            else:
                estimator.fit(
                    x_train,
                    y_train,
                    sample_weight=sample_weight,
                    eval_set=[(x_validation, y_validation)],
                    eval_metric=eval_metric,
                    callbacks=callbacks,
                )
            return estimator

        estimator.set_params(early_stopping_rounds=rounds, eval_metric=eval_metric)
        estimator.fit(
            x_train, y_train, sample_weight=sample_weight,
            eval_set=[(x_validation, y_validation)], verbose=False,
        )
        return estimator

    def _fit_ensemble(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame,
        y_validation: pd.Series,
        eval_metric: str,
        num_class: int,
        class_weight: str | None,
        sample_weight: np.ndarray | None,
    ) -> Any:
        """Fit ``ensemble_size`` independently-seeded classifiers; bag them.

        A single member is returned unwrapped (``ensemble_size=1`` fully
        disables ensembling with no wrapper overhead).
        """
        size: int = max(1, self._config.ensemble_size)
        estimators: list[Any] = []
        for seed_offset in range(size):
            estimator: Any = self._make_classifier(num_class, class_weight, seed_offset=seed_offset)
            estimator = self._fit_estimator(
                estimator, x_train, y_train, x_validation, y_validation, eval_metric, sample_weight
            )
            estimators.append(estimator)
        return estimators[0] if len(estimators) == 1 else _EnsembleClassifier(estimators=estimators)

    # ------------------------------------------------------------------
    # Feature alignment
    # ------------------------------------------------------------------
    def _align(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reindex incoming features onto the exact training-time column layout.

        A booster indexes features positionally, so silently reordered columns
        would produce confident nonsense.  Missing columns are filled with 0.0
        and logged - the model contract is enforced here, not hoped for.
        """
        missing: list[str] = [
            column for column in self._feature_columns if column not in features.columns
        ]
        if missing:
            _LOGGER.warning("%s: %d feature(s) missing at inference: %s", self.name, len(missing), missing[:5])

        aligned: pd.DataFrame = features.reindex(columns=list(self._feature_columns), fill_value=0.0)
        return aligned.astype(np.float64).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _require_model(self) -> Any:
        """Return the fitted estimator or raise."""
        if self._model is None:
            raise ModelNotLoadedError(f"{self.name} model is not loaded", head=self.name)
        return self._model

    @abstractmethod
    def train(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Fit the head and return its validation metrics."""


class DirectionModel(BaseModelHead):
    """Model 1 - multi-class market direction.

    Predicts the probability distribution over the three label classes produced
    by :class:`~module_b_features.labeler.TradeLabeler`
    (``LONG_SUCCESS`` / ``SHORT_SUCCESS`` / ``NO_TRADE_OR_FAIL``), which the
    schema then aggregates into LONG / SHORT / NO_TRADE mass.  Also the source
    of the out-of-fold stacked features every other head conditions on - see
    the module docstring.
    """

    name = "direction_model"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        #: Populated by :meth:`train`; consumed by
        #: :meth:`MLSubsystem.train_all` to build the stacked dataset for
        #: Entry/Exit/Risk. Not persisted - it is training-time-only scaffolding.
        self.last_oof_stack: dict[str, np.ndarray] | None = None
        self.last_oof_covered: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Out-of-fold predictions (calibration input + Entry/Exit/Risk stacking)
    # ------------------------------------------------------------------
    def generate_oof_predictions(
        self,
        dataset: ProcessedDataset,
        feature_columns: list[str],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Purged walk-forward out-of-fold predicted probabilities, per row.

        For every fold (see :mod:`module_c_ml.cross_validation`) a fresh
        classifier is fit on that fold's training block - with its OWN inner
        purged split for early stopping, never the fold's own validation
        block (see :func:`_inner_purged_split`) - and scored on the fold's
        validation block. Rows in the dataset's leading, un-covered block
        (before any fold's history was long enough to train on) fall back to
        the overall class prior rather than a fabricated confident guess.

        Returns:
            ``(probabilities, covered)`` where ``probabilities`` maps each
            :data:`~module_b_features.labeler.LABEL_ORDER` class name to a
            full-length float array, and ``covered`` is the boolean mask of
            rows an actual fold scored.
        """
        rows: int = len(dataset)
        horizon: int = self._settings.labels.max_holding_bars
        folds: list[CVFold] = purged_walk_forward_splits(
            rows, self._config.cv_folds, horizon, self._config.cv_embargo_bars
        )

        class_index: dict[str, int] = {name: index for index, name in enumerate(LABEL_ORDER)}
        encoded: pd.Series = dataset.direction_target.map(class_index)
        num_classes: int = len(LABEL_ORDER)
        proba: np.ndarray = np.full((rows, num_classes), np.nan, dtype=np.float64)
        covered: np.ndarray = np.zeros(rows, dtype=bool)

        prior: np.ndarray = (
            encoded.value_counts(normalize=True).reindex(range(num_classes), fill_value=0.0).to_numpy()
        )
        if prior.sum() <= 0.0:
            prior = np.full(num_classes, 1.0 / num_classes)

        if not folds:
            _LOGGER.warning(
                "Direction OOF: dataset too small for %d purged folds - "
                "calibration/stacking will fall back to the class prior",
                self._config.cv_folds,
            )
            for class_position in range(num_classes):
                proba[:, class_position] = prior[class_position]
            probabilities: dict[str, np.ndarray] = {
                name: proba[:, class_index[name]] for name in LABEL_ORDER
            }
            return probabilities, covered

        features: pd.DataFrame = dataset.features[feature_columns]
        weights: pd.Series = dataset.sample_weight

        for fold in folds:
            inner_train, inner_val = _inner_purged_split(fold.train_index, horizon)
            if len(inner_train) < _MIN_OOF_FOLD_TRAIN_ROWS or encoded.iloc[inner_train].nunique() < 2:
                continue

            estimator: Any = self._make_classifier(num_classes, self._config.class_weight)
            estimator = self._fit_estimator(
                estimator,
                features.iloc[inner_train],
                encoded.iloc[inner_train].astype(int),
                features.iloc[inner_val],
                encoded.iloc[inner_val].astype(int),
                eval_metric="multi_logloss",
                sample_weight=self._weight_array(weights.iloc[inner_train]),
            )

            fold_proba: np.ndarray = estimator.predict_proba(features.iloc[fold.validation_index])
            classes: list[int] = [int(value) for value in getattr(estimator, "classes_", [])]
            for position, class_position in enumerate(classes):
                if 0 <= class_position < num_classes:
                    proba[fold.validation_index, class_position] = fold_proba[:, position]
            covered[fold.validation_index] = True

        for class_position in range(num_classes):
            column: np.ndarray = proba[:, class_position]
            column[~covered] = prior[class_position]

        probabilities = {name: proba[:, class_index[name]] for name in LABEL_ORDER}
        return probabilities, covered

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        dataset: ProcessedDataset,
        feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fit the (optionally ensembled, calibrated) multi-class classifier.

        Args:
            dataset: Pooled, purged training dataset.
            feature_columns: Explicit feature set to train on (e.g. after
                null-importance pruning). Defaults to ``dataset.feature_columns``.
        """
        if dataset.is_empty:
            raise ModelTrainingError("direction model received an empty dataset")

        classes: list[str] = sorted(set(dataset.direction_target.unique()))
        if len(classes) < 2:
            raise ModelTrainingError("direction model needs >= 2 classes", classes=classes)

        class_index: dict[str, int] = {name: index for index, name in enumerate(LABEL_ORDER)}
        encoded: pd.Series = dataset.direction_target.map(class_index)
        if encoded.isna().any():
            raise ModelTrainingError("direction labels contain unknown classes")

        active_features: list[str] = feature_columns or list(dataset.feature_columns)

        # --- Out-of-fold predictions: feed calibration and (via MLSubsystem)
        # the Entry/Exit/Risk stacking features. ---------------------------
        self.last_oof_stack = None
        self.last_oof_covered = None
        oof_probabilities: dict[str, np.ndarray] | None = None
        oof_covered: np.ndarray | None = None
        if self._config.calibrate_probabilities or self._config.use_direction_stacking:
            oof_probabilities, oof_covered = self.generate_oof_predictions(dataset, active_features)
            if self._config.use_direction_stacking:
                self.last_oof_stack = _direction_stack_values(oof_probabilities)
                self.last_oof_covered = oof_covered

        # --- Final model: fit on (an ensembled bag over) the full training
        # block, early-stopped against the standard purged tail split. -----
        train_index, validation_index = dataset.train_validation_split(
            self._config.validation_fraction, self._config.purge_bars
        )
        features: pd.DataFrame = dataset.features[active_features]
        weights: pd.Series = dataset.sample_weight

        self._model = self._fit_ensemble(
            features.iloc[train_index],
            encoded.iloc[train_index].astype(int),
            features.iloc[validation_index],
            encoded.iloc[validation_index].astype(int),
            eval_metric="multi_logloss",
            num_class=len(LABEL_ORDER),
            class_weight=self._config.class_weight,
            sample_weight=self._weight_array(weights.iloc[train_index]),
        )
        self._feature_columns = tuple(active_features)

        # --- Calibration: fit on OOF predictions, never on rows the final
        # model's own bag members were trained on. --------------------------
        self._calibrator = None
        if self._config.calibrate_probabilities and oof_probabilities is not None and oof_covered is not None:
            if np.any(oof_covered):
                oof_matrix: np.ndarray = np.column_stack(
                    [oof_probabilities[name][oof_covered] for name in LABEL_ORDER]
                )
                self._calibrator = _IsotonicCalibrator.fit(
                    oof_matrix, encoded.to_numpy()[oof_covered]
                )

        metrics: dict[str, Any] = self._score(
            self._model,
            features.iloc[validation_index],
            encoded.iloc[validation_index].astype(int),
            self._calibrator,
        )
        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "classes": list(LABEL_ORDER),
            "distribution": dataset.class_distribution(),
            "ensemble_size": max(1, self._config.ensemble_size),
            "calibrated": self._calibrator is not None,
            "oof_coverage": float(np.mean(oof_covered)) if oof_covered is not None else 0.0,
            "metrics": metrics,
        }
        _LOGGER.info("Direction model trained: %s", metrics)
        return metrics

    @staticmethod
    def _score(
        estimator: Any,
        features: pd.DataFrame,
        target: pd.Series,
        calibrator: "_IsotonicCalibrator | None",
    ) -> dict[str, float]:
        """Compute validation accuracy, balanced accuracy and (raw + calibrated) log loss."""
        if features.empty:
            return {}
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

        predictions: np.ndarray = estimator.predict(features)
        probabilities: np.ndarray = estimator.predict_proba(features)
        labels: list[int] = list(range(len(LABEL_ORDER)))
        result: dict[str, float] = {
            "accuracy": float(accuracy_score(target, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        }
        try:
            result["log_loss"] = float(log_loss(target, probabilities, labels=labels))
        except ValueError:  # pragma: no cover - degenerate validation block
            result["log_loss"] = float("nan")

        if calibrator is not None:
            try:
                calibrated: np.ndarray = calibrator.transform(probabilities)
                result["log_loss_calibrated"] = float(log_loss(target, calibrated, labels=labels))
            except ValueError:  # pragma: no cover
                result["log_loss_calibrated"] = float("nan")
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> DirectionPrediction:
        """Score a single feature row (stateless, thread-safe).

        Falls back to a transparent trend/regime heuristic when no artifact is
        loaded, stamping the result as ``HEURISTIC``.
        """
        if self._model is None:
            return self._heuristic(features)

        aligned: pd.DataFrame = self._align(features)
        raw: np.ndarray = np.asarray(self._model.predict_proba(aligned), dtype=np.float64)
        if self._calibrator is not None:
            raw = self._calibrator.transform(raw)
        raw_row: np.ndarray = raw[0]

        classes: list[int] = [int(value) for value in getattr(self._model, "classes_", [])]
        probabilities: dict[str, float] = {name: 0.0 for name in LABEL_ORDER}
        if classes and len(classes) == raw_row.size:
            for position, class_index in enumerate(classes):
                if 0 <= class_index < len(LABEL_ORDER):
                    probabilities[LABEL_ORDER[class_index]] = float(raw_row[position])
        else:  # pragma: no cover - estimator without `classes_`
            for position, name in enumerate(LABEL_ORDER[: raw_row.size]):
                probabilities[name] = float(raw_row[position])

        return DirectionPrediction(probabilities=probabilities, source=ModelSource.TRAINED)

    @staticmethod
    def _heuristic(features: pd.DataFrame) -> DirectionPrediction:
        """Documented fallback: KAMA slope + DI spread, gated by the FDI regime.

        The score is squashed with a logistic so it behaves like a probability,
        then damped by ``fdi_trending`` - in a ranging market (FDI > 1.5) the
        directional read is deliberately pushed toward NO_TRADE.
        """
        row: pd.Series = features.iloc[0]
        kama_slope: float = float(row.get("kama_slope", 0.0) or 0.0)
        di_spread: float = float(row.get("di_spread", 0.0) or 0.0)
        trending: float = float(row.get("fdi_trending", 0.0) or 0.0)
        regime: float = float(row.get("hmm_regime", -1.0) or -1.0)

        score: float = 60.0 * kama_slope + 1.2 * di_spread
        if regime == float(HMMRegime.BULL_TREND):
            score += 0.25
        elif regime == float(HMMRegime.BEAR_TREND):
            score -= 0.25

        directional: float = 1.0 / (1.0 + float(np.exp(-clamp(score, -8.0, 8.0))))
        # Ranging markets keep at least 55 % of the mass on "do nothing".
        trade_mass: float = 0.45 * (0.35 + 0.65 * trending)
        long_mass: float = trade_mass * directional
        short_mass: float = trade_mass * (1.0 - directional)

        return DirectionPrediction(
            probabilities={
                LabelClass.LONG_SUCCESS.value: long_mass,
                LabelClass.SHORT_SUCCESS.value: short_mass,
                LabelClass.NO_TRADE_OR_FAIL.value: max(1e-6, 1.0 - long_mass - short_mass),
            },
            source=ModelSource.HEURISTIC,
        )


class EntryModel(BaseModelHead):
    """Model 2 - entry timing filter, and the system's meta-label classifier.

    Answers a narrow question: given that Direction has already proposed a
    side (available to this head as stacked input, see the module docstring),
    is *this* candle close a clean entry, or should we wait for the next
    5-minute bar?  Trained on ``entry_quality``, which marks bars whose
    winning trade never gave back more than ``low_risk_mae_ratio`` of its stop.

    Uses its own class-weighting policy (``ml.entry_class_weight``,
    independent of Direction's) and learns its own operating threshold from a
    held-out precision/recall sweep, rather than relying on a single fixed
    default cutoff - see :meth:`train`.
    """

    name = "entry_model"

    def train(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Fit the binary entry-quality classifier."""
        if dataset.is_empty:
            raise ModelTrainingError("entry model received an empty dataset")

        target: pd.Series = dataset.entry_target.astype(int)
        if target.nunique() < 2:
            raise ModelTrainingError("entry target is degenerate (single class)")

        train_index, validation_index = dataset.train_validation_split(
            self._config.validation_fraction, self._config.purge_bars
        )
        features: pd.DataFrame = dataset.features
        weights: pd.Series = dataset.sample_weight

        self._model = self._fit_ensemble(
            features.iloc[train_index],
            target.iloc[train_index],
            features.iloc[validation_index],
            target.iloc[validation_index],
            eval_metric="binary_logloss",
            num_class=2,
            class_weight=self._config.entry_class_weight,
            sample_weight=self._weight_array(weights.iloc[train_index]),
        )
        self._feature_columns = dataset.feature_columns

        metrics: dict[str, float] = {}
        self._calibrator = None
        optimal_threshold: float = self._settings.decision.min_entry_probability

        if len(validation_index) > 0:
            from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score

            validation_features: pd.DataFrame = features.iloc[validation_index]
            validation_target: pd.Series = target.iloc[validation_index]
            raw_probabilities: np.ndarray = np.asarray(
                self._model.predict_proba(validation_features), dtype=np.float64
            )

            if self._config.calibrate_probabilities and validation_target.nunique() > 1:
                self._calibrator = _IsotonicCalibrator.fit(
                    raw_probabilities, validation_target.to_numpy()
                )
                scored_probabilities: np.ndarray = self._calibrator.transform(raw_probabilities)
            else:
                scored_probabilities = raw_probabilities

            positive_probability: np.ndarray = scored_probabilities[:, -1]

            # Learn the operating threshold from a precision/recall sweep
            # (maximise F1) instead of hard-coding a fixed cutoff - "properly
            # fixing" the entry gate means the threshold reflects THIS
            # model's actual calibrated probability distribution.
            if validation_target.nunique() > 1:
                precision, recall, thresholds = precision_recall_curve(
                    validation_target, positive_probability
                )
                f1: np.ndarray = np.where(
                    (precision + recall) > 0,
                    2 * precision * recall / np.maximum(precision + recall, 1e-12),
                    0.0,
                )
                # precision_recall_curve returns one more precision/recall
                # point than thresholds (the all-positive endpoint); align.
                if len(thresholds) > 0:
                    best_index: int = int(np.argmax(f1[: len(thresholds)]))
                    optimal_threshold = float(thresholds[best_index])

            predictions: np.ndarray = (positive_probability >= optimal_threshold).astype(int)
            metrics = {
                "precision": float(precision_score(validation_target, predictions, zero_division=0)),
                "recall": float(recall_score(validation_target, predictions, zero_division=0)),
                "roc_auc": (
                    float(roc_auc_score(validation_target, positive_probability))
                    if validation_target.nunique() > 1
                    else float("nan")
                ),
                "optimal_threshold": optimal_threshold,
            }

        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(train_index)),
            "positive_rate": float(target.mean()),
            "ensemble_size": max(1, self._config.ensemble_size),
            "calibrated": self._calibrator is not None,
            "optimal_threshold": optimal_threshold,
            "metrics": metrics,
        }
        _LOGGER.info("Entry model trained: %s", metrics)
        return metrics

    def predict(
        self,
        features: pd.DataFrame,
        action: TradeAction = TradeAction.NO_TRADE,
        threshold: float | None = None,
    ) -> EntryPrediction:
        """Decide whether to act on this candle or wait for the next one.

        ``threshold`` overrides everything when supplied by the caller.
        Otherwise the model's own learned ``optimal_threshold`` (see
        :meth:`train`) is used, floored at ``decision.min_entry_probability``
        so an operator-configured minimum can never be undercut by training.
        """
        cutoff: float = threshold if threshold is not None else max(
            float(self._metadata.get("optimal_threshold", 0.0)),
            self._settings.decision.min_entry_probability,
        )
        if self._model is None:
            return self._heuristic(features, action, cutoff)

        aligned: pd.DataFrame = self._align(features)
        raw: np.ndarray = np.asarray(self._model.predict_proba(aligned), dtype=np.float64)
        if self._calibrator is not None:
            raw = self._calibrator.transform(raw)
        probability: float = float(raw[0, -1])
        return EntryPrediction(
            probability=clamp(probability, 0.0, 1.0),
            should_enter=probability >= cutoff,
            source=ModelSource.TRAINED,
            reason=f"p(clean entry)={probability:.3f} vs threshold {cutoff:.3f}",
        )

    @staticmethod
    def _heuristic(
        features: pd.DataFrame,
        action: TradeAction,
        cutoff: float,
    ) -> EntryPrediction:
        """Documented fallback based on order-flow alignment and structure.

        Three additive components, each in ``[-0.15, +0.15]``:

        * order-book imbalance pointing the same way as the intended trade,
        * a trending (low-FDI) structure rather than chop,
        * a spread that is not in the top decile of its own history.
        """
        row: pd.Series = features.iloc[0]
        imbalance: float = float(row.get("ob_imbalance", 0.0) or 0.0)
        trending: float = float(row.get("fdi_trending", 0.0) or 0.0)
        spread_rank: float = float(row.get("ob_spread_rank", 0.5) or 0.5)

        directional_flow: float = imbalance if action is TradeAction.LONG else -imbalance
        score: float = 0.5
        score += clamp(directional_flow, -1.0, 1.0) * 0.15
        score += (trending - 0.5) * 0.20
        score -= clamp(spread_rank - 0.5, -0.5, 0.5) * 0.20

        probability: float = clamp(score, 0.0, 1.0)
        return EntryPrediction(
            probability=probability,
            should_enter=probability >= cutoff,
            source=ModelSource.HEURISTIC,
            reason=(
                f"heuristic entry score={probability:.3f} "
                f"(flow={directional_flow:+.2f}, trending={trending:.0f})"
            ),
        )


class ExitModel(BaseModelHead):
    """Model 3 - dynamic trade management.

    Three independent regressors predict the take-profit distance, the stop-loss
    distance and the trailing-activation distance as fractions of the entry
    price.  They are trained only on rows where a directional trade was actually
    selected, because those are the only rows whose excursion geometry is
    meaningful.
    """

    name = "exit_model"

    _TARGETS: Final[tuple[str, ...]] = ("target_tp_pct", "target_sl_pct", "target_trailing_pct")

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model = None  # dict[str, estimator] once trained

    def train(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Fit one regressor per exit parameter."""
        if dataset.is_empty:
            raise ModelTrainingError("exit model received an empty dataset")

        usable: pd.Series = dataset.exit_targets.notna().all(axis=1)
        if int(usable.sum()) < 100:
            raise ModelTrainingError(
                "not enough directional rows to fit the exit model", rows=int(usable.sum())
            )

        features: pd.DataFrame = dataset.features[usable].reset_index(drop=True)
        targets: pd.DataFrame = dataset.exit_targets[usable].reset_index(drop=True)
        weights: pd.Series = dataset.sample_weight[usable].reset_index(drop=True)

        split_point: int = max(1, int(len(features) * (1.0 - self._config.validation_fraction)))
        train_end: int = max(1, split_point - self._config.purge_bars)
        if train_end < _MIN_REGRESSION_TRAIN_ROWS:
            raise ModelTrainingError(
                "purge_bars leaves too few training rows for the exit model",
                train_rows=train_end, purge_bars=self._config.purge_bars, usable_rows=len(features),
            )

        estimators: dict[str, Any] = {}
        metrics: dict[str, float] = {}
        from sklearn.metrics import mean_absolute_error

        for column in self._TARGETS:
            estimator: Any = self._make_regressor()
            estimator = self._fit_estimator(
                estimator,
                features.iloc[:train_end],
                targets[column].iloc[:train_end],
                features.iloc[split_point:],
                targets[column].iloc[split_point:],
                eval_metric="l1",
                sample_weight=self._weight_array(weights.iloc[:train_end]),
            )
            estimators[column] = estimator
            if split_point < len(features):
                predictions: np.ndarray = estimator.predict(features.iloc[split_point:])
                metrics[f"{column}_mae"] = float(
                    mean_absolute_error(targets[column].iloc[split_point:], predictions)
                )

        self._model = estimators
        self._feature_columns = dataset.feature_columns
        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "rows": int(train_end),
            "metrics": metrics,
        }
        _LOGGER.info("Exit model trained: %s", metrics)
        return metrics

    def predict(self, features: pd.DataFrame) -> ExitParameters:
        """Produce the trade-management geometry for the current bar."""
        if self._model is None:
            return self._heuristic(features)

        aligned: pd.DataFrame = self._align(features)
        estimators: dict[str, Any] = self._model
        take_profit: float = float(estimators["target_tp_pct"].predict(aligned)[0])
        stop_loss: float = float(estimators["target_sl_pct"].predict(aligned)[0])
        trailing: float = float(estimators["target_trailing_pct"].predict(aligned)[0])

        return self._assemble(
            take_profit, stop_loss, trailing, features.iloc[0], ModelSource.TRAINED
        )

    def _heuristic(self, features: pd.DataFrame) -> ExitParameters:
        """Documented fallback: ATR-scaled barriers, widened by regime.

        High-volatility and whipsaw regimes get wider stops (so normal noise does
        not take the trade out) and proportionally wider targets, preserving the
        reward-to-risk ratio.
        """
        row: pd.Series = features.iloc[0]
        atr_pct: float = float(row.get("atr_pct", 0.0) or 0.0)
        if atr_pct <= 0.0:
            atr_pct = max(float(row.get("garch_volatility", 0.0) or 0.0) * 2.0, 0.0025)

        regime: float = float(row.get("hmm_regime", -1.0) or -1.0)
        volatility_rank: float = float(row.get("garch_vol_rank", 0.5) or 0.5)

        widen: float = 1.0 + 0.6 * clamp(volatility_rank, 0.0, 1.0)
        if regime == float(HMMRegime.HIGH_VOLATILITY):
            widen *= 1.35
        elif regime == float(HMMRegime.SIDEWAYS):
            widen *= 0.85

        stop_loss: float = atr_pct * self._settings.labels.sl_atr_multiple * widen
        take_profit: float = atr_pct * self._settings.labels.tp_atr_multiple * widen
        trailing: float = take_profit * 0.5
        return self._assemble(take_profit, stop_loss, trailing, row, ModelSource.HEURISTIC)

    @staticmethod
    def _assemble(
        take_profit: float,
        stop_loss: float,
        trailing: float,
        row: pd.Series,
        source: ModelSource,
    ) -> ExitParameters:
        """Clamp raw predictions into a geometry that is always tradeable.

        A regressor can emit a negative or absurd distance; the exchange cannot.
        These rails guarantee a positive, ordered geometry regardless of what the
        model produced:

        * the stop is never tighter than half the current ATR, because a stop
          inside 5-minute noise gets hit for reasons unrelated to the thesis;
        * the target is never below 1.1x nor above 6x the stop, which trims the
          degenerate "16:1" geometries a collapsed stop regressor can produce.
        """
        atr_pct: float = float(row.get("atr_pct", 0.0) or 0.0)
        if atr_pct > 0.0:
            stop_loss = max(stop_loss, atr_pct * _MIN_SL_ATR_FRACTION)
        stop_loss = clamp(stop_loss, _MIN_SL_PCT, _MAX_SL_PCT)

        take_profit = clamp(take_profit, _MIN_TP_PCT, _MAX_TP_PCT)
        take_profit = clamp(take_profit, stop_loss * 1.1, stop_loss * _MAX_REWARD_RISK)
        take_profit = min(take_profit, _MAX_TP_PCT)

        trailing_activation: float = clamp(trailing, take_profit * 0.25, take_profit * 0.95)
        trailing_distance: float = clamp(stop_loss * 0.6, _MIN_SL_PCT, _MAX_SL_PCT)

        return ExitParameters(
            take_profit_pct=take_profit,
            stop_loss_pct=stop_loss,
            trailing_activation_pct=trailing_activation,
            trailing_distance_pct=trailing_distance,
            source=source,
        )


class RiskModel(BaseModelHead):
    """Model 4 - capital allocation and leverage sizing.

    A regressor estimates an opportunity score in ``[0, 1]`` (learned from the
    labeler's ``target_risk_score``, which fuses path heat and the volatility
    regime).  Deterministic sizing logic then converts that score, the direction
    model's confidence and the current GARCH percentile into an integer leverage
    and a capital allocation - returning ``0x`` whenever the trade should abort.
    """

    name = "risk_model"

    def train(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Fit the opportunity-score regressor."""
        if dataset.is_empty:
            raise ModelTrainingError("risk model received an empty dataset")

        target: pd.Series = dataset.risk_target.astype(float)
        train_index, validation_index = dataset.train_validation_split(
            self._config.validation_fraction, self._config.purge_bars
        )
        features: pd.DataFrame = dataset.features
        weights: pd.Series = dataset.sample_weight

        estimator: Any = self._make_regressor()
        estimator = self._fit_estimator(
            estimator,
            features.iloc[train_index],
            target.iloc[train_index],
            features.iloc[validation_index],
            target.iloc[validation_index],
            eval_metric="l2",
            sample_weight=self._weight_array(weights.iloc[train_index]),
        )

        self._model = estimator
        self._feature_columns = dataset.feature_columns

        metrics: dict[str, float] = {}
        if len(validation_index) > 0:
            from sklearn.metrics import mean_absolute_error, r2_score

            predictions: np.ndarray = estimator.predict(features.iloc[validation_index])
            metrics = {
                "mae": float(mean_absolute_error(target.iloc[validation_index], predictions)),
                "r2": float(r2_score(target.iloc[validation_index], predictions)),
            }

        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(train_index)),
            "metrics": metrics,
        }
        _LOGGER.info("Risk model trained: %s", metrics)
        return metrics

    def predict(
        self,
        features: pd.DataFrame,
        direction_confidence: float,
    ) -> RiskAllocation:
        """Size the trade, or abort it.

        Args:
            features: One feature row.
            direction_confidence: Winning probability mass from Model 1.

        Returns:
            A :class:`RiskAllocation`; ``leverage == 0`` means "do not trade".
        """
        row: pd.Series = features.iloc[0]
        volatility_rank: float = clamp(float(row.get("garch_vol_rank", 0.5) or 0.5), 0.0, 1.0)

        if self._model is None:
            score: float = self._heuristic_score(row, direction_confidence)
            source: ModelSource = ModelSource.HEURISTIC
        else:
            aligned: pd.DataFrame = self._align(features)
            score = clamp(float(self._model.predict(aligned)[0]), 0.0, 1.0)
            source = ModelSource.TRAINED

        decision = self._settings.decision
        risk = self._settings.risk

        # --- Hard vetoes -------------------------------------------------
        if volatility_rank >= decision.max_volatility_percentile:
            return RiskAllocation(
                leverage=0,
                capital_allocation_pct=0.0,
                risk_score=score,
                abort_reason=(
                    f"volatility percentile {volatility_rank:.2f} >= "
                    f"{decision.max_volatility_percentile:.2f}"
                ),
                source=source,
            )
        if direction_confidence < decision.min_direction_confidence:
            return RiskAllocation(
                leverage=0,
                capital_allocation_pct=0.0,
                risk_score=score,
                abort_reason=(
                    f"direction confidence {direction_confidence:.3f} < "
                    f"{decision.min_direction_confidence:.3f}"
                ),
                source=source,
            )

        # --- Sizing ------------------------------------------------------
        # Confidence is rescaled onto [0, 1] across the *tradeable* band, so a
        # 70 %-confidence signal sizes near the floor and a 100 % one near the cap.
        confidence_span: float = max(1e-6, 1.0 - decision.min_direction_confidence)
        confidence_factor: float = clamp(
            (direction_confidence - decision.min_direction_confidence) / confidence_span, 0.0, 1.0
        )
        volatility_factor: float = 1.0 - clamp(volatility_rank, 0.0, 1.0) ** 2

        composite: float = score * (0.35 + 0.65 * confidence_factor) * volatility_factor
        leverage_cap: int = min(decision.max_leverage, risk.max_leverage)
        leverage: int = int(np.floor(composite * leverage_cap))

        if leverage < decision.min_leverage:
            return RiskAllocation(
                leverage=0,
                capital_allocation_pct=0.0,
                risk_score=score,
                abort_reason=(
                    f"sized leverage {leverage}x below the minimum "
                    f"{decision.min_leverage}x (composite={composite:.3f})"
                ),
                source=source,
            )
        leverage = int(clamp(float(leverage), float(decision.min_leverage), float(leverage_cap)))

        allocation_span: float = (
            decision.max_capital_allocation_pct - decision.min_capital_allocation_pct
        )
        allocation: float = decision.min_capital_allocation_pct + allocation_span * composite
        allocation = clamp(
            allocation, decision.min_capital_allocation_pct, decision.max_capital_allocation_pct
        )

        return RiskAllocation(
            leverage=leverage,
            capital_allocation_pct=allocation,
            risk_score=score,
            abort_reason="",
            source=source,
        )

    @staticmethod
    def _heuristic_score(row: pd.Series, direction_confidence: float) -> float:
        """Fallback opportunity score from trend strength and volatility rank."""
        adx_value: float = clamp(float(row.get("adx", 0.0) or 0.0) / 50.0, 0.0, 1.0)
        trending: float = float(row.get("fdi_trending", 0.0) or 0.0)
        volatility_rank: float = clamp(float(row.get("garch_vol_rank", 0.5) or 0.5), 0.0, 1.0)
        raw: float = (
            0.4 * direction_confidence + 0.3 * adx_value + 0.2 * trending + 0.1 * (1.0 - volatility_rank)
        )
        return clamp(raw, 0.0, 1.0)


class MLSubsystem:
    """Owns the four heads and exposes a single async inference entry point."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self.direction: DirectionModel = DirectionModel(settings)
        self.entry: EntryModel = EntryModel(settings)
        self.exit: ExitModel = ExitModel(settings)
        self.risk: RiskModel = RiskModel(settings)
        self._inference_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            max(1, settings.ml.inference_workers * 2)
        )

    @property
    def heads(self) -> dict[str, BaseModelHead]:
        """All four heads keyed by their role."""
        return {
            "direction": self.direction,
            "entry": self.entry,
            "exit": self.exit,
            "risk": self.risk,
        }

    @property
    def all_loaded(self) -> bool:
        """``True`` when every head has a fitted artifact available."""
        return all(head.is_loaded for head in self.heads.values())

    @property
    def loaded_heads(self) -> dict[str, bool]:
        """Per-head readiness, surfaced on the dashboard."""
        return {name: head.is_loaded for name, head in self.heads.items()}

    def versions(self) -> dict[str, str]:
        """Artifact versions, recorded on every audit row."""
        return {name: head.version for name, head in self.heads.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load_all(self) -> dict[str, bool]:
        """Load every artifact from disk, reporting which succeeded."""
        results: dict[str, bool] = {name: head.load() for name, head in self.heads.items()}
        missing: list[str] = [name for name, ok in results.items() if not ok]
        if missing:
            _LOGGER.warning(
                "Models unavailable: %s - those heads will use their documented heuristics",
                ", ".join(missing),
            )
        return results

    def save_all(self) -> dict[str, str]:
        """Persist every fitted head; unfitted heads are skipped."""
        saved: dict[str, str] = {}
        for name, head in self.heads.items():
            if head.is_loaded:
                saved[name] = str(head.save())
        return saved

    async def train_all(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Train all four heads off the event loop, then persist them.

        Orchestrates the full pipeline described in the module docstring:

        1. Null-importance feature pruning (shared across all heads, keyed
           off Direction's target - see ``module_c_ml.feature_selection``).
        2. Direction: out-of-fold predictions (calibration + stacking input),
           then the final ensembled/calibrated model.
        3. Entry/Exit/Risk: trained on Direction's out-of-fold stacked
           features appended to the (pruned) base feature set, when
           ``ml.use_direction_stacking`` is on.

        A head that fails to train is reported in the result and left
        unloaded; it does not abort the training of the remaining heads.
        """
        report: dict[str, Any] = {}
        config = self._settings.ml
        active_features: list[str] = list(dataset.feature_columns)

        if config.prune_weak_features and not dataset.is_empty:
            class_index: dict[str, int] = {name: index for index, name in enumerate(LABEL_ORDER)}
            encoded_target: pd.Series = dataset.direction_target.map(class_index).fillna(-1).astype(int)
            try:
                pruning: FeaturePruningReport = await asyncio.to_thread(
                    select_features_by_null_importance,
                    dataset.features,
                    encoded_target,
                    dataset.sample_weight,
                    active_features,
                    self._settings,
                    self._settings.labels.max_holding_bars,
                )
                active_features = list(pruning.kept_features)
                report["feature_pruning"] = {
                    "kept": len(pruning.kept_features),
                    "dropped": list(pruning.dropped_features),
                    "folds_used": pruning.folds_used,
                    "fell_back": pruning.fell_back,
                }
            except Exception as error:  # pragma: no cover - pruning must not block training
                _LOGGER.error("Feature pruning failed, keeping all features: %s", error)

        try:
            report["direction"] = await asyncio.to_thread(self.direction.train, dataset, active_features)
            self.direction.save()
        except (ModelTrainingError, ValueError) as error:
            _LOGGER.error("Training failed for direction: %s", error)
            report["direction"] = {"error": str(error)}

        stacked_dataset: ProcessedDataset = dataset
        if (
            config.use_direction_stacking
            and self.direction.last_oof_stack is not None
            and self.direction.last_oof_covered is not None
            and np.any(self.direction.last_oof_covered)
        ):
            stacked_dataset = _augment_with_direction_stack(
                dataset, self.direction.last_oof_stack, self.direction.last_oof_covered, active_features
            )
            report["stacking"] = {
                "enabled": True,
                "rows": len(stacked_dataset),
                "coverage": float(np.mean(self.direction.last_oof_covered)),
            }
        else:
            report["stacking"] = {"enabled": False}

        for name, head in (("entry", self.entry), ("exit", self.exit), ("risk", self.risk)):
            try:
                report[name] = await asyncio.to_thread(head.train, stacked_dataset)
                head.save()
            except (ModelTrainingError, ValueError) as error:
                _LOGGER.error("Training failed for %s: %s", name, error)
                report[name] = {"error": str(error)}

        return report

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer_sync(self, payload: InferencePayload) -> ModelInferenceResult:
        """Run all four heads on one feature row (pure, blocking).

        Exposed separately so the backtester - which is already off the event
        loop - can call it directly without a thread hop.
        """
        started: float = _monotonic_ms()
        features: pd.DataFrame = payload.features

        direction: DirectionPrediction = self.direction.predict(features)

        stacked_features: pd.DataFrame = features
        if self._settings.ml.use_direction_stacking:
            stack_values: dict[str, Any] = _direction_stack_values(direction.probabilities)
            stack_frame: pd.DataFrame = pd.DataFrame(
                {key: [value] for key, value in stack_values.items()}, index=features.index
            )
            stacked_features = pd.concat([features, stack_frame], axis=1)

        entry: EntryPrediction = self.entry.predict(stacked_features, action=direction.action)
        exit_params: ExitParameters = self.exit.predict(stacked_features)
        risk: RiskAllocation = self.risk.predict(
            stacked_features,
            direction_confidence=direction.confidence,
        )

        return ModelInferenceResult(
            symbol=payload.symbol,
            timestamp=payload.timestamp,
            close_price=payload.close,
            direction=direction,
            entry=entry,
            exit_params=exit_params,
            risk=risk,
            feature_snapshot=payload.snapshot,
            latency_ms=_monotonic_ms() - started,
            model_versions=self.versions(),
        )

    async def infer(self, payload: InferencePayload) -> ModelInferenceResult:
        """Async wrapper - runs :meth:`infer_sync` on a worker thread."""
        async with self._inference_semaphore:
            return await asyncio.to_thread(self.infer_sync, payload)

    async def infer_many(
        self,
        payloads: Sequence[InferencePayload],
    ) -> dict[str, ModelInferenceResult]:
        """Score many symbols concurrently, isolating per-symbol failures."""

        async def _one(payload: InferencePayload) -> tuple[str, ModelInferenceResult | None]:
            try:
                return payload.symbol, await self.infer(payload)
            except (ModelNotLoadedError, ValueError) as error:
                _LOGGER.error("Inference failed for %s: %s", payload.symbol, error)
                return payload.symbol, None

        results: list[tuple[str, ModelInferenceResult | None]] = await asyncio.gather(
            *(_one(payload) for payload in payloads)
        )
        return {symbol: result for symbol, result in results if result is not None}


def _monotonic_ms() -> float:
    """Monotonic clock in milliseconds, used for latency accounting."""
    import time

    return time.perf_counter() * 1_000.0
