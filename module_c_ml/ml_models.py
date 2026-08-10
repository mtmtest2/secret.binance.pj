"""Module C - four independent, stateless machine-learning heads.

Architecture
------------
Each head answers exactly one question and knows nothing about the others:

============  ==========================================================
Model 1       *Where* is the market going?      -> probability distribution
Model 2       *Now* or wait one more candle?    -> boolean + probability
Model 3       *How* should the trade be managed? -> TP / SL / trailing geometry
Model 4       *How much* capital and leverage?   -> 0-10x + allocation %
============  ==========================================================

Inference contract
------------------
* **Stateless.**  ``predict`` reads the fitted booster and its arguments, and
  nothing else.  No instance attribute is mutated, so concurrent calls from
  several event-loop tasks are safe.
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
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

import joblib
import numpy as np
import pandas as pd

from config.settings import MLSettings, Settings
from core.exceptions import ModelNotLoadedError, ModelTrainingError
from core.logger import get_logger
from core.utils import clamp, git_commit_hash
from module_b_features.features import FEATURE_COLUMNS, HMMRegime
from module_b_features.labeler import LABEL_ORDER, LABEL_TO_INDEX, LabelClass, risk_tier_from_score
from module_b_features.processor import (
    ChronologicalSplit,
    InferencePayload,
    ProcessedDataset,
    SplitBoundaries,
    assign_split,
)
from module_c_ml import metrics as ml_metrics
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

_ARTIFACT_VERSION: Final[str] = "1.0.0"

#: Hyperparameter fields recorded verbatim in every artifact's metadata, so a
#: training run can be reproduced from the sidecar JSON alone.
_HYPERPARAMETER_FIELDS: Final[tuple[str, ...]] = (
    "booster",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "num_leaves",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "random_state",
    "train_months",
    "validation_months",
    "test_months",
    "purge_bars",
    "early_stopping_rounds",
)

# Hard sanity rails applied to every exit geometry, trained or heuristic.
# Mirrored in module_b_features/labeler.py so the training targets are
# clamped to the same range these rails allow at inference time - keep the
# two sets of constants in sync if either changes.
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


def _hyperparameter_snapshot(config: MLSettings) -> dict[str, Any]:
    """The exact booster hyperparameters used for this run, for reproducibility."""
    return {field: getattr(config, field) for field in _HYPERPARAMETER_FIELDS if hasattr(config, field)}


def focal_loss_binary(
    y_true: np.ndarray, y_pred_raw: np.ndarray, gamma: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Experimental LightGBM custom objective for the gate stage. Down-weights
    the easy, confidently-correct majority region and up-weights the ambiguous
    near-0.5 region the gate gets wrong most, where the standard log-loss
    objective currently spends most of its gradient on already-easy rows.

    NOT validated against production log-loss - needs a real gamma sweep and
    A/B comparison against the default objective before trusting it, and may
    destabilize LightGBM's early stopping (which expects a smoothly
    decreasing eval metric) if gamma is too aggressive. Disabled by default;
    enable via MLSettings.use_focal_loss_for_gate for experimentation only.
    """
    p = 1.0 / (1.0 + np.exp(-y_pred_raw))
    grad = (p - y_true) * ((1 - p) ** gamma * y_true + p ** gamma * (1 - y_true)) * gamma
    hess = p * (1 - p) * gamma  # simplified - not a rigorous 2nd derivative, flagged as experimental
    return grad, hess


class _SigmoidScoreClassifier:
    """Wraps a LightGBM estimator fit with a raw-margin custom objective.

    LightGBM's own ``predict_proba`` cannot invert an arbitrary custom
    objective back into a calibrated probability - with one set, it emits a
    warning ("Cannot compute class probabilities... Returning raw scores
    instead") and hands back a 1-D array of raw margins, which breaks every
    downstream consumer expecting an ``(n, 2)`` probability matrix. This
    manually applies the sigmoid :func:`focal_loss_binary` itself assumes
    when computing gradients, using LightGBM's own ``raw_score=True`` predict
    path (stable regardless of objective). Every other attribute delegates
    straight through to the wrapped estimator.
    """

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator

    def predict_proba(self, x: Any) -> np.ndarray:
        raw: np.ndarray = np.asarray(self._estimator.predict(x, raw_score=True), dtype=np.float64)
        positive: np.ndarray = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - positive, positive])

    def __getattr__(self, name: str) -> Any:
        # Guard against infinite recursion when `_estimator` itself is not
        # yet set (e.g. mid-unpickling, before `__init__`/state-restore runs).
        if name == "_estimator":
            raise AttributeError(name)
        return getattr(self._estimator, name)


class _FocalLossGateObjective:
    """A picklable, 2-argument ``(y_true, y_pred_raw)`` callable binding ``gamma``.

    Two independent constraints rule out the obvious alternatives:

    * LightGBM's sklearn wrapper decides whether to pass sample weights into
      a custom objective by counting ``inspect.signature(objective).
      parameters`` - a ``functools.partial`` that binds ``gamma`` as a
      keyword still reports 3 parameters (``y_true``, ``y_pred_raw``,
      ``gamma``) even though one is pre-bound, so LightGBM tries to call it
      with 3 positional arguments (labels, preds, sample_weight) and
      collides with the already-bound ``gamma`` keyword. A callable class
      instance's ``__call__`` reports the correct 2 parameters instead.
    * A local closure is not picklable, which would break the joblib
      artifact save the moment this experimental flag is enabled (the fitted
      LightGBM estimator keeps its ``objective`` constructor argument as an
      instance attribute). A class defined at module scope pickles fine.

    Sample weights (e.g. recency weighting) are consequently not applied to
    this objective's gradients while the experimental flag is on - a known,
    documented limitation of this not-yet-validated feature.
    """

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def __call__(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return focal_loss_binary(y_true, y_pred_raw, gamma=self.gamma)


def _temporal_half_split(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split an already-temporally-ordered block into an earlier and later half.

    Used to carve a calibration-fit slice (earlier) and a calibration-eval
    slice (later) out of the validation block, so calibration is fit and
    scored on genuinely non-overlapping, temporally ordered data rather than
    reusing the same rows for both - the same discipline as the primary
    train/validation split, one level down.
    """
    midpoint: int = len(features) // 2
    return (
        features.iloc[:midpoint],
        target.iloc[:midpoint],
        features.iloc[midpoint:],
        target.iloc[midpoint:],
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
        """Persist the fitted estimator plus its feature contract.

        Alongside the joblib artifact (the only thing inference needs), a
        human- and machine-readable ``{name}.metrics.json`` sidecar is also
        written with the full training metadata - hyperparameters, git
        commit, feature list, dataset size, class distribution and every
        computed metric.  The joblib blob is opaque to a diagnostic report
        generator; the JSON sidecar is what makes a run's results readable
        without unpickling an estimator.
        """
        if self._model is None:
            raise ModelNotLoadedError(f"{self.name}: nothing to save", head=self.name)

        destination: Path = path or self.artifact_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "artifact_version": _ARTIFACT_VERSION,
            "head": self.name,
            "model": self._model,
            "feature_columns": list(self._feature_columns),
            "metadata": self._metadata,
        }
        payload.update(self._extra_artifact_state())
        joblib.dump(payload, destination, compress=3)
        sidecar: Path = destination.with_suffix(".metrics.json")
        try:
            sidecar.write_text(
                json.dumps(
                    {
                        "artifact_version": _ARTIFACT_VERSION,
                        "head": self.name,
                        "feature_columns": list(self._feature_columns),
                        **self._metadata,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except OSError as error:  # pragma: no cover - disk full / permissions
            _LOGGER.error("Could not write metrics sidecar for %s: %s", self.name, error)
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
        self._restore_extra_artifact_state(payload)
        _LOGGER.info(
            "Loaded %s model (trained_at=%s, rows=%s)",
            self.name,
            self._metadata.get("trained_at", "?"),
            self._metadata.get("rows", "?"),
        )
        return True

    def _extra_artifact_state(self) -> dict[str, Any]:
        """Extra per-head state to persist in the joblib artifact.

        Override in a head that owns state beyond ``self._model`` /
        ``self._metadata`` (e.g. :class:`DirectionModel`'s joint-probability
        calibrators) that must survive a save/load round-trip.
        """
        return {}

    def _restore_extra_artifact_state(self, payload: dict[str, Any]) -> None:
        """Restore state written by :meth:`_extra_artifact_state`. No-op by default."""

    # ------------------------------------------------------------------
    # Estimator factories
    # ------------------------------------------------------------------
    def _n_jobs(self) -> int:
        """Thread budget for the booster (kept small: we run many in parallel)."""
        return max(1, self._config.inference_workers)

    def _make_classifier(self, num_class: int) -> Any:
        """Build an untrained gradient-boosted classifier."""
        config: MLSettings = self._config
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
                random_state=config.random_state,
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
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=self._n_jobs(),
            verbose=-1,
        )

    def _make_regressor(self, *, robust: bool = False) -> Any:
        """Build an untrained gradient-boosted regressor.

        Args:
            robust: When ``True``, fit an L1 (least-absolute-deviation)
                objective instead of the default L2 one.  L2 lets a handful of
                extreme-outlier rows dominate the loss and drag every
                prediction toward them; L1 is the standard fix when the
                target is heavy-tailed (as the exit-geometry percentages are)
                and matches the ``l1`` eval metric already used to early-stop
                these heads.
        """
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
                objective="reg:absoluteerror" if robust else "reg:squarederror",
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
            objective="regression_l1" if robust else "regression",
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
        """Fit with early stopping when a non-empty validation block exists.

        ``sample_weight`` (when given) applies only to the training rows, not
        the early-stopping validation block - early stopping should keep
        judging genuine, uniformly-weighted held-out performance rather than
        a recency-tilted one.
        """
        rounds: int = self._config.early_stopping_rounds
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if x_validation.empty or rounds <= 0:
            estimator.fit(x_train, y_train, **fit_kwargs)
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
                    eval_X=x_validation,
                    eval_y=y_validation,
                    eval_metric=eval_metric,
                    callbacks=callbacks,
                    **fit_kwargs,
                )
            else:
                estimator.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_validation, y_validation)],
                    eval_metric=eval_metric,
                    callbacks=callbacks,
                    **fit_kwargs,
                )
            return estimator

        estimator.set_params(early_stopping_rounds=rounds, eval_metric=eval_metric)
        estimator.fit(
            x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False, **fit_kwargs
        )
        return estimator

    # ------------------------------------------------------------------
    # Chronological train/validation/test split
    # ------------------------------------------------------------------
    def _split_boundaries(self, dataset: ProcessedDataset) -> SplitBoundaries | None:
        """This head's train/validation/test cut points for ``dataset``.

        Always computed from the *full* dataset's timestamp range (never a
        filtered subset) so every head - even one that later restricts
        itself to a row subset, like :class:`ExitModel`/:class:`RiskModel` -
        agrees on exactly the same calendar boundaries. ``test`` is the
        final backtest window: :meth:`train` must never fit, early-stop,
        calibrate or threshold-tune against it.
        """
        return dataset.split_boundaries(
            train_months=self._config.train_months,
            validation_months=self._config.validation_months,
            test_months=self._config.test_months,
            purge_bars=self._config.purge_bars,
            timeframe_ms=self._settings.data.timeframe_ms,
        )

    def _chronological_split(self, dataset: ProcessedDataset) -> ChronologicalSplit:
        """Strict train/validation/test row positions over the full dataset."""
        return dataset.chronological_split(self._split_boundaries(dataset))

    @staticmethod
    def _split_period_metadata(split: ChronologicalSplit) -> dict[str, Any]:
        """Human-readable train/validation/test date ranges for the metrics sidecar.

        ``test`` is recorded here purely as a date range / row count for
        audit purposes - this head's ``train()`` never reads ``test_index``
        for fitting, early stopping, calibration or threshold selection.
        """

        def _period(start_ms: int | None, end_ms: int | None, rows: int) -> dict[str, Any]:
            return {
                "start": datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc).isoformat()
                if start_ms is not None
                else None,
                "end": datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc).isoformat()
                if end_ms is not None
                else None,
                "rows": rows,
            }

        return {
            "train": _period(split.train_start_ms, split.train_end_ms, int(len(split.train_index))),
            "validation": _period(
                split.validation_start_ms, split.validation_end_ms, int(len(split.validation_index))
            ),
            "test": _period(split.test_start_ms, split.test_end_ms, int(len(split.test_index))),
            "embargo_ms": split.boundaries.embargo_ms,
            "scaled_down_from_nominal_months": split.boundaries.scaled_down,
        }

    @classmethod
    def _filtered_split_period_metadata(
        cls,
        timestamps: np.ndarray,
        train_index: np.ndarray,
        validation_index: np.ndarray,
        test_index: np.ndarray,
        boundaries: SplitBoundaries | None,
    ) -> dict[str, Any]:
        """:meth:`_split_period_metadata` for a head (Exit, Risk) that first
        filters ``dataset`` down to a row subset before splitting it.

        ``boundaries`` still comes from the full, unfiltered dataset (see
        :meth:`_split_boundaries`), so the reported dates line up with every
        other head's even though the row counts here only cover this head's
        own filtered subset (e.g. only rows with a directional trade).
        """
        if boundaries is None:
            empty: np.ndarray = np.array([], dtype=np.int64)
            split = ChronologicalSplit(empty, empty, empty, None, None, None, None, None, None, None)  # type: ignore[arg-type]
            return cls._split_period_metadata(split)

        def _bounds(index: np.ndarray) -> tuple[int | None, int | None]:
            if index.size == 0:
                return None, None
            subset: np.ndarray = timestamps[index]
            return int(subset.min()), int(subset.max())

        train_start, train_end = _bounds(train_index)
        validation_start, validation_end = _bounds(validation_index)
        test_start, test_end = _bounds(test_index)
        split = ChronologicalSplit(
            train_index=train_index,
            validation_index=validation_index,
            test_index=test_index,
            train_start_ms=train_start,
            train_end_ms=train_end,
            validation_start_ms=validation_start,
            validation_end_ms=validation_end,
            test_start_ms=test_start,
            test_end_ms=test_end,
            boundaries=boundaries,
        )
        return cls._split_period_metadata(split)

    # ------------------------------------------------------------------
    # Sample weighting
    # ------------------------------------------------------------------
    def _recency_weights(self, timestamps: np.ndarray) -> np.ndarray | None:
        """Exponential time-decay training weights: recent bars vote louder.

        Crypto regimes drift - a candle from a year ago is not as informative
        about tomorrow as one from yesterday.  Weight halves every
        ``recency_half_life_days`` days behind the most recent row in the
        slice being fit.  Returns ``None`` (uniform weighting) when the
        feature is disabled via config or no timestamps are available, so
        callers can pass the result straight through as an optional
        ``sample_weight``.
        """
        half_life_days: float = self._config.recency_half_life_days
        if half_life_days <= 0.0 or timestamps.size == 0:
            return None
        age_days: np.ndarray = (
            timestamps.astype(np.float64).max() - timestamps.astype(np.float64)
        ) / 86_400_000.0
        return np.power(0.5, age_days / half_life_days)

    @staticmethod
    def _timestamps_for(dataset: ProcessedDataset, index: np.ndarray) -> np.ndarray:
        """Row timestamps aligned to a positional index into ``dataset.features``.

        ``dataset.metadata`` and ``dataset.features`` are built from the same
        reset-index frame in :meth:`DatasetProcessor._to_dataset`, so a
        positional index into one is a positional index into the other.
        """
        if "timestamp" not in dataset.metadata.columns:
            return np.array([], dtype=np.float64)
        return dataset.metadata["timestamp"].to_numpy(dtype=np.float64)[index]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _fit_production_calibrator(
        self,
        estimator: Any,
        calibration_features: pd.DataFrame,
        calibration_target: pd.Series,
        *,
        min_rows: int = 200,
    ) -> Any | None:
        """Wrap a fitted estimator with isotonic calibration for live inference.

        ``module_c_ml.metrics.calibrate_classifier`` already measures whether
        isotonic calibration improves log loss, on a temporal half-split of
        the validation block held out purely for that honest before/after
        comparison.  This is the separate, production-facing step: fit the
        calibrator that will actually be used for inference, on the *entire*
        validation block (every held-out row available, not half of it),
        since a shipped artifact should not throw away data the diagnostic
        report doesn't need.

        Returns ``None`` - never raises - when there is not enough data or
        the fit fails; the caller then keeps using the raw estimator.
        Calibration is always a refinement, never a requirement.
        """
        if len(calibration_features) < min_rows:
            return None

        from sklearn.calibration import CalibratedClassifierCV

        try:
            try:
                from sklearn.frozen import FrozenEstimator

                calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="isotonic")
            except ImportError:  # pragma: no cover - exercised only on sklearn < 1.6
                calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv="prefit")
            calibrated.fit(calibration_features, calibration_target)
        except Exception as error:  # noqa: BLE001 - calibration is best-effort, never fatal
            _LOGGER.warning(
                "%s: production calibration fit failed, using the raw estimator: %s",
                self.name,
                error,
            )
            return None
        return calibrated

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

    Predicts the probability distribution over the three label classes
    (``LONG_SUCCESS`` / ``SHORT_SUCCESS`` / ``NO_TRADE_OR_FAIL``) produced by
    :class:`~module_b_features.labeler.TradeLabeler`.  Risk tiering is
    deliberately *not* fused into this target - see the labeler module
    docstring - so the schema's LONG / SHORT / NO_TRADE aggregation is just
    this model's raw output.
    """

    name = "direction_model"

    #: Minimum trade-labelled training rows required to fit stage 2
    #: (long-vs-short). Below this a single softmax split is too noisy to
    #: trust, so stage 2 is skipped and long/short are left at 50/50.
    _MIN_DIRECTION_TRAIN_ROWS: Final[int] = 50

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        #: One isotonic regressor per class (LABEL_ORDER order), calibrating
        #: the *joint* probability directly. ``None`` until training finds it
        #: measurably improves log loss - see :meth:`_calibrate_cascade`.
        self._joint_calibrators: list[Any] | None = None

    def _extra_artifact_state(self) -> dict[str, Any]:
        return {"joint_calibrators": self._joint_calibrators}

    def _restore_extra_artifact_state(self, payload: dict[str, Any]) -> None:
        self._joint_calibrators = payload.get("joint_calibrators")

    def train(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Fit the two-stage cascade on the pooled, purged dataset.

        Stage 1 (the "gate") answers a binary question: is this bar a trade
        at all, or NO_TRADE?  Stage 2 answers a second, independent binary
        question - given that it is a trade, is it LONG or SHORT? - trained
        only on the rows stage 1's ground truth calls a trade.  A single
        3-way softmax forces one decision boundary to serve both questions at
        once, even though they lean on different signal (whether-to-trade
        skews toward volatility/regime features, long-vs-short toward
        directional/momentum ones); splitting them is the same rationale that
        already moved risk tiering out of this label (see the labeler module
        docstring).
        """
        if dataset.is_empty:
            raise ModelTrainingError("direction model received an empty dataset")

        classes: list[str] = sorted(set(dataset.direction_target.unique()))
        if len(classes) < 2:
            raise ModelTrainingError("direction model needs >= 2 classes", classes=classes)

        encoded: pd.Series = dataset.direction_target.map(LABEL_TO_INDEX)
        if encoded.isna().any():
            raise ModelTrainingError("direction labels contain unknown classes")

        split: ChronologicalSplit = self._chronological_split(dataset)
        train_index, validation_index = split.train_index, split.validation_index
        features: pd.DataFrame = dataset.features
        sample_weight: np.ndarray | None = self._recency_weights(
            self._timestamps_for(dataset, train_index)
        )

        gate_estimator, direction_estimator = self._fit_cascade(
            features, encoded, train_index, validation_index, sample_weight, early_stopping=True
        )

        self._model = {"gate": gate_estimator, "direction": direction_estimator}
        self._feature_columns = dataset.feature_columns

        validation_features: pd.DataFrame = features.iloc[validation_index]
        validation_target: pd.Series = encoded.iloc[validation_index].astype(int)
        full_metrics: dict[str, Any] = {}
        importance: dict[str, Any] = {"status": "NOT_AVAILABLE", "reason": "no validation rows"}
        calibration: dict[str, Any] = {"status": "NOT_AVAILABLE", "reason": "no validation rows"}
        per_symbol: dict[str, Any] = {}
        production_calibration: dict[str, str] = {"gate": "raw", "direction": "raw"}
        gate_sweep: list[dict[str, Any]] = []
        recommended_gate_threshold: float = self._settings.decision.min_gate_confidence
        direction_sweep: list[dict[str, Any]] = []
        recommended_direction_threshold: float = self._settings.decision.min_direction_given_trade_confidence

        if not validation_features.empty:
            probabilities: np.ndarray = self._combined_probabilities(
                gate_estimator, direction_estimator, validation_features
            )
            full_metrics = ml_metrics.direction_metrics(validation_target, probabilities, LABEL_ORDER)
            importance = {
                "gate": ml_metrics.feature_importance(gate_estimator, dataset.feature_columns),
                "direction": (
                    ml_metrics.feature_importance(direction_estimator, dataset.feature_columns)
                    if direction_estimator is not None
                    else {"status": "NOT_AVAILABLE", "reason": "not enough trade rows to fit stage 2"}
                ),
            }
            if "symbol" in dataset.metadata.columns:
                symbols_validation: pd.Series = dataset.metadata["symbol"].iloc[validation_index]
                per_symbol = ml_metrics.per_symbol_direction_accuracy(
                    validation_target, probabilities.argmax(axis=1), symbols_validation
                )

            # Auto-tune recommended gate/direction thresholds from the model's
            # own validation sweep, the same way EntryModel already does for
            # its decision threshold - reported for the operator to review,
            # never auto-applied to the live DecisionSettings.
            no_trade_index: int = LABEL_TO_INDEX[LabelClass.NO_TRADE_OR_FAIL.value]
            long_index: int = LABEL_TO_INDEX[LabelClass.LONG_SUCCESS.value]
            is_trade_validation: pd.Series = (validation_target != no_trade_index).astype(int)
            gate_probabilities: np.ndarray = np.asarray(
                gate_estimator.predict_proba(validation_features)
            )[:, -1]
            gate_sweep = ml_metrics.gate_threshold_sweep(is_trade_validation, gate_probabilities)
            recommended_gate_threshold = EntryModel._select_recommended_threshold(
                gate_sweep, self._settings.decision.min_gate_confidence
            )

            trade_mask_validation: pd.Series = is_trade_validation == 1
            direction_validation_features: pd.DataFrame = validation_features[trade_mask_validation]
            if direction_estimator is not None and not direction_validation_features.empty:
                long_given_trade_validation: np.ndarray = np.asarray(
                    direction_estimator.predict_proba(direction_validation_features)
                )[:, -1]
                is_long_validation: pd.Series = (
                    validation_target[trade_mask_validation] == long_index
                ).astype(int)
                direction_sweep = ml_metrics.direction_threshold_sweep(
                    is_long_validation, long_given_trade_validation
                )
                recommended_direction_threshold = EntryModel._select_recommended_threshold(
                    direction_sweep, self._settings.decision.min_direction_given_trade_confidence
                )

            calibration, self._model, production_calibration = self._calibrate_cascade(
                gate_estimator, direction_estimator, validation_features, validation_target
            )

        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit_hash(),
            "rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "classes": list(LABEL_ORDER),
            "distribution": dataset.class_distribution(),
            "hyperparameters": _hyperparameter_snapshot(self._config),
            "metrics": full_metrics,
            "feature_importance": importance,
            "calibration": calibration,
            "production_calibration": production_calibration,
            "per_symbol": per_symbol,
            "architecture": "two_stage_cascade",
            "gate_threshold_sweep": gate_sweep,
            "recommended_gate_threshold": recommended_gate_threshold,
            "direction_threshold_sweep": direction_sweep,
            "recommended_direction_threshold": recommended_direction_threshold,
            "split": self._split_period_metadata(split),
        }
        headline: dict[str, Any] = {
            key: full_metrics[key]
            for key in ("accuracy", "balanced_accuracy", "log_loss")
            if key in full_metrics
        }
        _LOGGER.info("Direction model trained: %s", headline)
        return full_metrics

    def predict(self, features: pd.DataFrame) -> DirectionPrediction:
        """Score a single feature row (stateless, thread-safe).

        Falls back to a transparent trend/regime heuristic when no artifact is
        loaded, stamping the result as ``HEURISTIC``.
        """
        if self._model is None:
            return self._heuristic(features)

        aligned: pd.DataFrame = self._align(features)
        gate_estimator: Any = self._model["gate"]
        direction_estimator: Any | None = self._model.get("direction")

        trade_probability: float = float(
            np.asarray(gate_estimator.predict_proba(aligned), dtype=np.float64)[0, -1]
        )
        long_given_trade: float = (
            float(np.asarray(direction_estimator.predict_proba(aligned), dtype=np.float64)[0, -1])
            if direction_estimator is not None
            else 0.5
        )

        long_probability: float = trade_probability * long_given_trade
        short_probability: float = trade_probability * (1.0 - long_given_trade)
        no_trade_probability: float = max(0.0, 1.0 - long_probability - short_probability)

        if self._joint_calibrators is not None:
            # Adjusts only the *reported* joint distribution (R3's consistency
            # check, audit logging) - trade_probability/direction_given_trade_
            # probability below stay the raw per-stage values R1a/R1b gate on.
            raw_matrix: np.ndarray = np.array([[long_probability, short_probability, no_trade_probability]])
            calibrated_matrix: np.ndarray = self._apply_joint_calibrators(self._joint_calibrators, raw_matrix)
            long_probability, short_probability, no_trade_probability = (
                float(calibrated_matrix[0, LABEL_TO_INDEX[LabelClass.LONG_SUCCESS.value]]),
                float(calibrated_matrix[0, LABEL_TO_INDEX[LabelClass.SHORT_SUCCESS.value]]),
                float(calibrated_matrix[0, LABEL_TO_INDEX[LabelClass.NO_TRADE_OR_FAIL.value]]),
            )

        return DirectionPrediction(
            probabilities={
                LabelClass.LONG_SUCCESS.value: long_probability,
                LabelClass.SHORT_SUCCESS.value: short_probability,
                LabelClass.NO_TRADE_OR_FAIL.value: no_trade_probability,
            },
            source=ModelSource.TRAINED,
            trade_probability=trade_probability,
            direction_given_trade_probability=long_given_trade,
        )

    # ------------------------------------------------------------------
    # Two-stage cascade internals
    # ------------------------------------------------------------------
    def _fit_cascade(
        self,
        features: pd.DataFrame,
        encoded: pd.Series,
        train_positions: np.ndarray,
        validation_positions: np.ndarray,
        sample_weight: np.ndarray | None,
        *,
        early_stopping: bool,
    ) -> tuple[Any, Any | None]:
        """Fit the trade gate, then long-vs-short on the gate's true-trade rows.

        ``sample_weight`` (if given) is aligned with ``train_positions`` and
        is sliced down to the trade subset for stage 2. ``early_stopping``
        selects between the production path (uses the validation block for
        early stopping, like every other head) and the fast path used by
        :meth:`walk_forward`, where fitting many folds cheaply matters more
        than the last bit of accuracy an inner validation split would buy.
        """
        no_trade_index: int = LABEL_TO_INDEX[LabelClass.NO_TRADE_OR_FAIL.value]
        long_index: int = LABEL_TO_INDEX[LabelClass.LONG_SUCCESS.value]
        is_trade: pd.Series = (encoded != no_trade_index).astype(int)
        is_long: pd.Series = (encoded == long_index).astype(int)

        gate_estimator: Any = self._make_classifier(num_class=2)
        using_focal_loss: bool = (
            self._config.use_focal_loss_for_gate and self._config.booster == "lightgbm"
        )
        if self._config.use_focal_loss_for_gate:
            if using_focal_loss:
                gate_estimator.set_params(
                    objective=_FocalLossGateObjective(self._config.focal_loss_gamma)
                )
            else:  # pragma: no cover - exercised only with booster="xgboost"
                _LOGGER.warning(
                    "use_focal_loss_for_gate is only implemented for the lightgbm booster; "
                    "ignoring it for booster=%s",
                    self._config.booster,
                )
        if early_stopping:
            gate_estimator = self._fit_estimator(
                gate_estimator,
                features.iloc[train_positions],
                is_trade.iloc[train_positions],
                features.iloc[validation_positions],
                is_trade.iloc[validation_positions],
                eval_metric="binary_logloss",
                sample_weight=sample_weight,
            )
        else:
            fit_kwargs: dict[str, Any] = {} if sample_weight is None else {"sample_weight": sample_weight}
            gate_estimator.fit(features.iloc[train_positions], is_trade.iloc[train_positions], **fit_kwargs)
        if using_focal_loss:
            # LightGBM's own predict_proba cannot invert a custom objective -
            # see _SigmoidScoreClassifier for why this wrap is needed.
            gate_estimator = _SigmoidScoreClassifier(gate_estimator)

        trade_mask: np.ndarray = is_trade.to_numpy()[train_positions] == 1
        train_trade_positions: np.ndarray = train_positions[trade_mask]
        validation_trade_positions: np.ndarray = (
            validation_positions[is_trade.to_numpy()[validation_positions] == 1]
            if len(validation_positions)
            else validation_positions
        )

        direction_estimator: Any | None = None
        if (
            len(train_trade_positions) >= self._MIN_DIRECTION_TRAIN_ROWS
            and is_long.iloc[train_trade_positions].nunique() >= 2
        ):
            direction_sample_weight: np.ndarray | None = (
                sample_weight[trade_mask] if sample_weight is not None else None
            )
            direction_estimator = self._make_classifier(num_class=2)
            if early_stopping:
                direction_estimator = self._fit_estimator(
                    direction_estimator,
                    features.iloc[train_trade_positions],
                    is_long.iloc[train_trade_positions],
                    features.iloc[validation_trade_positions],
                    is_long.iloc[validation_trade_positions],
                    eval_metric="binary_logloss",
                    sample_weight=direction_sample_weight,
                )
            else:
                fit_kwargs = (
                    {} if direction_sample_weight is None else {"sample_weight": direction_sample_weight}
                )
                direction_estimator.fit(
                    features.iloc[train_trade_positions], is_long.iloc[train_trade_positions], **fit_kwargs
                )
        return gate_estimator, direction_estimator

    @staticmethod
    def _combined_probabilities(
        gate_estimator: Any, direction_estimator: Any | None, features: pd.DataFrame
    ) -> np.ndarray:
        """Recombine the two-stage cascade into a ``(n, len(LABEL_ORDER))`` matrix."""
        trade_probability: np.ndarray = np.asarray(gate_estimator.predict_proba(features))[:, -1]
        long_given_trade: np.ndarray = (
            np.asarray(direction_estimator.predict_proba(features))[:, -1]
            if direction_estimator is not None
            else np.full(len(features), 0.5)
        )

        matrix: np.ndarray = np.zeros((len(features), len(LABEL_ORDER)))
        matrix[:, LABEL_TO_INDEX[LabelClass.LONG_SUCCESS.value]] = trade_probability * long_given_trade
        matrix[:, LABEL_TO_INDEX[LabelClass.SHORT_SUCCESS.value]] = trade_probability * (
            1.0 - long_given_trade
        )
        matrix[:, LABEL_TO_INDEX[LabelClass.NO_TRADE_OR_FAIL.value]] = 1.0 - trade_probability
        return matrix

    def _calibrate_cascade(
        self,
        gate_estimator: Any,
        direction_estimator: Any | None,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
        """Measure and (when it helps) wire in isotonic calibration per stage.

        Each binary stage is calibrated independently - there is no single
        multiclass estimator to calibrate as a whole once the cascade is
        split in two. Returns ``(calibration_report, model_dict,
        production_calibration)`` so the caller can drop all three straight
        into ``self._model`` / ``self._metadata``.
        """
        no_trade_index: int = LABEL_TO_INDEX[LabelClass.NO_TRADE_OR_FAIL.value]
        long_index: int = LABEL_TO_INDEX[LabelClass.LONG_SUCCESS.value]
        is_trade_validation: pd.Series = (validation_target != no_trade_index).astype(int)

        model: dict[str, Any] = {"gate": gate_estimator, "direction": direction_estimator}
        production_calibration: dict[str, str] = {"gate": "raw", "direction": "raw"}

        calib_x, calib_y, eval_x, eval_y = _temporal_half_split(validation_features, is_trade_validation)
        gate_calibration: dict[str, Any] = ml_metrics.calibrate_classifier(
            gate_estimator, calib_x, calib_y, eval_x, eval_y, n_classes=2
        )
        if gate_calibration.get("status") == "AVAILABLE" and gate_calibration.get("improved"):
            calibrated_gate: Any = self._fit_production_calibrator(
                gate_estimator, validation_features, is_trade_validation
            )
            if calibrated_gate is not None:
                model["gate"] = calibrated_gate
                production_calibration["gate"] = "isotonic"

        direction_calibration: dict[str, Any] = {
            "status": "NOT_AVAILABLE",
            "reason": "stage 2 was not fitted (not enough trade rows)",
        }
        if direction_estimator is not None:
            trade_mask: pd.Series = is_trade_validation == 1
            direction_features: pd.DataFrame = validation_features[trade_mask]
            direction_target: pd.Series = (validation_target[trade_mask] == long_index).astype(int)
            if len(direction_features) >= 100:
                d_calib_x, d_calib_y, d_eval_x, d_eval_y = _temporal_half_split(
                    direction_features, direction_target
                )
                direction_calibration = ml_metrics.calibrate_classifier(
                    direction_estimator, d_calib_x, d_calib_y, d_eval_x, d_eval_y, n_classes=2
                )
                if direction_calibration.get("status") == "AVAILABLE" and direction_calibration.get(
                    "improved"
                ):
                    calibrated_direction: Any = self._fit_production_calibrator(
                        direction_estimator, direction_features, direction_target
                    )
                    if calibrated_direction is not None:
                        model["direction"] = calibrated_direction
                        production_calibration["direction"] = "isotonic"
            else:
                direction_calibration = {
                    "status": "NOT_AVAILABLE",
                    "reason": "not enough trade rows in validation for a calibration split",
                }

        joint_calibration: dict[str, Any] = self._evaluate_joint_calibration(
            gate_estimator, direction_estimator, validation_features, validation_target
        )
        self._joint_calibrators = (
            self._calibrate_joint_probabilities(
                self._combined_probabilities(gate_estimator, direction_estimator, validation_features),
                validation_target.to_numpy(),
                len(LABEL_ORDER),
            )
            if joint_calibration.get("status") == "AVAILABLE" and joint_calibration.get("improved")
            else None
        )

        calibration: dict[str, Any] = {
            "gate": gate_calibration,
            "direction": direction_calibration,
            "joint": joint_calibration,
        }
        return calibration, model, production_calibration

    def _evaluate_joint_calibration(
        self,
        gate_estimator: Any,
        direction_estimator: Any | None,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
        *,
        min_calibration_rows: int = 50,
        min_eval_rows: int = 20,
    ) -> dict[str, Any]:
        """Honestly measure whether calibrating the joint probability helps.

        Mirrors :func:`module_c_ml.metrics.calibrate_classifier`'s temporal
        half-split discipline (fit on the earlier half, score on the later
        one) but operates on the *joint* long/short/no_trade matrix the
        two-stage cascade produces, since there is no single sklearn
        estimator here to hand to that function.
        """
        n_classes: int = len(LABEL_ORDER)
        midpoint: int = len(validation_features) // 2
        calib_probs: np.ndarray = self._combined_probabilities(
            gate_estimator, direction_estimator, validation_features.iloc[:midpoint]
        )
        eval_probs: np.ndarray = self._combined_probabilities(
            gate_estimator, direction_estimator, validation_features.iloc[midpoint:]
        )
        calib_target: np.ndarray = validation_target.iloc[:midpoint].to_numpy()
        eval_target: np.ndarray = validation_target.iloc[midpoint:].to_numpy()

        if len(calib_probs) < min_calibration_rows or len(eval_probs) < min_eval_rows:
            return {
                "status": "NOT_AVAILABLE",
                "reason": (
                    f"insufficient rows for a temporally safe calibration split "
                    f"(calibration={len(calib_probs)}, eval={len(eval_probs)})"
                ),
            }

        from sklearn.metrics import log_loss

        labels: list[int] = list(range(n_classes))
        one_hot_eval: np.ndarray = np.eye(n_classes)[eval_target]
        raw_brier: float = float(np.mean(np.sum((eval_probs - one_hot_eval) ** 2, axis=1)))
        try:
            raw_logloss: float = float(log_loss(eval_target, eval_probs, labels=labels))
        except ValueError:  # pragma: no cover - degenerate eval slice
            raw_logloss = float("nan")

        calibrators: list[Any] = self._calibrate_joint_probabilities(calib_probs, calib_target, n_classes)
        calibrated_eval_probs: np.ndarray = self._apply_joint_calibrators(calibrators, eval_probs)
        calibrated_brier: float = float(
            np.mean(np.sum((calibrated_eval_probs - one_hot_eval) ** 2, axis=1))
        )
        try:
            calibrated_logloss: float = float(log_loss(eval_target, calibrated_eval_probs, labels=labels))
        except ValueError:  # pragma: no cover - degenerate eval slice
            calibrated_logloss = float("nan")

        improved: bool = calibrated_logloss < raw_logloss
        return {
            "status": "AVAILABLE",
            "method": "isotonic_per_class",
            "calibration_rows": int(len(calib_probs)),
            "eval_rows": int(len(eval_probs)),
            "brier_score_raw": raw_brier,
            "brier_score_calibrated": calibrated_brier,
            "log_loss_raw": raw_logloss,
            "log_loss_calibrated": calibrated_logloss,
            "improved": bool(improved),
            "recommended_for_production": bool(improved),
            "note": (
                "Calibrates the joint long/short/no_trade probability directly, "
                "on top of whatever per-stage calibration already happened above - "
                "corrects residual miscalibration that the product of two "
                "independently-calibrated probabilities can still leave behind, "
                "which per-stage calibration alone cannot see. Applied only to "
                "the reported `probabilities` dict at inference (R3's consistency "
                "check, audit logging); trade_probability and "
                "direction_given_trade_probability, which R1a/R1b gate on, are "
                "never touched by this."
            ),
        }

    @staticmethod
    def _calibrate_joint_probabilities(
        calib_raw_probs: np.ndarray, calib_target: np.ndarray, n_classes: int
    ) -> list[Any]:
        """One isotonic regressor per class, fit directly on the joint
        long/short/no_trade probability vs. the true one-hot outcome - corrects
        residual miscalibration the product of two per-stage-calibrated
        probabilities leaves behind, which per-stage calibration alone cannot see.
        """
        from sklearn.isotonic import IsotonicRegression

        calibrators: list[Any] = []
        for class_index in range(n_classes):
            y_binary = (calib_target == class_index).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(calib_raw_probs[:, class_index], y_binary)
            calibrators.append(iso)
        return calibrators

    @staticmethod
    def _apply_joint_calibrators(calibrators: list[Any], raw_probs: np.ndarray) -> np.ndarray:
        """Map each class column through its isotonic calibrator and renormalise to 1."""
        calibrated: np.ndarray = np.column_stack(
            [
                calibrator.predict(raw_probs[:, class_index])
                for class_index, calibrator in enumerate(calibrators)
            ]
        )
        row_sums: np.ndarray = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        return calibrated / row_sums

    def walk_forward(self, dataset: ProcessedDataset, n_folds: int = 4) -> dict[str, Any]:
        """Expanding-window walk-forward evaluation across multiple rolling folds.

        :meth:`train` fits and scores one production cascade on a single
        temporal train/validation split - the ML diagnostic report's own
        recommendations flag that as the single biggest validation gap ("only
        a single split is currently performed"), since a lucky split can make
        a model look better than it generalises.  This fits ``n_folds``
        independent cascades (same two-stage architecture as production,
        minus early stopping and calibration - see :meth:`_fit_cascade`),
        each trained only on data strictly preceding its own validation block
        (with the same purge gap used in production), and reports per-fold
        plus aggregate accuracy so a wide spread across folds - not just the
        headline number - is visible.

        Every fold is a genuinely separate fit (no artifact is mutated or
        reused): this never touches ``self._model``.

        Restricted entirely to the train+validation region (everything
        strictly before the held-out test split's start): walk-forward is a
        development-time diagnostic, and the test split must stay unseen by
        every development decision, not just the final production fit - see
        the module's train/validation/test split discipline
        (``BaseModelHead._split_boundaries``).

        Returns ``{"status": "NOT_AVAILABLE", "reason": ...}`` when the
        dataset is too small to carve out ``n_folds`` honest folds, rather
        than fabricating a result from folds too small to mean anything.
        """
        min_rows_per_fold: int = 200
        boundaries: SplitBoundaries | None = self._split_boundaries(dataset)
        if boundaries is None:
            return {"status": "NOT_AVAILABLE", "reason": "no timestamps available"}

        all_timestamps: np.ndarray = self._timestamps_for(dataset, np.arange(len(dataset.features)))
        in_sample_positions: np.ndarray = np.nonzero(all_timestamps < boundaries.test_start_ms)[0]
        rows: int = in_sample_positions.size
        if rows < (n_folds + 1) * min_rows_per_fold:
            return {
                "status": "NOT_AVAILABLE",
                "reason": (
                    f"only {rows} train+validation rows available (test split excluded); "
                    f"walk-forward needs at least {(n_folds + 1) * min_rows_per_fold} for "
                    f"{n_folds} honest folds"
                ),
            }

        encoded_full: pd.Series = dataset.direction_target.map(LABEL_TO_INDEX)
        if encoded_full.isna().any():
            return {"status": "NOT_AVAILABLE", "reason": "direction labels contain unknown classes"}

        features: pd.DataFrame = dataset.features.iloc[in_sample_positions].reset_index(drop=True)
        encoded: pd.Series = encoded_full.iloc[in_sample_positions].reset_index(drop=True)
        timestamps: np.ndarray = all_timestamps[in_sample_positions]
        embargo_ms: int = boundaries.embargo_ms
        # n_folds+1 expanding blocks: block 0 is a seed reserved purely for
        # the first fold's training data (there is nothing before it to
        # validate against), and each of the remaining n_folds blocks is one
        # validation fold, trained on everything strictly before it.
        fold_boundaries: np.ndarray = np.linspace(0, rows, n_folds + 2, dtype=int)

        folds: list[dict[str, Any]] = []
        for fold_number in range(1, n_folds + 1):
            val_start, val_end = int(fold_boundaries[fold_number]), int(fold_boundaries[fold_number + 1])
            if val_start >= rows or val_end <= val_start:
                continue
            # Embargo is a *time* gap, not a row-count gap: the pooled
            # dataset interleaves every symbol at each timestamp, so
            # subtracting a fixed row count would purge far less real time
            # than `purge_bars` once more than one symbol is present.
            train_mask: np.ndarray = timestamps < (timestamps[val_start] - embargo_ms)
            train_positions: np.ndarray = np.nonzero(train_mask)[0]
            validation_positions: np.ndarray = np.arange(val_start, val_end)
            if train_positions.size < min_rows_per_fold or (val_end - val_start) < min_rows_per_fold // 2:
                continue
            if encoded.iloc[train_positions].nunique() < 2:
                continue

            sample_weight = self._recency_weights(timestamps[train_positions])
            gate_estimator, direction_estimator = self._fit_cascade(
                features, encoded, train_positions, validation_positions, sample_weight, early_stopping=False
            )

            val_target: pd.Series = encoded.iloc[validation_positions].astype(int)
            probabilities: np.ndarray = self._combined_probabilities(
                gate_estimator, direction_estimator, features.iloc[validation_positions]
            )
            fold_metrics: dict[str, Any] = ml_metrics.direction_metrics(
                val_target, probabilities, LABEL_ORDER
            )
            folds.append(
                {
                    "fold": fold_number,
                    "train_rows": int(train_positions.size),
                    "validation_rows": int(val_end - val_start),
                    "accuracy": fold_metrics.get("accuracy"),
                    "balanced_accuracy": fold_metrics.get("balanced_accuracy"),
                    "log_loss": fold_metrics.get("log_loss"),
                    "macro_f1": fold_metrics.get("macro_f1"),
                }
            )

        if not folds:
            return {"status": "NOT_AVAILABLE", "reason": "no fold had enough rows on both sides"}

        accuracies: list[float] = [f["accuracy"] for f in folds if f["accuracy"] is not None]
        balanced: list[float] = [f["balanced_accuracy"] for f in folds if f["balanced_accuracy"] is not None]
        return {
            "status": "AVAILABLE",
            "method": "expanding_window",
            "n_folds": len(folds),
            "folds": folds,
            "accuracy_mean": float(np.mean(accuracies)) if accuracies else None,
            "accuracy_std": float(np.std(accuracies)) if accuracies else None,
            "balanced_accuracy_mean": float(np.mean(balanced)) if balanced else None,
            "balanced_accuracy_std": float(np.std(balanced)) if balanced else None,
            "note": (
                "Each fold fits an independent two-stage cascade (not the "
                "production model) purely to measure how much accuracy varies "
                "across different time periods."
            ),
        }

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
            trade_probability=trade_mass,
            direction_given_trade_probability=directional,
        )


class EntryModel(BaseModelHead):
    """Model 2 - entry timing filter.

    Answers a narrow question: given that a direction has been chosen, is *this*
    candle close a clean entry, or should we wait for the next 5-minute bar?
    Trained on ``entry_quality``, which marks bars whose winning trade never gave
    back more than ``low_risk_mae_ratio`` of its stop.
    """

    name = "entry_model"

    def train(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Fit the binary entry-quality classifier."""
        if dataset.is_empty:
            raise ModelTrainingError("entry model received an empty dataset")

        target: pd.Series = dataset.entry_target.astype(int)
        if target.nunique() < 2:
            raise ModelTrainingError("entry target is degenerate (single class)")

        split: ChronologicalSplit = self._chronological_split(dataset)
        train_index, validation_index = split.train_index, split.validation_index
        features: pd.DataFrame = dataset.features
        estimator: Any = self._make_classifier(num_class=2)
        estimator = self._fit_estimator(
            estimator,
            features.iloc[train_index],
            target.iloc[train_index],
            features.iloc[validation_index],
            target.iloc[validation_index],
            eval_metric="binary_logloss",
            sample_weight=self._recency_weights(self._timestamps_for(dataset, train_index)),
        )

        self._model = estimator
        self._feature_columns = dataset.feature_columns

        full_metrics: dict[str, Any] = {}
        threshold_sweep: list[dict[str, Any]] = []
        importance: dict[str, Any] = {"status": "NOT_AVAILABLE", "reason": "no validation rows"}
        calibration: dict[str, Any] = {"status": "NOT_AVAILABLE", "reason": "no validation rows"}
        production_calibration: str = "raw"
        configured_floor: float = self._settings.decision.min_entry_probability
        cutoff: float = configured_floor
        if len(validation_index) > 0:
            validation_features: pd.DataFrame = features.iloc[validation_index]
            validation_target: pd.Series = target.iloc[validation_index]
            probabilities: np.ndarray = np.asarray(
                estimator.predict_proba(validation_features)
            )[:, 1]
            threshold_sweep = ml_metrics.entry_threshold_sweep(validation_target, probabilities)
            # Auto-tune the decision threshold from the model's own validation
            # sweep instead of trusting one hand-picked config constant - see
            # _select_recommended_threshold for why precision is weighted over
            # recall (a false-positive entry costs real capital; a missed
            # true positive only costs a smaller position count).
            cutoff = self._select_recommended_threshold(threshold_sweep, configured_floor)
            full_metrics = ml_metrics.entry_metrics(validation_target, probabilities, cutoff)
            importance = ml_metrics.feature_importance(estimator, dataset.feature_columns)
            calib_x, calib_y, eval_x, eval_y = _temporal_half_split(
                validation_features, validation_target
            )
            calibration = ml_metrics.calibrate_classifier(
                estimator, calib_x, calib_y, eval_x, eval_y, n_classes=2
            )
            if calibration.get("status") == "AVAILABLE" and calibration.get("improved"):
                calibrated_model: Any = self._fit_production_calibrator(
                    estimator, validation_features, validation_target
                )
                if calibrated_model is not None:
                    self._model = calibrated_model
                    production_calibration = "isotonic"

        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit_hash(),
            "rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "positive_rate": float(target.mean()),
            "decision_threshold": cutoff,
            "configured_floor_threshold": configured_floor,
            "hyperparameters": _hyperparameter_snapshot(self._config),
            "metrics": full_metrics,
            "threshold_sweep": threshold_sweep,
            "feature_importance": importance,
            "calibration": calibration,
            "production_calibration": production_calibration,
            "split": self._split_period_metadata(split),
        }
        headline: dict[str, Any] = {
            key: full_metrics[key] for key in ("precision", "recall", "roc_auc") if key in full_metrics
        }
        _LOGGER.info("Entry model trained: %s (threshold=%.2f)", headline, cutoff)
        return full_metrics

    @staticmethod
    def _select_recommended_threshold(sweep: list[dict[str, Any]], floor: float) -> float:
        """Pick the best threshold from the sweep by F-beta=0.5 (precision
        weighted 2x over recall), restricted to thresholds carrying enough
        signals to trust (``meets_min_sample_size``).

        A plain F1 pick tends to land on the loosest threshold in the sweep
        (highest recall), which is the wrong bias for an entry filter: a
        false-positive entry commits real capital, while a missed true
        positive only costs a smaller position count later. Falls back to
        ``floor`` (the configured default) when nothing in the sweep
        qualifies, so a sparse validation slice can never hand back a
        threshold nobody could act on.
        """
        beta_squared: float = 0.25  # beta = 0.5
        candidates: list[dict[str, Any]] = [row for row in sweep if row.get("meets_min_sample_size")]
        if not candidates:
            return floor

        def f_beta(row: dict[str, Any]) -> float:
            precision: float = float(row.get("precision", 0.0) or 0.0)
            recall: float = float(row.get("recall", 0.0) or 0.0)
            denominator: float = beta_squared * precision + recall
            if denominator <= 0.0:
                return 0.0
            return (1.0 + beta_squared) * precision * recall / denominator

        best: dict[str, Any] = max(candidates, key=f_beta)
        return float(best["threshold"])

    def predict(
        self,
        features: pd.DataFrame,
        action: TradeAction = TradeAction.NO_TRADE,
        threshold: float | None = None,
    ) -> EntryPrediction:
        """Decide whether to act on this candle or wait for the next one."""
        cutoff: float = (
            threshold
            if threshold is not None
            else self._metadata.get("decision_threshold", self._settings.decision.min_entry_probability)
        )
        if self._model is None:
            return self._heuristic(features, action, cutoff)

        aligned: pd.DataFrame = self._align(features)
        probability: float = float(
            np.asarray(self._model.predict_proba(aligned), dtype=np.float64)[0, -1]
        )
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

        * taker buy/sell flow pointing the same way as the intended trade,
        * a trending (low-FDI) structure rather than chop,
        * a volatility percentile that is not in the top decile of its own
          history (a proxy for "conditions are not adverse/wide").

        Substitution note: ``ob_imbalance`` and ``ob_spread_rank`` were
        removed from ``FEATURE_COLUMNS`` entirely (Binance has no historical
        order-book depth endpoint, so those columns could never be
        backfilled for training - see the commit that removed them). This
        fallback - only ever exercised when no trained booster is loaded -
        is redesigned around two features that still exist:
        ``taker_buy_sell_ratio`` (a log taker buy/sell volume ratio - a
        genuine, if Binance-retention-limited, order-flow-alignment proxy)
        replaces the order-book imbalance term, and ``atr_rank`` (realized
        volatility percentile) replaces the spread-rank term as the closest
        available proxy for "conditions are not adverse" absent any real
        spread metric. Neither substitute is a measured equivalent of the
        original - they are directionally reasonable stand-ins for a
        heuristic that is itself only a documented fallback, not the
        trained model's decision path.
        """
        row: pd.Series = features.iloc[0]
        order_flow: float = clamp(float(row.get("taker_buy_sell_ratio", 0.0) or 0.0), -1.0, 1.0)
        trending: float = float(row.get("fdi_trending", 0.0) or 0.0)
        volatility_rank: float = float(row.get("atr_rank", 0.5) or 0.5)

        directional_flow: float = order_flow if action is TradeAction.LONG else -order_flow
        score: float = 0.5
        score += clamp(directional_flow, -1.0, 1.0) * 0.15
        score += (trending - 0.5) * 0.20
        score -= clamp(volatility_rank - 0.5, -0.5, 0.5) * 0.20

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
        metadata_usable: pd.DataFrame = dataset.metadata[usable].reset_index(drop=True)

        # Boundaries come from the *full* dataset (not this filtered subset)
        # so Exit shares the exact same train/validation/test calendar cut
        # points as every other head - see BaseModelHead._split_boundaries.
        boundaries: SplitBoundaries | None = self._split_boundaries(dataset)
        usable_timestamps: np.ndarray = (
            metadata_usable["timestamp"].to_numpy(dtype=np.int64)
            if "timestamp" in metadata_usable.columns
            else np.array([], dtype=np.int64)
        )
        if boundaries is not None and usable_timestamps.size > 0:
            train_index, validation_index, test_index = assign_split(usable_timestamps, boundaries)
        else:
            train_index = np.arange(len(features), dtype=np.int64)
            validation_index = np.array([], dtype=np.int64)
            test_index = np.array([], dtype=np.int64)

        estimators: dict[str, Any] = {}
        full_metrics: dict[str, Any] = {}
        importance: dict[str, Any] = {}

        validation_features: pd.DataFrame = features.iloc[validation_index]
        baseline: dict[str, np.ndarray] = self._baseline_predictions(validation_features)
        sample_weight: np.ndarray | None = self._recency_weights(usable_timestamps[train_index])

        for column in self._TARGETS:
            estimator: Any = self._make_regressor(robust=True)
            estimator = self._fit_estimator(
                estimator,
                features.iloc[train_index],
                targets[column].iloc[train_index],
                validation_features,
                targets[column].iloc[validation_index],
                eval_metric="l1",
                sample_weight=sample_weight,
            )
            estimators[column] = estimator
            if len(validation_index) > 0:
                predictions: np.ndarray = estimator.predict(validation_features)
                validation_target: np.ndarray = targets[column].iloc[validation_index].to_numpy()
                target_metrics: dict[str, Any] = ml_metrics.regression_metrics(
                    validation_target, predictions
                )
                baseline_metrics: dict[str, Any] = ml_metrics.regression_metrics(
                    validation_target, baseline[column]
                )
                target_metrics["baseline_rule_based_mae"] = baseline_metrics.get("mae")
                target_metrics["beats_rule_based_baseline"] = bool(
                    target_metrics.get("mae", float("inf")) < baseline_metrics.get("mae", float("inf"))
                )
                full_metrics[column] = target_metrics
                importance[column] = ml_metrics.feature_importance(estimator, dataset.feature_columns)

        self._model = estimators
        self._feature_columns = dataset.feature_columns
        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit_hash(),
            "rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "hyperparameters": _hyperparameter_snapshot(self._config),
            "metrics": full_metrics,
            "feature_importance": importance,
            "split": self._filtered_split_period_metadata(
                usable_timestamps, train_index, validation_index, test_index, boundaries
            ),
        }
        headline: dict[str, Any] = {
            column: full_metrics[column].get("mae") for column in full_metrics
        }
        _LOGGER.info("Exit model trained: %s", headline)
        return full_metrics

    def _baseline_predictions(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        """Vectorised twin of :meth:`_heuristic`'s ATR-scaled rule, for comparison.

        Per Part 15 of the diagnostic reporting requirements: the ML exit
        model must prove it beats the plain rule-based baseline that would
        run in its place, rather than being trusted on its validation MAE
        alone.
        """
        atr_pct: pd.Series = features.get("atr_pct", pd.Series(0.0, index=features.index)).fillna(0.0)
        garch: pd.Series = features.get(
            "garch_volatility", pd.Series(0.0, index=features.index)
        ).fillna(0.0)
        effective_atr: pd.Series = atr_pct.where(atr_pct > 0.0, (garch * 2.0).clip(lower=0.0025))

        vol_rank: pd.Series = features.get(
            "garch_vol_rank", pd.Series(0.5, index=features.index)
        ).fillna(0.5)
        widen: pd.Series = 1.0 + 0.6 * vol_rank.clip(0.0, 1.0)

        stop_loss: np.ndarray = (effective_atr * self._settings.labels.sl_atr_multiple * widen).to_numpy()
        take_profit: np.ndarray = (
            effective_atr * self._settings.labels.tp_atr_multiple * widen
        ).to_numpy()
        return {
            "target_tp_pct": take_profit,
            "target_sl_pct": stop_loss,
            "target_trailing_pct": take_profit * 0.5,
        }

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
        """Fit the opportunity-score regressor.

        Trained only on rows where a directional trade was actually selected
        (``direction_target != NO_TRADE_OR_FAIL``) - the same restriction
        :class:`ExitModel` already applies to its own targets, and for the
        same underlying reason. ``target_risk_score`` is deterministically
        ``0.0`` for every ``NO_TRADE_OR_FAIL`` row (see
        ``module_b_features.labeler.TradeLabeler._attach_model_targets``),
        which is roughly 46% of the pooled dataset in a typical run. Training
        on the full, unfiltered population forced this regressor to also
        re-learn Direction's own hard "will either side ever reach
        take-profit" question on the same 55 features Direction itself only
        solves at barely-above-random balanced accuracy - diluting the signal
        this head actually needs (how clean is the path of a trade
        Direction/Entry have *already* approved, which is the only question
        :meth:`predict` is ever asked at inference time) and measuring R^2
        against an out-of-distribution population it will never see live.
        This was the single largest driver of this head's R^2 sitting far
        below Exit's, despite identical hyperparameters and features.
        """
        if dataset.is_empty:
            raise ModelTrainingError("risk model received an empty dataset")

        usable: pd.Series = dataset.direction_target != LabelClass.NO_TRADE_OR_FAIL.value
        if int(usable.sum()) < 100:
            raise ModelTrainingError(
                "not enough directional rows to fit the risk model", rows=int(usable.sum())
            )

        features: pd.DataFrame = dataset.features[usable].reset_index(drop=True)
        target: pd.Series = dataset.risk_target[usable].astype(float).reset_index(drop=True)
        metadata_usable: pd.DataFrame = dataset.metadata[usable].reset_index(drop=True)

        # Boundaries come from the *full* dataset, matching every other head -
        # see BaseModelHead._split_boundaries.
        boundaries: SplitBoundaries | None = self._split_boundaries(dataset)
        usable_timestamps: np.ndarray = (
            metadata_usable["timestamp"].to_numpy(dtype=np.int64)
            if "timestamp" in metadata_usable.columns
            else np.array([], dtype=np.int64)
        )
        if boundaries is not None and usable_timestamps.size > 0:
            train_index, validation_index, test_index = assign_split(usable_timestamps, boundaries)
        else:
            train_index = np.arange(len(features), dtype=np.int64)
            validation_index = np.array([], dtype=np.int64)
            test_index = np.array([], dtype=np.int64)
        validation_features: pd.DataFrame = features.iloc[validation_index]

        # target_risk_score is right-skewed (mean 0.29, median 0.20 in the
        # first production run) the same way the exit-geometry percentages
        # are, so it gets the same L1 (robust) objective rather than L2 -
        # a handful of extreme-heat rows should not dominate the loss and
        # drag every other prediction toward them.
        estimator: Any = self._make_regressor(robust=True)
        estimator = self._fit_estimator(
            estimator,
            features.iloc[train_index],
            target.iloc[train_index],
            validation_features,
            target.iloc[validation_index],
            eval_metric="l1",
            sample_weight=self._recency_weights(usable_timestamps[train_index]),
        )

        self._model = estimator
        self._feature_columns = dataset.feature_columns

        full_metrics: dict[str, Any] = {}
        importance: dict[str, Any] = {"status": "NOT_AVAILABLE", "reason": "no validation rows"}
        if len(validation_features) > 0:
            predictions: np.ndarray = estimator.predict(validation_features)
            full_metrics = ml_metrics.regression_metrics(
                target.iloc[validation_index].to_numpy(), predictions
            )
            importance = ml_metrics.feature_importance(estimator, dataset.feature_columns)

        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit_hash(),
            "rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "hyperparameters": _hyperparameter_snapshot(self._config),
            "metrics": full_metrics,
            "feature_importance": importance,
            "training_row_filter": (
                "direction_target != NO_TRADE_OR_FAIL - see RiskModel.train docstring"
            ),
            "split": self._filtered_split_period_metadata(
                usable_timestamps, train_index, validation_index, test_index, boundaries
            ),
        }
        headline: dict[str, Any] = {
            key: full_metrics[key] for key in ("mae", "r2") if key in full_metrics
        }
        _LOGGER.info("Risk model trained: %s", headline)
        return full_metrics

    def predict(
        self,
        features: pd.DataFrame,
        direction_confidence: float,
    ) -> RiskAllocation:
        """Size the trade, or abort it.

        Args:
            features: One feature row.
            direction_confidence: The independent direction-given-trade
                conditional confidence (``max(p, 1 - p)`` of
                ``DirectionPrediction.direction_given_trade_probability``) -
                the same 0.5-1.0 conditional-confidence scale the Decision
                Engine's own R1B rule (``Rule.DIRECTION_CONFIDENCE``) gates
                on. This head's hard veto and sizing curve are gated against
                ``DecisionSettings.min_direction_given_trade_confidence`` for
                consistency with that scale. Prior to the fix that added this
                docstring note, callers passed the stale *joint*
                long/short/no_trade distribution's max
                (``DirectionPrediction.confidence``) here, which meant a
                confident direction call could still be silently downsized or
                vetoed by this head even after clearing the Decision Engine's
                own independent gate/direction rules - see
                ``MLSubsystem.infer_sync`` for the call site.

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

        # The tier is derived from the score itself rather than taken from the
        # Direction model - see labeler.risk_tier_from_score for why the two
        # questions ("which way" and "how clean is the path") are kept apart.
        risk_tier: str = risk_tier_from_score(score, self._settings.labels)

        decision = self._settings.decision
        risk = self._settings.risk

        # --- Hard vetoes -------------------------------------------------
        if volatility_rank >= decision.max_volatility_percentile:
            return RiskAllocation(
                leverage=0,
                capital_allocation_pct=0.0,
                risk_score=score,
                risk_tier=risk_tier,
                abort_reason=(
                    f"volatility percentile {volatility_rank:.2f} >= "
                    f"{decision.max_volatility_percentile:.2f}"
                ),
                source=source,
            )
        if direction_confidence < decision.min_direction_given_trade_confidence:
            return RiskAllocation(
                leverage=0,
                capital_allocation_pct=0.0,
                risk_score=score,
                risk_tier=risk_tier,
                abort_reason=(
                    f"direction confidence {direction_confidence:.3f} < "
                    f"{decision.min_direction_given_trade_confidence:.3f}"
                ),
                source=source,
            )

        # --- Sizing ------------------------------------------------------
        # Confidence is rescaled onto [0, 1] across the *tradeable* band, so a
        # 70 %-confidence signal sizes near the floor and a 100 % one near the cap.
        confidence_span: float = max(1e-6, 1.0 - decision.min_direction_given_trade_confidence)
        confidence_factor: float = clamp(
            (direction_confidence - decision.min_direction_given_trade_confidence) / confidence_span, 0.0, 1.0
        )
        volatility_factor: float = 1.0 - clamp(volatility_rank, 0.0, 1.0) ** 2
        tier_factor: float = {"LOW": 1.0, "MEDIUM": 0.75, "HIGH": 0.5}.get(risk_tier, 0.6)

        composite: float = score * (0.35 + 0.65 * confidence_factor) * volatility_factor * tier_factor
        leverage_cap: int = min(decision.max_leverage, risk.max_leverage)
        leverage: int = int(np.floor(composite * leverage_cap))

        if leverage < decision.min_leverage:
            return RiskAllocation(
                leverage=0,
                capital_allocation_pct=0.0,
                risk_score=score,
                risk_tier=risk_tier,
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
            risk_tier=risk_tier,
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

        A head that fails to train is reported in the result and left unloaded;
        it does not abort the training of the remaining heads.
        """
        report: dict[str, Any] = {}
        for name, head in self.heads.items():
            try:
                report[name] = await asyncio.to_thread(head.train, dataset)
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
        entry: EntryPrediction = self.entry.predict(features, action=direction.action)
        exit_params: ExitParameters = self.exit.predict(features)
        # RiskModel's hard veto/sizing curve reads the same independent
        # direction-given-trade conditional confidence the Decision Engine's
        # R1B rule gates on - not the stale *joint* long/short/no_trade
        # distribution's max (`direction.confidence`). Passing the joint
        # metric here meant a confident direction call that had already
        # cleared the Decision Engine's own gate/direction rules could still
        # be silently vetoed or downsized by this head reading a different,
        # harder-to-clear scale.
        direction_given_trade_confidence: float = max(
            direction.direction_given_trade_probability,
            1.0 - direction.direction_given_trade_probability,
        )
        risk: RiskAllocation = self.risk.predict(
            features,
            direction_confidence=direction_given_trade_confidence,
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
