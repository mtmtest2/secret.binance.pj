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
from core.utils import clamp
from module_b_features.features import FEATURE_COLUMNS, HEAD_FEATURE_COLUMNS, HMMRegime
from module_b_features.labeler import (
    LABEL_ORDER,
    LONG_LABELS,
    SHORT_LABELS,
    LabelClass,
)
from module_b_features.processor import InferencePayload, ProcessedDataset
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


class BaseModelHead(ABC):
    """Shared training / persistence plumbing for the four heads."""

    #: Filename stem of the artifact, unique per head.
    name: str = "base"
    #: Key into :data:`HEAD_FEATURE_COLUMNS`; decides which columns this head
    #: consumes.  Heads see different feature sets on purpose - a feature is
    #: routed to the head whose question it answers, and nowhere else.
    feature_set: str = "direction"

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._config: MLSettings = settings.ml
        self._model: Any = None
        self._feature_columns: tuple[str, ...] = HEAD_FEATURE_COLUMNS.get(
            self.feature_set, FEATURE_COLUMNS
        )
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Feature contract & splitting
    # ------------------------------------------------------------------
    def _training_columns(self, dataset: ProcessedDataset) -> list[str]:
        """Columns this head trains on, intersected with what the dataset has."""
        wanted: tuple[str, ...] = HEAD_FEATURE_COLUMNS.get(self.feature_set, FEATURE_COLUMNS)
        available: set[str] = set(dataset.features.columns)
        columns: list[str] = [column for column in wanted if column in available]
        missing: list[str] = [column for column in wanted if column not in available]
        if missing:
            _LOGGER.warning(
                "%s: %d feature(s) absent from the dataset and dropped from the contract: %s",
                self.name,
                len(missing),
                missing,
            )
        return columns

    def _design_matrix(self, dataset: ProcessedDataset) -> pd.DataFrame:
        """The head's own view of the pooled feature matrix."""
        return dataset.features[self._training_columns(dataset)]

    def _split(self, dataset: ProcessedDataset) -> tuple[np.ndarray, np.ndarray]:
        """Purged, embargoed train/validation positions with the test tail removed.

        Every head fits through here, which is what makes "the test block is never
        trained on" a structural property rather than a convention.
        """
        return dataset.train_validation_split(
            validation_fraction=self._config.validation_fraction,
            purge_bars=self._config.purge_bars,
            test_fraction=self._config.test_fraction,
            embargo_bars=self._config.embargo_bars,
            timeframe_ms=self._settings.data.timeframe_ms,
        )

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
    ) -> Any:
        """Fit with early stopping when a non-empty validation block exists."""
        rounds: int = self._config.early_stopping_rounds
        if x_validation.empty or rounds <= 0:
            estimator.fit(x_train, y_train)
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
                )
            else:
                estimator.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_validation, y_validation)],
                    eval_metric=eval_metric,
                    callbacks=callbacks,
                )
            return estimator

        estimator.set_params(early_stopping_rounds=rounds, eval_metric=eval_metric)
        estimator.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
        return estimator

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
    """Model 1 - market direction, as a two-stage cascade.

    Why not one flat 5-class booster
    --------------------------------
    The flat head has to separate "trade vs no-trade" and "long vs short" with
    the same trees and the same loss.  Since ~45 % of rows are
    ``NO_TRADE_OR_FAIL`` and the two directional outcomes are near-symmetric,
    almost all of the achievable log-loss reduction sits in the first question,
    so that is where the capacity goes - and LONG-vs-SHORT discrimination, the
    thing the strategy actually monetises, is learned only incidentally.

    The cascade splits the problem instead:

    ``gate``       ``P(tradeable | x)`` over every row.
    ``direction``  ``P(LONG | x, tradeable)`` fitted **only on tradeable rows**,
                   so every split it makes is spent on long-vs-short.
    ``tier``       ``P(low risk | x, tradeable)``, which keeps the five-class
                   output contract (and therefore the implied risk tier) intact.

    The joint distribution is reassembled as

    .. code-block:: text

        p(NO_TRADE)        = 1 - p_gate
        p(LONG_*)          = p_gate *      p_dir  * {p_tier, 1 - p_tier}
        p(SHORT_*)         = p_gate * (1 - p_dir) * {p_tier, 1 - p_tier}

    so :class:`DirectionPrediction`, the Decision Engine cascade and **every
    threshold** see exactly the shape they saw before.  No threshold anywhere in
    the system is changed by this head; the improvement has to come from better
    probabilities at the same cut-off, which is the only kind that is real.

    Setting ``ml.direction_architecture = "single_stage"`` restores the original
    flat 5-class booster verbatim - that is what
    :class:`DirectionBaselineModel` uses to stay a frozen comparison point.
    """

    name = "direction_model"
    feature_set = "direction"

    #: Architecture used when the instance does not override it.
    forced_architecture: str | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @property
    def architecture(self) -> str:
        """``two_stage_cascade`` or ``single_stage``."""
        return self.forced_architecture or self._config.direction_architecture

    def train(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Fit the direction head on the pooled, purged, test-free dataset."""
        if dataset.is_empty:
            raise ModelTrainingError("direction model received an empty dataset")

        classes: list[str] = sorted(set(dataset.direction_target.unique()))
        if len(classes) < 2:
            raise ModelTrainingError("direction model needs >= 2 classes", classes=classes)

        train_index, validation_index = self._split(dataset)
        if train_index.size == 0:
            raise ModelTrainingError("direction model has no training rows after purging")

        features: pd.DataFrame = self._design_matrix(dataset)
        self._feature_columns = tuple(features.columns)

        if self.architecture == "single_stage":
            metrics: dict[str, Any] = self._train_single_stage(
                dataset, features, train_index, validation_index
            )
        else:
            metrics = self._train_cascade(dataset, features, train_index, validation_index)

        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "architecture": self.architecture,
            "rows": int(train_index.size),
            "validation_rows": int(validation_index.size),
            "classes": list(LABEL_ORDER),
            "feature_columns": list(self._feature_columns),
            "distribution": dataset.class_distribution(),
            "metrics": metrics,
        }
        _LOGGER.info("Direction model (%s) trained: %s", self.architecture, metrics)
        return metrics

    def _train_single_stage(
        self,
        dataset: ProcessedDataset,
        features: pd.DataFrame,
        train_index: np.ndarray,
        validation_index: np.ndarray,
    ) -> dict[str, Any]:
        """The original flat 5-class booster - the frozen baseline path."""
        class_index: dict[str, int] = {name: index for index, name in enumerate(LABEL_ORDER)}
        encoded: pd.Series = dataset.direction_target.map(class_index)
        if encoded.isna().any():
            raise ModelTrainingError("direction labels contain unknown classes")

        estimator: Any = self._make_classifier(num_class=len(LABEL_ORDER))
        estimator = self._fit_estimator(
            estimator,
            features.iloc[train_index],
            encoded.iloc[train_index].astype(int),
            features.iloc[validation_index],
            encoded.iloc[validation_index].astype(int),
            eval_metric="multi_logloss",
        )
        self._model = {"architecture": "single_stage", "flat": estimator}
        return self._score_validation(features.iloc[validation_index], dataset, validation_index)

    def _train_cascade(
        self,
        dataset: ProcessedDataset,
        features: pd.DataFrame,
        train_index: np.ndarray,
        validation_index: np.ndarray,
    ) -> dict[str, Any]:
        """Fit gate, direction and tier stages, then calibrate the first two."""
        labels: pd.Series = dataset.direction_target
        is_trade: pd.Series = labels.isin(LONG_LABELS | SHORT_LABELS).astype(int)
        is_long: pd.Series = labels.isin(LONG_LABELS).astype(int)
        is_low_risk: pd.Series = labels.isin(
            {
                LabelClass.LONG_SUCCESS_LOW_RISK.value,
                LabelClass.SHORT_SUCCESS_LOW_RISK.value,
            }
        ).astype(int)

        fit_index, calibration_index = self._carve_calibration_slice(dataset, train_index)
        if fit_index.size == 0:
            fit_index, calibration_index = train_index, np.array([], dtype=np.int64)

        # --- Stage 1: tradeable gate --------------------------------------
        gate: Any = self._make_classifier(num_class=2)
        gate = self._fit_estimator(
            gate,
            features.iloc[fit_index],
            is_trade.iloc[fit_index],
            features.iloc[validation_index],
            is_trade.iloc[validation_index],
            eval_metric="binary_logloss",
        )

        # --- Stage 2: LONG vs SHORT, tradeable rows only ------------------
        # This restriction is the point of the cascade: the stage never sees a
        # NO_TRADE row, so none of its capacity is spent re-deriving the gate.
        directional_fit: np.ndarray = fit_index[is_trade.iloc[fit_index].to_numpy() == 1]
        directional_validation: np.ndarray = validation_index[
            is_trade.iloc[validation_index].to_numpy() == 1
        ]
        if directional_fit.size < 100:
            raise ModelTrainingError(
                "not enough directional rows to fit the long/short stage",
                rows=int(directional_fit.size),
            )

        direction_stage: Any = self._make_stage2_classifier()
        direction_stage = self._fit_estimator(
            direction_stage,
            features.iloc[directional_fit],
            is_long.iloc[directional_fit],
            features.iloc[directional_validation],
            is_long.iloc[directional_validation],
            eval_metric="binary_logloss",
        )

        # --- Stage 3: risk tier, so the 5-class contract survives ----------
        tier_stage: Any = self._make_stage2_classifier()
        tier_stage = self._fit_estimator(
            tier_stage,
            features.iloc[directional_fit],
            is_low_risk.iloc[directional_fit],
            features.iloc[directional_validation],
            is_low_risk.iloc[directional_validation],
            eval_metric="binary_logloss",
        )

        self._model = {
            "architecture": "two_stage_cascade",
            "gate": gate,
            "direction": direction_stage,
            "tier": tier_stage,
            "gate_calibrator": None,
            "direction_calibrator": None,
        }

        calibration: dict[str, Any] = self._fit_calibrators(
            features, is_trade, is_long, calibration_index
        )
        metrics: dict[str, Any] = self._score_validation(
            features.iloc[validation_index], dataset, validation_index
        )
        metrics["calibration"] = calibration
        metrics["stage_rows"] = {
            "gate_fit": int(fit_index.size),
            "direction_fit": int(directional_fit.size),
            "calibration": int(calibration_index.size),
        }
        return metrics

    def _carve_calibration_slice(
        self,
        dataset: ProcessedDataset,
        train_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split the *training* block into fit and calibration parts.

        The calibration slice is the chronological tail of the training block,
        separated from the fit part by the same purge+embargo gap used between
        train and validation.  It deliberately comes out of train and not out of
        validation: validation is already spent on early stopping, and the test
        block is untouchable.
        """
        if not self._config.direction_calibrate or train_index.size < 500:
            return train_index, np.array([], dtype=np.int64)

        stamps: np.ndarray = dataset.timestamps()[train_index]
        order: np.ndarray = np.argsort(stamps, kind="stable")
        ordered: np.ndarray = train_index[order]
        sorted_stamps: np.ndarray = stamps[order]

        cut_position: int = int(ordered.size * (1.0 - self._config.direction_calibration_fraction))
        cut_position = max(1, min(ordered.size - 1, cut_position))
        cut_ts: int = int(sorted_stamps[cut_position])
        gap_ms: int = (self._config.purge_bars + self._config.embargo_bars) * (
            self._settings.data.timeframe_ms
        )

        fit_mask: np.ndarray = sorted_stamps < cut_ts - gap_ms
        calibration_mask: np.ndarray = sorted_stamps >= cut_ts
        return ordered[fit_mask], ordered[calibration_mask]

    def _fit_calibrators(
        self,
        features: pd.DataFrame,
        is_trade: pd.Series,
        is_long: pd.Series,
        calibration_index: np.ndarray,
    ) -> dict[str, Any]:
        """Fit isotonic calibrators, keeping them only when they help.

        A calibrator that does not improve the Brier score on the slice it was
        fitted on is not a calibrator, it is noise with extra steps - so it is
        discarded and the raw estimator stays in production.  The decision and
        the numbers behind it are recorded in the artifact metadata.
        """
        report: dict[str, Any] = {"status": "SKIPPED", "gate": None, "direction": None}
        if calibration_index.size < 200 or self._model is None:
            return report

        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss

        report["status"] = "AVAILABLE"
        report["rows"] = int(calibration_index.size)
        matrix: pd.DataFrame = features.iloc[calibration_index]

        def _calibrate(stage: str, estimator: Any, target: np.ndarray) -> dict[str, Any] | None:
            if target.size == 0 or len(set(target.tolist())) < 2:
                return None
            raw: np.ndarray = np.asarray(
                estimator.predict_proba(matrix), dtype=np.float64
            )[:, -1]
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(raw, target)
            calibrated: np.ndarray = np.asarray(calibrator.predict(raw), dtype=np.float64)

            raw_brier: float = float(brier_score_loss(target, raw))
            calibrated_brier: float = float(brier_score_loss(target, calibrated))
            improved: bool = calibrated_brier < raw_brier
            if improved:
                self._model[f"{stage}_calibrator"] = calibrator
            return {
                "brier_raw": raw_brier,
                "brier_calibrated": calibrated_brier,
                "applied": improved,
            }

        report["gate"] = _calibrate(
            "gate", self._model["gate"], is_trade.iloc[calibration_index].to_numpy()
        )

        directional: np.ndarray = calibration_index[
            is_trade.iloc[calibration_index].to_numpy() == 1
        ]
        if directional.size >= 200:
            directional_matrix: pd.DataFrame = features.iloc[directional]
            raw_direction: np.ndarray = np.asarray(
                self._model["direction"].predict_proba(directional_matrix), dtype=np.float64
            )[:, -1]
            target: np.ndarray = is_long.iloc[directional].to_numpy()
            if len(set(target.tolist())) > 1:
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                calibrator.fit(raw_direction, target)
                calibrated = np.asarray(calibrator.predict(raw_direction), dtype=np.float64)
                raw_brier = float(brier_score_loss(target, raw_direction))
                calibrated_brier = float(brier_score_loss(target, calibrated))
                if calibrated_brier < raw_brier:
                    self._model["direction_calibrator"] = calibrator
                report["direction"] = {
                    "brier_raw": raw_brier,
                    "brier_calibrated": calibrated_brier,
                    "applied": calibrated_brier < raw_brier,
                }
        return report

    def _make_stage2_classifier(self) -> Any:
        """Binary classifier for the LONG/SHORT and tier stages.

        These stages see a fraction of the rows the gate does and overfit far
        more readily, so they get their own, harder-regularised hyper-parameters
        rather than inheriting the gate's.
        """
        config: MLSettings = self._config
        if config.booster == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=config.direction_stage2_n_estimators,
                learning_rate=config.direction_stage2_learning_rate,
                max_depth=config.direction_stage2_max_depth,
                subsample=config.direction_stage2_subsample,
                colsample_bytree=config.direction_stage2_colsample_bytree,
                reg_lambda=config.direction_stage2_reg_lambda,
                reg_alpha=config.direction_stage2_reg_alpha,
                min_child_weight=config.direction_stage2_min_child_samples,
                objective="binary:logistic",
                random_state=config.random_state,
                n_jobs=self._n_jobs(),
                tree_method="hist",
                verbosity=0,
            )

        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=config.direction_stage2_n_estimators,
            learning_rate=config.direction_stage2_learning_rate,
            max_depth=config.direction_stage2_max_depth,
            num_leaves=config.direction_stage2_num_leaves,
            subsample=config.direction_stage2_subsample,
            subsample_freq=1,
            colsample_bytree=config.direction_stage2_colsample_bytree,
            reg_lambda=config.direction_stage2_reg_lambda,
            reg_alpha=config.direction_stage2_reg_alpha,
            min_child_samples=config.direction_stage2_min_child_samples,
            objective="binary",
            num_class=1,
            # Long and short are close to balanced, but not exactly; weighting
            # keeps a mild imbalance from becoming a systematic long or short
            # bias in the decision boundary.
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=self._n_jobs(),
            verbose=-1,
        )

    def _score_validation(
        self,
        matrix: pd.DataFrame,
        dataset: ProcessedDataset,
        validation_index: np.ndarray,
    ) -> dict[str, Any]:
        """Aggregated LONG/SHORT/NO_TRADE metrics on the validation block."""
        if matrix.empty or validation_index.size == 0:
            return {}

        from module_c_ml.evaluation import direction_metrics

        probabilities: np.ndarray = self._joint_probabilities(matrix)
        return direction_metrics(
            probabilities=probabilities,
            labels=dataset.direction_target.iloc[validation_index].to_numpy(),
            settings=self._settings,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _joint_probabilities(self, matrix: pd.DataFrame) -> np.ndarray:
        """Return the ``(n, 5)`` class distribution for a design matrix."""
        model: dict[str, Any] = self._require_model()
        rows: int = len(matrix)
        output: np.ndarray = np.zeros((rows, len(LABEL_ORDER)), dtype=np.float64)

        index_of: dict[str, int] = {name: position for position, name in enumerate(LABEL_ORDER)}

        if model.get("architecture") == "single_stage":
            estimator: Any = model["flat"]
            raw: np.ndarray = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
            classes: list[int] = [int(value) for value in getattr(estimator, "classes_", [])]
            if classes and len(classes) == raw.shape[1]:
                for position, class_index in enumerate(classes):
                    if 0 <= class_index < len(LABEL_ORDER):
                        output[:, class_index] = raw[:, position]
            else:  # pragma: no cover - estimator without `classes_`
                output[:, : raw.shape[1]] = raw
            return output

        gate: np.ndarray = self._stage_probability(model["gate"], model["gate_calibrator"], matrix)
        direction: np.ndarray = self._stage_probability(
            model["direction"], model["direction_calibrator"], matrix
        )
        tier: np.ndarray = self._stage_probability(model["tier"], None, matrix)

        long_mass: np.ndarray = gate * direction
        short_mass: np.ndarray = gate * (1.0 - direction)

        output[:, index_of[LabelClass.LONG_SUCCESS_LOW_RISK.value]] = long_mass * tier
        output[:, index_of[LabelClass.LONG_SUCCESS_HIGH_RISK.value]] = long_mass * (1.0 - tier)
        output[:, index_of[LabelClass.SHORT_SUCCESS_LOW_RISK.value]] = short_mass * tier
        output[:, index_of[LabelClass.SHORT_SUCCESS_HIGH_RISK.value]] = short_mass * (1.0 - tier)
        output[:, index_of[LabelClass.NO_TRADE_OR_FAIL.value]] = 1.0 - gate
        return output

    @staticmethod
    def _stage_probability(
        estimator: Any,
        calibrator: Any,
        matrix: pd.DataFrame,
    ) -> np.ndarray:
        """Positive-class probability of one stage, optionally calibrated."""
        raw: np.ndarray = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)[:, -1]
        if calibrator is not None:
            raw = np.asarray(calibrator.predict(raw), dtype=np.float64)
        return np.clip(raw, 1e-6, 1.0 - 1e-6)

    def predict(self, features: pd.DataFrame) -> DirectionPrediction:
        """Score a single feature row (stateless, thread-safe).

        Falls back to a transparent trend/regime heuristic when no artifact is
        loaded, stamping the result as ``HEURISTIC``.
        """
        if self._model is None:
            return self._heuristic(features)

        aligned: pd.DataFrame = self._align(features)
        raw: np.ndarray = self._joint_probabilities(aligned)[0]
        probabilities: dict[str, float] = {
            name: float(raw[position]) for position, name in enumerate(LABEL_ORDER)
        }
        return DirectionPrediction(probabilities=probabilities, source=ModelSource.TRAINED)

    def predict_proba_frame(self, features: pd.DataFrame) -> np.ndarray:
        """Batch scoring used by the evaluation harness (``(n, 5)`` array)."""
        return self._joint_probabilities(self._align(features))

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
                LabelClass.LONG_SUCCESS_LOW_RISK.value: long_mass * 0.6,
                LabelClass.LONG_SUCCESS_HIGH_RISK.value: long_mass * 0.4,
                LabelClass.SHORT_SUCCESS_LOW_RISK.value: short_mass * 0.6,
                LabelClass.SHORT_SUCCESS_HIGH_RISK.value: short_mass * 0.4,
                LabelClass.NO_TRADE_OR_FAIL.value: max(1e-6, 1.0 - long_mass - short_mass),
            },
            source=ModelSource.HEURISTIC,
        )


class DirectionBaselineModel(DirectionModel):
    """The frozen pre-change Direction head, kept purely for comparison.

    It is pinned to the flat 5-class architecture and to the *base* feature block
    (no order-flow columns), which is exactly what the system trained before this
    change.  It writes its own artifact, so retraining the production head can
    never overwrite the baseline, and the two are always scored by the same
    evaluator at the same, unchanged decision threshold.
    """

    name = "direction_model_baseline"
    feature_set = "baseline_direction"
    forced_architecture = "single_stage"


class EntryModel(BaseModelHead):
    """Model 2 - entry timing filter.

    Answers a narrow question: given that a direction has been chosen, is *this*
    candle close a clean entry, or should we wait for the next 5-minute bar?
    Trained on ``entry_quality``, which marks bars whose winning trade never gave
    back more than ``low_risk_mae_ratio`` of its stop.
    """

    name = "entry_model"
    feature_set = "entry"

    def train(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Fit the binary entry-quality classifier."""
        if dataset.is_empty:
            raise ModelTrainingError("entry model received an empty dataset")

        target: pd.Series = dataset.entry_target.astype(int)
        if target.nunique() < 2:
            raise ModelTrainingError("entry target is degenerate (single class)")

        train_index, validation_index = self._split(dataset)
        features: pd.DataFrame = self._design_matrix(dataset)
        estimator: Any = self._make_classifier(num_class=2)
        estimator = self._fit_estimator(
            estimator,
            features.iloc[train_index],
            target.iloc[train_index],
            features.iloc[validation_index],
            target.iloc[validation_index],
            eval_metric="binary_logloss",
        )

        self._model = estimator
        self._feature_columns = tuple(features.columns)

        metrics: dict[str, float] = {}
        if len(validation_index) > 0:
            from sklearn.metrics import precision_score, recall_score, roc_auc_score

            validation_features: pd.DataFrame = features.iloc[validation_index]
            validation_target: pd.Series = target.iloc[validation_index]
            predictions: np.ndarray = estimator.predict(validation_features)
            probabilities: np.ndarray = estimator.predict_proba(validation_features)[:, 1]
            metrics = {
                "precision": float(precision_score(validation_target, predictions, zero_division=0)),
                "recall": float(recall_score(validation_target, predictions, zero_division=0)),
                "roc_auc": (
                    float(roc_auc_score(validation_target, probabilities))
                    if validation_target.nunique() > 1
                    else float("nan")
                ),
            }

        self._metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(train_index)),
            "positive_rate": float(target.mean()),
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
        """Decide whether to act on this candle or wait for the next one."""
        cutoff: float = (
            threshold if threshold is not None else self._settings.decision.min_entry_probability
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
    feature_set = "exit"

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

        features: pd.DataFrame = self._design_matrix(dataset)[usable].reset_index(drop=True)
        targets: pd.DataFrame = dataset.exit_targets[usable].reset_index(drop=True)

        # The exit head sub-selects rows, so it cannot reuse the shared position
        # split directly; it reproduces the same three-way geometry by fraction,
        # with the test tail removed first so it is never fitted on.
        usable_rows: int = len(features)
        test_start: int = max(
            1, int(usable_rows * (1.0 - self._config.test_fraction))
        )
        split_point: int = max(
            1, int(test_start * (1.0 - self._config.validation_fraction))
        )
        train_end: int = max(1, split_point - self._config.purge_bars - self._config.embargo_bars)

        estimators: dict[str, Any] = {}
        metrics: dict[str, float] = {}
        from sklearn.metrics import mean_absolute_error

        for column in self._TARGETS:
            estimator: Any = self._make_regressor()
            estimator = self._fit_estimator(
                estimator,
                features.iloc[:train_end],
                targets[column].iloc[:train_end],
                features.iloc[split_point:test_start],
                targets[column].iloc[split_point:test_start],
                eval_metric="l1",
            )
            estimators[column] = estimator
            if split_point < test_start:
                predictions: np.ndarray = estimator.predict(features.iloc[split_point:test_start])
                metrics[f"{column}_mae"] = float(
                    mean_absolute_error(targets[column].iloc[split_point:test_start], predictions)
                )

        self._model = estimators
        self._feature_columns = tuple(features.columns)
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
    feature_set = "risk"

    def train(self, dataset: ProcessedDataset) -> dict[str, Any]:
        """Fit the opportunity-score regressor."""
        if dataset.is_empty:
            raise ModelTrainingError("risk model received an empty dataset")

        target: pd.Series = dataset.risk_target.astype(float)
        train_index, validation_index = self._split(dataset)
        features: pd.DataFrame = self._design_matrix(dataset)

        estimator: Any = self._make_regressor()
        estimator = self._fit_estimator(
            estimator,
            features.iloc[train_index],
            target.iloc[train_index],
            features.iloc[validation_index],
            target.iloc[validation_index],
            eval_metric="l2",
        )

        self._model = estimator
        self._feature_columns = tuple(features.columns)

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
        risk_tier: str = "UNKNOWN",
    ) -> RiskAllocation:
        """Size the trade, or abort it.

        Args:
            features: One feature row.
            direction_confidence: Winning probability mass from Model 1.
            risk_tier: Tier implied by the direction model's top class.

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
                risk_tier=risk_tier,
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
                risk_tier=risk_tier,
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
        #: Frozen comparison head.  Deliberately *not* part of :attr:`heads`, so
        #: routine training never overwrites it and it can never serve a live
        #: prediction - it exists only for the before/after evaluation.
        self.direction_baseline: DirectionBaselineModel = DirectionBaselineModel(settings)
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
        risk: RiskAllocation = self.risk.predict(
            features,
            direction_confidence=direction.confidence,
            risk_tier=direction.implied_risk_tier,
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
