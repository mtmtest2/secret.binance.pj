"""Module C (ML Subsystem) and Module D (Decision Engine)."""

from __future__ import annotations

from module_c_ml.decision_engine import DecisionEngine
from module_c_ml.evaluation import ComparisonReport, DirectionComparison, direction_metrics
from module_c_ml.ml_models import (
    DirectionBaselineModel,
    DirectionModel,
    EntryModel,
    ExitModel,
    MLSubsystem,
    RiskModel,
)
from module_c_ml.schemas import (
    DecisionResult,
    DecisionVerdict,
    DirectionPrediction,
    EntryPrediction,
    ExitParameters,
    ModelInferenceResult,
    RiskAllocation,
    TradeAction,
    TradeSignal,
)

__all__: list[str] = [
    "ComparisonReport",
    "DecisionEngine",
    "DecisionResult",
    "DecisionVerdict",
    "DirectionBaselineModel",
    "DirectionComparison",
    "DirectionModel",
    "DirectionPrediction",
    "EntryModel",
    "EntryPrediction",
    "ExitModel",
    "ExitParameters",
    "MLSubsystem",
    "ModelInferenceResult",
    "RiskAllocation",
    "RiskModel",
    "TradeAction",
    "TradeSignal",
    "direction_metrics",
]
