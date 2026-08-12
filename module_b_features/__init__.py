"""Module B - Feature Engineering & Risk-Based Labeling."""

from __future__ import annotations

from module_b_features.features import (
    BASE_FEATURE_COLUMNS,
    DIRECTION_FEATURE_COLUMNS,
    ENTRY_FEATURE_COLUMNS,
    EXIT_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    HEAD_FEATURE_COLUMNS,
    ORDER_FLOW_FEATURES,
    RISK_FEATURE_COLUMNS,
    FeatureEngineer,
    FeatureService,
    HMMRegime,
)
from module_b_features.labeler import LabelClass, RiskTier, TradeLabeler
from module_b_features.processor import DatasetProcessor, DatasetSplit, ProcessedDataset

__all__: list[str] = [
    "BASE_FEATURE_COLUMNS",
    "DIRECTION_FEATURE_COLUMNS",
    "ENTRY_FEATURE_COLUMNS",
    "EXIT_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "HEAD_FEATURE_COLUMNS",
    "ORDER_FLOW_FEATURES",
    "RISK_FEATURE_COLUMNS",
    "DatasetProcessor",
    "DatasetSplit",
    "FeatureEngineer",
    "FeatureService",
    "HMMRegime",
    "LabelClass",
    "ProcessedDataset",
    "RiskTier",
    "TradeLabeler",
]
