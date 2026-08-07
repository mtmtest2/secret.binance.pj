"""Cross-cutting primitives shared by every module (logging, errors, helpers)."""

from __future__ import annotations

from core.exceptions import (
    ConfigurationError,
    DataIntegrityError,
    ExecutionError,
    FeatureEngineeringError,
    InsufficientDataError,
    KillSwitchEngaged,
    ModelNotLoadedError,
    QuantSystemError,
)
from core.logger import configure_logging, get_logger

__all__: list[str] = [
    "ConfigurationError",
    "DataIntegrityError",
    "ExecutionError",
    "FeatureEngineeringError",
    "InsufficientDataError",
    "KillSwitchEngaged",
    "ModelNotLoadedError",
    "QuantSystemError",
    "configure_logging",
    "get_logger",
]
