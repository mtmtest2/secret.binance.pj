"""Typed exception hierarchy.

Granular error handling is mandated by the coding standards: every module raises
domain-specific exceptions so callers can react precisely instead of swallowing
bare :class:`Exception` objects.
"""

from __future__ import annotations

from typing import Any


class QuantSystemError(Exception):
    """Base class for every error raised inside the trading system."""

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = dict(context)

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        if not self.context:
            return self.message
        rendered: str = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({rendered})"


class ConfigurationError(QuantSystemError):
    """Raised when settings are internally inconsistent or missing."""


class DataFetchError(QuantSystemError):
    """Raised when the exchange could not deliver data after all retries."""


class DataIntegrityError(QuantSystemError):
    """Raised by the QC gatekeeper when data cannot be healed into a valid state."""


class InsufficientDataError(QuantSystemError):
    """Raised when fewer rows are available than a computation requires."""


class FeatureEngineeringError(QuantSystemError):
    """Raised when a feature transform fails irrecoverably."""


class LabelingError(QuantSystemError):
    """Raised when the forward-looking labeler receives inconsistent input."""


class ModelNotLoadedError(QuantSystemError):
    """Raised when inference is attempted before a model artifact is loaded."""


class ModelTrainingError(QuantSystemError):
    """Raised when a model cannot be fitted (degenerate labels, empty data, ...)."""


class DecisionEngineError(QuantSystemError):
    """Raised when the decision engine receives malformed inference results."""


class ExecutionError(QuantSystemError):
    """Raised when an order could not be placed, amended or cancelled."""


class KillSwitchEngaged(QuantSystemError):
    """Raised when the Risk Guard is RED and new risk is therefore forbidden."""


class DatabaseError(QuantSystemError):
    """Raised when a persistence operation fails."""
