"""Exponential time-decay sample weighting (module_c_ml/ml_models.py).

Crypto regimes drift, so training rows are weighted by recency: a row
``recency_half_life_days`` behind the most recent one in the slice gets half
the weight, one that far again gets a quarter, and so on.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import Settings
from module_c_ml.ml_models import DirectionModel


def test_recency_weights_halve_at_the_configured_half_life() -> None:
    settings = Settings(ml={"recency_half_life_days": 10.0})
    model = DirectionModel(settings)

    day_ms = 86_400_000.0
    now = 1_000_000.0 * day_ms
    timestamps = np.array([now, now - 10 * day_ms, now - 20 * day_ms, now - 30 * day_ms])

    weights = model._recency_weights(timestamps)
    assert weights is not None
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5, rel=1e-6)
    assert weights[2] == pytest.approx(0.25, rel=1e-6)
    assert weights[3] == pytest.approx(0.125, rel=1e-6)
    # Monotonically non-increasing as rows get older.
    assert list(weights) == sorted(weights, reverse=True)


def test_recency_weights_disabled_returns_none() -> None:
    settings = Settings(ml={"recency_half_life_days": 0.0})
    model = DirectionModel(settings)
    timestamps = np.array([1.0, 2.0, 3.0])
    assert model._recency_weights(timestamps) is None


def test_recency_weights_empty_timestamps_returns_none() -> None:
    settings = Settings()
    model = DirectionModel(settings)
    assert model._recency_weights(np.array([])) is None
