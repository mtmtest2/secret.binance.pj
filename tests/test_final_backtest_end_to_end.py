"""End-to-end check: real feature engineering -> real labeling -> the strict
chronological split -> real training -> the final backtest replay - wired
together exactly the way ``main.TradingSystem`` runs them, minus the real
database/exchange.

This is the integration-level counterpart to:

* ``tests/test_chronological_split.py`` (pure split math),
* ``tests/test_backtester_realism.py`` (pure fee/slippage arithmetic),
* ``tests/test_diagnostic_backtest_window.py`` (pure window-sizing math).

Here those pieces run together against one real, causally-computed
``ProcessedDataset`` to confirm the model heads never see the held-out test
split during training, and that the final backtest replay window falls
entirely inside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from module_b_features.processor import DatasetProcessor, ProcessedDataset
from module_b_features.features import FeatureService
from module_c_ml.decision_engine import DecisionEngine
from module_c_ml.ml_models import MLSubsystem
from module_e_execution.backtester import Backtester
from main import _final_backtest_window

_TIMEFRAME_MS = 5 * 60 * 1_000
_SYMBOL = "BTC/USDT:USDT"


def _synthetic_ohlcv(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """A plausible, gap-free 5m OHLCV history shaped exactly like
    ``DatabaseHandler.load_ohlcv_dataframe``'s real return value (DatetimeIndex
    named "open_time" plus a parallel int ms "timestamp" column)."""
    returns = rng.normal(0.0, 0.004, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, size=n)))
    volume = np.abs(rng.normal(1_000.0, 200.0, size=n)) + 1.0

    # Built as plain integer ms arithmetic (never derived from a DatetimeIndex
    # via `.view`/`.astype`) so this is immune to pandas' datetime64 unit
    # (ns/us/ms) varying by version - exactly the epoch-ms integers
    # DatabaseHandler.load_ohlcv_dataframe reads straight out of SQLite.
    base_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    timestamps = base_ms + np.arange(n, dtype=np.int64) * _TIMEFRAME_MS
    index = pd.to_datetime(timestamps, unit="ms", utc=True)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    frame.index.name = "open_time"
    return frame


class FakeDatabase:
    """Serves one symbol's worth of synthetic OHLCV, matching the real
    ``DatabaseHandler`` contract closely enough for ``DatasetProcessor`` and
    ``Backtester`` to both run their real code paths against it."""

    def __init__(self, ohlcv: pd.DataFrame) -> None:
        self._ohlcv = ohlcv

    async def load_ohlcv_dataframe(self, symbol: str, limit: int | None = None) -> pd.DataFrame:
        if limit is None:
            return self._ohlcv
        return self._ohlcv.iloc[-limit:]

    async def load_futures_metrics_frame(self, symbol: str, limit: int = 1_000) -> pd.DataFrame:
        return pd.DataFrame()


async def _empty_book_frame(symbol: str, depth: int) -> pd.DataFrame:
    return pd.DataFrame()


@pytest.mark.asyncio
async def test_final_backtest_replays_only_the_held_out_test_split() -> None:
    settings = Settings(
        ml={
            "n_estimators": 15,
            "early_stopping_rounds": 5,
            "purge_bars": 10,
            # Tiny (~13.9-day) history relative to the nominal 24-month split
            # window - split_boundaries scales all three blocks down
            # proportionally (2:1:1), which is exactly the "shorter than 2
            # years of history" case a newly-bootstrapped system starts in.
        },
    )
    rng = np.random.default_rng(11)
    ohlcv = _synthetic_ohlcv(rng, n=4_000)
    database = FakeDatabase(ohlcv)

    feature_service = FeatureService(settings)
    processor = DatasetProcessor(settings, database, feature_service=feature_service)
    processor._load_order_book_frame = _empty_book_frame  # avoid touching a real DB

    dataset: ProcessedDataset = await processor.build_training_dataset(symbols=[_SYMBOL])
    assert not dataset.is_empty

    boundaries = dataset.split_boundaries(
        train_months=settings.ml.train_months,
        validation_months=settings.ml.validation_months,
        test_months=settings.ml.test_months,
        purge_bars=settings.ml.purge_bars,
        timeframe_ms=settings.data.timeframe_ms,
    )
    assert boundaries is not None and boundaries.scaled_down is True
    split = dataset.chronological_split(boundaries)
    assert len(split.train_index) > 0
    assert len(split.validation_index) > 0
    assert len(split.test_index) > 0

    # --- Train all four heads: none of this may touch the test split -------
    ml = MLSubsystem(settings)
    report = await ml.train_all(dataset)
    for head, head_report in report.items():
        assert "error" not in head_report, f"{head} failed to train: {head_report}"

    # Direction's metadata carries the full split disclosure - check it
    # directly against the ground-truth boundaries.
    direction_split = ml.direction.metadata["split"]
    assert direction_split["train"]["rows"] == len(split.train_index)
    assert direction_split["validation"]["rows"] == len(split.validation_index)
    assert direction_split["test"]["rows"] == len(split.test_index)
    if direction_split["train"]["end"] and direction_split["validation"]["start"]:
        assert direction_split["train"]["end"] < direction_split["validation"]["start"]
    if direction_split["validation"]["end"] and direction_split["test"]["start"]:
        assert direction_split["validation"]["end"] < direction_split["test"]["start"]

    # --- Final backtest: must replay only the test window ------------------
    assert split.test_start_ms is not None and split.test_end_ms is not None
    warmup_padding = feature_service.engineer.minimum_rows()
    max_candles = _final_backtest_window(
        test_start_ms=split.test_start_ms,
        test_end_ms=split.test_end_ms,
        timeframe_ms=settings.data.timeframe_ms,
        warmup_padding=warmup_padding,
    )

    decisions = DecisionEngine(settings)
    backtester = Backtester(settings, database, feature_service, ml, decisions)
    backtest_report = await backtester.run(
        symbols=[_SYMBOL], max_candles=max_candles, warmup_bars=warmup_padding
    )

    replay_start_ms = int(backtest_report.start.timestamp() * 1000)
    replay_end_ms = int(backtest_report.end.timestamp() * 1000)

    # The replay must land inside the held-out test window - it must never
    # reach back into rows the model actually trained or validated on. Its
    # *tail* may extend up to `max_holding_bars` candles past the dataset's
    # own test_end_ms: those trailing raw candles have real price data but
    # no computable label (the labeler cannot simulate a trade outcome
    # without future bars to resolve it against), so they were dropped from
    # the labeled dataset entirely - never seen by training OR validation -
    # while still being genuine, freshly-available price history the
    # backtest can and should replay through, exactly like a live deployment
    # would trade the newest candle before its own eventual outcome is known.
    unlabelable_tail_ms = settings.labels.max_holding_bars * _TIMEFRAME_MS
    assert replay_start_ms >= split.test_start_ms - _TIMEFRAME_MS
    assert replay_end_ms <= split.test_end_ms + unlabelable_tail_ms + _TIMEFRAME_MS
    assert replay_start_ms > (split.validation_end_ms or 0)

    # Realistic costs are wired in end-to-end: any trade that happened must
    # carry a real fee (taker_fee default > 0).
    for trade in backtest_report.trades:
        assert trade["fees_paid"] > 0.0
