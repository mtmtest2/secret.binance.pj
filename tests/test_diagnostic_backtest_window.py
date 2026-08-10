"""Tests for the final-backtest replay window sizing.

Covers:
* ``main._final_backtest_window`` - the pure bar-count math used by
  ``TradingSystem._run_final_backtest`` to size the replay depth so it
  covers exactly the model's held-out test split (never more, never less).
* ``BacktestReport.oos_disclosure`` round-tripping through ``to_dict()``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from main import _final_backtest_window
from module_e_execution.backtester import BacktestReport


def test_final_backtest_window_covers_exactly_the_test_span_plus_warmup() -> None:
    """A 6-month test split (5m bars) should translate to the matching bar
    count, plus whatever warm-up padding the caller asked for."""
    timeframe_ms = 5 * 60 * 1_000
    test_start_ms = 0
    test_end_ms = 30 * timeframe_ms  # 30 bars wide

    bars = _final_backtest_window(
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
        timeframe_ms=timeframe_ms,
        warmup_padding=1_000,
    )

    assert bars == 30 + 1 + 1_000  # inclusive of both endpoints, plus warm-up


def test_final_backtest_window_handles_zero_width_test_split() -> None:
    bars = _final_backtest_window(
        test_start_ms=1_000, test_end_ms=1_000, timeframe_ms=300_000, warmup_padding=50
    )
    assert bars == 1 + 50


def test_backtest_report_to_dict_round_trips_oos_disclosure() -> None:
    report = BacktestReport(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 6, 1, tzinfo=timezone.utc),
        initial_equity=1_000.0,
        final_equity=1_050.0,
    )
    assert report.oos_disclosure is None
    assert report.to_dict()["oos_disclosure"] is None

    report.oos_disclosure = {"oos_fraction": 1.0}
    assert report.to_dict()["oos_disclosure"] == {"oos_fraction": 1.0}
