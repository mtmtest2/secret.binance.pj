"""Tests for the diagnostic backtest window sizing and its OOS disclosure.

Covers:
* ``main._diagnostic_backtest_window`` - the pure per-symbol bar-count and
  out-of-sample/in-sample split math used by
  ``TradingSystem._run_validation_backtest`` (see task 2, "decouple the
  diagnostic backtest window from validation_fraction").
* ``BacktestReport.oos_disclosure`` round-tripping through ``to_dict()``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from main import _diagnostic_backtest_window
from module_e_execution.backtester import BacktestReport


def test_diagnostic_window_full_overlap_when_target_exceeds_oos_tail() -> None:
    """A full-year diagnostic window against a much shorter OOS tail must be
    disclosed as mostly in-sample, not silently presented as a clean holdout.
    """
    diagnostic_bars_per_symbol, disclosure = _diagnostic_backtest_window(
        oos_validation_rows=10_000,  # e.g. ~0.3y OOS tail spread over 10 symbols
        diagnostic_backtest_bars=105_120,  # full year target
        symbol_count=10,
    )

    assert diagnostic_bars_per_symbol == 10_512
    assert disclosure["out_of_sample_bars_per_symbol"] == 1_000
    assert disclosure["in_sample_bars_per_symbol"] == 9_512
    assert 0.0 < disclosure["oos_fraction"] < 1.0
    assert "in_sample_bars_per_symbol" in disclosure["note"]


def test_diagnostic_window_fully_oos_when_target_is_smaller_than_validation_tail() -> None:
    """When the diagnostic target window is smaller than (or equal to) the
    genuinely out-of-sample tail, the whole replay is a clean holdout.
    """
    diagnostic_bars_per_symbol, disclosure = _diagnostic_backtest_window(
        oos_validation_rows=200_000,
        diagnostic_backtest_bars=1_000,
        symbol_count=10,
    )

    assert diagnostic_bars_per_symbol == 100
    assert disclosure["out_of_sample_bars_per_symbol"] == 100
    assert disclosure["in_sample_bars_per_symbol"] == 0
    assert disclosure["oos_fraction"] == 1.0


def test_diagnostic_window_rejects_non_positive_symbol_count() -> None:
    try:
        _diagnostic_backtest_window(
            oos_validation_rows=100, diagnostic_backtest_bars=100, symbol_count=0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for symbol_count=0")


def test_backtest_report_to_dict_round_trips_oos_disclosure() -> None:
    report = BacktestReport(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 6, 1, tzinfo=timezone.utc),
        initial_equity=1_000.0,
        final_equity=1_050.0,
    )
    assert report.oos_disclosure is None
    assert report.to_dict()["oos_disclosure"] is None

    report.oos_disclosure = {"oos_fraction": 0.42}
    assert report.to_dict()["oos_disclosure"] == {"oos_fraction": 0.42}
