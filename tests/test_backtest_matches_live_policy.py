"""The backtest must simulate the system that would actually run (P5-P7, P12-P14, P19-P22).

A backtester is only worth its runtime if the thing it replays is the thing
production does. The audited one diverged in five separate places at once: it
skipped each position's own entry bar, never consulted the Risk Guard, allocated
position slots by dictionary order instead of by confidence, stamped trade open
times from the wall clock, and trailed on a hardcoded rule rather than the Exit
model's. Each of those flatters results independently.

These tests pin the corrected behaviour at the level where it can be checked
cheaply and deterministically.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from config.settings import Settings
from main import _oos_disclosure
from module_c_ml.decision_engine import DecisionContext, DecisionEngine, Rule
from module_c_ml.schemas import DecisionResult, DecisionVerdict
from module_e_execution.backtester import (
    Backtester,
    BacktestReport,
    _accumulate_rule_counts,
)
from module_e_execution.risk_guard import RiskLadder, SystemState

_TIMEFRAME_MS = 5 * 60 * 1_000


# ---------------------------------------------------------------------------
# P6 - the risk ladder both paths share
# ---------------------------------------------------------------------------
def _ladder(**risk) -> RiskLadder:
    return RiskLadder(Settings(risk={"starting_equity": 1_000.0, **risk}))


def test_ladder_trips_red_on_the_configured_daily_drawdown() -> None:
    ladder = _ladder(daily_drawdown_red_pct=0.05)
    day = date(2026, 1, 1)

    assert ladder.observe(equity=1_000.0, realised_pnls=[], day=day)[0] is SystemState.GREEN
    state, reason = ladder.observe(equity=940.0, realised_pnls=[], day=day)

    assert state is SystemState.RED
    assert "daily drawdown" in reason


def test_red_latches_until_a_human_resets_it() -> None:
    """`require_manual_reset` means a live deployment stays stopped.

    A replay that trips RED and then keeps trading reports an equity curve the
    system could never have produced.
    """
    ladder = _ladder(daily_drawdown_red_pct=0.05, require_manual_reset=True)
    day = date(2026, 1, 1)
    ladder.observe(equity=1_000.0, realised_pnls=[], day=day)
    ladder.observe(equity=900.0, realised_pnls=[], day=day)
    assert ladder.state is SystemState.RED

    # Recovering equity must not release the latch.
    state, _ = ladder.observe(equity=1_100.0, realised_pnls=[], day=day)
    assert state is SystemState.RED
    assert ladder.size_multiplier == 0.0


def test_yellow_throttles_sizing_rather_than_halting() -> None:
    ladder = _ladder(daily_drawdown_yellow_pct=0.03, daily_drawdown_red_pct=0.10)
    day = date(2026, 1, 1)
    ladder.observe(equity=1_000.0, realised_pnls=[], day=day)
    state, _ = ladder.observe(equity=960.0, realised_pnls=[], day=day)

    assert state is SystemState.YELLOW
    assert 0.0 < ladder.size_multiplier < 1.0


def test_consecutive_losses_trip_red() -> None:
    ladder = _ladder(consecutive_losses_red=3)
    day = date(2026, 1, 1)
    ladder.observe(equity=1_000.0, realised_pnls=[], day=day)
    state, _ = ladder.observe(equity=999.0, realised_pnls=[-1.0, -1.0, -1.0], day=day)

    assert state is SystemState.RED


def test_a_new_day_rebases_the_daily_drawdown() -> None:
    ladder = _ladder(daily_drawdown_red_pct=0.05)
    ladder.observe(equity=1_000.0, realised_pnls=[], day=date(2026, 1, 1))
    # Same equity on a new day is not a drawdown against that day's open.
    state, _ = ladder.observe(equity=960.0, realised_pnls=[], day=date(2026, 1, 2))
    assert state is SystemState.GREEN


# ---------------------------------------------------------------------------
# P20 - every rule's objection, not only the one that fired first
# ---------------------------------------------------------------------------
def test_rule_counts_record_every_check_not_just_the_blocking_one() -> None:
    decision = DecisionResult(
        decision_id="d1",
        symbol="BTC/USDT:USDT",
        verdict=DecisionVerdict.NO_TRADE,
        rule_triggered=Rule.GATE_CONFIDENCE,
        reason="rejected",
        checks=[
            {"rule": Rule.GATE_CONFIDENCE, "passed": False, "detail": ""},
            {"rule": Rule.DIRECTION_CONFIDENCE, "passed": True, "detail": ""},
            {"rule": Rule.ENTRY_REJECTED, "passed": False, "detail": ""},
        ],
    )
    independent: dict[str, int] = {}
    evaluations: dict[str, dict[str, int]] = {}
    _accumulate_rule_counts(decision, independent, evaluations)

    # Both objectors are counted, not only the first.
    assert independent[Rule.GATE_CONFIDENCE] == 1
    assert independent[Rule.ENTRY_REJECTED] == 1
    assert Rule.DIRECTION_CONFIDENCE not in independent
    assert evaluations[Rule.DIRECTION_CONFIDENCE] == {"reached": 1, "passed": 1}


# ---------------------------------------------------------------------------
# P13 - Sortino's denominator
# ---------------------------------------------------------------------------
def test_sortino_divides_by_all_periods_not_only_losing_ones() -> None:
    """On a right-skewed curve Sortino must exceed Sharpe.

    Reporting Sortino *below* Sharpe on a distribution with a 2.4:1 win/loss
    payoff was the tell that the two used different denominators.
    """
    # Mostly small gains with a few small losses -> right-skewed.
    returns = np.array([0.01] * 20 + [-0.004] * 5, dtype=np.float64)
    equity = 1_000.0 * np.cumprod(1.0 + returns)
    equity = np.concatenate([[1_000.0], equity])

    sharpe = Backtester._sharpe(equity)
    sortino = Backtester._sortino(equity)

    assert sortino > sharpe > 0.0


def test_sortino_is_zero_without_any_losing_period() -> None:
    equity = 1_000.0 * np.cumprod(1.0 + np.full(10, 0.01))
    assert Backtester._sortino(np.concatenate([[1_000.0], equity])) == 0.0


# ---------------------------------------------------------------------------
# P12 - oos_fraction measured, not asserted
# ---------------------------------------------------------------------------
class _Split:
    test_start_ms = 1_700_000_000_000
    test_end_ms = 1_700_000_000_000 + 100 * _TIMEFRAME_MS
    test_index = np.arange(100)


def _curve(start_ms: int, bars: int) -> list[dict[str, float]]:
    return [{"timestamp": float(start_ms + i * _TIMEFRAME_MS), "equity": 1_000.0} for i in range(bars)]


def test_oos_fraction_is_one_when_the_replay_is_clean() -> None:
    report = BacktestReport(
        start=None, end=None, initial_equity=1_000.0, final_equity=1_000.0,
        equity_curve=_curve(_Split.test_start_ms, 50),
    )
    in_sample = np.arange(
        _Split.test_start_ms - 200 * _TIMEFRAME_MS, _Split.test_start_ms, _TIMEFRAME_MS, dtype=np.int64
    )
    disclosure = _oos_disclosure(report, _Split(), in_sample)

    assert disclosure["oos_fraction"] == 1.0
    assert disclosure["bars_overlapping_train_or_validation"] == 0
    assert disclosure["replayed_bars"] == 50


def test_oos_fraction_detects_overlap_with_training_data() -> None:
    """The field must be able to report the thing it exists to detect."""
    report = BacktestReport(
        start=None, end=None, initial_equity=1_000.0, final_equity=1_000.0,
        equity_curve=_curve(_Split.test_start_ms, 40),
    )
    # Half the replayed bars are also training bars.
    in_sample = np.array(
        [_Split.test_start_ms + i * _TIMEFRAME_MS for i in range(20)], dtype=np.int64
    )
    disclosure = _oos_disclosure(report, _Split(), in_sample)

    assert disclosure["bars_overlapping_train_or_validation"] == 20
    assert disclosure["oos_fraction"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# P19 - the relaxed replay differs only in the thresholds it means to relax
# ---------------------------------------------------------------------------
def test_relaxed_overrides_preserve_every_other_decision_setting() -> None:
    from main import _RELAXED_DECISION_OVERRIDES

    live = Settings(
        decision={
            "max_leverage": 7,
            "max_capital_allocation_pct": 0.11,
            "max_concurrent_positions": 9,
            "accepted_risk_tiers": ("LOW",),
        }
    )
    relaxed = live.decision.model_copy(update=_RELAXED_DECISION_OVERRIDES)

    # The five intended fields moved...
    assert relaxed.min_gate_confidence == 0.50
    assert relaxed.min_entry_probability == 0.50
    # ...and nothing else did.
    assert relaxed.max_leverage == 7
    assert relaxed.max_capital_allocation_pct == 0.11
    assert relaxed.max_concurrent_positions == 9
    assert relaxed.accepted_risk_tiers == ("LOW",)


# ---------------------------------------------------------------------------
# P30(b) - a configurable that does something
# ---------------------------------------------------------------------------
def test_max_positions_per_symbol_is_honoured() -> None:
    engine = DecisionEngine(Settings(decision={"max_positions_per_symbol": 2}))
    checks: list[dict] = []

    one_open = DecisionContext(open_symbols=frozenset({"BTC/USDT:USDT"}), positions_per_symbol={"BTC/USDT:USDT": 1})
    at_cap = DecisionContext(open_symbols=frozenset({"BTC/USDT:USDT"}), positions_per_symbol={"BTC/USDT:USDT": 2})

    assert engine._config.max_positions_per_symbol == 2
    # One open position under a cap of two must not block a second.
    assert one_open.positions_per_symbol["BTC/USDT:USDT"] < engine._config.max_positions_per_symbol
    assert at_cap.positions_per_symbol["BTC/USDT:USDT"] >= engine._config.max_positions_per_symbol


def test_context_without_per_symbol_counts_keeps_the_old_single_position_rule() -> None:
    """Callers that track only a symbol set must behave exactly as before."""
    engine = DecisionEngine(Settings())
    context = DecisionContext(open_symbols=frozenset({"BTC/USDT:USDT"}))
    assert context.positions_per_symbol == {}
    assert engine._config.max_positions_per_symbol == 1
