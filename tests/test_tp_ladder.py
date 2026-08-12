"""The three-stage take-profit ladder and its stage-driven stop loss."""

from __future__ import annotations

import pytest

from config.settings import Settings, TakeProfitSettings
from module_e_execution.models import CloseReason, Position
from module_e_execution.tp_ladder import LadderStage, StopProtection, TakeProfitLadder


def build_ladder(is_long: bool = True, config: TakeProfitSettings | None = None) -> TakeProfitLadder:
    """The specification's worked example: entry 100, SL 98, TP 101/102/103."""
    return TakeProfitLadder.build(
        is_long=is_long,
        entry_price=100.0,
        take_profit_pct=0.03,
        stop_loss_pct=0.02,
        config=config or TakeProfitSettings(),
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_worked_example_geometry() -> None:
    """A long at 100 with a 3 % target and a 2 % stop gives 101/102/103 and 98."""
    ladder = build_ladder()
    assert ladder.tp_prices[0] == pytest.approx(101.0)
    assert ladder.tp_prices[1] == pytest.approx(102.0)
    assert ladder.tp_prices[2] == pytest.approx(103.0)
    assert ladder.initial_stop == pytest.approx(98.0)
    assert ladder.current_stop() == pytest.approx(98.0)
    assert ladder.protection is StopProtection.INITIAL


def test_short_geometry_is_the_exact_inverse() -> None:
    """A short at 100 gives 99/98/97 and a stop at 102."""
    ladder = build_ladder(is_long=False)
    assert ladder.tp_prices[0] == pytest.approx(99.0)
    assert ladder.tp_prices[1] == pytest.approx(98.0)
    assert ladder.tp_prices[2] == pytest.approx(97.0)
    assert ladder.initial_stop == pytest.approx(102.0)


# ---------------------------------------------------------------------------
# Stage transitions
# ---------------------------------------------------------------------------
def test_tp1_moves_the_stop_to_breakeven_without_closing_the_position() -> None:
    """TP1 scales out 30 % and protects the rest at entry."""
    ladder = build_ladder()
    events = ladder.on_tick(101.0)

    assert [event.kind for event in events] == ["TAKE_PROFIT_1"]
    assert events[0].fraction == pytest.approx(0.30)
    assert ladder.stage == int(LadderStage.TP1_FILLED)
    assert ladder.remaining_fraction == pytest.approx(0.70)
    assert not ladder.is_closed
    assert ladder.current_stop() == pytest.approx(100.0)
    assert ladder.protection is StopProtection.BREAKEVEN


def test_tp2_moves_the_stop_to_tp1() -> None:
    """TP2 takes another 30 % and locks the TP1 price in as the stop."""
    ladder = build_ladder()
    ladder.on_tick(101.0)
    events = ladder.on_tick(102.0)

    assert [event.kind for event in events] == ["TAKE_PROFIT_2"]
    assert ladder.stage == int(LadderStage.TP2_FILLED)
    assert ladder.remaining_fraction == pytest.approx(0.40)
    assert ladder.current_stop() == pytest.approx(101.0)
    assert ladder.protection is StopProtection.TP1_LOCKED


def test_tp3_closes_the_remainder() -> None:
    """TP3 takes everything that is left and retires the ladder."""
    ladder = build_ladder()
    for price in (101.0, 102.0, 103.0):
        ladder.on_tick(price)

    assert ladder.is_closed
    assert ladder.remaining_fraction == pytest.approx(0.0)
    assert [leg["level"] for leg in ladder.filled] == [1, 2, 3]
    assert sum(leg["fraction"] for leg in ladder.filled) == pytest.approx(1.0)


def test_short_ladder_transitions_are_inverted() -> None:
    """The same sequence, mirrored, for a short."""
    ladder = build_ladder(is_long=False)
    assert [event.kind for event in ladder.on_tick(99.0)] == ["TAKE_PROFIT_1"]
    assert ladder.current_stop() == pytest.approx(100.0)
    assert [event.kind for event in ladder.on_tick(98.0)] == ["TAKE_PROFIT_2"]
    assert ladder.current_stop() == pytest.approx(99.0)
    ladder.on_tick(97.0)
    assert ladder.is_closed


def test_allocations_are_configurable() -> None:
    """The 30/30/40 default is a default, not a hard-coded rule."""
    config = TakeProfitSettings(close_fractions=(0.5, 0.25, 0.25))
    ladder = build_ladder(config=config)
    assert ladder.on_tick(101.0)[0].fraction == pytest.approx(0.50)
    assert ladder.on_tick(102.0)[0].fraction == pytest.approx(0.25)
    assert ladder.on_tick(103.0)[0].fraction == pytest.approx(0.25)


def test_level_fractions_are_configurable() -> None:
    """The ladder spacing is configurable too, and TP3 stays the model's target."""
    config = TakeProfitSettings(level_fractions=(0.25, 0.5, 1.0))
    ladder = build_ladder(config=config)
    assert ladder.tp_prices == pytest.approx((100.75, 101.5, 103.0))


def test_dust_legs_are_merged_forward() -> None:
    """A leg below ``min_leg_fraction`` folds into the next one, total preserved."""
    config = TakeProfitSettings(close_fractions=(0.02, 0.28, 0.70), min_leg_fraction=0.05)
    ladder = build_ladder(config=config)
    assert ladder.close_fractions[0] == pytest.approx(0.0)
    assert ladder.close_fractions[1] == pytest.approx(0.30)
    assert sum(ladder.close_fractions) == pytest.approx(1.0)


def test_configuration_rejects_allocations_that_do_not_sum_to_one() -> None:
    """A ladder that leaves part of the position unmanaged is a config error."""
    with pytest.raises(ValueError):
        TakeProfitSettings(close_fractions=(0.3, 0.3, 0.3))
    with pytest.raises(ValueError):
        TakeProfitSettings(level_fractions=(0.5, 0.4, 1.0))
    with pytest.raises(ValueError):
        TakeProfitSettings(level_fractions=(0.3, 0.6, 0.9))


# ---------------------------------------------------------------------------
# Intrabar ordering - the conservative rule
# ---------------------------------------------------------------------------
def test_a_bar_touching_both_tp1_and_the_stop_is_booked_as_a_stop() -> None:
    """No favourable ordering is ever assumed from OHLC alone."""
    ladder = build_ladder()
    events = ladder.on_bar(high=101.5, low=97.5)

    assert len(events) == 1
    assert events[0].is_stop
    assert events[0].protection is StopProtection.INITIAL
    assert events[0].price == pytest.approx(98.0)
    assert events[0].fraction == pytest.approx(1.0)
    assert ladder.is_closed


def test_a_bar_that_runs_to_tp2_and_reverses_gives_back_the_locked_stop() -> None:
    """The tightened stop is re-tested against the *same* bar, not the next one."""
    ladder = build_ladder()
    events = ladder.on_bar(high=102.5, low=100.5)

    kinds = [event.kind for event in events]
    assert kinds == ["TAKE_PROFIT_1", "TAKE_PROFIT_2", "STOP"]
    # 30 % at 101, 30 % at 102, and the remaining 40 % given back at the
    # TP1-locked stop of 101 - inside the very bar that reached TP2.
    assert events[-1].protection is StopProtection.TP1_LOCKED
    assert events[-1].price == pytest.approx(101.0)
    assert events[-1].fraction == pytest.approx(0.40)
    assert ladder.is_closed


def test_a_bar_reaching_tp1_and_holding_does_not_stop_out() -> None:
    """Breakeven protection only fires if the bar actually traded back to entry."""
    ladder = build_ladder()
    events = ladder.on_bar(high=101.5, low=100.5)

    assert [event.kind for event in events] == ["TAKE_PROFIT_1"]
    assert not ladder.is_closed
    assert ladder.remaining_fraction == pytest.approx(0.70)


def test_a_bar_reaching_tp1_then_returning_to_entry_stops_at_breakeven() -> None:
    """TP1 banked, remainder scratched at entry - the ladder's intended worst case."""
    ladder = build_ladder()
    events = ladder.on_bar(high=101.5, low=99.5)

    assert [event.kind for event in events] == ["TAKE_PROFIT_1", "STOP"]
    assert events[-1].protection is StopProtection.BREAKEVEN
    assert events[-1].price == pytest.approx(100.0)


def test_levels_fill_in_order_within_one_bar() -> None:
    """A bar spanning all three levels fills TP1, TP2 then TP3 - never out of order."""
    ladder = build_ladder()
    events = ladder.on_bar(high=104.0, low=99.5)

    assert [event.kind for event in events] == [
        "TAKE_PROFIT_1",
        "TAKE_PROFIT_2",
        "TAKE_PROFIT_3",
    ]
    assert ladder.is_closed


def test_short_bar_ordering_is_inverted() -> None:
    """For a short the adverse extreme is the high, and it is still tested first."""
    ladder = build_ladder(is_long=False)
    events = ladder.on_bar(high=102.5, low=98.5)

    assert len(events) == 1
    assert events[0].is_stop
    assert events[0].price == pytest.approx(102.0)


def test_reached_flags_report_fills_not_stage() -> None:
    """A stop-out must not be reported as having reached TP1 and TP2.

    Stopping also advances the stage to CLOSED, so deriving the "reached"
    statistics from the stage would report 100 % TP1 attainment on a strategy
    that never took a single profit.
    """
    stopped = build_ladder()
    stopped.on_bar(high=100.5, low=97.5)
    snapshot = stopped.to_dict()
    assert snapshot["reached_tp1"] is False
    assert snapshot["reached_tp2"] is False
    assert snapshot["reached_tp3"] is False

    partial = build_ladder()
    partial.on_bar(high=101.2, low=100.5)
    partial.on_bar(high=100.6, low=99.0)
    snapshot = partial.to_dict()
    assert snapshot["reached_tp1"] is True
    assert snapshot["reached_tp2"] is False
    assert snapshot["protection"] == "BREAKEVEN"


def test_a_closed_ladder_never_fires_again() -> None:
    """Idempotence: further bars on a finished ladder produce nothing."""
    ladder = build_ladder()
    ladder.on_bar(high=104.0, low=99.5)
    assert ladder.on_bar(high=110.0, low=90.0) == []
    assert ladder.on_tick(110.0) == []


# ---------------------------------------------------------------------------
# Position integration
# ---------------------------------------------------------------------------
def _position(settings: Settings) -> Position:
    from module_c_ml.schemas import TradeAction, TradeSignal

    signal = TradeSignal(
        symbol="TEST/USDT:USDT",
        action=TradeAction.LONG,
        reference_price=100.0,
        leverage=2,
        capital_allocation_pct=0.1,
        take_profit=103.0,
        stop_loss=98.0,
        trailing_trigger=101.5,
        trailing_distance_pct=0.01,
        take_profit_pct=0.03,
        stop_loss_pct=0.02,
        confidence=0.8,
    )
    return Position.from_signal(
        signal,
        equity=1_000.0,
        entry_price=100.0,
        quantity=10.0,
        maintenance_margin_rate=0.005,
        mode="backtest",
        take_profit_config=settings.take_profit,
    )


def test_partial_closes_reduce_the_position_and_bank_profit(settings: Settings) -> None:
    """Each leg shrinks the position and adds its own PnL, fees included."""
    position = _position(settings)
    assert position.initial_quantity == pytest.approx(10.0)

    events = position.ladder.on_tick(101.0) if position.ladder else []
    delta, reason = position.apply_ladder_event(events[0], exit_price=101.0, fee_rate=0.0005)

    assert reason is CloseReason.TAKE_PROFIT_1
    assert position.quantity == pytest.approx(7.0)
    # 3 units * 1.0 profit, less the exit fee on 3 * 101.
    assert delta == pytest.approx(3.0 - 101.0 * 3.0 * 0.0005)
    assert position.realized_pnl == pytest.approx(delta)
    assert len(position.partial_fills) == 1


def test_effective_stop_takes_the_most_protective_level(settings: Settings) -> None:
    """A ladder stop is never loosened by the trailing stop, or vice versa."""
    position = _position(settings)
    assert position.effective_stop() == pytest.approx(98.0)

    position.ladder.on_tick(101.0)
    assert position.effective_stop() == pytest.approx(100.0)

    # A trailing stop below the ladder's breakeven must not win.
    position.trailing_active = True
    position.trailing_stop = 99.0
    assert position.effective_stop() == pytest.approx(100.0)

    # A tighter trailing stop does.
    position.trailing_stop = 100.5
    assert position.effective_stop() == pytest.approx(100.5)


def test_stop_reasons_distinguish_breakeven_from_a_real_loss(settings: Settings) -> None:
    """"Stopped at breakeven" is reported as its own outcome, not as STOP_LOSS."""
    position = _position(settings)
    assert position.stop_close_reason() is CloseReason.STOP_LOSS

    position.ladder.on_tick(101.0)
    assert position.stop_close_reason() is CloseReason.BREAKEVEN_STOP

    position.ladder.on_tick(102.0)
    assert position.stop_close_reason() is CloseReason.PROFIT_STOP


def test_realized_r_is_measured_against_the_original_risk(settings: Settings) -> None:
    """R is anchored to the *initial* stop, not to wherever the stop ended up."""
    position = _position(settings)
    assert position.risk_per_unit == pytest.approx(2.0)

    position.realized_pnl = 20.0  # 10 units * 2.0 of risk == 1R
    assert position.realized_r == pytest.approx(1.0)


def test_disabling_the_ladder_restores_the_single_target(settings: Settings) -> None:
    """The whole mechanism can be switched off end to end."""
    settings.take_profit.enabled = False
    position = _position(settings)
    assert position.ladder is None
    assert position.effective_stop() == pytest.approx(98.0)
    assert position.stop_close_reason() is CloseReason.STOP_LOSS
