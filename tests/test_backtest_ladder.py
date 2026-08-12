"""The backtester's bar mechanics once the ladder is in play."""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import Settings
from module_c_ml.schemas import TradeAction, TradeSignal
from module_e_execution.backtester import Backtester
from module_e_execution.models import CloseReason, Position, PositionStatus


def _backtester(settings: Settings) -> Backtester:
    """A backtester with no collaborators: only the bar mechanics are exercised."""
    return Backtester(settings, database=None, feature_service=None, ml_subsystem=None, decision_engine=None)  # type: ignore[arg-type]


def _position(settings: Settings, is_long: bool = True) -> Position:
    signal = TradeSignal(
        symbol="TEST/USDT:USDT",
        action=TradeAction.LONG if is_long else TradeAction.SHORT,
        reference_price=100.0,
        leverage=2,
        capital_allocation_pct=0.1,
        take_profit=103.0 if is_long else 97.0,
        stop_loss=98.0 if is_long else 102.0,
        trailing_trigger=101.5 if is_long else 98.5,
        trailing_distance_pct=0.01,
        take_profit_pct=0.03,
        stop_loss_pct=0.02,
        confidence=0.8,
    )
    position = Position.from_signal(
        signal,
        equity=1_000.0,
        entry_price=100.0,
        quantity=10.0,
        maintenance_margin_rate=0.005,
        mode="backtest",
        take_profit_config=settings.take_profit,
    )
    # Push liquidation far away so it never pre-empts the ladder in these tests.
    position.liquidation_price = 1.0 if is_long else 10_000.0
    return position


def _bar(high: float, low: float, close: float | None = None) -> pd.Series:
    return pd.Series(
        {
            "timestamp": 1_700_000_000_000,
            "open": 100.0,
            "high": high,
            "low": low,
            "close": close if close is not None else (high + low) / 2.0,
            "volume": 1_000.0,
        }
    )


def test_partial_fill_keeps_the_position_open(settings: Settings) -> None:
    """A TP1 fill banks cash but does not retire the position."""
    backtester = _backtester(settings)
    position = _position(settings)

    delta = backtester._advance_bar(position, _bar(high=101.2, low=100.4), 1)  # noqa: SLF001

    assert delta > 0.0
    assert position.status is PositionStatus.OPEN
    assert position.quantity == pytest.approx(7.0)
    assert len(position.partial_fills) == 1
    assert position.partial_fills[0]["reason"] == CloseReason.TAKE_PROFIT_1.value


def test_full_ladder_run_closes_and_books_every_leg(settings: Settings) -> None:
    """TP1 -> TP2 -> TP3 across bars leaves the position flat and profitable."""
    backtester = _backtester(settings)
    position = _position(settings)

    total = 0.0
    for high, low in ((101.2, 100.4), (102.2, 101.4), (103.2, 102.4)):
        total += backtester._advance_bar(position, _bar(high, low), 1)  # noqa: SLF001

    assert position.is_flat
    assert position.status is PositionStatus.CLOSED
    assert position.close_reason == CloseReason.TAKE_PROFIT_3.value
    assert len(position.partial_fills) == 3
    assert total > 0.0
    assert position.realized_r > 0.0


def test_a_bar_hitting_both_tp1_and_the_stop_books_the_stop(settings: Settings) -> None:
    """The backtester never assumes the favourable intrabar sequence."""
    backtester = _backtester(settings)
    position = _position(settings)

    delta = backtester._advance_bar(position, _bar(high=101.5, low=97.5), 1)  # noqa: SLF001

    assert position.is_flat
    assert position.close_reason == CloseReason.STOP_LOSS.value
    assert delta < 0.0
    assert not position.partial_fills or position.partial_fills[0]["reason"] == (
        CloseReason.STOP_LOSS.value
    )


def test_a_reversal_after_tp1_is_booked_as_a_breakeven_stop(settings: Settings) -> None:
    """Banked TP1 plus a scratch on the remainder - the ladder's design case."""
    backtester = _backtester(settings)
    position = _position(settings)

    backtester._advance_bar(position, _bar(high=101.2, low=100.6), 1)  # noqa: SLF001
    backtester._advance_bar(position, _bar(high=100.8, low=99.0), 2)  # noqa: SLF001

    assert position.is_flat
    assert position.close_reason == CloseReason.BREAKEVEN_STOP.value
    assert position.stop_protection == "BREAKEVEN"
    # The banked TP1 leg, minus costs on the scratched remainder: a far better
    # outcome than the full -1R the un-laddered position would have taken.
    assert position.realized_r > -1.0


def test_liquidation_pre_empts_the_ladder(settings: Settings) -> None:
    """A liquidated position cannot go on to take a profit."""
    backtester = _backtester(settings)
    position = _position(settings)
    position.liquidation_price = 99.0  # above the 98.0 stop, so it is hit first

    delta = backtester._advance_bar(position, _bar(high=101.5, low=98.5), 1)  # noqa: SLF001

    assert position.close_reason == CloseReason.LIQUIDATION.value
    assert position.status is PositionStatus.LIQUIDATED
    assert delta >= -position.margin


def test_short_position_mirrors_the_long_behaviour(settings: Settings) -> None:
    """The inverse logic is exercised through the same code path."""
    backtester = _backtester(settings)
    position = _position(settings, is_long=False)

    backtester._advance_bar(position, _bar(high=99.4, low=98.8), 1)  # noqa: SLF001
    assert position.quantity == pytest.approx(7.0)
    assert position.ladder is not None
    assert position.ladder.current_stop() == pytest.approx(100.0)

    backtester._advance_bar(position, _bar(high=101.0, low=98.9), 2)  # noqa: SLF001
    assert position.is_flat
    assert position.close_reason == CloseReason.BREAKEVEN_STOP.value


def test_ladder_metrics_are_reported(settings: Settings) -> None:
    """The report exposes exactly the TP/SL statistics the comparison needs."""
    backtester = _backtester(settings)
    position = _position(settings)
    for high, low in ((101.2, 100.4), (102.2, 101.4), (103.2, 102.4)):
        backtester._advance_bar(position, _bar(high, low), 1)  # noqa: SLF001

    metrics = backtester._ladder_metrics([position.to_row()])  # noqa: SLF001
    assert metrics["pct_reached_tp1"] == pytest.approx(1.0)
    assert metrics["pct_reached_tp2"] == pytest.approx(1.0)
    assert metrics["pct_reached_tp3"] == pytest.approx(1.0)
    assert metrics["pct_stopped_breakeven"] == pytest.approx(0.0)
    assert metrics["average_r"] > 0.0
    assert "average_trade_return" in metrics


def test_side_metrics_split_long_from_short(settings: Settings) -> None:
    """LONG and SHORT performance are reported separately."""
    backtester = _backtester(settings)
    rows = [
        {"side": "LONG", "realized_pnl": 5.0},
        {"side": "LONG", "realized_pnl": -2.0},
        {"side": "SHORT", "realized_pnl": -1.0},
    ]
    metrics = backtester._side_metrics(rows)  # noqa: SLF001

    assert metrics["long_trades"] == 2
    assert metrics["long_win_rate"] == pytest.approx(0.5)
    assert metrics["long_net_profit"] == pytest.approx(3.0)
    assert metrics["short_trades"] == 1
    assert metrics["short_win_rate"] == pytest.approx(0.0)
    assert metrics["short_net_profit"] == pytest.approx(-1.0)


@pytest.fixture(name="settings")
def _settings_fixture() -> Settings:
    """Default settings - the ladder defaults are what these tests pin."""
    return Settings()


def test_scaled_out_trade_is_recorded_at_its_full_opened_size(settings: Settings) -> None:
    """``quantity`` on the persisted row is the size actually traded.

    The live ``quantity`` shrinks as ladder legs fill, so reading it for the
    trade record would persist a fully scaled-out trade as zero-size with zero
    notional - the position would look like it never happened.
    """
    backtester = _backtester(settings)
    position = _position(settings)
    opened_notional = position.notional

    for high, low in ((101.2, 100.4), (102.2, 101.4), (103.2, 102.4)):
        backtester._advance_bar(position, _bar(high, low), 1)  # noqa: SLF001

    assert position.is_flat            # nothing left open ...
    assert position.quantity == pytest.approx(0.0)
    row = position.to_row()
    assert row["quantity"] == pytest.approx(10.0)   # ... but the row is full size
    assert row["notional"] == pytest.approx(opened_notional)
    assert position.max_favorable_excursion > 0.0


def test_partially_scaled_trade_still_reports_full_size(settings: Settings) -> None:
    """Half-laddered positions report the opened size too, not the remainder."""
    backtester = _backtester(settings)
    position = _position(settings)
    backtester._advance_bar(position, _bar(101.2, 100.4), 1)  # noqa: SLF001

    assert position.quantity == pytest.approx(7.0)
    assert position.to_row()["quantity"] == pytest.approx(10.0)
