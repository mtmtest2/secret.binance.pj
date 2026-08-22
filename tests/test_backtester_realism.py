"""Direct tests of the backtester's realism guarantees (module_e_execution/backtester.py).

These pin down the two properties the final held-out backtest depends on to
be a trustworthy "simulate real deployment" estimate rather than an
optimistic fantasy:

* **Costs are real.** Every fill pays taker fees on both legs and slippage
  in the adverse direction (never favourable) - see ``Backtester._fill`` /
  ``Backtester._book_close``.
* **Fills happen strictly after the decision.** A signal raised from bar
  ``t``'s close can only fill at bar ``t+1``'s open - never at the close
  that produced it - which is what ``Backtester._simulate`` enforces by
  queuing a signal in ``pending`` and only filling it on the *next* loop
  iteration. This file does not re-derive that from ``_simulate`` (which
  needs a full ML/decision stack to drive); it is verified by inspection in
  the module docstring and the ordering of steps 1-3 in ``_simulate``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import Settings
from module_c_ml.schemas import TradeAction, TradeSignal
from module_e_execution.backtester import Backtester
from module_e_execution.models import CloseReason, Position


def _backtester(settings: Settings) -> Backtester:
    """A Backtester with no DB/ML/decision wiring - enough to exercise the
    pure fill/close arithmetic, which only reads ``self._config``."""
    return Backtester(settings, database=None, feature_service=None, ml_subsystem=None, decision_engine=None)  # type: ignore[arg-type]


def _signal(action: TradeAction, reference_price: float = 100.0) -> TradeSignal:
    if action is TradeAction.LONG:
        take_profit, stop_loss, trailing_trigger = 104.0, 98.0, 102.0
    else:
        take_profit, stop_loss, trailing_trigger = 96.0, 102.0, 98.0
    return TradeSignal(
        symbol="BTC/USDT:USDT",
        action=action,
        reference_price=reference_price,
        leverage=2,
        capital_allocation_pct=0.10,
        take_profit=take_profit,
        stop_loss=stop_loss,
        trailing_trigger=trailing_trigger,
        trailing_distance_pct=0.01,
        take_profit_pct=0.04,
        stop_loss_pct=0.02,
        confidence=0.8,
    )


class TestFillAppliesSlippageAndFees:
    def test_long_fill_pays_slippage_upward_and_a_taker_entry_fee(self) -> None:
        settings = Settings(execution={"slippage_bps": 10.0, "taker_fee": 0.0004})
        backtester = _backtester(settings)
        signal = _signal(TradeAction.LONG, reference_price=100.0)
        bar_open = 100.0
        row = pd.Series({"open": bar_open, "high": 101.0, "low": 99.0, "close": 100.5, "timestamp": 0})
        equity = 1_000.0

        position = backtester._fill(signal, row, equity)

        assert position is not None
        # Slippage always moves the fill *against* the trader: a long buys higher.
        expected_fill_price = bar_open * (1.0 + 10.0 / 10_000.0)
        assert position.entry_price == pytest.approx(expected_fill_price)
        assert position.entry_price > bar_open

        expected_quantity = (signal.margin_for(equity) * signal.leverage) / expected_fill_price
        assert position.quantity == pytest.approx(expected_quantity)

        expected_fee = expected_fill_price * expected_quantity * 0.0004
        assert position.fees_paid == pytest.approx(expected_fee)
        assert position.fees_paid > 0.0

    def test_short_fill_pays_slippage_downward(self) -> None:
        settings = Settings(execution={"slippage_bps": 10.0, "taker_fee": 0.0004})
        backtester = _backtester(settings)
        signal = _signal(TradeAction.SHORT, reference_price=100.0)
        bar_open = 100.0
        row = pd.Series({"open": bar_open, "high": 101.0, "low": 99.0, "close": 100.5, "timestamp": 0})

        position = backtester._fill(signal, row, 1_000.0)

        assert position is not None
        expected_fill_price = bar_open * (1.0 - 10.0 / 10_000.0)
        assert position.entry_price == pytest.approx(expected_fill_price)
        assert position.entry_price < bar_open

    def test_zero_slippage_and_fee_config_reproduces_the_raw_open_price(self) -> None:
        settings = Settings(execution={"slippage_bps": 0.0, "taker_fee": 0.0})
        backtester = _backtester(settings)
        signal = _signal(TradeAction.LONG, reference_price=100.0)
        row = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "timestamp": 0})

        position = backtester._fill(signal, row, 1_000.0)

        assert position is not None
        assert position.entry_price == pytest.approx(100.0)
        assert position.fees_paid == pytest.approx(0.0)


class TestBookCloseAppliesExitSlippageFeeAndFunding:
    def test_take_profit_close_pays_adverse_slippage_and_exit_fee(self) -> None:
        settings = Settings(execution={"slippage_bps": 10.0, "taker_fee": 0.0004})
        backtester = _backtester(settings)
        signal = _signal(TradeAction.LONG, reference_price=100.0)
        position = Position.from_signal(
            signal,
            equity=1_000.0,
            entry_price=100.0,
            quantity=2.0,
            maintenance_margin_rate=settings.execution.maintenance_margin_rate,
            mode="backtest",
        )
        position.funding_paid = 0.05

        raw_exit_price = 104.0  # the take-profit level
        delta = backtester._book_close(position, raw_exit_price, CloseReason.TAKE_PROFIT, timestamp=1)

        # A long's exit slippage always shaves the exit price down (adverse).
        expected_exit_price = raw_exit_price * (1.0 - 10.0 / 10_000.0)
        assert position.exit_price == pytest.approx(expected_exit_price)
        assert position.exit_price < raw_exit_price

        expected_exit_fee = expected_exit_price * position.quantity * 0.0004
        expected_gross = (expected_exit_price - position.entry_price) * position.quantity * 1
        expected_delta = expected_gross - expected_exit_fee - position.funding_paid
        assert delta == pytest.approx(expected_delta)
        assert position.realized_pnl == pytest.approx(expected_delta)

    def test_liquidation_close_applies_slippage_and_still_caps_the_loss(self) -> None:
        settings = Settings(execution={"slippage_bps": 10.0, "taker_fee": 0.0004})
        backtester = _backtester(settings)
        signal = _signal(TradeAction.LONG, reference_price=100.0)
        position = Position.from_signal(
            signal,
            equity=1_000.0,
            entry_price=100.0,
            quantity=2.0,
            maintenance_margin_rate=settings.execution.maintenance_margin_rate,
            mode="backtest",
        )

        liquidation_price = 90.0
        delta = backtester._book_close(position, liquidation_price, CloseReason.LIQUIDATION, timestamp=1)

        # A forced close during the move that triggered it is the fill most
        # likely to be *worse* than its trigger price. Exempting liquidation
        # from slippage - as this path used to - modelled it as the single best
        # fill in the system, understating tail risk on the trades that matter
        # most. It now pays the same penalty as any other adverse exit.
        expected_exit_price = liquidation_price * (1.0 - 10.0 / 10_000.0)
        assert position.exit_price == pytest.approx(expected_exit_price)
        assert position.exit_price < liquidation_price
        # The loss is still bounded at the committed margin, never more - the
        # isolated-margin cap is applied after slippage, not instead of it.
        assert delta >= -position.margin - 1e-9

    def test_zero_cost_config_reproduces_pure_price_pnl(self) -> None:
        settings = Settings(execution={"slippage_bps": 0.0, "taker_fee": 0.0})
        backtester = _backtester(settings)
        signal = _signal(TradeAction.LONG, reference_price=100.0)
        position = Position.from_signal(
            signal,
            equity=1_000.0,
            entry_price=100.0,
            quantity=2.0,
            maintenance_margin_rate=settings.execution.maintenance_margin_rate,
            mode="backtest",
        )

        delta = backtester._book_close(position, 104.0, CloseReason.TAKE_PROFIT, timestamp=1)

        assert delta == pytest.approx((104.0 - 100.0) * 2.0)
