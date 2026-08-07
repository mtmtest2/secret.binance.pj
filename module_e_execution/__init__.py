"""Module E - Execution, Paper Trading, Backtesting & Risk Management."""

from __future__ import annotations

from module_e_execution.backtester import BacktestReport, Backtester
from module_e_execution.executor import LiveExecutor
from module_e_execution.models import (
    AccountState,
    ExecutionReport,
    Position,
    PositionStatus,
)
from module_e_execution.paper_trader import PaperTrader
from module_e_execution.risk_guard import RiskGuard, SystemState

__all__: list[str] = [
    "AccountState",
    "BacktestReport",
    "Backtester",
    "ExecutionReport",
    "LiveExecutor",
    "PaperTrader",
    "Position",
    "PositionStatus",
    "RiskGuard",
    "SystemState",
]
