"""Module A - Data Ingestion & Quality Control.

Public surface:

* :class:`~module_a_data.models.OHLCVCandle` and friends - strict Pydantic schemas.
* :class:`~module_a_data.fetcher.BinanceDataFetcher` - async ccxt data acquisition.
* :class:`~module_a_data.qc_validator.QCValidator` - the QC gatekeeper.
* :class:`~module_a_data.db_handler.DatabaseHandler` - async SQLite persistence.
* :class:`~module_a_data.pipeline.DataPipeline` - fetch -> validate -> heal -> store.
"""

from __future__ import annotations

from module_a_data.db_handler import DatabaseHandler
from module_a_data.fetcher import BinanceDataFetcher
from module_a_data.models import (
    FuturesMetrics,
    MarketDataBundle,
    OHLCVCandle,
    OrderBookSnapshot,
    QCIssue,
    QCReport,
    QCSeverity,
)
from module_a_data.pipeline import DataPipeline
from module_a_data.qc_validator import QCValidator
from module_a_data.universe import SymbolCandidate, UniverseManager

__all__: list[str] = [
    "BinanceDataFetcher",
    "DataPipeline",
    "DatabaseHandler",
    "FuturesMetrics",
    "MarketDataBundle",
    "OHLCVCandle",
    "OrderBookSnapshot",
    "QCIssue",
    "QCReport",
    "QCSeverity",
    "QCValidator",
    "SymbolCandidate",
    "UniverseManager",
]
