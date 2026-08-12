"""Symbols with corrupted/incomplete stored candles must not reach the ML pipeline.

Covers module_a_data/qc_validator.py::QCValidator.validate_stored_frame and its
wiring into module_b_features/processor.py::DatasetProcessor._load_symbol_inputs,
which is the second gate: it runs on whatever Module B reads back from SQLite,
independent of the ingestion-time QC gate in Module A.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import Settings
from core.exceptions import InsufficientDataError
from core.utils import last_closed_candle_open_ms
from module_a_data.models import QCSeverity
from module_a_data.qc_validator import QCValidator
from module_b_features.processor import DatasetProcessor

TF_MS = 5 * 60 * 1_000
END_MS = last_closed_candle_open_ms(TF_MS)


def clean_frame(n: int = 50) -> pd.DataFrame:
    base = END_MS - (n - 1) * TF_MS
    timestamps = [base + i * TF_MS for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * n,
            "high": [100.1] * n,
            "low": [99.9] * n,
            "close": [100.0] * n,
            "volume": [10.0] * n,
        }
    )


class FakeDatabase:
    def __init__(self, ohlcv: pd.DataFrame) -> None:
        self._ohlcv = ohlcv

    async def load_ohlcv_dataframe(self, symbol: str, limit: int | None = None) -> pd.DataFrame:
        return self._ohlcv

    async def load_futures_metrics_frame(self, symbol: str, limit: int = 1_000) -> pd.DataFrame:
        return pd.DataFrame()

    async def load_agg_trade_flow_frame(
        self, symbol: str, limit: int | None = None
    ) -> pd.DataFrame:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# QCValidator.validate_stored_frame - pure structural checks
# ---------------------------------------------------------------------------
def test_validate_stored_frame_accepts_clean_data() -> None:
    validator = QCValidator(Settings())
    issues = validator.validate_stored_frame("BTC/USDT:USDT", clean_frame())
    assert not any(issue.severity is QCSeverity.CRITICAL for issue in issues)


def test_validate_stored_frame_flags_a_gap() -> None:
    validator = QCValidator(Settings())
    frame = clean_frame(50)
    gapped = pd.concat([frame.iloc[:20], frame.iloc[25:]], ignore_index=True)
    issues = validator.validate_stored_frame("BTC/USDT:USDT", gapped)
    critical_codes = {issue.code.value for issue in issues if issue.severity is QCSeverity.CRITICAL}
    assert "MISSING_CANDLES" in critical_codes


def test_validate_stored_frame_flags_duplicates() -> None:
    validator = QCValidator(Settings())
    frame = clean_frame(20)
    duped = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    issues = validator.validate_stored_frame("BTC/USDT:USDT", duped)
    critical_codes = {issue.code.value for issue in issues if issue.severity is QCSeverity.CRITICAL}
    assert "DUPLICATE_TIMESTAMP" in critical_codes


def test_validate_stored_frame_flags_bad_price_geometry() -> None:
    validator = QCValidator(Settings())
    frame = clean_frame(10)
    frame.loc[3, "high"] = frame.loc[3, "low"] - 1.0  # high < low: impossible
    issues = validator.validate_stored_frame("BTC/USDT:USDT", frame)
    critical_codes = {issue.code.value for issue in issues if issue.severity is QCSeverity.CRITICAL}
    assert "PRICE_LOGIC_VIOLATION" in critical_codes


def test_validate_stored_frame_flags_nan() -> None:
    validator = QCValidator(Settings())
    frame = clean_frame(10)
    frame.loc[4, "close"] = float("nan")
    issues = validator.validate_stored_frame("BTC/USDT:USDT", frame)
    critical_codes = {issue.code.value for issue in issues if issue.severity is QCSeverity.CRITICAL}
    assert "NAN_VALUES" in critical_codes


def test_validate_stored_frame_flags_empty() -> None:
    validator = QCValidator(Settings())
    issues = validator.validate_stored_frame("BTC/USDT:USDT", pd.DataFrame())
    assert any(issue.severity is QCSeverity.CRITICAL for issue in issues)


# ---------------------------------------------------------------------------
# DatasetProcessor wiring: a gappy/corrupt symbol must never reach Module B
# uncleaned, but a symbol should not lose its *entire* history over one old,
# already-superseded defect either - it should trim to the clean trailing run
# and only be rejected outright when nothing usable survives that trim.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_load_symbol_inputs_trims_to_clean_trailing_run_on_gap() -> None:
    settings = Settings()
    frame = clean_frame(50)
    gapped = pd.concat([frame.iloc[:20], frame.iloc[30:]], ignore_index=True)
    processor = DatasetProcessor(settings, FakeDatabase(gapped))
    processor._load_order_book_frame = _empty_book_frame  # avoid touching a real DB

    ohlcv, futures, book, flow = await processor._load_symbol_inputs("BTC/USDT:USDT", depth=50)

    # The pre-gap segment (candles 0..19) is dropped; only the clean trailing
    # run after the gap (candles 30..49) survives - not an outright rejection.
    assert len(ohlcv) == 20
    assert int(ohlcv["timestamp"].iloc[0]) == int(frame["timestamp"].iloc[30])
    assert int(ohlcv["timestamp"].iloc[-1]) == int(frame["timestamp"].iloc[-1])


@pytest.mark.asyncio
async def test_load_symbol_inputs_raises_when_nothing_survives_trimming() -> None:
    """When corruption spans the whole window, there is no clean run to fall
    back to, so the symbol must still be rejected outright."""
    settings = Settings()
    frame = clean_frame(50)
    frame["high"] = frame["low"] - 1.0  # every row violates OHLC geometry
    processor = DatasetProcessor(settings, FakeDatabase(frame))

    with pytest.raises(InsufficientDataError):
        await processor._load_symbol_inputs("BTC/USDT:USDT", depth=50)


@pytest.mark.asyncio
async def test_load_symbol_inputs_accepts_clean_storage() -> None:
    settings = Settings()
    processor = DatasetProcessor(settings, FakeDatabase(clean_frame(50)))
    processor._load_order_book_frame = _empty_book_frame  # avoid touching a real DB

    ohlcv, futures, book, flow = await processor._load_symbol_inputs("BTC/USDT:USDT", depth=50)
    assert len(ohlcv) == 50
    assert futures is None
    assert book is None


async def _empty_book_frame(symbol: str, depth: int) -> pd.DataFrame:
    return pd.DataFrame()


@pytest.mark.asyncio
async def test_build_inference_payload_skips_symbol_when_nothing_survives_trimming() -> None:
    """End-to-end: a symbol corrupt everywhere - no clean trailing run to fall
    back to - must still be refused by the live-inference entry point."""
    settings = Settings()
    frame = clean_frame(50)
    frame["close"] = float("nan")  # every row fails validation; nothing to trim to
    processor = DatasetProcessor(settings, FakeDatabase(frame))

    payload = await processor.build_inference_payload("BTC/USDT:USDT", lookback_candles=50)
    assert payload is None
