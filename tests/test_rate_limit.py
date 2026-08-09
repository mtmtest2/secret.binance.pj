"""API request-rate reduction (config/settings.py + module_a_data/fetcher.py)."""

from __future__ import annotations

from config.settings import ExchangeSettings, Settings
from module_a_data.fetcher import BinanceDataFetcher


def _settings(**exchange_overrides: object) -> Settings:
    return Settings(exchange=ExchangeSettings(**exchange_overrides))


def test_default_scale_is_80_percent() -> None:
    assert Settings().exchange.request_rate_scale == 0.8


def test_rate_scale_slows_ccxt_dispatch_pacing() -> None:
    settings = _settings()
    fetcher = BinanceDataFetcher(settings)
    try:
        baseline = 50  # ccxt's stock binance rateLimit (ms/weight-1 request)
        expected = int(round(baseline / settings.exchange.request_rate_scale))
        assert fetcher.exchange.rateLimit == expected
        # 80% of the original speed means requests are paced ~25% slower.
        assert fetcher.exchange.rateLimit > baseline
        assert abs(fetcher.exchange.rateLimit - baseline / 0.8) < 1.0
    finally:
        fetcher._owns_exchange = False  # avoid an event-loop-less close() warning


def test_scale_of_one_keeps_ccxt_default() -> None:
    settings = _settings(request_rate_scale=1.0)
    fetcher = BinanceDataFetcher(settings)
    try:
        assert fetcher.exchange.rateLimit == 50
    finally:
        fetcher._owns_exchange = False


def test_scale_out_of_bounds_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExchangeSettings(request_rate_scale=1.5)
    with pytest.raises(ValidationError):
        ExchangeSettings(request_rate_scale=0.0)
