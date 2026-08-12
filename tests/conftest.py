"""Shared fixtures: a deterministic synthetic market with known order flow.

The tests must not depend on a populated database or on Binance being
reachable, so the market is generated here.  It is *not* random noise: the price
path carries a mild autocorrelated drift and the aggressive-flow buckets are
generated from that same drift, which means the order-flow features have a real
(if modest) relationship to future returns.  That is what lets a test assert
"the model learned something" rather than merely "the code ran".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings

_BUCKET_MS: int = 5 * 60 * 1_000
_START_MS: int = 1_700_000_000_000 // _BUCKET_MS * _BUCKET_MS


@pytest.fixture()
def settings() -> Settings:
    """Settings with the windows shrunk so tests run in seconds, not minutes."""
    config = Settings()
    config.features.garch_window = 120
    config.features.hmm_window = 200
    config.features.rank_window = 60
    config.features.relative_volume_lookback = 20
    config.ml.n_estimators = 40
    config.ml.direction_stage2_n_estimators = 40
    config.ml.early_stopping_rounds = 0
    config.ml.walk_forward_folds = 2
    return config


def make_market(
    rows: int = 2_400,
    seed: int = 7,
    symbol: str = "TEST/USDT:USDT",
    drift_strength: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``(ohlcv, agg_trade_flow)`` for one synthetic symbol.

    A slow mean-reverting latent state drives both the return of the *next* bar
    and the aggressive buy/sell split of the *current* one, so order flow leads
    price the way it does in a real book - which is precisely the structure the
    new features are supposed to capture.
    """
    generator = np.random.default_rng(seed)

    latent: np.ndarray = np.zeros(rows, dtype=np.float64)
    for index in range(1, rows):
        latent[index] = 0.92 * latent[index - 1] + generator.normal(0.0, 1.0)
    latent /= max(1e-9, float(np.std(latent)))

    noise: np.ndarray = generator.normal(0.0, 1.0, rows)
    returns: np.ndarray = 0.0006 * (
        drift_strength * np.concatenate([[0.0], latent[:-1]]) + (1.0 - drift_strength) * noise
    )
    close: np.ndarray = 100.0 * np.exp(np.cumsum(returns))
    open_price: np.ndarray = np.concatenate([[100.0], close[:-1]])

    spread: np.ndarray = np.abs(generator.normal(0.0, 0.0008, rows)) * close
    high: np.ndarray = np.maximum(open_price, close) + spread
    low: np.ndarray = np.minimum(open_price, close) - spread

    base_volume: np.ndarray = np.abs(generator.lognormal(6.0, 0.45, rows))
    timestamps: np.ndarray = _START_MS + np.arange(rows, dtype=np.int64) * _BUCKET_MS

    ohlcv = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": base_volume,
        }
    )
    ohlcv.index = pd.to_datetime(ohlcv["timestamp"], unit="ms", utc=True)
    ohlcv.index.name = "open_time"

    # The aggressive split follows the latent state, so the imbalance genuinely
    # leads the next bar's return.
    buy_share: np.ndarray = np.clip(0.5 + 0.18 * latent + generator.normal(0.0, 0.06, rows), 0.02, 0.98)
    flow = pd.DataFrame(
        {
            "timestamp": timestamps,
            "buy_volume": base_volume * buy_share,
            "sell_volume": base_volume * (1.0 - buy_share),
            "trades": np.maximum(1, (base_volume / 10.0).astype(int)),
        }
    )
    return ohlcv, flow


def make_pooled_dataset(settings: Settings, symbols: int = 3, rows: int = 2_400) -> Any:
    """Feature+label dataset pooled across several synthetic symbols."""
    from module_b_features.features import FEATURE_COLUMNS, FeatureEngineer
    from module_b_features.labeler import TradeLabeler
    from module_b_features.processor import ProcessedDataset

    engineer = FeatureEngineer(settings)
    labeler = TradeLabeler(settings)

    frames: list[pd.DataFrame] = []
    for index in range(symbols):
        ohlcv, flow = make_market(rows=rows, seed=11 + index)
        featured: pd.DataFrame = engineer.build(ohlcv, agg_trade_flow=flow)
        labeled: pd.DataFrame = labeler.generate(featured)
        labeled["symbol"] = f"SYM{index}/USDT:USDT"
        frames.append(labeled)

    pooled: pd.DataFrame = pd.concat(frames, ignore_index=True)
    pooled = pooled[pooled["label_is_valid"].fillna(False)]
    pooled = pooled.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(FEATURE_COLUMNS) + ["label", "target_risk_score"]
    )
    pooled = pooled.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    return ProcessedDataset(
        features=pooled[list(FEATURE_COLUMNS)].astype(float),
        direction_target=pooled["label"].astype(str),
        entry_target=pooled["entry_quality"].astype(int),
        exit_targets=pooled[["target_tp_pct", "target_sl_pct", "target_trailing_pct"]].astype(float),
        risk_target=pooled["target_risk_score"].astype(float),
        metadata=pooled[["symbol", "timestamp", "open", "high", "low", "close", "volume"]],
        symbols=tuple(f"SYM{index}/USDT:USDT" for index in range(symbols)),
        feature_columns=tuple(FEATURE_COLUMNS),
    )
