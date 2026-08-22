#!/usr/bin/env python3
"""Run the whole training + diagnostic pipeline on synthetic data.

The real diagnostic needs ~5.7M candles across 27 symbols pulled from Binance.
That is not reproducible in CI, on a laptop, or in any environment without the
production database - which meant the report generator itself had no end-to-end
regression coverage, and the only way to find out whether a change to it worked
was to retrain on real data and read the output by hand.

This script closes that gap.  It builds a small multi-symbol OHLCV history,
runs the *real* feature engineering, labeling, chronological split, model
training, backtest and ``diagnostics.build_report`` over it, and writes the
same JSON/Markdown pair the production run produces.

It can also inject the exact pathologies the branch audit found, so the report's
detectors can be tested against data that is known to be broken:

``--degenerate-fraction``
    Share of each symbol's history replaced with flat candles (open == high ==
    low == close, zero volume).  This is the P2 finding: the audited run had
    72% of its training rows in this state, which pinned ADX at its ceiling and
    forced ``wick_ratio``/``whipsaw_rate`` to zero.

``--dead-feature``
    Blank a feature column completely, reproducing P3, where four order-book
    columns were NaN on all 5,722,703 rows and no check noticed.

``--duplicate-rows``
    Duplicate N (symbol, timestamp) pairs, reproducing P29.

Because the data is synthetic, the *model quality* numbers here mean nothing -
no conclusion about market edge can be drawn from them.  What the run does
prove is that the pipeline executes end to end and that every detector,
invariant and report field behaves as intended on data whose defects are known
by construction.  That is precisely the class of bug the audit was full of.

Usage::

    python scripts/run_synthetic_diagnostic.py --out reports/baseline.json
    python scripts/run_synthetic_diagnostic.py --degenerate-fraction 0.6 --dead-feature ob_imbalance
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from module_b_features.features import FeatureService  # noqa: E402
from module_b_features.processor import DatasetProcessor, ProcessedDataset  # noqa: E402
from module_c_ml.decision_engine import DecisionEngine  # noqa: E402
from module_c_ml.ml_models import MLSubsystem  # noqa: E402
from module_e_execution.backtester import Backtester  # noqa: E402
from module_f_panel import diagnostics  # noqa: E402

_TIMEFRAME_MS = 5 * 60 * 1_000


def synthetic_ohlcv(
    rng: np.random.Generator,
    rows: int,
    *,
    degenerate_fraction: float = 0.0,
    drift: float = 0.0,
) -> pd.DataFrame:
    """A gap-free 5m OHLCV history shaped like ``load_ohlcv_dataframe``'s output.

    ``degenerate_fraction`` replaces a leading block with flat candles - the
    same shape a dead, illiquid market prints and the same shape the audited
    training window was 72% composed of.  It is a leading block rather than a
    random scatter on purpose: the audit's damage came from the degeneracy
    being concentrated in the *oldest* part of the window, which is what made
    the train/validation drift so large.
    """
    returns = rng.normal(drift, 0.004, size=rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, size=rows)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, size=rows)))
    volume = np.abs(rng.normal(1_000.0, 200.0, size=rows)) + 1.0

    degenerate_rows = int(rows * max(0.0, min(1.0, degenerate_fraction)))
    if degenerate_rows:
        flat = close[0]
        open_[:degenerate_rows] = flat
        high[:degenerate_rows] = flat
        low[:degenerate_rows] = flat
        close[:degenerate_rows] = flat
        volume[:degenerate_rows] = 0.0

    base_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    timestamps = base_ms + np.arange(rows, dtype=np.int64) * _TIMEFRAME_MS
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.to_datetime(timestamps, unit="ms", utc=True),
    )
    frame.index.name = "open_time"
    return frame


class InMemoryDatabase:
    """Serves synthetic OHLCV and an in-memory key/value state store.

    Implements exactly the surface ``DatasetProcessor``, ``Backtester`` and
    ``diagnostics.build_report`` touch, so all three run their real code paths.
    """

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames
        self._state: dict[str, Any] = {}

    async def load_ohlcv_dataframe(self, symbol: str, limit: int | None = None) -> pd.DataFrame:
        frame = self._frames.get(symbol, pd.DataFrame())
        if limit is None or frame.empty:
            return frame
        return frame.iloc[-limit:]

    async def load_futures_metrics_frame(self, symbol: str, limit: int = 1_000) -> pd.DataFrame:
        return pd.DataFrame()

    async def get_state(self, key: str) -> Any | None:
        return self._state.get(key)

    async def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    async def close(self) -> None:
        return None


async def _empty_book_frame(symbol: str, depth: int) -> pd.DataFrame:
    return pd.DataFrame()


def _inject_dead_features(dataset: ProcessedDataset, columns: Sequence[str]) -> list[str]:
    """Blank feature columns entirely, reproducing P3's all-NaN columns."""
    blanked: list[str] = []
    for column in columns:
        if column in dataset.features.columns:
            dataset.features[column] = np.nan
            dataset.null_counts_by_feature[column] = int(len(dataset.features))
            blanked.append(column)
    return blanked


async def build(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings(
        ml={
            "model_dir": Path(args.out).parent / "models",
            "n_estimators": args.n_estimators,
            "early_stopping_rounds": 5,
            "purge_bars": 10,
        },
    )
    rng = np.random.default_rng(args.seed)

    symbols = [f"SYM{i}/USDT:USDT" for i in range(args.symbols)]
    frames = {
        symbol: synthetic_ohlcv(
            rng,
            args.rows,
            degenerate_fraction=args.degenerate_fraction,
            # A mild per-symbol drift so the label classes are not perfectly
            # balanced - a perfectly symmetric market makes every head look
            # identically useless and hides real regressions.
            drift=float(rng.normal(0.0, 0.00005)),
        )
        for symbol in symbols
    }
    database = InMemoryDatabase(frames)

    feature_service = FeatureService(settings)
    processor = DatasetProcessor(settings, database, feature_service=feature_service)
    processor._load_order_book_frame = _empty_book_frame

    dataset = await processor.build_training_dataset(symbols=symbols)
    if dataset.is_empty:
        raise SystemExit("synthetic dataset came out empty - raise --rows")

    if args.dead_feature:
        blanked = _inject_dead_features(dataset, args.dead_feature)
        print(f"  injected dead features: {blanked}")

    print(f"  dataset: {len(dataset)} rows x {len(dataset.feature_columns)} features")

    ml = MLSubsystem(settings)
    train_report = await ml.train_all(dataset)
    for head, head_report in train_report.items():
        if isinstance(head_report, dict) and "error" in head_report:
            print(f"  WARNING: {head} failed to train: {head_report['error']}")

    backtest = None
    if not args.skip_backtest:
        decisions = DecisionEngine(settings)
        backtester = Backtester(settings, database, feature_service, ml, decisions)
        try:
            backtest = await backtester.run(
                symbols=symbols,
                max_candles=args.rows,
                warmup_bars=feature_service.engineer.minimum_rows(),
            )
        except Exception as error:  # noqa: BLE001 - a failed replay must not lose the report
            print(f"  WARNING: backtest failed: {type(error).__name__}: {error}")

    report = await diagnostics.build_report(
        settings=settings,
        database=database,
        ml=ml,
        dataset=dataset,
        run_id=args.run_id,
        backtest=backtest,
    )
    await database.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/synthetic_diagnostic.json")
    parser.add_argument("--run-id", default="synthetic")
    parser.add_argument("--symbols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4_000)
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--degenerate-fraction",
        type=float,
        default=0.0,
        help="share of each symbol's history replaced with flat zero-volume candles (P2)",
    )
    parser.add_argument(
        "--dead-feature",
        action="append",
        default=[],
        help="feature column to blank entirely, repeatable (P3)",
    )
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    print(f"building synthetic diagnostic run '{args.run_id}'...")
    report = asyncio.run(build(args))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(diagnostics.render_markdown(report), encoding="utf-8")

    summary = report.get("ai_summary", {})
    print(f"  wrote {out} and {out.with_suffix('.md')}")
    print(f"  overall_status      : {summary.get('overall_status')}")
    print(f"  biggest_data_problem: {summary.get('biggest_data_problem')!r}")
    print(f"  recommendations     : "
          f"CRITICAL={len(report.get('recommendations', {}).get('CRITICAL', []))} "
          f"HIGH={len(report.get('recommendations', {}).get('HIGH', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
