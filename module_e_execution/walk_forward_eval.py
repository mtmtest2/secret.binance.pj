"""Walk-forward, purged out-of-sample performance evaluation.

This is the tool that answers the actual question a change to the ML
pipeline needs to answer - not "did validation accuracy go up" but "would
this have made money, walk-forward, retraining periodically, only ever
trading forward in time."

For each purged walk-forward fold (see :mod:`module_c_ml.cross_validation`):

1. A completely fresh :class:`~module_c_ml.ml_models.MLSubsystem` is trained
   from scratch on that fold's training window only
   (:meth:`~module_b_features.processor.DatasetProcessor.build_training_dataset`
   with ``end_ms`` capping it strictly before the fold's validation window).
2. That fold's *held-out* validation window is replayed through the real
   :class:`~module_e_execution.backtester.Backtester` - full fee, slippage,
   funding and liquidation simulation, next-bar-open fills, the same engine
   that would run in paper/live mode. The model never saw this window during
   training.

Every fold's out-of-sample trades and equity curve are then stitched into one
continuous out-of-sample record (folds are combined by chaining their
*period returns*, not their raw equity levels, since each fold restarts from
the same starting equity rather than compounding fold-to-fold - see
:func:`_combine_metrics`), and the standard performance metrics (Sharpe,
Sortino, profit factor, expectancy, max drawdown) are computed on that
combined record.

Usage - an honest before/after comparison of any pipeline change:

    report_new = await run_walk_forward_evaluation(settings, db, symbols)
    settings.ml.use_direction_stacking = False   # or any other toggle
    report_old = await run_walk_forward_evaluation(settings, db, symbols)
    print(report_new.summary())
    print(report_old.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from core.exceptions import InsufficientDataError
from core.logger import get_logger
from core.utils import ms_to_datetime
from module_a_data.db_handler import DatabaseHandler
from module_b_features.features import FeatureService
from module_b_features.processor import DatasetProcessor, ProcessedDataset
from module_c_ml.cross_validation import CVFold, purged_walk_forward_splits
from module_c_ml.decision_engine import DecisionEngine
from module_c_ml.ml_models import MLSubsystem
from module_e_execution.backtester import BacktestReport, Backtester

_LOGGER = get_logger(__name__)

_BARS_PER_YEAR: int = 365 * 24 * 12


@dataclass(slots=True)
class WalkForwardFoldResult:
    """One fold's training report plus its held-out backtest."""

    fold_index: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    training_report: dict[str, Any]
    backtest: BacktestReport


@dataclass(slots=True)
class WalkForwardReport:
    """The full walk-forward evaluation: per-fold detail plus combined metrics."""

    folds: list[WalkForwardFoldResult] = field(default_factory=list)
    combined_metrics: dict[str, float] = field(default_factory=dict)
    combined_equity_curve: list[dict[str, float]] = field(default_factory=list)
    combined_trades: list[dict[str, Any]] = field(default_factory=list)
    settings_summary: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Multi-line, human-readable report for logs and the CLI."""
        lines: list[str] = ["=" * 78, "WALK-FORWARD OUT-OF-SAMPLE EVALUATION", "=" * 78]
        for fold in self.folds:
            m = fold.backtest.metrics
            lines.append(
                f"Fold {fold.fold_index}: {fold.validation_start} -> {fold.validation_end}  |  "
                f"trades={int(m.get('total_trades', 0)):4d}  "
                f"win={m.get('win_rate', 0.0):6.1%}  "
                f"PF={m.get('profit_factor', 0.0):6.2f}  "
                f"sharpe={m.get('sharpe_ratio', 0.0):6.2f}  "
                f"maxDD={m.get('max_drawdown_pct', 0.0):6.1%}"
            )
        lines.append("-" * 78)
        cm = self.combined_metrics
        lines.append(f"COMBINED OUT-OF-SAMPLE ({len(self.folds)} folds, chained returns):")
        lines.append(f"  trades              : {int(cm.get('total_trades', 0))}")
        lines.append(f"  win rate            : {cm.get('win_rate', 0.0):.2%}")
        lines.append(f"  profit factor       : {cm.get('profit_factor', 0.0):.3f}")
        lines.append(f"  expectancy          : {cm.get('expectancy', 0.0):,.4f} USDT/trade")
        lines.append(f"  sharpe ratio        : {cm.get('sharpe_ratio', 0.0):.3f}")
        lines.append(f"  sortino ratio       : {cm.get('sortino_ratio', 0.0):.3f}")
        lines.append(f"  max drawdown        : {cm.get('max_drawdown_pct', 0.0):.2%}")
        lines.append(f"  chained return      : {cm.get('total_return_pct', 0.0):.2%}")
        lines.append(f"  total fees / funding: {cm.get('total_fees', 0.0):,.2f} / "
                      f"{cm.get('total_funding', 0.0):,.2f} USDT")
        lines.append(f"  liquidations        : {int(cm.get('liquidations', 0))}")
        lines.append("=" * 78)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "settings_summary": self.settings_summary,
            "folds": [
                {
                    "fold_index": fold.fold_index,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "validation_start": fold.validation_start,
                    "validation_end": fold.validation_end,
                    "training_report": fold.training_report,
                    "metrics": fold.backtest.metrics,
                }
                for fold in self.folds
            ],
            "combined_metrics": self.combined_metrics,
        }


async def run_walk_forward_evaluation(
    settings: Settings,
    database: DatabaseHandler,
    symbols: Sequence[str],
    n_splits: int | None = None,
    max_candles_per_symbol: int | None = None,
) -> WalkForwardReport:
    """Run the full retrain-then-backtest walk-forward evaluation.

    Args:
        settings: Config to evaluate. To A/B a pipeline change, run this
            twice with two ``Settings`` objects that differ only in the
            setting under test (e.g. ``ml.use_direction_stacking``).
        database: An already-initialised handler over real historical data.
        symbols: Universe to evaluate.
        n_splits: Overrides ``ml.cv_folds`` for this run.
        max_candles_per_symbol: History depth cap (defaults to
            ``data.history_bootstrap_candles``).

    Returns:
        A :class:`WalkForwardReport`.

    Raises:
        InsufficientDataError: When there is not enough stored history for
            even one purged fold.
    """
    if not symbols:
        raise InsufficientDataError("walk-forward evaluation requires at least one symbol")

    splits: int = n_splits or settings.ml.cv_folds
    depth: int = max_candles_per_symbol or settings.data.history_bootstrap_candles
    horizon: int = settings.labels.max_holding_bars

    # The shortest/reference symbol's own timeline defines every fold's time
    # boundaries, applied uniformly across the whole universe - simpler and
    # safer than trying to intersect every symbol's own available range.
    reference_symbol: str = symbols[0]
    reference: pd.DataFrame = await database.load_ohlcv_dataframe(reference_symbol, limit=depth)
    if reference.empty:
        raise InsufficientDataError(f"no stored candles for reference symbol {reference_symbol}")
    timestamps: np.ndarray = reference["timestamp"].to_numpy()
    n_rows: int = len(timestamps)

    folds: list[CVFold] = purged_walk_forward_splits(n_rows, splits, horizon, settings.ml.cv_embargo_bars)
    if not folds:
        raise InsufficientDataError(
            f"not enough stored history ({n_rows} bars for {reference_symbol}) "
            f"for {splits} purged walk-forward folds"
        )

    feature_service: FeatureService = FeatureService(settings)
    warmup_bars: int = feature_service.engineer.minimum_rows()
    timeframe_ms: int = settings.data.timeframe_ms
    starting_equity: float = settings.execution.paper_starting_balance

    report = WalkForwardReport(
        settings_summary={
            "cv_folds": splits,
            "use_direction_stacking": settings.ml.use_direction_stacking,
            "calibrate_probabilities": settings.ml.calibrate_probabilities,
            "prune_weak_features": settings.ml.prune_weak_features,
            "use_sample_weighting": settings.ml.use_sample_weighting,
            "ensemble_size": settings.ml.ensemble_size,
            "class_weight": settings.ml.class_weight,
            "entry_class_weight": settings.ml.entry_class_weight,
            "max_holding_bars": horizon,
        }
    )

    for fold in folds:
        train_end_ms: int = int(timestamps[fold.train_index[-1]])
        val_start_ms: int = int(timestamps[fold.validation_index[0]])
        val_end_ms: int = int(timestamps[fold.validation_index[-1]])
        warmup_start_ms: int = val_start_ms - warmup_bars * timeframe_ms

        _LOGGER.info(
            "Walk-forward fold %d/%d: train through %s, validate %s -> %s",
            fold.fold_index + 1, len(folds),
            ms_to_datetime(train_end_ms), ms_to_datetime(val_start_ms), ms_to_datetime(val_end_ms),
        )

        # --- Train a completely fresh MLSubsystem on this fold's past only ---
        processor = DatasetProcessor(settings, database, feature_service)
        dataset: ProcessedDataset = await processor.build_training_dataset(
            symbols=list(symbols), max_candles_per_symbol=depth, end_ms=train_end_ms,
        )
        if dataset.is_empty:
            _LOGGER.warning("Fold %d: empty training dataset - skipping", fold.fold_index)
            continue

        ml = MLSubsystem(settings)
        training_report: dict[str, Any] = await ml.train_all(dataset)

        # --- Backtest strictly on the held-out window; warm-up bars are
        # loaded but excluded from scoring via `warmup_bars`. -----------------
        decisions = DecisionEngine(settings)
        backtester = Backtester(settings, database, feature_service, ml, decisions)
        try:
            fold_backtest: BacktestReport = await backtester.run(
                symbols=list(symbols),
                max_candles=depth,
                initial_equity=starting_equity,
                start_ms=warmup_start_ms,
                end_ms=val_end_ms,
                warmup_bars=warmup_bars,
            )
        except InsufficientDataError as error:
            _LOGGER.warning("Fold %d: backtest could not run (%s) - skipping", fold.fold_index, error)
            continue

        report.folds.append(
            WalkForwardFoldResult(
                fold_index=fold.fold_index,
                train_start=str(ms_to_datetime(int(timestamps[fold.train_index[0]]))),
                train_end=str(ms_to_datetime(train_end_ms)),
                validation_start=str(ms_to_datetime(val_start_ms)),
                validation_end=str(ms_to_datetime(val_end_ms)),
                training_report=training_report,
                backtest=fold_backtest,
            )
        )

    if not report.folds:
        raise InsufficientDataError("no walk-forward fold produced a usable backtest")

    report.combined_trades = [trade for fold in report.folds for trade in fold.backtest.trades]
    report.combined_metrics = _combine_metrics(report.folds, starting_equity)
    return report


def _combine_metrics(folds: list[WalkForwardFoldResult], starting_equity: float) -> dict[str, float]:
    """Stitch each fold's out-of-sample equity curve into one continuous record.

    Each fold is trained and backtested independently and restarts from the
    same starting equity - deliberately, so an early fold's drawdown cannot
    mechanically shrink the position sizing (a fraction of *current* equity)
    available to a later, unrelated fold.  Combining them correctly therefore
    means chaining their *period returns* into one synthetic continuous
    curve, not concatenating their raw equity levels (which would introduce
    a discontinuity at every fold boundary).
    """
    all_returns: list[float] = []
    for fold in folds:
        curve: list[dict[str, float]] = fold.backtest.equity_curve
        if len(curve) < 2:
            continue
        equity: np.ndarray = np.asarray([point["equity"] for point in curve], dtype=np.float64)
        previous: np.ndarray = equity[:-1]
        safe_previous: np.ndarray = np.where(previous != 0.0, previous, np.nan)
        returns: np.ndarray = (equity[1:] - previous) / safe_previous
        all_returns.extend(returns[np.isfinite(returns)].tolist())

    stitched: list[float] = [starting_equity]
    for period_return in all_returns:
        stitched.append(stitched[-1] * (1.0 + period_return))
    stitched_equity: np.ndarray = np.asarray(stitched, dtype=np.float64)

    trades: list[dict[str, Any]] = [trade for fold in folds for trade in fold.backtest.trades]
    pnls: np.ndarray = np.asarray(
        [float(trade.get("realized_pnl", 0.0)) for trade in trades], dtype=np.float64
    )
    wins: np.ndarray = pnls[pnls > 0.0]
    losses: np.ndarray = pnls[pnls < 0.0]
    gross_profit: float = float(wins.sum()) if wins.size else 0.0
    gross_loss: float = float(-losses.sum()) if losses.size else 0.0

    running_peak: np.ndarray = np.maximum.accumulate(stitched_equity)
    safe_peak: np.ndarray = np.where(running_peak > 0.0, running_peak, 1.0)
    drawdowns: np.ndarray = (running_peak - stitched_equity) / safe_peak
    max_drawdown: float = float(np.max(drawdowns)) if drawdowns.size else 0.0

    returns_array: np.ndarray = np.asarray(all_returns, dtype=np.float64)
    deviation: float = float(returns_array.std(ddof=1)) if returns_array.size > 1 else 0.0
    sharpe: float = (
        float(returns_array.mean() / deviation * math.sqrt(_BARS_PER_YEAR)) if deviation > 0.0 else 0.0
    )
    downside: np.ndarray = returns_array[returns_array < 0.0]
    downside_deviation: float = float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
    sortino: float = (
        float(returns_array.mean() / downside_deviation * math.sqrt(_BARS_PER_YEAR))
        if downside_deviation > 0.0 else 0.0
    )

    total_return: float = (
        (stitched_equity[-1] - stitched_equity[0]) / stitched_equity[0]
        if stitched_equity[0] > 0.0 else 0.0
    )

    return {
        "total_trades": float(pnls.size),
        "winning_trades": float(wins.size),
        "losing_trades": float(losses.size),
        "win_rate": float(wins.size / pnls.size) if pnls.size else 0.0,
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0.0 else (math.inf if gross_profit > 0.0 else 0.0)
        ),
        "expectancy": float(pnls.mean()) if pnls.size else 0.0,
        "average_win": float(wins.mean()) if wins.size else 0.0,
        "average_loss": float(losses.mean()) if losses.size else 0.0,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "total_fees": float(sum(float(trade.get("fees_paid", 0.0)) for trade in trades)),
        "total_funding": float(sum(float(trade.get("funding_paid", 0.0)) for trade in trades)),
        "liquidations": float(
            sum(1 for trade in trades if trade.get("close_reason") == "LIQUIDATION")
        ),
        "folds_used": float(len(folds)),
    }


def label_barrier_consistency_report(
    settings: Settings, ml: MLSubsystem, dataset: ProcessedDataset
) -> dict[str, Any]:
    """Diagnose the train/inference barrier-width mismatch noted during review.

    ``TradeLabeler`` generates its LONG_SUCCESS/SHORT_SUCCESS labels against
    *fixed* ATR-multiple barriers (``labels.tp_atr_multiple`` /
    ``sl_atr_multiple``), but at live inference the Exit model predicts a
    *different*, per-row dynamic barrier - the Decision Engine trades on the
    Exit model's geometry, not the labeling one. This computes the Exit
    model's predicted reward/risk ratio across a sample of the training
    dataset and compares it with the fixed labeling barriers' own ratio, so
    an operator can see whether the two have drifted apart.

    There is no one-time code fix for this: the natural "fix" (deriving
    labels from the Exit model's own predicted barrier) is circular - Exit's
    target depends on which side Direction chose, which depends on the label
    the barrier width defines in the first place. The practical mitigation is
    operational: periodically run this report and re-tune
    ``tp_atr_multiple``/``sl_atr_multiple`` toward whatever ratio the Exit
    model has actually converged on, rather than treating them as fixed
    constants set once.
    """
    if dataset.is_empty or not ml.exit.is_loaded:
        return {"available": False, "reason": "empty dataset or exit model not trained"}

    sample: pd.DataFrame = dataset.features.sample(n=min(2000, len(dataset)), random_state=0)
    aligned: pd.DataFrame = ml.exit._align(sample)  # noqa: SLF001 - diagnostic tool, not a hot path
    estimators: dict[str, Any] = ml.exit._model  # noqa: SLF001
    predicted_tp: np.ndarray = np.asarray(estimators["target_tp_pct"].predict(aligned))
    predicted_sl: np.ndarray = np.asarray(estimators["target_sl_pct"].predict(aligned))
    predicted_rr: np.ndarray = np.divide(
        predicted_tp, predicted_sl, out=np.zeros_like(predicted_tp), where=predicted_sl > 0
    )

    label_rr: float = settings.labels.tp_atr_multiple / settings.labels.sl_atr_multiple

    return {
        "available": True,
        "label_barrier_reward_risk": label_rr,
        "exit_model_predicted_rr_mean": float(np.mean(predicted_rr)),
        "exit_model_predicted_rr_median": float(np.median(predicted_rr)),
        "exit_model_predicted_rr_std": float(np.std(predicted_rr)),
        "drift_pct": float(abs(np.median(predicted_rr) - label_rr) / label_rr) if label_rr > 0 else 0.0,
        "sample_rows": int(len(sample)),
    }
