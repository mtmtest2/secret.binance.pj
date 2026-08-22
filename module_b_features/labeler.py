"""Forward-looking, risk-tiered multi-class label generation.

This module is the *only* place in the system that is allowed to look into the
future, because it manufactures the ground truth used for supervised training.
It must therefore never be applied to the live-inference path - the processor
enforces that by requiring an explicit ``mode="train"``.

Simulation model
----------------
For every candle close the labeler opens a hypothetical **long** and a
hypothetical **short**, each with volatility-scaled barriers::

    tp_distance = tp_atr_multiple * ATR_t
    sl_distance = sl_atr_multiple * ATR_t

and scans forward at most ``max_holding_bars`` candles for the first barrier
touch.  Three conservative modelling choices keep the labels honest:

1. **Worst-case intra-candle ordering.**  5-minute OHLC data cannot tell us
   whether the high or the low came first.  When a single candle touches both
   barriers we assume the **stop** was hit first.  Any other assumption
   manufactures phantom winners.
2. **Barrier fills, not close fills.**  A resolved trade exits exactly at its
   barrier price; only expired trades mark to the horizon close.
3. **Heat is measured, not ignored.**  The maximum adverse excursion (MAE) along
   the path is recorded and expressed as a fraction of the stop distance.  A
   winner that spent the trade sitting at 90 % of its stop is *not* the same
   trade as one that never went offside, and the risk tier says so.

Risk tiering
------------
The tier combines path risk (MAE ratio) with the volatility regime at entry
(rolling GARCH percentile).  Extreme volatility escalates the tier by one step
and caps it at ``VERY_HIGH``, which - by configuration - is folded into
``NO_TRADE_OR_FAIL``: a trade that only worked because the market was violent is
not an edge worth learning.

The tier is *not* fused into the Direction model's label.  Earlier revisions
split each direction into a LOW_RISK/HIGH_RISK pair of classes (five classes
total); that made the label a noisy compound of "which way" and "how clean
was the path", and the two questions have very different feature signatures.
The path-risk half is only knowable from information a pre-trade feature
vector barely carries (it depends on the exact intra-trade excursion), so
fusing it into the direction target diluted the one signal the Direction
model can actually learn well - forecasting the primary discrete outcome -
and swamped a 44%-majority NO_TRADE class in five-way search noise, causing
class-balanced training to overcorrect against it.  Direction now predicts
only ``LONG_SUCCESS`` / ``SHORT_SUCCESS`` / ``NO_TRADE_OR_FAIL``; the tier is
still computed here (and still trains the Risk model's continuous score via
``target_risk_score``) and is turned back into a discrete tier for gating and
sizing purposes by :func:`risk_tier_from_score`, which the Risk model uses at
inference time.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

import numpy as np
import pandas as pd

from config.settings import LabelSettings, Settings
from core.exceptions import InsufficientDataError, LabelingError
from core.logger import get_logger

_LOGGER = get_logger(__name__)

_EPSILON: Final[float] = 1e-12
#: Rows processed per vectorised chunk, bounding peak memory on long histories.
_CHUNK_ROWS: Final[int] = 20_000

#: Exit-target clamp rails.  Mirror the hard sanity rails the Exit model's
#: ``_assemble`` applies at inference time (``module_c_ml.ml_models``) so the
#: training target is never wider than what a trade could ever actually use -
#: without this, rare extreme-excursion candles blow up the regressor's loss
#: and its predictions on ordinary rows along with it.
#:
#: Investigated (task 7b) but deliberately left unchanged: a production
#: diagnostic report showed the SL target's median sitting exactly at
#: _MIN_SL_PCT, and a synthetic-data check here confirmed it - for typical
#: liquid-5m ATR (~0.2% of price) and sl_atr_multiple=1.0, the ATR-scaled
#: floor_sl computed just above this constant (0.5 * sl_dist) already lands
#: below 0.0015 for a large share of rows (any trade whose actual heat was
#: small - e.g. a clean winner - naturally wants a much tighter stop), so
#: this absolute floor - not the ATR-relative one - is what actually binds
#: and collapses their true, smaller optimal_sl values into one constant,
#: destroying the variance the regressor needs (matches the reported
#: r^2 ~= 0.065, far below TP/trailing's ~0.31).
#:
#: Lowering _MIN_SL_PCT was considered and rejected without real data: at
#: ExecutionSettings defaults (taker_fee=0.05%, slippage_bps=0.05%), a round
#: trip already costs ~0.2% in fees + slippage alone - *above* the current
#: 0.15% floor. A stop any tighter would be economically closer to (or
#: past) a guaranteed net loss purely from trading costs, independent of
#: whether the price move itself was adverse. Trading a statistically
#: stronger-looking label for an economically unsound live stop-loss is not
#: a confident net-positive change on synthetic data alone - left as-is
#: pending a real retrain + backtest the operator can actually validate
#: against live fee/slippage reality.
_MIN_TP_PCT: Final[float] = 0.0020
_MAX_TP_PCT: Final[float] = 0.1500
_MIN_SL_PCT: Final[float] = 0.0015
_MAX_SL_PCT: Final[float] = 0.0800


class LabelClass(str, Enum):
    """The multi-class target consumed by the Market Direction model."""

    LONG_SUCCESS = "LONG_SUCCESS"
    SHORT_SUCCESS = "SHORT_SUCCESS"
    NO_TRADE_OR_FAIL = "NO_TRADE_OR_FAIL"


class RiskTier(str, Enum):
    """Path-risk classification of a simulated trade."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    NONE = "NONE"


class TradeOutcome(str, Enum):
    """Which barrier resolved the simulated trade."""

    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    EXPIRED = "EXPIRED"
    UNRESOLVED = "UNRESOLVED"


#: Stable class ordering shared by the labeler and the Direction model.
LABEL_ORDER: Final[tuple[str, ...]] = (
    LabelClass.LONG_SUCCESS.value,
    LabelClass.SHORT_SUCCESS.value,
    LabelClass.NO_TRADE_OR_FAIL.value,
)

LABEL_TO_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(LABEL_ORDER)}

#: Which labels represent a tradeable long / short opportunity.
LONG_LABELS: Final[frozenset[str]] = frozenset({LabelClass.LONG_SUCCESS.value})
SHORT_LABELS: Final[frozenset[str]] = frozenset({LabelClass.SHORT_SUCCESS.value})

_TIER_ORDER: Final[tuple[str, ...]] = (
    RiskTier.LOW.value,
    RiskTier.MEDIUM.value,
    RiskTier.HIGH.value,
    RiskTier.VERY_HIGH.value,
)

#: ``0.5 + 0.5 * volatility_component`` at a neutral (median, 0.5) volatility
#: reading - the same multiplier :meth:`TradeLabeler._attach_model_targets`
#: applies to ``heat_component`` when building ``target_risk_score``.
_NEUTRAL_SCORE_MULTIPLIER: Final[float] = 0.75


def risk_tier_from_score(score: float, config: LabelSettings) -> str:
    """Map the Risk model's continuous opportunity score back onto a tier.

    At inference time there is no simulated path to measure a real MAE ratio
    from, so the Risk model's tier can no longer be read off the Direction
    label the way earlier revisions did (see the module docstring).  Instead
    this inverts the training-time score formula -
    ``heat_component * (0.5 + 0.5 * volatility_component)`` where
    ``heat_component = 1 - mae_ratio`` - at a neutral (median) volatility
    reading, which turns each of the labeler's own MAE-ratio tier boundaries
    into an equivalent score cut point.  The cut points therefore move
    automatically with ``config`` instead of being separately hand-tuned
    constants.
    """
    low_boundary: float = (1.0 - config.low_risk_mae_ratio) * _NEUTRAL_SCORE_MULTIPLIER
    medium_boundary: float = (1.0 - config.medium_risk_mae_ratio) * _NEUTRAL_SCORE_MULTIPLIER
    high_boundary: float = (1.0 - config.high_risk_mae_ratio) * _NEUTRAL_SCORE_MULTIPLIER

    if score >= low_boundary:
        return RiskTier.LOW.value
    if score >= medium_boundary:
        return RiskTier.MEDIUM.value
    if score >= high_boundary:
        return RiskTier.HIGH.value
    return RiskTier.VERY_HIGH.value


class _SideSimulation:
    """Container for the vectorised simulation output of one side."""

    __slots__ = (
        "outcome",
        "mae_ratio",
        "mfe_ratio",
        "bars_to_exit",
        "pnl_pct",
        "optimal_tp_pct",
        "optimal_sl_pct",
        "optimal_trailing_pct",
    )

    def __init__(self, rows: int) -> None:
        self.outcome: np.ndarray = np.full(rows, TradeOutcome.UNRESOLVED.value, dtype=object)
        self.mae_ratio: np.ndarray = np.full(rows, np.nan, dtype=np.float64)
        self.mfe_ratio: np.ndarray = np.full(rows, np.nan, dtype=np.float64)
        self.bars_to_exit: np.ndarray = np.full(rows, np.nan, dtype=np.float64)
        self.pnl_pct: np.ndarray = np.full(rows, np.nan, dtype=np.float64)
        self.optimal_tp_pct: np.ndarray = np.full(rows, np.nan, dtype=np.float64)
        self.optimal_sl_pct: np.ndarray = np.full(rows, np.nan, dtype=np.float64)
        self.optimal_trailing_pct: np.ndarray = np.full(rows, np.nan, dtype=np.float64)


class TradeLabeler:
    """Generates multi-class, risk-tiered labels and per-model regression targets."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._config: LabelSettings = settings.labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach every training target to a feature frame.

        Args:
            frame: Feature frame from :class:`FeatureEngineer` - must contain
                ``high``, ``low``, ``close`` and ``atr`` (``garch_volatility`` is
                used when ATR has not warmed up).

        Returns:
            A copy of ``frame`` with the following columns added:

            ============================  ==========================================
            ``label``                     Multi-class target (Direction model)
            ``label_index``               Integer encoding of ``label``
            ``risk_tier``                 ``LOW`` / ``MEDIUM`` / ``HIGH`` / ``VERY_HIGH``
            ``entry_quality``             Binary target for the Entry model
            ``target_tp_pct``             Regression target for the Exit model
            ``target_sl_pct``             Regression target for the Exit model
            ``target_trailing_pct``       Regression target for the Exit model
            ``target_risk_score``         Regression target for the Risk model
            ``long_outcome`` / ``short_outcome``   Which barrier resolved each side
            ``long_mae_ratio`` / ``short_mae_ratio``  Path heat as a share of the stop
            ``long_pnl_pct`` / ``short_pnl_pct``       Barrier-accurate simulated PnL
            ``label_is_valid``            ``False`` for the un-simulatable tail
            ============================  ==========================================

        Raises:
            InsufficientDataError: When the frame is shorter than the holding
                horizon plus one bar.
            LabelingError: When required columns are absent.
        """
        required: set[str] = {"high", "low", "close"}
        missing: set[str] = required - set(frame.columns)
        if missing:
            raise LabelingError("labeling requires OHLC columns", missing=sorted(missing))

        horizon: int = self._config.max_holding_bars
        if len(frame) <= horizon + 1:
            raise InsufficientDataError(
                "not enough candles to simulate the holding horizon",
                rows=len(frame),
                horizon=horizon,
            )

        result: pd.DataFrame = frame.copy()
        high: np.ndarray = frame["high"].to_numpy(dtype=np.float64)
        low: np.ndarray = frame["low"].to_numpy(dtype=np.float64)
        close: np.ndarray = frame["close"].to_numpy(dtype=np.float64)

        tp_distance, sl_distance = self._barrier_distances(frame, close)

        long_side: _SideSimulation = self._simulate(
            high, low, close, tp_distance, sl_distance, horizon, is_long=True
        )
        short_side: _SideSimulation = self._simulate(
            high, low, close, tp_distance, sl_distance, horizon, is_long=False
        )

        volatility_percentile: np.ndarray = self._volatility_percentile(frame)
        labels, tiers, chosen_side = self._classify(long_side, short_side, volatility_percentile)

        result["label"] = labels
        result["label_index"] = [LABEL_TO_INDEX.get(name, LABEL_TO_INDEX[
            LabelClass.NO_TRADE_OR_FAIL.value
        ]) for name in labels]
        result["risk_tier"] = tiers
        result["label_is_valid"] = long_side.outcome != TradeOutcome.UNRESOLVED.value

        result["long_outcome"] = long_side.outcome
        result["short_outcome"] = short_side.outcome
        result["long_mae_ratio"] = long_side.mae_ratio
        result["short_mae_ratio"] = short_side.mae_ratio
        result["long_mfe_ratio"] = long_side.mfe_ratio
        result["short_mfe_ratio"] = short_side.mfe_ratio
        result["long_pnl_pct"] = long_side.pnl_pct
        result["short_pnl_pct"] = short_side.pnl_pct
        result["long_bars_to_exit"] = long_side.bars_to_exit
        result["short_bars_to_exit"] = short_side.bars_to_exit

        result = self._attach_model_targets(
            result, long_side, short_side, chosen_side, tiers, volatility_percentile
        )

        distribution: dict[str, int] = (
            result.loc[result["label_is_valid"], "label"].value_counts().to_dict()
        )
        _LOGGER.info("Label distribution: %s", distribution)
        return result

    # ------------------------------------------------------------------
    # Barriers
    # ------------------------------------------------------------------
    def _barrier_distances(
        self,
        frame: pd.DataFrame,
        close: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-row TP/SL distances in price units.

        ATR is the primary volatility scale.  Where ATR has not warmed up (or is
        degenerate) the GARCH one-step forecast is used instead, converted from a
        return to a price distance.  Both are known at bar ``t``, so barrier
        placement introduces no look-ahead.
        """
        if "atr" in frame.columns:
            atr_values: np.ndarray = frame["atr"].to_numpy(dtype=np.float64)
        else:
            atr_values = np.full(close.size, np.nan, dtype=np.float64)

        if "garch_volatility" in frame.columns:
            garch: np.ndarray = frame["garch_volatility"].to_numpy(dtype=np.float64) * close
        else:
            garch = np.full(close.size, np.nan, dtype=np.float64)

        scale: np.ndarray = np.where(np.isfinite(atr_values) & (atr_values > 0.0), atr_values, garch)
        # Final fallback: 0.25 % of price, a sane floor for liquid 5m perps.
        fallback: np.ndarray = close * 0.0025
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, fallback)

        tp_distance: np.ndarray = scale * self._config.tp_atr_multiple
        sl_distance: np.ndarray = scale * self._config.sl_atr_multiple
        return tp_distance, np.maximum(sl_distance, close * 1e-5)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _simulate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        tp_distance: np.ndarray,
        sl_distance: np.ndarray,
        horizon: int,
        is_long: bool,
    ) -> _SideSimulation:
        """Vectorised triple-barrier scan for one side.

        Implemented with ``sliding_window_view`` plus running extrema
        (``np.minimum.accumulate`` / ``np.maximum.accumulate``) so the whole
        history resolves in a handful of array passes instead of a Python
        double loop.  Work is chunked to bound peak memory at ``O(chunk * horizon)``.
        """
        rows: int = close.size
        simulation = _SideSimulation(rows)
        simulatable: int = rows - horizon
        if simulatable <= 0:
            return simulation

        for chunk_start in range(0, simulatable, _CHUNK_ROWS):
            chunk_end: int = min(chunk_start + _CHUNK_ROWS, simulatable)
            self._simulate_chunk(
                simulation,
                high,
                low,
                close,
                tp_distance,
                sl_distance,
                horizon,
                is_long,
                chunk_start,
                chunk_end,
            )
        return simulation

    def _simulate_chunk(
        self,
        simulation: _SideSimulation,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        tp_distance: np.ndarray,
        sl_distance: np.ndarray,
        horizon: int,
        is_long: bool,
        chunk_start: int,
        chunk_end: int,
    ) -> None:
        """Resolve one chunk of entries; writes straight into ``simulation``."""
        count: int = chunk_end - chunk_start
        if count <= 0:
            return

        # Forward windows: row i sees bars i+1 ... i+horizon.
        window_slice: slice = slice(chunk_start + 1, chunk_end + horizon)
        future_high: np.ndarray = np.lib.stride_tricks.sliding_window_view(
            high[window_slice], horizon
        )[:count]
        future_low: np.ndarray = np.lib.stride_tricks.sliding_window_view(
            low[window_slice], horizon
        )[:count]

        entry: np.ndarray = close[chunk_start:chunk_end]
        tp_dist: np.ndarray = tp_distance[chunk_start:chunk_end]
        sl_dist: np.ndarray = sl_distance[chunk_start:chunk_end]

        if is_long:
            tp_price: np.ndarray = entry + tp_dist
            sl_price: np.ndarray = entry - sl_dist
            tp_touched: np.ndarray = future_high >= tp_price[:, np.newaxis]
            sl_touched: np.ndarray = future_low <= sl_price[:, np.newaxis]
            running_adverse: np.ndarray = np.minimum.accumulate(future_low, axis=1)
            running_favorable: np.ndarray = np.maximum.accumulate(future_high, axis=1)
        else:
            tp_price = entry - tp_dist
            sl_price = entry + sl_dist
            tp_touched = future_low <= tp_price[:, np.newaxis]
            sl_touched = future_high >= sl_price[:, np.newaxis]
            running_adverse = np.maximum.accumulate(future_high, axis=1)
            running_favorable = np.minimum.accumulate(future_low, axis=1)

        tp_any: np.ndarray = tp_touched.any(axis=1)
        sl_any: np.ndarray = sl_touched.any(axis=1)
        first_tp: np.ndarray = np.argmax(tp_touched, axis=1)
        first_sl: np.ndarray = np.argmax(sl_touched, axis=1)

        # Conservative tie-break: when both barriers are touched on the same
        # candle we cannot know the intra-candle order, so the stop wins.
        stop_first: np.ndarray = sl_any & (~tp_any | (first_sl <= first_tp))
        target_first: np.ndarray = tp_any & ~stop_first

        exit_index: np.ndarray = np.where(
            stop_first, first_sl, np.where(target_first, first_tp, horizon - 1)
        )

        outcome: np.ndarray = np.full(count, TradeOutcome.EXPIRED.value, dtype=object)
        outcome[stop_first] = TradeOutcome.STOP_LOSS.value
        outcome[target_first] = TradeOutcome.TAKE_PROFIT.value

        rows_index: np.ndarray = np.arange(count)
        worst_price: np.ndarray = running_adverse[rows_index, exit_index]
        best_price: np.ndarray = running_favorable[rows_index, exit_index]

        if is_long:
            adverse_move: np.ndarray = entry - worst_price
            favorable_move: np.ndarray = best_price - entry
            expiry_price: np.ndarray = close[chunk_start + horizon : chunk_end + horizon]
            exit_price: np.ndarray = np.where(
                stop_first, sl_price, np.where(target_first, tp_price, expiry_price)
            )
            pnl_pct: np.ndarray = (exit_price - entry) / np.maximum(entry, _EPSILON)
        else:
            adverse_move = worst_price - entry
            favorable_move = entry - best_price
            expiry_price = close[chunk_start + horizon : chunk_end + horizon]
            exit_price = np.where(
                stop_first, sl_price, np.where(target_first, tp_price, expiry_price)
            )
            pnl_pct = (entry - exit_price) / np.maximum(entry, _EPSILON)

        adverse_move = np.maximum(adverse_move, 0.0)
        favorable_move = np.maximum(favorable_move, 0.0)

        # --- Exit-model targets ------------------------------------------
        # The best price reached anywhere inside the horizon defines the TP that
        # would have been optimal; the heat suffered on the way there defines the
        # SL that would have survived it.
        horizon_favorable: np.ndarray = running_favorable[:, -1]
        if is_long:
            peak_index: np.ndarray = np.argmax(future_high, axis=1)
            heat_to_peak: np.ndarray = entry - running_adverse[rows_index, peak_index]
            optimal_tp: np.ndarray = (horizon_favorable - entry) / np.maximum(entry, _EPSILON)
        else:
            peak_index = np.argmin(future_low, axis=1)
            heat_to_peak = running_adverse[rows_index, peak_index] - entry
            optimal_tp = (entry - horizon_favorable) / np.maximum(entry, _EPSILON)

        heat_to_peak = np.maximum(heat_to_peak, 0.0)
        optimal_tp = np.maximum(optimal_tp, 0.0) * 0.9  # leave room for slippage
        # Floor the stop at half the volatility-scaled distance: a stop tighter
        # than that sits inside ordinary 5m noise and is not a realistic target
        # for the exit model to learn.
        optimal_sl: np.ndarray = (heat_to_peak / np.maximum(entry, _EPSILON)) * 1.15
        floor_sl: np.ndarray = 0.5 * sl_dist / np.maximum(entry, _EPSILON)
        optimal_sl = np.maximum(optimal_sl, floor_sl)

        # A handful of altcoin candles carry genuine >50%-in-4h excursions
        # (flash pumps, thin-book dumps).  Left unclipped, those rare rows
        # dominate the L2 loss the exit regressors are fit with and drag their
        # predictions into the same blown-up range on ordinary rows too - the
        # exact failure mode the ML diagnostic report's exit-model R^2 (deeply
        # negative, predictions reaching into the thousands of percent) was
        # pointing at.  Clamping to the same hard rails the trading engine
        # already enforces on every exit geometry (see
        # ``module_c_ml.ml_models._MIN_TP_PCT`` / ``_MAX_TP_PCT`` / etc.) keeps
        # the *label* consistent with what a trade could ever actually use,
        # instead of training the regressor to chase unusable outliers.
        optimal_tp = np.clip(optimal_tp, _MIN_TP_PCT, _MAX_TP_PCT)
        optimal_sl = np.clip(optimal_sl, _MIN_SL_PCT, _MAX_SL_PCT)

        target = slice(chunk_start, chunk_end)
        simulation.outcome[target] = outcome
        simulation.mae_ratio[target] = np.clip(
            adverse_move / np.maximum(sl_dist, _EPSILON), 0.0, 2.0
        )
        simulation.mfe_ratio[target] = favorable_move / np.maximum(tp_dist, _EPSILON)
        simulation.bars_to_exit[target] = exit_index.astype(np.float64) + 1.0
        simulation.pnl_pct[target] = pnl_pct
        simulation.optimal_tp_pct[target] = optimal_tp
        simulation.optimal_sl_pct[target] = optimal_sl
        simulation.optimal_trailing_pct[target] = optimal_tp * 0.5

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def _volatility_percentile(self, frame: pd.DataFrame) -> np.ndarray:
        """Rolling GARCH volatility percentile at entry, defaulting to the median."""
        for column in ("garch_vol_rank", "atr_rank"):
            if column in frame.columns:
                values: np.ndarray = frame[column].to_numpy(dtype=np.float64)
                return np.nan_to_num(values, nan=0.5)
        return np.full(len(frame), 0.5, dtype=np.float64)

    def _classify(
        self,
        long_side: _SideSimulation,
        short_side: _SideSimulation,
        volatility_percentile: np.ndarray,
    ) -> tuple[list[str], list[str], np.ndarray]:
        """Fuse both simulated sides into a single label and risk tier.

        Selection rules, applied in order:

        1. A side qualifies only when its take-profit barrier resolved the trade.
        2. If both sides qualify (a whipsaw that ran both ways inside the
           horizon), the side that resolved *first* wins; ties are broken by the
           lower MAE ratio, i.e. the less painful path.
        3. The winner's tier is derived from its MAE ratio, then escalated by the
           volatility regime.  A ``VERY_HIGH`` tier collapses to
           ``NO_TRADE_OR_FAIL`` when ``discard_very_high_risk`` is set.  The
           tier is recorded (for the Risk model target) but no longer changes
           *which* label the Direction model sees - see the module docstring.
        """
        rows: int = long_side.outcome.size
        labels: list[str] = []
        tiers: list[str] = []
        chosen: np.ndarray = np.zeros(rows, dtype=np.int8)  # +1 long, -1 short, 0 none

        long_win: np.ndarray = long_side.outcome == TradeOutcome.TAKE_PROFIT.value
        short_win: np.ndarray = short_side.outcome == TradeOutcome.TAKE_PROFIT.value
        unresolved: np.ndarray = long_side.outcome == TradeOutcome.UNRESOLVED.value

        for index in range(rows):
            if unresolved[index]:
                labels.append(LabelClass.NO_TRADE_OR_FAIL.value)
                tiers.append(RiskTier.NONE.value)
                continue

            take_long: bool = bool(long_win[index])
            take_short: bool = bool(short_win[index])

            if take_long and take_short:
                long_bars: float = float(long_side.bars_to_exit[index])
                short_bars: float = float(short_side.bars_to_exit[index])
                if long_bars < short_bars:
                    take_short = False
                elif short_bars < long_bars:
                    take_long = False
                elif float(long_side.mae_ratio[index]) <= float(short_side.mae_ratio[index]):
                    take_short = False
                else:
                    take_long = False

            if not take_long and not take_short:
                labels.append(LabelClass.NO_TRADE_OR_FAIL.value)
                tiers.append(RiskTier.NONE.value)
                continue

            side: _SideSimulation = long_side if take_long else short_side
            tier: str = self._risk_tier(
                float(side.mae_ratio[index]), float(volatility_percentile[index])
            )

            if tier == RiskTier.VERY_HIGH.value and self._config.discard_very_high_risk:
                labels.append(LabelClass.NO_TRADE_OR_FAIL.value)
                tiers.append(tier)
                continue

            if take_long:
                labels.append(LabelClass.LONG_SUCCESS.value)
                chosen[index] = 1
            else:
                labels.append(LabelClass.SHORT_SUCCESS.value)
                chosen[index] = -1
            tiers.append(tier)

        return labels, tiers, chosen

    def _risk_tier(self, mae_ratio: float, volatility_percentile: float) -> str:
        """Map path heat plus the entry volatility regime onto a risk tier.

        ``mae_ratio`` is the maximum adverse excursion expressed as a fraction of
        the stop distance: 0.0 means the trade never went offside, 1.0 means it
        touched the stop.  Volatility escalates the tier because the same heat is
        far more dangerous when the next candle can be three times as large.
        """
        if not np.isfinite(mae_ratio):
            return RiskTier.VERY_HIGH.value

        config: LabelSettings = self._config
        if mae_ratio <= config.low_risk_mae_ratio:
            base: int = 0
        elif mae_ratio <= config.medium_risk_mae_ratio:
            base = 1
        elif mae_ratio <= config.high_risk_mae_ratio:
            base = 2
        else:
            base = 3

        if volatility_percentile >= config.extreme_volatility_percentile:
            base = 3
        elif volatility_percentile >= config.high_volatility_percentile:
            base = min(3, base + 1)

        return _TIER_ORDER[base]

    # ------------------------------------------------------------------
    # Per-model targets
    # ------------------------------------------------------------------
    def _attach_model_targets(
        self,
        frame: pd.DataFrame,
        long_side: _SideSimulation,
        short_side: _SideSimulation,
        chosen_side: np.ndarray,
        tiers: list[str],
        volatility_percentile: np.ndarray,
    ) -> pd.DataFrame:
        """Derive the Entry, Exit and Risk model targets from the chosen side.

        * **Entry model** - binary: was this bar a *well-timed* entry?  A bar
          qualifies only when the selected side won *and* the trade never gave
          back more than ``low_risk_mae_ratio`` of its stop.  That is precisely
          the "enter now vs wait one candle" question the model must answer.
        * **Exit model** - three regression targets (TP, SL, trailing trigger)
          taken from the winning side's realised excursion geometry.
        * **Risk model** - a scalar opportunity score in ``[0, 1]`` combining win
          quality, path heat and the volatility regime, which the sizing head
          converts into leverage and capital allocation.
        """
        rows: int = len(frame)
        entry_quality: np.ndarray = np.zeros(rows, dtype=np.float64)
        target_tp: np.ndarray = np.full(rows, np.nan, dtype=np.float64)
        target_sl: np.ndarray = np.full(rows, np.nan, dtype=np.float64)
        target_trailing: np.ndarray = np.full(rows, np.nan, dtype=np.float64)
        risk_score: np.ndarray = np.zeros(rows, dtype=np.float64)

        long_selected: np.ndarray = chosen_side == 1
        short_selected: np.ndarray = chosen_side == -1
        any_selected: np.ndarray = chosen_side != 0

        mae_ratio: np.ndarray = np.where(
            long_selected, long_side.mae_ratio, np.where(short_selected, short_side.mae_ratio, np.nan)
        )
        for source, mask in ((long_side, long_selected), (short_side, short_selected)):
            target_tp[mask] = source.optimal_tp_pct[mask]
            target_sl[mask] = source.optimal_sl_pct[mask]
            target_trailing[mask] = source.optimal_trailing_pct[mask]

        clean_entry: np.ndarray = any_selected & (
            np.nan_to_num(mae_ratio, nan=1.0) <= self._config.low_risk_mae_ratio
        )
        entry_quality[clean_entry] = 1.0

        heat_component: np.ndarray = 1.0 - np.clip(np.nan_to_num(mae_ratio, nan=1.0), 0.0, 1.0)
        volatility_component: np.ndarray = 1.0 - np.clip(volatility_percentile, 0.0, 1.0)
        raw_score: np.ndarray = heat_component * (0.5 + 0.5 * volatility_component)
        risk_score[any_selected] = np.clip(raw_score[any_selected], 0.0, 1.0)

        frame["entry_quality"] = entry_quality
        frame["target_tp_pct"] = target_tp
        frame["target_sl_pct"] = target_sl
        frame["target_trailing_pct"] = target_trailing
        frame["target_risk_score"] = risk_score
        frame["selected_side"] = chosen_side
        frame["selected_mae_ratio"] = mae_ratio
        frame["risk_tier"] = tiers
        return frame
