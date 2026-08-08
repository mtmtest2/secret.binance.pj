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

1. **Intra-candle ordering.**  5-minute OHLC data alone cannot tell us whether
   the high or the low came first within a bar.  When a single candle touches
   both barriers, the labeler asks the caller for that candle's own
   **1-minute sub-candles** (``IntrabarLookup``, see :meth:`TradeLabeler.generate`)
   and walks them in time order to find out which barrier was actually struck
   first - real data, not a guess.  Whenever that data was not supplied (or a
   bar remains ambiguous even at 1-minute resolution), the labeler falls back
   to the conservative assumption that the **stop** was hit first; any other
   fallback would manufacture phantom winners.  ``generate()`` reports every
   candle it had to fall back on via ``result.attrs["ambiguous_candle_timestamps"]``
   so the caller can fetch the missing 1-minute data and re-run labeling.
2. **Barrier fills, not close fills.**  A resolved trade exits exactly at its
   barrier price; only expired trades mark to the horizon close.
3. **Heat is measured, not ignored.**  The maximum adverse excursion (MAE) along
   the path is recorded and expressed as a fraction of the stop distance.  A
   winner whose path spent most of its time near the stop (and/or fired during
   an extreme-volatility regime) only worked because the market was violent,
   not because of a real edge, so it is folded back into ``NO_TRADE_OR_FAIL``
   (see ``discard_very_high_risk`` on :class:`~config.settings.LabelSettings`)
   instead of being counted as a win.

Output classes
---------------
The Direction model is deliberately kept to three classes - ``LONG_SUCCESS``,
``SHORT_SUCCESS`` and ``NO_TRADE_OR_FAIL`` - rather than further split by path
risk.  MAE ratio and the volatility regime still gate *whether* a trade counts
as a win at all (see above); they no longer produce a separate LOW/HIGH_RISK
label for the model to learn.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import LabelSettings, Settings
from core.exceptions import InsufficientDataError, LabelingError
from core.logger import get_logger

_LOGGER = get_logger(__name__)

_EPSILON: Final[float] = 1e-12
#: Rows processed per vectorised chunk, bounding peak memory on long histories.
_CHUNK_ROWS: Final[int] = 20_000

#: One 1-minute sub-candle as ``(open, high, low, close)``, supplied by the
#: caller for the intra-candle barrier-order refinement (see ``TradeLabeler``).
SubCandle = tuple[float, float, float, float]
#: 5m-candle-open-timestamp (ms) -> its 1-minute sub-candles, ascending in time.
IntrabarLookup = Mapping[int, Sequence[SubCandle]]


class LabelClass(str, Enum):
    """The 3-class target consumed by the Market Direction model.

    Deliberately kept flat - no LOW/HIGH_RISK split - so the model learns a
    single, undiluted notion of "this setup worked" per direction instead of
    four overlapping success classes.
    """

    LONG_SUCCESS = "LONG_SUCCESS"
    SHORT_SUCCESS = "SHORT_SUCCESS"
    NO_TRADE_OR_FAIL = "NO_TRADE_OR_FAIL"


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
        "tie",
        "exit_bar_index",
        "pending_intrabar",
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
        #: ``True`` where this side's exit bar touched *both* barriers, so raw
        #: OHLC alone could not order them (see ``_resolve_ties_with_intrabar``).
        self.tie: np.ndarray = np.zeros(rows, dtype=bool)
        #: Global row index of the (would-be) exit bar for tie rows, ``-1``
        #: elsewhere. Used to look up that candle's 1-minute sub-candles.
        self.exit_bar_index: np.ndarray = np.full(rows, -1, dtype=np.int64)
        #: Tie rows still resolved via the conservative fallback because no
        #: (or insufficiently precise) 1-minute data was supplied.
        self.pending_intrabar: np.ndarray = np.zeros(rows, dtype=bool)


class TradeLabeler:
    """Generates multi-class, risk-tiered labels and per-model regression targets."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._config: LabelSettings = settings.labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        frame: pd.DataFrame,
        intrabar: IntrabarLookup | None = None,
    ) -> pd.DataFrame:
        """Attach every training target to a feature frame.

        Args:
            frame: Feature frame from :class:`FeatureEngineer` - must contain
                ``timestamp``, ``high``, ``low``, ``close`` and ``atr``
                (``garch_volatility`` is used when ATR has not warmed up).
            intrabar: Optional ``{5m candle open timestamp (ms): [(open, high,
                low, close), ...]}`` map of 1-minute sub-candles, ascending in
                time, used to resolve same-bar TP/SL ambiguity precisely (see
                the module docstring). Candles absent from this mapping fall
                back to the conservative stop-first assumption and are listed
                in the returned frame's ``result.attrs["ambiguous_candle_timestamps"]``
                so the caller can fetch them and call ``generate`` again.

        Returns:
            A copy of ``frame`` with the following columns added:

            ============================  ==========================================
            ``label``                     3-class target (Direction model)
            ``label_index``               Integer encoding of ``label``
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

            Also sets ``result.attrs["ambiguous_candle_timestamps"]``: the sorted
            list of 5m candle open timestamps (ms) whose barrier order is still
            resolved via the conservative fallback because ``intrabar`` did not
            cover them (or was ``None``).

        Raises:
            InsufficientDataError: When the frame is shorter than the holding
                horizon plus one bar.
            LabelingError: When required columns are absent.
        """
        required: set[str] = {"timestamp", "high", "low", "close"}
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
        timestamps: np.ndarray = frame["timestamp"].to_numpy(dtype=np.int64)
        high: np.ndarray = frame["high"].to_numpy(dtype=np.float64)
        low: np.ndarray = frame["low"].to_numpy(dtype=np.float64)
        close: np.ndarray = frame["close"].to_numpy(dtype=np.float64)

        tp_distance, sl_distance = self._barrier_distances(frame, close)

        long_side: _SideSimulation = self._simulate(
            high, low, close, timestamps, tp_distance, sl_distance, horizon,
            is_long=True, intrabar=intrabar,
        )
        short_side: _SideSimulation = self._simulate(
            high, low, close, timestamps, tp_distance, sl_distance, horizon,
            is_long=False, intrabar=intrabar,
        )

        ambiguous_timestamps: list[int] = []
        if self._config.refine_ambiguous_barriers:
            total_tie: int = int(np.count_nonzero(long_side.tie)) + int(
                np.count_nonzero(short_side.tie)
            )
            total_pending: int = int(np.count_nonzero(long_side.pending_intrabar)) + int(
                np.count_nonzero(short_side.pending_intrabar)
            )
            if total_tie:
                _LOGGER.info(
                    "Intra-candle barrier refinement: %d/%d ambiguous same-bar "
                    "TP/SL touches resolved with 1-minute data (%d still on the "
                    "conservative stop-first fallback)",
                    total_tie - total_pending,
                    total_tie,
                    total_pending,
                )
            needed: set[int] = set()
            for side in (long_side, short_side):
                pending_index: np.ndarray = side.exit_bar_index[side.pending_intrabar]
                needed.update(int(value) for value in timestamps[pending_index])
            ambiguous_timestamps = sorted(needed)

        volatility_percentile: np.ndarray = self._volatility_percentile(frame)
        labels, chosen_side = self._classify(long_side, short_side, volatility_percentile)

        result["label"] = labels
        result["label_index"] = [LABEL_TO_INDEX.get(name, LABEL_TO_INDEX[
            LabelClass.NO_TRADE_OR_FAIL.value
        ]) for name in labels]
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
            result, long_side, short_side, chosen_side, volatility_percentile
        )

        distribution: dict[str, int] = (
            result.loc[result["label_is_valid"], "label"].value_counts().to_dict()
        )
        _LOGGER.info("Label distribution: %s", distribution)
        result.attrs["ambiguous_candle_timestamps"] = ambiguous_timestamps
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
        timestamps: np.ndarray,
        tp_distance: np.ndarray,
        sl_distance: np.ndarray,
        horizon: int,
        is_long: bool,
        intrabar: IntrabarLookup | None,
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
                timestamps,
                tp_distance,
                sl_distance,
                horizon,
                is_long,
                chunk_start,
                chunk_end,
                intrabar,
            )
        return simulation

    def _simulate_chunk(
        self,
        simulation: _SideSimulation,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timestamps: np.ndarray,
        tp_distance: np.ndarray,
        sl_distance: np.ndarray,
        horizon: int,
        is_long: bool,
        chunk_start: int,
        chunk_end: int,
        intrabar: IntrabarLookup | None,
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

        # Default, conservative tie-break: when both barriers are touched on
        # the same candle we cannot know the intra-candle order from raw OHLC
        # alone, so the stop wins unless the refinement below overturns it.
        stop_first: np.ndarray = sl_any & (~tp_any | (first_sl <= first_tp))
        target_first: np.ndarray = tp_any & ~stop_first

        # True ambiguity is *only* the same-bar case (first_tp == first_sl):
        # whenever the two first-touch indices differ, one barrier genuinely
        # resolved on an earlier candle and there is nothing to refine.
        rows_index: np.ndarray = np.arange(count)
        tie: np.ndarray = tp_any & sl_any & (first_tp == first_sl)
        exit_bar_index: np.ndarray = chunk_start + 1 + rows_index + first_sl
        pending: np.ndarray = np.zeros(count, dtype=bool)
        if self._config.refine_ambiguous_barriers and np.any(tie):
            flip_to_target, pending = self._resolve_ties_with_intrabar(
                tie, exit_bar_index, timestamps, tp_price, sl_price, is_long, intrabar
            )
            stop_first = stop_first & ~flip_to_target
            target_first = tp_any & (target_first | flip_to_target)
            simulation.tie[chunk_start:chunk_end] = tie
            simulation.exit_bar_index[chunk_start:chunk_end] = exit_bar_index
            simulation.pending_intrabar[chunk_start:chunk_end] = pending

        exit_index: np.ndarray = np.where(
            stop_first, first_sl, np.where(target_first, first_tp, horizon - 1)
        )

        outcome: np.ndarray = np.full(count, TradeOutcome.EXPIRED.value, dtype=object)
        outcome[stop_first] = TradeOutcome.STOP_LOSS.value
        outcome[target_first] = TradeOutcome.TAKE_PROFIT.value

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
    # Intra-candle barrier-order refinement
    # ------------------------------------------------------------------
    def _resolve_ties_with_intrabar(
        self,
        tie: np.ndarray,
        exit_bar_index: np.ndarray,
        timestamps: np.ndarray,
        tp_price: np.ndarray,
        sl_price: np.ndarray,
        is_long: bool,
        intrabar: IntrabarLookup | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve same-bar TP/SL touches using that candle's real 1-minute path.

        5-minute bars only record four prices; when both barriers sit inside a
        single bar's ``[low, high]`` range, the raw data cannot say which was
        struck first.  Rather than guessing, this looks up the exit candle's
        own 1-minute sub-candles (supplied by the caller in ``intrabar``, keyed
        by the 5m candle's open timestamp) and walks them in time order to see
        which barrier price was actually crossed first.

        Args:
            tie: Boolean mask, rows where TP and SL first-touch on the same bar.
            exit_bar_index: Global row index of that bar, one per row.
            timestamps: Full-length candle-open-timestamp array (ms).
            tp_price, sl_price: Absolute barrier prices for this chunk's rows.
            is_long: Which side is being resolved.
            intrabar: ``{5m timestamp: [(open, high, low, close), ...]}`` or
                ``None`` when no 1-minute data has been fetched yet.

        Returns:
            ``(flip_to_target, still_pending)``, both chunk-local boolean masks.
            ``flip_to_target`` marks rows where the sub-candles showed TP struck
            first.  ``still_pending`` marks tie rows that kept the conservative
            stop-first default because no data was available for that candle,
            or the sub-candles were themselves still ambiguous down to the
            minute - these are reported back to the caller so it can fetch the
            missing window and try again.
        """
        flip_to_target: np.ndarray = np.zeros(tie.shape, dtype=bool)
        still_pending: np.ndarray = np.zeros(tie.shape, dtype=bool)
        if not np.any(tie):
            return flip_to_target, still_pending

        for position in np.nonzero(tie)[0]:
            bar_timestamp: int = int(timestamps[exit_bar_index[position]])
            sub_candles: Sequence[SubCandle] | None = (
                intrabar.get(bar_timestamp) if intrabar is not None else None
            )
            if not sub_candles:
                still_pending[position] = True
                continue

            winner: str | None = self._first_touch_from_subcandles(
                sub_candles, float(tp_price[position]), float(sl_price[position]), is_long
            )
            if winner == "tp":
                flip_to_target[position] = True
            elif winner is None:
                # The 1-minute candles didn't disambiguate either (e.g. both
                # barriers sit inside one of them) - stay conservative, and
                # there is nothing coarser left to fetch, so this is *not*
                # reported as still-pending.
                continue

        return flip_to_target, still_pending

    @staticmethod
    def _first_touch_from_subcandles(
        sub_candles: Sequence[SubCandle],
        tp_price: float,
        sl_price: float,
        is_long: bool,
    ) -> str | None:
        """Walk 1-minute sub-candles in time order to find the first barrier hit.

        Returns ``"tp"``, ``"sl"``, or ``None`` when neither was touched in the
        supplied candles or a single sub-candle touches both (still ambiguous,
        just five times less often than at 5m resolution).
        """
        for _open, high, low, close in sub_candles:
            tp_hit: bool = high >= tp_price if is_long else low <= tp_price
            sl_hit: bool = low <= sl_price if is_long else high >= sl_price
            if tp_hit and sl_hit:
                return None
            if tp_hit:
                return "tp"
            if sl_hit:
                return "sl"
        return None

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
    ) -> tuple[list[str], np.ndarray]:
        """Fuse both simulated sides into a single 3-class label.

        Selection rules, applied in order:

        1. A side qualifies only when its take-profit barrier resolved the trade.
        2. If both sides qualify (a whipsaw that ran both ways inside the
           horizon), the side that resolved *first* wins; ties are broken by the
           lower MAE ratio, i.e. the less painful path.
        3. A winner is discarded back to ``NO_TRADE_OR_FAIL`` when its path heat
           and the entry volatility regime mark it as very-high-risk (see
           :meth:`_is_very_high_risk`) - a trade that only worked because the
           market was violent is not an edge worth learning, but this no longer
           produces a separate output class.
        """
        rows: int = long_side.outcome.size
        labels: list[str] = []
        chosen: np.ndarray = np.zeros(rows, dtype=np.int8)  # +1 long, -1 short, 0 none

        long_win: np.ndarray = long_side.outcome == TradeOutcome.TAKE_PROFIT.value
        short_win: np.ndarray = short_side.outcome == TradeOutcome.TAKE_PROFIT.value
        unresolved: np.ndarray = long_side.outcome == TradeOutcome.UNRESOLVED.value

        for index in range(rows):
            if unresolved[index]:
                labels.append(LabelClass.NO_TRADE_OR_FAIL.value)
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
                continue

            side: _SideSimulation = long_side if take_long else short_side
            if self._config.discard_very_high_risk and self._is_very_high_risk(
                float(side.mae_ratio[index]), float(volatility_percentile[index])
            ):
                labels.append(LabelClass.NO_TRADE_OR_FAIL.value)
                continue

            if take_long:
                labels.append(LabelClass.LONG_SUCCESS.value)
                chosen[index] = 1
            else:
                labels.append(LabelClass.SHORT_SUCCESS.value)
                chosen[index] = -1

        return labels, chosen

    def _is_very_high_risk(self, mae_ratio: float, volatility_percentile: float) -> bool:
        """Whether path heat plus the entry volatility regime make a win too risky to trust.

        ``mae_ratio`` is the maximum adverse excursion expressed as a fraction of
        the stop distance: 0.0 means the trade never went offside, 1.0 means it
        touched the stop.  Volatility escalates the assessment because the same
        heat is far more dangerous when the next candle can be three times as
        large.  This mirrors the old 4-tier (LOW/MEDIUM/HIGH/VERY_HIGH) escalation
        logic, collapsed to the single boolean the 3-class label needs.
        """
        if not np.isfinite(mae_ratio):
            return True

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

        return base == 3

    # ------------------------------------------------------------------
    # Per-model targets
    # ------------------------------------------------------------------
    def _attach_model_targets(
        self,
        frame: pd.DataFrame,
        long_side: _SideSimulation,
        short_side: _SideSimulation,
        chosen_side: np.ndarray,
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
        return frame
