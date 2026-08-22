"""Vectorised technical indicators implemented directly on NumPy/pandas.

``ta-lib`` requires a compiled C extension, which is a poor fit for a portable
VPS deployment, so every indicator the system needs is implemented here from
first principles.  Two properties are non-negotiable throughout this module:

* **Causality.**  The value at index ``t`` is computed exclusively from data at
  indices ``<= t``.  No ``center=True`` rolling windows, no backward fills, no
  ``shift(-k)``.
* **Determinism.**  Given the same input frame the output is bit-for-bit
  reproducible, which is what makes backtest results comparable to live ones.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range.

    ``TR = max(high - low, |high - prev_close|, |low - prev_close|)``.
    """
    previous_close: pd.Series = close.shift(1)
    ranges: pd.DataFrame = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1, skipna=False)


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing (an EMA with ``alpha = 1 / window``)."""
    return series.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    return wilder_smooth(true_range(high, low, close), window)


def rsi(close: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index in ``[0, 100]``.

    Gains and losses are smoothed with Wilder's method.  A period with no
    average loss yields 100 (and no average gain yields 0) instead of ``inf``.
    """
    delta: pd.Series = close.diff()
    gains: pd.Series = delta.clip(lower=0.0)
    losses: pd.Series = (-delta).clip(lower=0.0)

    avg_gain: pd.Series = wilder_smooth(gains, window)
    avg_loss: pd.Series = wilder_smooth(losses, window)

    relative_strength: pd.Series = avg_gain / avg_loss.replace(0.0, np.nan)
    result: pd.Series = 100.0 - (100.0 / (1.0 + relative_strength))
    result = result.where(avg_loss > 0.0, 100.0)
    result = result.where(avg_gain > 0.0, other=result.where(avg_loss <= 0.0, 0.0))
    return result.clip(lower=0.0, upper=100.0)


def kama(
    close: pd.Series,
    window: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """Kaufman's Adaptive Moving Average.

    KAMA adapts its smoothing constant to the *efficiency ratio* - the share of
    net directional movement in the total path travelled::

        ER_t   = |close_t - close_{t-n}| / sum(|close_i - close_{i-1}|, i in (t-n, t])
        SC_t   = (ER_t * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        KAMA_t = KAMA_{t-1} + SC_t * (close_t - KAMA_{t-1})

    In a clean trend ``ER -> 1`` and KAMA tracks price almost tick for tick; in
    chop ``ER -> 0`` and KAMA flattens out, which is exactly the noise filter the
    direction model needs.

    The recursion is inherently sequential, so it is evaluated in a single
    ``O(n)`` NumPy loop rather than a (much slower) ``rolling.apply``.
    """
    if window < 2:
        raise ValueError("kama window must be >= 2")

    values: np.ndarray = close.to_numpy(dtype=np.float64, copy=True)
    size: int = values.size
    output: np.ndarray = np.full(size, np.nan, dtype=np.float64)
    if size <= window:
        return pd.Series(output, index=close.index, name="kama")

    absolute_change: np.ndarray = np.abs(np.diff(values, prepend=values[0]))
    # Rolling sum of |price changes| over the efficiency window (volatility term).
    volatility: np.ndarray = (
        pd.Series(absolute_change).rolling(window=window, min_periods=window).sum().to_numpy()
    )
    direction: np.ndarray = np.abs(values - np.roll(values, window))
    direction[:window] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency_ratio: np.ndarray = np.where(volatility > 0.0, direction / volatility, 0.0)
    efficiency_ratio = np.nan_to_num(efficiency_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    efficiency_ratio = np.clip(efficiency_ratio, 0.0, 1.0)

    fastest: float = 2.0 / (float(fast) + 1.0)
    slowest: float = 2.0 / (float(slow) + 1.0)
    smoothing: np.ndarray = np.square(efficiency_ratio * (fastest - slowest) + slowest)

    # Seed with the simple average of the first `window` closes.
    output[window] = float(np.mean(values[: window + 1]))
    for index in range(window + 1, size):
        previous: float = output[index - 1]
        output[index] = previous + smoothing[index] * (values[index] - previous)

    return pd.Series(output, index=close.index, name="kama")


def fractal_dimension_index(close: pd.Series, window: int = 30) -> pd.Series:
    """Fractal Dimension Index (FDI) over a rolling window.

    The FDI measures how much of the plane a price path fills.  Within each
    window the closes are min-max normalised to ``[0, 1]``, the path length is
    accumulated on a grid whose vertical step is the normalised price change and
    whose horizontal step is ``1/n``, and the dimension follows from::

        FDI = 1 + (ln(path_length) + ln(2)) / ln(2n)

    Interpretation:

    * ``FDI ~ 1.0`` - a straight line, i.e. a pure trend.
    * ``FDI ~ 1.5`` - a random walk (no exploitable structure).
    * ``FDI ~ 2.0`` - the path fills the plane, i.e. violent chop / ranging.

    The rolling window is evaluated with ``sliding_window_view`` so the whole
    series is computed in a handful of vectorised NumPy operations.
    """
    if window < 3:
        raise ValueError("fdi window must be >= 3")

    values: np.ndarray = close.to_numpy(dtype=np.float64)
    size: int = values.size
    output: np.ndarray = np.full(size, np.nan, dtype=np.float64)
    if size < window:
        return pd.Series(output, index=close.index, name="fdi")

    windows: np.ndarray = np.lib.stride_tricks.sliding_window_view(values, window)
    window_min: np.ndarray = windows.min(axis=1, keepdims=True)
    window_max: np.ndarray = windows.max(axis=1, keepdims=True)
    span: np.ndarray = window_max - window_min

    # A perfectly flat window has no fractal structure; treat it as a random walk.
    flat_mask: np.ndarray = (span <= 0.0).ravel()
    safe_span: np.ndarray = np.where(span > 0.0, span, 1.0)
    normalised: np.ndarray = (windows - window_min) / safe_span

    vertical_steps: np.ndarray = np.diff(normalised, axis=1)
    horizontal_step: float = 1.0 / float(window)
    segment_lengths: np.ndarray = np.sqrt(
        np.square(vertical_steps) + (horizontal_step * horizontal_step)
    )
    path_length: np.ndarray = segment_lengths.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        dimension: np.ndarray = 1.0 + (np.log(path_length) + np.log(2.0)) / np.log(
            2.0 * float(window)
        )
    dimension = np.where(flat_mask, 1.5, dimension)
    dimension = np.clip(np.nan_to_num(dimension, nan=1.5, posinf=2.0, neginf=1.0), 1.0, 2.0)

    output[window - 1 :] = dimension
    return pd.Series(output, index=close.index, name="fdi")


#: Below this a directional index carries no usable movement; DX computed
#: against it is a ratio of noise to noise.
_MIN_DIRECTIONAL_INDEX: Final[float] = 1e-9


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index with its +DI / -DI components."""
    up_move: pd.Series = high.diff()
    down_move: pd.Series = -low.diff()

    plus_dm: pd.Series = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=high.index
    )
    minus_dm: pd.Series = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=high.index
    )

    atr_values: pd.Series = atr(high, low, close, window)
    safe_atr: pd.Series = atr_values.replace(0.0, np.nan)

    plus_di: pd.Series = 100.0 * wilder_smooth(plus_dm, window) / safe_atr
    minus_di: pd.Series = 100.0 * wilder_smooth(minus_dm, window) / safe_atr

    di_sum: pd.Series = (plus_di + minus_di).replace(0.0, np.nan)
    directional_index: pd.Series = 100.0 * (plus_di - minus_di).abs() / di_sum

    # DX is scale-invariant: 100*|a-b|/(a+b) is exactly 100 whenever one DI is
    # zero, however small the other. On a flat stretch punctuated by a single
    # directional tick that is precisely the situation - one DM is one tick, the
    # other is 0 - so DX pins at its ceiling without any division by zero, and
    # the guards on `safe_atr`/`di_sum` never fire. Requiring both sides to carry
    # movement is what keeps the reading meaningful; where they do not, the value
    # is undefined and NaN says so. Filling it with 0.0 (as this did) injects a
    # value into the Wilder smoother that then propagates across the window, and
    # 0 is an ordinary ADX reading rather than a marker for "no information".
    both_present: pd.Series = (plus_di > _MIN_DIRECTIONAL_INDEX) & (minus_di > _MIN_DIRECTIONAL_INDEX)
    directional_index = directional_index.where(both_present)
    adx_values: pd.Series = wilder_smooth(directional_index, window)

    return adx_values, plus_di, minus_di


def bollinger(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger bands, returned as ``(middle, width_pct, position)``.

    ``position`` is ``%b``: 0 at the lower band, 1 at the upper band.
    """
    middle: pd.Series = close.rolling(window=window, min_periods=window).mean()
    deviation: pd.Series = close.rolling(window=window, min_periods=window).std(ddof=0)

    upper: pd.Series = middle + num_std * deviation
    lower: pd.Series = middle - num_std * deviation

    width: pd.Series = (upper - lower) / middle.replace(0.0, np.nan)
    band_span: pd.Series = (upper - lower).replace(0.0, np.nan)
    position: pd.Series = (close - lower) / band_span

    return middle, width.fillna(0.0), position.clip(lower=-1.0, upper=2.0)


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rank the current value inside its trailing window, scaled to ``[0, 1]``.

    This is the workhorse that turns non-stationary quantities (volatility,
    volume, open interest) into stationary features a gradient-boosted tree can
    actually generalise from.  ``Rolling.rank`` ranks the *last* element of each
    window, so the transform is causal by construction.
    """
    if window < 2:
        raise ValueError("rank window must be >= 2")
    return series.rolling(window=window, min_periods=max(5, window // 10)).rank(pct=True)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score, guarded against zero-variance windows."""
    mean: pd.Series = series.rolling(window=window, min_periods=max(5, window // 10)).mean()
    std: pd.Series = series.rolling(window=window, min_periods=max(5, window // 10)).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).fillna(0.0)


def realized_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    """Trailing realised volatility (standard deviation of log returns)."""
    return log_returns.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=0)


def slope(series: pd.Series, window: int) -> pd.Series:
    """Normalised slope: the ``window``-bar change divided by the current level.

    Expressed relative to price so the value is comparable across symbols whose
    absolute prices differ by six orders of magnitude (BTC vs PEPE).
    """
    change: pd.Series = series.diff(window)
    return change / series.abs().replace(0.0, np.nan)
