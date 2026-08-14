"""Advanced, strictly causal feature engineering (Module B).

Look-ahead bias is the single failure mode that turns a profitable backtest into
a losing live system, so every transform in this file obeys one rule: **the
value at bar ``t`` may only depend on information available at the close of bar
``t``.**  Concretely:

* Rolling statistics use trailing windows (``min_periods`` set, never
  ``center=True``).
* GARCH is refitted on a *trailing* window and its output at ``t`` is the
  one-step-ahead variance forecast - a quantity that is, by definition, known at
  ``t``.  Between refits the conditional-variance recursion is rolled forward
  with realised returns, which keeps it causal *and* cheap.
* The HMM is refitted on a trailing window, and regimes are produced by a
  **forward filter** (``alpha`` recursion), never by Viterbi/forward-backward
  smoothing over the whole series - smoothing at ``t`` would use data from
  ``t+1 ... T``.
* Micro-structure joins are ``merge_asof(direction="backward")``, so a bar can
  only see order-book/funding snapshots taken at or before its own close.

Feature computation is CPU bound.  :class:`FeatureService` therefore offloads it
to a worker thread (or a process pool) so the asyncio event loop that drives the
5-minute cycle and the FastAPI panel is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Final, Iterator, Sequence

import numpy as np
import pandas as pd

from config.settings import FeatureSettings, Settings
from core.exceptions import FeatureEngineeringError, InsufficientDataError
from core.logger import get_logger
from module_b_features import indicators as ind

_LOGGER = get_logger(__name__)

_EPSILON: Final[float] = 1e-12
#: arch works best on percent-scaled returns; we divide the result back out.
_GARCH_SCALE: Final[float] = 100.0
#: Floor for any denominator derived from `realized_vol_12`.  One tick on a
#: coarse-tick alt is ~1e-4 in log-return terms, so 1e-8 is far below anything
#: a real market prints and only ever binds on genuinely flat candles - where
#: the companion `realized_vol_12_is_zero` flag carries the information instead.
_MIN_REALIZED_VOL: Final[float] = 1e-8


@contextmanager
def _muted_logger(name: str) -> Iterator[None]:
    """Temporarily silence a third-party logger.

    ``hmmlearn``'s EM convergence monitor reports a stalled fit through
    ``logging.warning`` rather than the ``warnings`` module, so it slips past
    the ``warnings.catch_warnings()`` guard already used around every HMM fit
    below.  A short window failing to fully converge is expected and benign -
    the fit still returns its best estimate - so this is pure log noise on a
    hot path called on every rolling refit; muting the logger for the
    duration of the fit call is what the surrounding ``catch_warnings`` block
    already intended to do.
    """
    logger: logging.Logger = logging.getLogger(name)
    previous_level: int = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


class HMMRegime(IntEnum):
    """Canonical, semantically stable market regimes.

    ``hmmlearn`` labels states arbitrarily and the labelling changes on every
    refit.  Feeding a raw state id to a downstream model would therefore be
    meaningless, so raw states are remapped onto this fixed taxonomy using the
    fitted emission means (see :meth:`FeatureEngineer._canonical_state_map`).
    """

    BULL_TREND = 0
    BEAR_TREND = 1
    HIGH_VOLATILITY = 2
    SIDEWAYS = 3
    UNKNOWN = -1


#: Ordered list of columns the ML subsystem consumes.  Order is part of the
#: model contract: a saved booster expects its features in exactly this layout.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # --- trend / direction --------------------------------------------------
    "kama_distance",
    "kama_slope",
    "kama_slope_fast",
    "ema_fast_slow_spread",
    "close_ema_slow_ratio",
    "adx",
    "di_spread",
    # --- structure ----------------------------------------------------------
    "fdi",
    "fdi_trending",
    "fdi_delta",
    "bb_width",
    "bb_position",
    # --- momentum -----------------------------------------------------------
    "rsi",
    "rsi_delta",
    "log_return_1",
    "log_return_3",
    "log_return_12",
    "log_return_48",
    "momentum_rank",
    # --- volatility ---------------------------------------------------------
    "atr_pct",
    "atr_rank",
    "realized_vol_12",
    "realized_vol_48",
    "garch_volatility",
    "garch_vol_rank",
    "garch_vol_ratio",
    "vol_of_vol",
    #: 1.0 when the trailing 12 candles were completely flat.  This used to be
    #: expressed as a NaN in `vol_of_vol`/`garch_vol_ratio`, which cost the whole
    #: row in the processor's `dropna`; it is real information (a dead market),
    #: so it is carried explicitly instead of destroying the sample.
    "realized_vol_12_is_zero",
    # --- path heat / whipsaw -------------------------------------------------
    "wick_ratio",
    "whipsaw_rate",
    # --- regime -------------------------------------------------------------
    "hmm_regime",
    "hmm_prob_bull",
    "hmm_prob_bear",
    "hmm_prob_high_vol",
    "hmm_prob_sideways",
    "hmm_regime_age",
    # --- volume -------------------------------------------------------------
    "volume_zscore",
    "volume_rank",
    "volume_trend",
    "dollar_volume_rank",
    # --- micro-structure / derivatives ---------------------------------------
    # The five order-book/liquidation columns below were removed in 2b56c8a on
    # the premise that "Binance exposes no historical endpoint for either,
    # ever".  That is true of the REST API and false of Binance's own bulk
    # archive: data.binance.vision publishes `bookTicker`, `bookDepth` and
    # `liquidationSnapshot` as daily ZIPs for the full life of each contract.
    # They are restored here and backfilled by
    # module_a_data/archive_loader.py::BinanceArchiveLoader, aggregated onto the
    # 5m candle grid by module_a_data/pipeline.py::backfill_market_microstructure.
    #
    # Unlike the previous incarnation these are *not* zero-filled when absent:
    # a missing snapshot is carried as NaN plus an explicit `_is_missing` flag,
    # so the booster can split on "unobserved" instead of being taught that a
    # neutral book is a market state (see the module docstring on missingness).
    "ob_imbalance",
    "ob_imbalance_delta",
    "ob_spread_bps",
    "ob_spread_rank",
    "liquidation_imbalance",
    "microstructure_is_missing",
    # funding_rate has full history since contract inception; open interest and
    # the positioning ratios are ~30-day-retention limited on the REST API but
    # are also published in full by the archive's `metrics` dataset - see
    # module_a_data/pipeline.py::backfill_futures_metrics.
    "funding_rate",
    "funding_rate_delta",
    "funding_rate_rank",
    "open_interest_change",
    "open_interest_rank",
    "long_short_ratio",
    "taker_buy_sell_ratio",
    # --- session ------------------------------------------------------------
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)


#: Features whose absence is a legitimate, expected state rather than a broken
#: pipeline: the archive's micro-structure coverage starts later than klines for
#: some symbols, and a live cycle may simply have no book reading for the bar.
#: Training keeps such rows (the boosters split on NaN); live inference is
#: allowed to score them too, because refusing to trade whenever the order book
#: is momentarily unobserved would silently halt the whole system.
OPTIONAL_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "ob_imbalance",
        "ob_imbalance_delta",
        "ob_spread_bps",
        "ob_spread_rank",
        "liquidation_imbalance",
    }
)

#: Everything else must be finite before a bar is tradeable - these are the
#: rolling-window features whose absence means "not warmed up yet".
REQUIRED_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in FEATURE_COLUMNS if column not in OPTIONAL_FEATURE_COLUMNS
)


class FeatureEngineer:
    """Builds the full feature matrix for a single symbol.

    The class is *stateless between calls*: :meth:`build` derives everything from
    its arguments, which makes it safe to run inside a thread or a process pool.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._config: FeatureSettings = settings.features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def minimum_rows(self) -> int:
        """Rows required before the slowest feature produces a value."""
        return (
            max(
                self._config.garch_window,
                self._config.hmm_window,
                self._config.rank_window,
                self._config.fdi_window,
                self._config.kama_slow,
                48,
            )
            + 10
        )

    def build(
        self,
        ohlcv: pd.DataFrame,
        futures: pd.DataFrame | None = None,
        order_book: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute every engineered feature for one symbol.

        Args:
            ohlcv: Frame indexed by UTC open time with ``timestamp, open, high,
                low, close, volume`` columns (as produced by
                :meth:`DatabaseHandler.load_ohlcv_dataframe`).
            futures: Optional funding / open-interest / positioning history with
                a ``timestamp`` column.  Joined backward-asof.
            order_book: Accepted for backward compatibility but no longer
                consumed - the order-book depth features it fed (ob_imbalance,
                ob_imbalance_delta, ob_spread_bps, ob_spread_rank) were removed
                from FEATURE_COLUMNS; Binance has no historical order-book
                depth endpoint, so they could never be backfilled for training.

        Returns:
            The input frame plus every column in :data:`FEATURE_COLUMNS`.  Rows
            whose slow features have not warmed up still carry ``NaN`` - dropping
            them is the processor's job, not this function's.

        Raises:
            InsufficientDataError: When fewer than 60 candles are supplied - no
                feature in this module is meaningful below that.
            FeatureEngineeringError: When a transform fails irrecoverably.
        """
        required_columns: set[str] = {"open", "high", "low", "close", "volume"}
        missing: set[str] = required_columns - set(ohlcv.columns)
        if missing:
            raise FeatureEngineeringError("OHLCV frame is missing columns", missing=sorted(missing))
        if len(ohlcv) < 60:
            raise InsufficientDataError(
                "at least 60 candles are required to engineer features", rows=len(ohlcv)
            )

        frame: pd.DataFrame = ohlcv.sort_index().copy()
        if "timestamp" not in frame.columns:
            frame["timestamp"] = (frame.index.view("int64") // 1_000_000).astype("int64")

        try:
            frame = self._add_price_features(frame)
            frame = self._add_volatility_features(frame)
            frame = self._add_risk_features(frame)
            frame = self._add_volume_features(frame)
            frame = self._add_garch_features(frame)
            frame = self._add_hmm_features(frame)
            frame = self._add_microstructure_features(frame, futures, order_book)
            frame = self._add_session_features(frame)
        except (InsufficientDataError, FeatureEngineeringError):
            raise
        except Exception as error:  # pragma: no cover - defensive catch-all
            raise FeatureEngineeringError(
                "feature computation failed", cause=type(error).__name__, detail=str(error)
            ) from error

        for column in FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan

        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame

    # ------------------------------------------------------------------
    # Price / trend / momentum
    # ------------------------------------------------------------------
    def _add_price_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """KAMA, EMA spreads, ADX, FDI, Bollinger, RSI and log returns."""
        close: pd.Series = frame["close"].astype(float)
        high: pd.Series = frame["high"].astype(float)
        low: pd.Series = frame["low"].astype(float)
        config: FeatureSettings = self._config

        log_close: pd.Series = np.log(close.clip(lower=_EPSILON))
        frame["log_return_1"] = log_close.diff(1)
        frame["log_return_3"] = log_close.diff(3)
        frame["log_return_12"] = log_close.diff(12)
        frame["log_return_48"] = log_close.diff(48)
        frame["momentum_rank"] = ind.rolling_percentile_rank(
            frame["log_return_12"].fillna(0.0), config.rank_window
        )

        # --- KAMA ---------------------------------------------------------
        kama_values: pd.Series = ind.kama(
            close, window=config.kama_window, fast=config.kama_fast, slow=config.kama_slow
        )
        frame["kama"] = kama_values
        frame["kama_distance"] = (close - kama_values) / close.replace(0.0, np.nan)
        frame["kama_slope"] = ind.slope(kama_values, window=5)
        frame["kama_slope_fast"] = ind.slope(kama_values, window=2)

        # --- EMA structure -------------------------------------------------
        ema_fast: pd.Series = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_slow: pd.Series = close.ewm(span=48, adjust=False, min_periods=48).mean()
        frame["ema_fast_slow_spread"] = (ema_fast - ema_slow) / close.replace(0.0, np.nan)
        frame["close_ema_slow_ratio"] = close / ema_slow.replace(0.0, np.nan) - 1.0

        # --- Directional movement -----------------------------------------
        adx_values, plus_di, minus_di = ind.adx(high, low, close, window=config.adx_window)
        frame["adx"] = adx_values
        frame["di_spread"] = (plus_di - minus_di) / 100.0

        # --- Fractal dimension --------------------------------------------
        fdi_values: pd.Series = ind.fractal_dimension_index(close, window=config.fdi_window)
        frame["fdi"] = fdi_values
        # 1 when the market is trending (FDI < 1.5), 0 when it is ranging.
        frame["fdi_trending"] = (fdi_values < 1.5).astype(float)
        frame["fdi_delta"] = fdi_values.diff(3)

        # --- Bands and oscillators -----------------------------------------
        _, bb_width, bb_position = ind.bollinger(
            close, window=config.bb_window, num_std=config.bb_std
        )
        frame["bb_width"] = bb_width
        frame["bb_position"] = bb_position

        rsi_values: pd.Series = ind.rsi(close, window=config.rsi_window)
        frame["rsi"] = rsi_values / 100.0
        frame["rsi_delta"] = rsi_values.diff(3) / 100.0

        return frame

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------
    def _add_volatility_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """ATR (absolute and relative), realised volatility and vol-of-vol."""
        close: pd.Series = frame["close"].astype(float)
        config: FeatureSettings = self._config

        atr_values: pd.Series = ind.atr(
            frame["high"].astype(float),
            frame["low"].astype(float),
            close,
            window=config.atr_window,
        )
        frame["atr"] = atr_values
        frame["atr_pct"] = atr_values / close.replace(0.0, np.nan)
        frame["atr_rank"] = ind.rolling_percentile_rank(
            frame["atr_pct"].fillna(0.0), config.rank_window
        )

        log_returns: pd.Series = frame["log_return_1"].fillna(0.0)
        frame["realized_vol_12"] = ind.realized_volatility(log_returns, 12)
        frame["realized_vol_48"] = ind.realized_volatility(log_returns, 48)

        # A dead market - 12 consecutive candles that all closed at the same
        # price, which is routine for a coarse-tick alt in a quiet hour - makes
        # `realized_vol_12` exactly 0.0.  Dividing by it used to emit NaN here
        # and in `garch_vol_ratio`, and the processor's `dropna` across every
        # feature column then destroyed the *whole row*.  Because flat candles
        # cluster in exactly the low-liquidity symbols and quiet periods, that
        # deleted 73% of the training window while leaving validation and test
        # nearly intact - a silent, systematic sample-selection bias rather
        # than the harmless warm-up trim it was logged as.  Flooring the
        # denominator keeps the ratio finite; `realized_vol_12_is_zero`
        # preserves the fact that the market was dead, which is real
        # information the models should be able to split on.
        is_zero_vol: pd.Series = frame["realized_vol_12"].fillna(0.0).abs() <= _EPSILON
        floored_vol: pd.Series = frame["realized_vol_12"].clip(lower=_MIN_REALIZED_VOL)
        frame["realized_vol_12_is_zero"] = is_zero_vol.astype(float)
        frame["vol_of_vol"] = (
            frame["realized_vol_12"]
            .rolling(window=48, min_periods=12)
            .std(ddof=0)
            .div(floored_vol)
        )
        return frame

    # ------------------------------------------------------------------
    # Path heat / whipsaw
    # ------------------------------------------------------------------
    def _add_risk_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Whipsaw / rejection-wick proxies for the Risk model's path-heat target.

        The Risk model predicts how much a trade is likely to get whipsawed
        before resolving (the labeler's ``target_risk_score``, built from the
        max-adverse-excursion ratio). The existing volatility features (ATR,
        realised vol, GARCH) only capture the *magnitude* of price moves, not
        how often direction reverses or how much of a candle's range was a
        rejected wick rather than a genuine directional move - two signals
        that are conceptually closer to "how choppy has this been" than
        "how big have moves been". Both are purely trailing/rolling, so they
        stay causal like every other feature in this module.
        """
        high: pd.Series = frame["high"].astype(float)
        low: pd.Series = frame["low"].astype(float)
        open_price: pd.Series = frame["open"].astype(float)
        close: pd.Series = frame["close"].astype(float)

        candle_range: pd.Series = (high - low).replace(0.0, np.nan)
        body: pd.Series = (close - open_price).abs()
        raw_wick_ratio: pd.Series = 1.0 - (body / candle_range).clip(0.0, 1.0)
        frame["wick_ratio"] = raw_wick_ratio.rolling(window=6, min_periods=3).mean().fillna(0.0)

        log_return: pd.Series = frame["log_return_1"].fillna(0.0)
        sign_flip: pd.Series = (np.sign(log_return) != np.sign(log_return.shift(1))).astype(float)
        frame["whipsaw_rate"] = sign_flip.rolling(window=12, min_periods=6).mean().fillna(0.0)
        return frame

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------
    def _add_volume_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Trailing volume statistics, all expressed as stationary ranks."""
        volume: pd.Series = frame["volume"].astype(float)
        config: FeatureSettings = self._config

        frame["volume_zscore"] = ind.rolling_zscore(volume, config.rank_window)
        frame["volume_rank"] = ind.rolling_percentile_rank(volume, config.rank_window)
        short_mean: pd.Series = volume.rolling(window=12, min_periods=6).mean()
        long_mean: pd.Series = volume.rolling(window=96, min_periods=24).mean()
        frame["volume_trend"] = short_mean / long_mean.replace(0.0, np.nan) - 1.0

        dollar_volume: pd.Series = volume * frame["close"].astype(float)
        frame["dollar_volume_rank"] = ind.rolling_percentile_rank(
            dollar_volume, config.rank_window
        )
        return frame

    # ------------------------------------------------------------------
    # GARCH
    # ------------------------------------------------------------------
    def _add_garch_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach the one-step-ahead GARCH volatility forecast and its rank."""
        log_returns: pd.Series = frame["log_return_1"].fillna(0.0)
        forecast: pd.Series = self._rolling_garch_forecast(log_returns)

        frame["garch_volatility"] = forecast
        frame["garch_vol_rank"] = ind.rolling_percentile_rank(
            forecast.ffill().fillna(0.0), self._config.rank_window
        )
        # Floored, not NaN-ed, for the same reason as `vol_of_vol` above.
        realized: pd.Series = frame["realized_vol_12"].clip(lower=_MIN_REALIZED_VOL)
        frame["garch_vol_ratio"] = forecast / realized
        return frame

    def _rolling_garch_forecast(self, log_returns: pd.Series) -> pd.Series:
        """Causal one-step-ahead conditional volatility from a rolling GARCH fit.

        The model is refitted every ``garch_refit_every`` bars on the trailing
        ``garch_window``.  Between refits the GARCH(1,1) variance recursion is
        rolled forward with realised returns::

            sigma2_{t+1} = omega + alpha * eps_t^2 + beta * sigma2_t

        which uses nothing beyond bar ``t``.  For non-(1,1) orders the last
        refit's forecast is held until the next refit (still causal, just less
        responsive).

        If ``arch`` is unavailable or a fit fails, the series degrades to an
        EWMA volatility estimate rather than emitting ``NaN`` for the rest of the
        run - a degraded feature is recoverable, a hole is not.
        """
        values: np.ndarray = log_returns.to_numpy(dtype=np.float64)
        size: int = values.size
        output: np.ndarray = np.full(size, np.nan, dtype=np.float64)

        config: FeatureSettings = self._config
        window: int = min(config.garch_window, max(100, size // 2))
        if size < window + 5:
            return self._ewma_volatility(log_returns)

        try:
            from arch import arch_model  # imported lazily: heavy optional import
        except ImportError:  # pragma: no cover - exercised only without `arch`
            _LOGGER.warning("`arch` is not installed - falling back to EWMA volatility")
            return self._ewma_volatility(log_returns)

        scaled: np.ndarray = values * _GARCH_SCALE
        is_garch_11: bool = config.garch_p == 1 and config.garch_q == 1

        # Precomputed per-row causal fallback, used for any bar the GARCH
        # recursion cannot cover (see the `not fitted` branch below).  EWMA is
        # itself strictly trailing, so substituting it introduces no look-ahead.
        ewma_fallback: np.ndarray = self._ewma_volatility(log_returns).to_numpy(dtype=np.float64)

        omega: float = 0.0
        alpha: float = 0.0
        beta: float = 0.0
        mu: float = 0.0
        sigma2: float = float(np.var(scaled[:window])) or 1.0
        held_forecast: float = np.nan
        fitted: bool = False

        for index in range(window, size):
            needs_refit: bool = (not fitted) or ((index - window) % config.garch_refit_every == 0)
            if needs_refit:
                sample: np.ndarray = scaled[index - window + 1 : index + 1]
                params: dict[str, float] | None = self._fit_garch(
                    arch_model, sample, config.garch_p, config.garch_q
                )
                if params is not None:
                    omega = params["omega"]
                    alpha = params["alpha"]
                    beta = params["beta"]
                    mu = params["mu"]
                    sigma2 = params["last_variance"]
                    held_forecast = params["forecast_variance"]
                    fitted = True

            if not fitted:
                # Not yet a single usable fit.  `_fit_garch` rejects any
                # parameter set with `alpha + beta >= 1`, which is the *normal*
                # outcome for GARCH(1,1) on high-frequency crypto returns
                # (near-IGARCH behaviour), so this branch can persist for tens
                # of thousands of bars.  Leaving NaN here used to cost the whole
                # row downstream; the series-wide EWMA fallback below only fires
                # when *every* row is NaN, so a partial failure produced a long
                # hole rather than a fallback.  Emit the causal EWMA estimate
                # for this row instead and keep looking for a fit.
                output[index] = ewma_fallback[index]
                continue

            if is_garch_11:
                residual: float = scaled[index] - mu
                forecast_variance: float = omega + alpha * residual * residual + beta * sigma2
                forecast_variance = max(forecast_variance, _EPSILON)
                output[index] = float(np.sqrt(forecast_variance)) / _GARCH_SCALE
                sigma2 = forecast_variance
            else:
                output[index] = float(np.sqrt(max(held_forecast, _EPSILON))) / _GARCH_SCALE

        result: pd.Series = pd.Series(output, index=log_returns.index, name="garch_volatility")
        if bool(result.notna().sum() == 0):
            return self._ewma_volatility(log_returns)
        return result

    @staticmethod
    def _fit_garch(
        arch_model: Any,
        sample: np.ndarray,
        p_order: int,
        q_order: int,
    ) -> dict[str, float] | None:
        """Fit GARCH(p, q) on a trailing sample and extract its parameters.

        Returns ``None`` when the optimiser fails or produces a degenerate
        (non-stationary / non-positive) parameter set, so the caller can keep
        using the previous fit instead of propagating garbage.
        """
        if sample.size < 50 or float(np.std(sample)) <= 0.0:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(
                    sample,
                    mean="Constant",
                    vol="GARCH",
                    p=p_order,
                    q=q_order,
                    dist="normal",
                    rescale=False,
                )
                result = model.fit(disp="off", show_warning=False, options={"maxiter": 200})
        except Exception as error:  # pragma: no cover - optimiser instability
            _LOGGER.debug("GARCH fit failed: %s", error)
            return None

        params: pd.Series = result.params
        omega: float = float(params.get("omega", 0.0))
        alpha: float = float(params.get("alpha[1]", 0.0))
        beta: float = float(params.get("beta[1]", 0.0))
        mu: float = float(params.get("mu", 0.0))

        if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or (alpha + beta) >= 1.0:
            return None

        conditional: np.ndarray = np.asarray(result.conditional_volatility, dtype=np.float64)
        if conditional.size == 0 or not np.isfinite(conditional[-1]):
            return None
        last_variance: float = float(conditional[-1] ** 2)

        try:
            forecast_variance: float = float(
                np.asarray(result.forecast(horizon=1, reindex=False).variance)[-1, 0]
            )
        except Exception:  # pragma: no cover - older arch releases
            residual: float = float(sample[-1] - mu)
            forecast_variance = omega + alpha * residual * residual + beta * last_variance

        if not np.isfinite(forecast_variance) or forecast_variance <= 0.0:
            return None

        return {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "mu": mu,
            "last_variance": last_variance,
            "forecast_variance": forecast_variance,
        }

    @staticmethod
    def _ewma_volatility(log_returns: pd.Series, span: int = 48) -> pd.Series:
        """RiskMetrics-style EWMA volatility - the GARCH fallback."""
        variance: pd.Series = (
            log_returns.pow(2).ewm(span=span, adjust=False, min_periods=span // 2).mean()
        )
        return np.sqrt(variance).rename("garch_volatility")

    # ------------------------------------------------------------------
    # Hidden Markov Model regimes
    # ------------------------------------------------------------------
    def _add_hmm_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach the canonical HMM regime plus its filtered probabilities."""
        observations: np.ndarray = self._hmm_observations(frame)
        regimes, probabilities = self._rolling_hmm_filter(observations)

        frame["hmm_regime"] = regimes.astype(float)
        frame["hmm_prob_bull"] = probabilities[:, HMMRegime.BULL_TREND]
        frame["hmm_prob_bear"] = probabilities[:, HMMRegime.BEAR_TREND]
        frame["hmm_prob_high_vol"] = probabilities[:, HMMRegime.HIGH_VOLATILITY]
        frame["hmm_prob_sideways"] = probabilities[:, HMMRegime.SIDEWAYS]

        # How many consecutive bars the current regime has persisted, normalised.
        regime_series: pd.Series = pd.Series(regimes, index=frame.index)
        block_id: pd.Series = (regime_series != regime_series.shift(1)).cumsum()
        frame["hmm_regime_age"] = regime_series.groupby(block_id).cumcount().div(48.0).clip(upper=5.0)
        return frame

    @staticmethod
    def _hmm_observations(frame: pd.DataFrame) -> np.ndarray:
        """Build the ``(n, 3)`` observation matrix fed to the Gaussian HMM.

        The three channels - signed return, log high/low range and relative
        volume - separate the four regimes the taxonomy targets: direction comes
        from the return channel, whipsaw from the range channel, and
        participation from the volume channel.
        """
        log_return: np.ndarray = frame["log_return_1"].fillna(0.0).to_numpy(dtype=np.float64)

        high: np.ndarray = frame["high"].to_numpy(dtype=np.float64)
        low: np.ndarray = frame["low"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_range: np.ndarray = np.log(np.maximum(high, _EPSILON) / np.maximum(low, _EPSILON))
        log_range = np.nan_to_num(log_range, nan=0.0, posinf=0.0, neginf=0.0)

        volume: pd.Series = frame["volume"].astype(float)
        relative_volume: np.ndarray = (
            volume.div(volume.rolling(window=96, min_periods=12).mean().replace(0.0, np.nan))
            .fillna(1.0)
            .to_numpy(dtype=np.float64)
        )
        relative_volume = np.clip(np.nan_to_num(relative_volume, nan=1.0), 0.0, 20.0)

        return np.column_stack([log_return, log_range, relative_volume])

    def _rolling_hmm_filter(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Refit-and-filter loop producing causal regime labels.

        Returns:
            ``(regimes, probabilities)`` where ``regimes`` holds canonical
            :class:`HMMRegime` values (``-1`` before warm-up) and
            ``probabilities`` is ``(n, 4)`` of filtered regime probabilities.
        """
        size: int = observations.shape[0]
        regimes: np.ndarray = np.full(size, float(HMMRegime.UNKNOWN), dtype=np.float64)
        probabilities: np.ndarray = np.full((size, 4), 0.25, dtype=np.float64)

        config: FeatureSettings = self._config
        window: int = min(config.hmm_window, max(200, size // 2))
        if size < window + 5:
            return regimes, probabilities

        try:
            from hmmlearn.hmm import GaussianHMM  # lazy: heavy optional import
        except ImportError:  # pragma: no cover - exercised only without hmmlearn
            _LOGGER.warning("`hmmlearn` is not installed - regimes degrade to UNKNOWN")
            return regimes, probabilities

        model: Any = None
        state_map: np.ndarray = np.array([], dtype=np.int64)
        scaler_mean: np.ndarray = np.zeros(observations.shape[1])
        scaler_std: np.ndarray = np.ones(observations.shape[1])
        log_transmat: np.ndarray = np.zeros((0, 0))
        means: np.ndarray = np.zeros((0, 0))
        variances: np.ndarray = np.zeros((0, 0))
        alpha_log: np.ndarray = np.zeros(0)

        for index in range(window, size):
            needs_refit: bool = model is None or ((index - window) % config.hmm_refit_every == 0)
            if needs_refit:
                sample: np.ndarray = observations[index - window + 1 : index + 1]
                fitted = self._fit_hmm(GaussianHMM, sample, config)
                if fitted is not None:
                    model = fitted["model"]
                    scaler_mean = fitted["mean"]
                    scaler_std = fitted["std"]
                    state_map = fitted["state_map"]
                    means = fitted["means"]
                    variances = fitted["variances"]
                    log_transmat = fitted["log_transmat"]
                    # Re-seed the forward filter over the entire fitting window so
                    # `alpha` reflects everything known at `index` - and nothing more.
                    standardised: np.ndarray = (sample - scaler_mean) / scaler_std
                    alpha_log = self._forward_filter(
                        standardised,
                        fitted["log_startprob"],
                        log_transmat,
                        means,
                        variances,
                    )

            if model is None:
                continue

            if not needs_refit:
                point: np.ndarray = (observations[index] - scaler_mean) / scaler_std
                emission: np.ndarray = self._diag_gaussian_logpdf(
                    point[np.newaxis, :], means, variances
                )[0]
                alpha_log = self._log_matmul(alpha_log, log_transmat) + emission
                alpha_log -= self._log_sum_exp(alpha_log)

            posterior: np.ndarray = np.exp(alpha_log - self._log_sum_exp(alpha_log))
            canonical: np.ndarray = np.zeros(4, dtype=np.float64)
            for raw_state, mapped in enumerate(state_map):
                canonical[int(mapped)] += float(posterior[raw_state])

            total: float = float(canonical.sum())
            if total > 0.0:
                canonical /= total
            probabilities[index] = canonical
            regimes[index] = float(int(np.argmax(canonical)))

        return regimes, probabilities

    def _fit_hmm(
        self,
        gaussian_hmm: Any,
        sample: np.ndarray,
        config: FeatureSettings,
    ) -> dict[str, Any] | None:
        """Fit a Gaussian HMM on a trailing window and extract its parameters."""
        mean: np.ndarray = sample.mean(axis=0)
        std: np.ndarray = sample.std(axis=0)
        std = np.where(std > _EPSILON, std, 1.0)
        standardised: np.ndarray = (sample - mean) / std

        try:
            with warnings.catch_warnings(), _muted_logger("hmmlearn.base"):
                warnings.simplefilter("ignore")
                model = gaussian_hmm(
                    n_components=config.hmm_states,
                    covariance_type="diag",
                    n_iter=60,
                    tol=1e-3,
                    random_state=config.hmm_seed,
                    init_params="stmc",
                )
                model.fit(standardised)
        except Exception as error:  # pragma: no cover - EM instability
            _LOGGER.debug("HMM fit failed: %s", error)
            return None

        covars: np.ndarray = np.asarray(model.covars_, dtype=np.float64)
        if covars.ndim == 3:
            variances: np.ndarray = np.diagonal(covars, axis1=1, axis2=2).copy()
        else:
            variances = covars.copy()
        variances = np.maximum(variances, 1e-6)

        means: np.ndarray = np.asarray(model.means_, dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_startprob: np.ndarray = np.log(
                np.maximum(np.asarray(model.startprob_, dtype=np.float64), _EPSILON)
            )
            log_transmat: np.ndarray = np.log(
                np.maximum(np.asarray(model.transmat_, dtype=np.float64), _EPSILON)
            )

        return {
            "model": model,
            "mean": mean,
            "std": std,
            "means": means,
            "variances": variances,
            "log_startprob": log_startprob,
            "log_transmat": log_transmat,
            "state_map": self._canonical_state_map(means),
        }

    @staticmethod
    def _canonical_state_map(means: np.ndarray) -> np.ndarray:
        """Map arbitrary HMM state ids onto the stable :class:`HMMRegime` taxonomy.

        The observation channels are ``(return, range, relative_volume)``, all
        standardised, so the emission means are directly interpretable:

        * The state with the largest *range* mean is the whipsaw / high-volatility
          regime.
        * Of the remainder, the largest *return* mean is the bull trend and the
          smallest is the bear trend.
        * Anything left over is the quiet sideways regime.

        Without this remapping the regime feature would change meaning on every
        refit and the downstream models would learn noise.
        """
        state_count: int = means.shape[0]
        mapping: np.ndarray = np.full(state_count, int(HMMRegime.SIDEWAYS), dtype=np.int64)
        if state_count == 0:
            return mapping

        return_means: np.ndarray = means[:, 0]
        range_means: np.ndarray = means[:, 1] if means.shape[1] > 1 else np.zeros(state_count)

        high_vol_state: int = int(np.argmax(range_means))
        mapping[high_vol_state] = int(HMMRegime.HIGH_VOLATILITY)

        remaining: list[int] = [state for state in range(state_count) if state != high_vol_state]
        if not remaining:
            return mapping

        bull_state: int = max(remaining, key=lambda state: float(return_means[state]))
        mapping[bull_state] = int(HMMRegime.BULL_TREND)

        remaining = [state for state in remaining if state != bull_state]
        if remaining:
            bear_state: int = min(remaining, key=lambda state: float(return_means[state]))
            mapping[bear_state] = int(HMMRegime.BEAR_TREND)

        return mapping

    @staticmethod
    def _diag_gaussian_logpdf(
        points: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> np.ndarray:
        """Log density of diagonal-covariance Gaussians, shape ``(n, k)``.

        Implemented directly instead of calling ``hmmlearn``'s private
        ``_compute_log_likelihood`` so the filter does not depend on library
        internals that change between releases.
        """
        deviation: np.ndarray = points[:, np.newaxis, :] - means[np.newaxis, :, :]
        exponent: np.ndarray = -0.5 * np.sum(
            np.square(deviation) / variances[np.newaxis, :, :], axis=2
        )
        normaliser: np.ndarray = -0.5 * np.sum(
            np.log(2.0 * np.pi * variances), axis=1
        )
        return exponent + normaliser[np.newaxis, :]

    @classmethod
    def _forward_filter(
        cls,
        observations: np.ndarray,
        log_startprob: np.ndarray,
        log_transmat: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> np.ndarray:
        """Run the log-domain forward (``alpha``) recursion over a window.

        Only the *filtered* distribution is produced: ``alpha_t`` depends on
        ``x_1 ... x_t`` and never on future observations, which is precisely what
        keeps the regime feature free of look-ahead bias.
        """
        emissions: np.ndarray = cls._diag_gaussian_logpdf(observations, means, variances)
        alpha_log: np.ndarray = log_startprob + emissions[0]
        alpha_log -= cls._log_sum_exp(alpha_log)
        for step in range(1, observations.shape[0]):
            alpha_log = cls._log_matmul(alpha_log, log_transmat) + emissions[step]
            alpha_log -= cls._log_sum_exp(alpha_log)
        return alpha_log

    @staticmethod
    def _log_sum_exp(values: np.ndarray) -> float:
        """Numerically stable ``log(sum(exp(values)))``."""
        maximum: float = float(np.max(values))
        if not np.isfinite(maximum):
            return maximum
        return maximum + float(np.log(np.sum(np.exp(values - maximum))))

    @staticmethod
    def _log_matmul(log_vector: np.ndarray, log_matrix: np.ndarray) -> np.ndarray:
        """Stable ``log(exp(v) @ exp(M))`` for the forward transition step."""
        combined: np.ndarray = log_vector[:, np.newaxis] + log_matrix
        maximum: np.ndarray = np.max(combined, axis=0)
        return maximum + np.log(np.sum(np.exp(combined - maximum[np.newaxis, :]), axis=0))

    # ------------------------------------------------------------------
    # Micro-structure & session
    # ------------------------------------------------------------------
    def _add_microstructure_features(
        self,
        frame: pd.DataFrame,
        futures: pd.DataFrame | None,
        order_book: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Join futures snapshots backward-asof onto the candles.

        ``direction="backward"`` is what enforces causality here: a candle can
        only be matched with a snapshot whose timestamp is ``<=`` its own open
        time.  Missing feeds degrade to neutral constants (zero funding, etc.),
        never to forward-filled future values.

        ``order_book`` carries the 5m-bucketed micro-structure aggregates built
        by :meth:`module_a_data.pipeline.PipelineOrchestrator.
        backfill_market_microstructure` from Binance's ``bookTicker`` and
        ``liquidationSnapshot`` daily archives.  Because those buckets sit on
        the candles' own 5-minute grid, they are joined on the **exact** bucket
        key rather than as-of: an as-of match would silently carry a book
        reading from hours (or days) earlier into a bar that had none, which is
        precisely the "no data looks like a market state" failure that got the
        previous incarnation of these columns deleted.

        Missing buckets are therefore left as ``NaN`` and flagged by
        ``microstructure_is_missing`` rather than zero-filled.  ``NaN`` is a
        value LightGBM splits on natively, so "we could not see the book" stays
        distinguishable from "the book was balanced" all the way into the model.
        """
        config: FeatureSettings = self._config
        frame = frame.copy()
        frame["timestamp"] = frame["timestamp"].astype("int64")

        merged: pd.DataFrame = self._asof_join(frame, futures, "futures")
        merged = self._exact_bucket_join(merged, order_book, "microstructure")
        merged = self._add_order_book_features(merged, config)

        funding: pd.Series = (
            merged.get("funding_rate", pd.Series(0.0, index=merged.index)).astype(float).fillna(0.0)
        )
        merged["funding_rate"] = funding
        merged["funding_rate_delta"] = funding.diff(12).fillna(0.0)
        merged["funding_rate_rank"] = ind.rolling_percentile_rank(
            funding, config.rank_window
        ).fillna(0.5)

        open_interest: pd.Series = (
            merged.get("open_interest", pd.Series(0.0, index=merged.index)).astype(float).fillna(0.0)
        )
        merged["open_interest_change"] = (
            open_interest.pct_change(12).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-5.0, 5.0)
        )
        merged["open_interest_rank"] = ind.rolling_percentile_rank(
            open_interest, config.rank_window
        ).fillna(0.5)

        long_short: pd.Series = (
            merged.get("long_short_ratio", pd.Series(1.0, index=merged.index))
            .astype(float)
            .fillna(1.0)
        )
        merged["long_short_ratio"] = np.log(long_short.clip(lower=0.01))
        taker: pd.Series = (
            merged.get("taker_buy_sell_ratio", pd.Series(1.0, index=merged.index))
            .astype(float)
            .fillna(1.0)
        )
        merged["taker_buy_sell_ratio"] = np.log(taker.clip(lower=0.01))

        merged.index = frame.index
        return merged

    @staticmethod
    def _add_order_book_features(
        frame: pd.DataFrame,
        config: FeatureSettings,
    ) -> pd.DataFrame:
        """Derive the order-book and liquidation columns from joined buckets.

        Every column here is deliberately ``NaN``-preserving.  ``bid_qty``,
        ``ask_qty``, ``spread_bps`` and the two liquidation volumes arrive as
        ``NaN`` for any 5m bucket the archive did not cover; each derived
        feature inherits that, and ``microstructure_is_missing`` records it as a
        first-class signal.
        """
        index: pd.Index = frame.index
        missing_column: pd.Series = pd.Series(np.nan, index=index, dtype="float64")

        bid_qty: pd.Series = frame.get("bid_qty", missing_column).astype(float)
        ask_qty: pd.Series = frame.get("ask_qty", missing_column).astype(float)
        depth: pd.Series = bid_qty + ask_qty
        # A bucket whose top-of-book was observed but empty on both sides is
        # genuinely undefined, not balanced - keep it NaN rather than 0.
        frame["ob_imbalance"] = ((bid_qty - ask_qty) / depth.where(depth > 0.0)).clip(-1.0, 1.0)
        frame["ob_imbalance_delta"] = frame["ob_imbalance"].diff(3)

        spread: pd.Series = frame.get("spread_bps", missing_column).astype(float)
        frame["ob_spread_bps"] = spread.clip(lower=0.0)
        # `Rolling.rank` skips NaN within the window, so a sparse stretch yields
        # a rank over whatever was actually observed instead of a fabricated 0.5.
        frame["ob_spread_rank"] = ind.rolling_percentile_rank(
            frame["ob_spread_bps"], config.rank_window
        )

        liquidation_buy: pd.Series = frame.get("liquidation_buy_volume", missing_column).astype(float)
        liquidation_sell: pd.Series = frame.get("liquidation_sell_volume", missing_column).astype(float)
        liquidation_total: pd.Series = liquidation_buy + liquidation_sell
        # A covered bucket with no liquidations reads 0/0.  That is a real,
        # informative state ("nobody got liquidated"), so it maps to 0.0 - but
        # only when the bucket was covered at all, which the loader guarantees
        # by writing an explicit zero row for every bucket of every covered day.
        frame["liquidation_imbalance"] = np.where(
            liquidation_total.to_numpy() > 0.0,
            (liquidation_buy - liquidation_sell) / liquidation_total.where(liquidation_total > 0.0),
            np.where(liquidation_total.notna().to_numpy(), 0.0, np.nan),
        )

        frame["microstructure_is_missing"] = (
            frame["ob_imbalance"].isna() & frame["ob_spread_bps"].isna()
        ).astype(float)
        return frame

    @staticmethod
    def _exact_bucket_join(
        frame: pd.DataFrame,
        other: pd.DataFrame | None,
        label: str,
    ) -> pd.DataFrame:
        """Join pre-bucketed 5m aggregates on the exact ``timestamp`` key.

        Unlike :meth:`_asof_join` this never carries a stale reading forward: a
        bucket the archive did not cover simply stays ``NaN``.  Causality is
        preserved because the bucket key *is* the candle's own open time and the
        loader only ever emits buckets that have fully closed.
        """
        if other is None or other.empty or "timestamp" not in other.columns:
            _LOGGER.debug("No %s data supplied; those columns stay NaN", label)
            return frame

        right: pd.DataFrame = other.copy()
        right["timestamp"] = right["timestamp"].astype("int64")
        right = right.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

        overlapping: list[str] = [
            column for column in right.columns if column != "timestamp" and column in frame.columns
        ]
        right = right.drop(columns=overlapping)

        original_index: pd.Index = frame.index
        joined: pd.DataFrame = frame.merge(right, on="timestamp", how="left")
        joined.index = original_index
        return joined

    @staticmethod
    def _asof_join(
        frame: pd.DataFrame,
        other: pd.DataFrame | None,
        label: str,
    ) -> pd.DataFrame:
        """Backward as-of join on ``timestamp`` - the causal way to attach snapshots."""
        if other is None or other.empty or "timestamp" not in other.columns:
            _LOGGER.debug("No %s data supplied; micro-structure features stay neutral", label)
            return frame

        right: pd.DataFrame = other.copy()
        right["timestamp"] = right["timestamp"].astype("int64")
        right = right.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

        overlapping: list[str] = [
            column
            for column in right.columns
            if column != "timestamp" and column in frame.columns
        ]
        right = right.drop(columns=overlapping)

        original_index: pd.Index = frame.index
        left: pd.DataFrame = frame.sort_values("timestamp").reset_index(drop=False)
        joined: pd.DataFrame = pd.merge_asof(
            left,
            right,
            on="timestamp",
            direction="backward",
        )
        index_column: str = original_index.name or "index"
        joined = joined.set_index(index_column)
        joined.index.name = original_index.name
        return joined

    @staticmethod
    def _add_session_features(frame: pd.DataFrame) -> pd.DataFrame:
        """Cyclical encodings of the time of day and day of week.

        Crypto trades 24/7 but liquidity is far from uniform; sine/cosine pairs
        let a tree split on "the Asia session" without an artificial 23 -> 0
        discontinuity.
        """
        index: pd.DatetimeIndex = pd.DatetimeIndex(frame.index)
        hour_fraction: np.ndarray = (
            index.hour.to_numpy(dtype=np.float64)
            + index.minute.to_numpy(dtype=np.float64) / 60.0
        ) / 24.0
        day_fraction: np.ndarray = index.dayofweek.to_numpy(dtype=np.float64) / 7.0

        frame["hour_sin"] = np.sin(2.0 * np.pi * hour_fraction)
        frame["hour_cos"] = np.cos(2.0 * np.pi * hour_fraction)
        frame["dow_sin"] = np.sin(2.0 * np.pi * day_fraction)
        frame["dow_cos"] = np.cos(2.0 * np.pi * day_fraction)
        return frame


# ---------------------------------------------------------------------------
# Process-pool entry point (must be module level so it can be pickled).
# ---------------------------------------------------------------------------
def compute_features_worker(
    settings: Settings,
    ohlcv_records: list[dict[str, Any]],
    futures_records: list[dict[str, Any]],
    book_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build features from plain records - picklable wrapper for a process pool."""
    ohlcv: pd.DataFrame = pd.DataFrame.from_records(ohlcv_records)
    if ohlcv.empty:
        return []
    ohlcv.index = pd.to_datetime(ohlcv["timestamp"], unit="ms", utc=True)
    ohlcv.index.name = "open_time"

    futures: pd.DataFrame | None = (
        pd.DataFrame.from_records(futures_records) if futures_records else None
    )
    book: pd.DataFrame | None = pd.DataFrame.from_records(book_records) if book_records else None

    engineer = FeatureEngineer(settings)
    result: pd.DataFrame = engineer.build(ohlcv, futures=futures, order_book=book)
    return result.reset_index().to_dict(orient="records")


class FeatureService:
    """Async facade that keeps CPU-bound feature work off the event loop.

    By default the work is dispatched with :func:`asyncio.to_thread`: pandas and
    NumPy release the GIL for the heavy numeric kernels, so threads already give
    real parallelism, and they avoid the pickling cost of a process pool.  Set
    ``use_process_pool=True`` when running wide backtests where the pure-Python
    portions (GARCH/HMM refit loops) dominate.
    """

    def __init__(self, settings: Settings, use_process_pool: bool = False) -> None:
        self._settings: Settings = settings
        self._engineer: FeatureEngineer = FeatureEngineer(settings)
        self._use_process_pool: bool = use_process_pool
        self._executor: ProcessPoolExecutor | None = None
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(settings.features.max_feature_workers)

    @property
    def engineer(self) -> FeatureEngineer:
        """Expose the underlying synchronous engineer (used by the backtester)."""
        return self._engineer

    def _pool(self) -> ProcessPoolExecutor:
        """Lazily create the process pool."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._settings.features.max_feature_workers
            )
        return self._executor

    async def build(
        self,
        ohlcv: pd.DataFrame,
        futures: pd.DataFrame | None = None,
        order_book: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute features without blocking the caller's event loop."""
        async with self._semaphore:
            if not self._use_process_pool:
                return await asyncio.to_thread(
                    self._engineer.build, ohlcv, futures, order_book
                )

            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            records: list[dict[str, Any]] = await loop.run_in_executor(
                self._pool(),
                compute_features_worker,
                self._settings,
                ohlcv.reset_index(drop=True).to_dict(orient="records"),
                [] if futures is None else futures.to_dict(orient="records"),
                [] if order_book is None else order_book.to_dict(orient="records"),
            )
            if not records:
                return pd.DataFrame()
            frame: pd.DataFrame = pd.DataFrame.from_records(records)
            if "open_time" in frame.columns:
                frame = frame.set_index("open_time")
            return frame

    async def build_many(
        self,
        payloads: Sequence[tuple[str, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]],
    ) -> dict[str, pd.DataFrame]:
        """Build features for many symbols concurrently.

        Symbols whose computation raises are logged and omitted from the result
        rather than failing the entire cycle.
        """

        async def _one(
            symbol: str,
            ohlcv: pd.DataFrame,
            futures: pd.DataFrame | None,
            book: pd.DataFrame | None,
        ) -> tuple[str, pd.DataFrame | None]:
            try:
                return symbol, await self.build(ohlcv, futures, book)
            except (FeatureEngineeringError, InsufficientDataError) as error:
                _LOGGER.error("Feature build failed for %s: %s", symbol, error)
                return symbol, None

        results: list[tuple[str, pd.DataFrame | None]] = await asyncio.gather(
            *(_one(symbol, ohlcv, futures, book) for symbol, ohlcv, futures, book in payloads)
        )
        return {symbol: frame for symbol, frame in results if frame is not None}

    async def shutdown(self) -> None:
        """Tear down the process pool, if one was created."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
