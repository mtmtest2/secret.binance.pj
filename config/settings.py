"""Central, strictly-typed configuration for the AI Quant Trading System.

Every tunable constant in the system lives here.  No module is allowed to
hard-code magic numbers: they must be injected from :class:`Settings` so that
behaviour can be audited and reproduced from a single source of truth.

Configuration is loaded from (in order of increasing precedence):

1. The defaults declared in this file.
2. A ``.env`` file located at the project root.
3. Real process environment variables.

Nested groups use a double-underscore delimiter, e.g.::

    DECISION__MIN_DIRECTION_CONFIDENCE=0.72
    RISK__MAX_LEVERAGE=5
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.logger import get_logger

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

TradingMode = Literal["backtest", "paper", "live"]

# ---------------------------------------------------------------------------
# Default universe: ~30 of the deepest USDT-M perpetual futures order books.
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS: Final[tuple[str, ...]] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "TON/USDT:USDT",
    "TRX/USDT:USDT",
    "DOT/USDT:USDT",
    "NEAR/USDT:USDT",
    "LTC/USDT:USDT",
    "BCH/USDT:USDT",
    "MATIC/USDT:USDT",
    "UNI/USDT:USDT",
    "APT/USDT:USDT",
    "FIL/USDT:USDT",
    "ARB/USDT:USDT",
    "OP/USDT:USDT",
    "ATOM/USDT:USDT",
    "INJ/USDT:USDT",
    "SUI/USDT:USDT",
    "SEI/USDT:USDT",
    "TIA/USDT:USDT",
    "AAVE/USDT:USDT",
    "ETC/USDT:USDT",
    "RUNE/USDT:USDT",
    "PEPE/USDT:USDT",
)


class ExchangeSettings(BaseModel):
    """Binance USDT-M futures connectivity parameters."""

    api_key: str = Field(default="", description="Binance API key (futures enabled).")
    api_secret: str = Field(default="", description="Binance API secret.")
    testnet: bool = Field(default=True, description="Route orders to the futures testnet.")
    default_type: str = Field(default="future", description="ccxt defaultType option.")
    request_timeout_ms: int = Field(default=20_000, ge=1_000, le=120_000)
    enable_rate_limit: bool = Field(default=True)

    max_retries: int = Field(default=5, ge=1, le=12, description="Retries per REST call.")
    backoff_base_seconds: float = Field(default=1.0, gt=0.0)
    backoff_max_seconds: float = Field(default=60.0, gt=0.0)
    backoff_jitter: float = Field(default=0.25, ge=0.0, le=1.0)

    max_concurrent_requests: int = Field(default=8, ge=1, le=64)

    #: Throttle every request to this fraction of ccxt's default pacing, so the
    #: exchange's per-IP weight budget is never approached under normal load.
    #: ccxt's built-in throttler paces *dispatch* of every call (independent of
    #: ``max_concurrent_requests``, which only bounds in-flight I/O) by sleeping
    #: ``exchange.rateLimit`` ms between weight-1 requests; dividing that budget
    #: by this fraction is what actually slows the request rate to 80 % of the
    #: exchange's default speed - 1.0 keeps ccxt's own default pacing.
    request_rate_scale: float = Field(default=0.8, gt=0.0, le=1.0)


class DataSettings(BaseModel):
    """Data-ingestion parameters for the 5-minute pipeline."""

    #: Fallback universe used only until the operator saves a selection in the
    #: web panel.  The live universe is discovered from Binance and stored in
    #: the database - see :class:`UniverseSettings` and ``UniverseManager``.
    symbols: tuple[str, ...] = Field(default=DEFAULT_SYMBOLS)
    timeframe: Literal["5m"] = Field(default="5m")
    timeframe_ms: int = Field(default=5 * 60 * 1_000)

    ohlcv_limit: int = Field(default=1_500, ge=50, le=1_500)
    #: 2 full years of 5-minute bars (24 months * 30.4375 days * 288 bars/day
    #: = 210_384), plus a buffer for feature warm-up (the slowest feature -
    #: the rolling HMM/GARCH windows - needs ~1_010 bars before it produces a
    #: value) and QC trimming/gaps. Sized to exactly cover
    #: ``MLSettings.train_months + validation_months + test_months`` (default
    #: 12 + 6 + 6 = 24) with room to spare, so the model's held-out test
    #: split is a genuine, full-width final backtest rather than a window
    #: truncated by how much history was actually fetched.
    history_bootstrap_candles: int = Field(default=212_400, ge=500)
    orderbook_depth: int = Field(default=20, ge=5, le=100)
    orderbook_levels_for_imbalance: int = Field(default=10, ge=1, le=100)

    #: Exclude the still-forming candle: Binance always returns it as the last row.
    drop_unclosed_candle: bool = Field(default=True)

    #: Seconds past the 5-minute boundary at which the cycle fires.  Firing at
    #: exactly :00 races the exchange's own candle close and regularly yields a
    #: missing last bar, so a small offset is the operationally correct default.
    cycle_second_offset: int = Field(default=10, ge=0, le=59)


class UniverseSettings(BaseModel):
    """Screening rules for the tradeable symbol universe.

    The universe is *discovered* from Binance rather than hard-coded: the panel
    lists every USDT-M perpetual, annotated with the metrics below, and the
    operator ticks the ones to trade.  These thresholds decide which rows are
    marked eligible and which are pre-selected by the "suggest" button.
    """

    target_count: int = Field(default=30, ge=1, le=200)
    quote_currency: str = Field(default="USDT")

    #: 24 h quote volume floor - the primary liquidity screen.
    min_quote_volume_24h: float = Field(default=50_000_000.0, ge=0.0)
    #: Bid/ask spread ceiling in basis points, measured at discovery time.
    max_spread_bps: float = Field(default=6.0, gt=0.0)
    #: Days since listing.  Below this there is not enough 5m history to train.
    #: Deliberately left at 365 rather than raised to 547 (1.5x, matching
    #: ``DataSettings.history_bootstrap_candles``): several symbols already in
    #: ``DEFAULT_SYMBOLS`` (e.g. APT, ARB, OP, SUI, SEI, TIA) listed well under
    #: 1.5 years ago, so requiring 547 days would shrink the tradeable universe
    #: for the sake of uniform series length. Per-symbol walk-forward splits
    #: already tolerate ragged history lengths - a newer symbol simply
    #: contributes a shorter, still-valid training/validation series rather
    #: than being padded or excluded. Revisit if the universe should instead
    #: favour fewer, longer-lived symbols.
    min_history_days: int = Field(default=365, ge=1)

    #: Account size the small-capital screens are calibrated against.
    reference_equity: float = Field(default=1_000.0, gt=0.0)
    #: One lot-size step must not cost more than this share of the smallest
    #: position the risk model can open.  This is what rejects coins whose
    #: quantity granularity is too coarse for a small account (e.g. a 0.001 BTC
    #: step is ~100 USDT of notional, unusable when the smallest position is 10).
    max_granularity_fraction: float = Field(default=0.25, gt=0.0, le=1.0)

    #: Re-discovery interval; market metadata does not change minute to minute.
    discovery_cache_seconds: float = Field(default=900.0, gt=0.0)

    #: Relative weights of the ranking score (normalised internally).
    weight_liquidity: float = Field(default=0.45, ge=0.0)
    weight_spread: float = Field(default=0.25, ge=0.0)
    weight_affordability: float = Field(default=0.20, ge=0.0)
    weight_history: float = Field(default=0.10, ge=0.0)


class QCSettings(BaseModel):
    """Quality-Control gatekeeper thresholds (see ``qc_validator.py``)."""

    max_heal_attempts: int = Field(default=4, ge=1, le=10)
    heal_backoff_seconds: float = Field(default=2.0, gt=0.0)
    #: Hard wall-clock ceiling on one symbol's total heal loop, regardless of how
    #: many attempts remain in the budget.  Bounds worst-case latency so a symbol
    #: stuck healing cannot indefinitely hold the shared request-rate budget and
    #: starve every other symbol's cycle.
    max_heal_duration_seconds: float = Field(default=90.0, gt=0.0)
    #: Suspicious timestamps within this many bars of each other are healed as
    #: one contiguous re-fetch window instead of two separate ones.
    heal_merge_gap_bars: int = Field(default=3, ge=0)
    #: When a heal attempt would otherwise need more distinct windows than this,
    #: it falls back to batched windows spanning the damaged range - fragmenting
    #: further would trade a handful of extra requests for no real precision.
    max_heal_window_groups: int = Field(default=12, ge=1)
    #: Hard cap, in bars, on the span of any single fallback batch window. Without
    #: this, widespread damage across a long history could otherwise collapse
    #: into one unbounded re-fetch of tens of thousands of candles; instead the
    #: full damaged range is split into controlled, bounded-size batches.
    max_heal_window_bars: int = Field(default=2_000, ge=50)

    #: A candle whose volume exceeds ``median * this`` is flagged as an anomaly.
    volume_spike_median_multiple: float = Field(default=50.0, gt=1.0)
    #: Robust z-score (MAD based) above which a log-return is flagged.
    return_mad_zscore_limit: float = Field(default=14.0, gt=1.0)
    #: Absolute per-candle return ceiling (fraction, 0.35 == 35 %).
    max_abs_candle_return: float = Field(default=0.35, gt=0.0, le=1.0)
    #: Tolerated fraction of zero-volume candles inside a fetched block.
    max_zero_volume_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Maximum acceptable staleness of the newest closed candle, in bars.
    max_lag_bars: int = Field(default=3, ge=1, le=50)
    #: Minimum rows required before statistical (non-structural) checks run.
    min_rows_for_statistics: int = Field(default=30, ge=5)


class FeatureSettings(BaseModel):
    """Feature-engineering hyper-parameters (Module B)."""

    kama_window: int = Field(default=10, ge=2)
    kama_fast: int = Field(default=2, ge=1)
    kama_slow: int = Field(default=30, ge=2)

    fdi_window: int = Field(default=30, ge=5)
    atr_window: int = Field(default=14, ge=2)
    rsi_window: int = Field(default=14, ge=2)
    adx_window: int = Field(default=14, ge=2)
    bb_window: int = Field(default=20, ge=2)
    bb_std: float = Field(default=2.0, gt=0.0)

    garch_window: int = Field(default=500, ge=100)
    garch_refit_every: int = Field(default=50, ge=1)
    garch_p: int = Field(default=1, ge=1, le=3)
    garch_q: int = Field(default=1, ge=1, le=3)

    hmm_states: int = Field(default=4, ge=2, le=8)
    hmm_window: int = Field(default=1_000, ge=200)
    hmm_refit_every: int = Field(default=100, ge=1)
    hmm_seed: int = Field(default=42)

    #: Rolling window used to convert raw values into stationary percentiles.
    rank_window: int = Field(default=288, ge=20)  # 288 bars == 24 h of 5m candles

    max_feature_workers: int = Field(default=4, ge=1, le=32)


class LabelSettings(BaseModel):
    """Forward-looking, risk-tiered label generation (Module B)."""

    tp_atr_multiple: float = Field(default=2.0, gt=0.0)
    sl_atr_multiple: float = Field(default=1.0, gt=0.0)
    max_holding_bars: int = Field(default=48, ge=2)  # 48 * 5m == 4 h

    #: MAE (max adverse excursion) expressed as a fraction of the SL distance.
    low_risk_mae_ratio: float = Field(default=0.35, gt=0.0, lt=1.0)
    medium_risk_mae_ratio: float = Field(default=0.60, gt=0.0, lt=1.0)
    high_risk_mae_ratio: float = Field(default=0.85, gt=0.0, le=1.0)

    #: Rolling percentile of GARCH volatility above which the regime is "extreme".
    extreme_volatility_percentile: float = Field(default=0.95, gt=0.0, lt=1.0)
    high_volatility_percentile: float = Field(default=0.80, gt=0.0, lt=1.0)

    #: Trades whose tier resolves to VERY_HIGH are folded into NO_TRADE_OR_FAIL.
    discard_very_high_risk: bool = Field(default=True)


class MLSettings(BaseModel):
    """Machine-learning subsystem configuration (Module C)."""

    model_dir: Path = Field(default=PROJECT_ROOT / "artifacts" / "models")
    booster: Literal["lightgbm", "xgboost"] = Field(default="lightgbm")
    random_state: int = Field(default=42)

    n_estimators: int = Field(default=400, ge=10)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    max_depth: int = Field(default=6, ge=1, le=32)
    num_leaves: int = Field(default=63, ge=2)
    subsample: float = Field(default=0.85, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.85, gt=0.0, le=1.0)
    min_child_samples: int = Field(default=40, ge=1)
    reg_lambda: float = Field(default=1.0, ge=0.0)

    #: Strict, chronological 3-way split (never a random shuffle). ``test`` is
    #: anchored to the most recent data and is the model's final backtest
    #: window: it is never touched by training, early stopping, calibration,
    #: threshold selection or model selection - only by the one-shot backtest
    #: replay run after everything else is already frozen. ``validation`` is
    #: the block immediately before it, used for every development decision;
    #: ``train`` is everything older. Defaults sum to 24 months (2 full years)
    #: - see ``DataSettings.history_bootstrap_candles``.
    train_months: float = Field(default=12.0, gt=0.0)
    validation_months: float = Field(default=6.0, gt=0.0)
    test_months: float = Field(default=6.0, gt=0.0)
    #: Embargo gap (bars) cut from the trailing edge of train and of
    #: validation, converted to a *time* duration (``purge_bars *
    #: DataSettings.timeframe_ms``) and applied uniformly regardless of how
    #: many symbols share a timestamp in the pooled dataset. Should stay >=
    #: ``LabelSettings.max_holding_bars`` (flagged, if not, by
    #: ``Settings._warn_if_purge_too_short_for_label_horizon``) - a label
    #: simulated from a row inside the embargo can look forward past the
    #: split boundary into the next block, which is exactly the leakage this
    #: gap exists to prevent.
    purge_bars: int = Field(default=60, ge=0)
    early_stopping_rounds: int = Field(default=50, ge=0)

    #: Exponential time-decay half-life (days) for training sample weights: a
    #: row this many days behind the most recent training row gets half the
    #: weight, one that far again gets a quarter, and so on.  Crypto regimes
    #: drift, so a year-old candle should not vote as loudly as yesterday's.
    #: ``0`` disables recency weighting (every row weighted equally).
    recency_half_life_days: float = Field(default=45.0, ge=0.0)

    inference_workers: int = Field(default=2, ge=1, le=16)

    #: Experimental. Swaps the gate stage's (Direction model, stage 1) LightGBM
    #: objective for a focal-loss custom objective that down-weights the easy,
    #: confidently-correct majority region and up-weights the ambiguous
    #: near-0.5 region - see ``module_c_ml.ml_models.focal_loss_binary`` for why
    #: this is not yet validated against production log-loss. Off by default;
    #: the direction (stage 2) estimator and every other head are unaffected
    #: regardless of this flag.
    use_focal_loss_for_gate: bool = Field(default=False)
    focal_loss_gamma: float = Field(default=2.0, gt=0.0)


class DecisionSettings(BaseModel):
    """Decision Engine thresholds (Module D)."""

    #: Removed: this field used to be RiskModel's own hard-veto/sizing floor,
    #: read against the stale *joint* long/short/no_trade confidence
    #: (`DirectionPrediction.confidence`). RiskModel.predict now receives the
    #: same independent direction-given-trade conditional confidence the
    #: Decision Engine's own R1B rule gates on (see MLSubsystem.infer_sync),
    #: so it was repointed at `min_direction_given_trade_confidence` below
    #: instead - a second, differently-scoped threshold for the same
    #: quantity would just be a second place to forget to update. Nothing in
    #: the repo reads `min_direction_confidence` any more (grepped clean).
    max_no_trade_probability: float = Field(default=0.35, gt=0.0, le=1.0)
    min_entry_probability: float = Field(default=0.55, gt=0.0, lt=1.0)

    #: Stage-1: is this bar worth trading at all (gate's own probability,
    #: not the multiplied joint one). Replaces gating on the product, which
    #: silently discarded confident direction calls whenever the gate alone
    #: was <0.5.
    min_gate_confidence: float = Field(default=0.55, gt=0.0, lt=1.0)
    #: Stage-2, conditional on the gate already saying "trade": how sure is
    #: LONG vs SHORT. Also the threshold RiskModel.predict gates its own hard
    #: veto and sizing curve against (see min_direction_confidence's removal
    #: note above) - both consumers now read the identical, independent
    #: conditional-confidence signal off the same threshold.
    min_direction_given_trade_confidence: float = Field(default=0.60, gt=0.0, lt=1.0)

    min_leverage: int = Field(default=1, ge=0, le=10)
    max_leverage: int = Field(default=10, ge=1, le=10)
    min_capital_allocation_pct: float = Field(default=0.01, gt=0.0, le=1.0)
    max_capital_allocation_pct: float = Field(default=0.20, gt=0.0, le=1.0)

    #: Reject when the model-implied reward/risk ratio drops below this.
    min_reward_risk_ratio: float = Field(default=1.3, gt=0.0)
    #: Rolling GARCH volatility percentile above which trading is suspended.
    max_volatility_percentile: float = Field(default=0.97, gt=0.0, le=1.0)

    accepted_risk_tiers: tuple[str, ...] = Field(default=("LOW", "MEDIUM", "HIGH"))
    blocked_hmm_regimes: tuple[int, ...] = Field(default=())

    max_concurrent_positions: int = Field(default=5, ge=1, le=50)
    max_positions_per_symbol: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def _validate_leverage_band(self) -> "DecisionSettings":
        if self.min_leverage > self.max_leverage:
            raise ValueError("min_leverage cannot exceed max_leverage")
        if self.min_capital_allocation_pct > self.max_capital_allocation_pct:
            raise ValueError("min_capital_allocation_pct cannot exceed the max")
        return self


class RiskSettings(BaseModel):
    """Risk Guard / kill-switch configuration (Module E)."""

    starting_equity: float = Field(default=1_000.0, gt=0.0)
    max_leverage: int = Field(default=10, ge=1, le=10)

    daily_drawdown_red_pct: float = Field(default=0.05, gt=0.0, lt=1.0)
    daily_drawdown_yellow_pct: float = Field(default=0.03, gt=0.0, lt=1.0)
    total_drawdown_red_pct: float = Field(default=0.20, gt=0.0, lt=1.0)

    consecutive_losses_yellow: int = Field(default=3, ge=1)
    consecutive_losses_red: int = Field(default=5, ge=1)

    api_errors_yellow: int = Field(default=3, ge=1)
    api_errors_red: int = Field(default=6, ge=1)
    api_error_window_seconds: float = Field(default=300.0, gt=0.0)

    #: Multiplier applied to position sizing while the system sits in YELLOW.
    yellow_size_multiplier: float = Field(default=0.4, gt=0.0, le=1.0)

    max_daily_trades: int = Field(default=40, ge=1)
    require_manual_reset: bool = Field(default=True)

    @model_validator(mode="after")
    def _validate_ladders(self) -> "RiskSettings":
        if self.daily_drawdown_yellow_pct >= self.daily_drawdown_red_pct:
            raise ValueError("YELLOW drawdown threshold must be below the RED one")
        if self.consecutive_losses_yellow > self.consecutive_losses_red:
            raise ValueError("YELLOW loss streak must not exceed the RED one")
        if self.api_errors_yellow > self.api_errors_red:
            raise ValueError("YELLOW API error budget must not exceed the RED one")
        return self


class ExecutionSettings(BaseModel):
    """Order-routing, fee and simulation parameters (Module E)."""

    margin_mode: Literal["isolated", "cross"] = Field(default="isolated")
    order_type: Literal["market", "limit"] = Field(default="market")
    limit_offset_bps: float = Field(default=2.0, ge=0.0)
    limit_order_timeout_seconds: float = Field(default=45.0, gt=0.0)

    taker_fee: float = Field(default=0.0005, ge=0.0)  # 0.05 %
    maker_fee: float = Field(default=0.0002, ge=0.0)  # 0.02 %
    slippage_bps: float = Field(default=5.0, ge=0.0)  # 0.05 %

    position_monitor_interval_seconds: float = Field(default=5.0, gt=0.0)
    reconcile_interval_seconds: float = Field(default=60.0, gt=0.0)

    #: Binance charges/credits funding every 8 hours (00:00, 08:00, 16:00 UTC).
    funding_interval_hours: int = Field(default=8, ge=1)
    maintenance_margin_rate: float = Field(default=0.005, gt=0.0, lt=1.0)

    paper_starting_balance: float = Field(default=1_000.0, gt=0.0)


class WebSettings(BaseModel):
    """FastAPI monitoring panel settings (Module F)."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65_535)
    title: str = Field(default="AI Quant Futures Panel")
    audit_page_size: int = Field(default=100, ge=10, le=1_000)
    #: Optional shared secret required by the mutating ``/api/*`` endpoints.
    api_token: str = Field(default="")


class DatabaseSettings(BaseModel):
    """SQLite / SQLAlchemy async engine settings."""

    path: Path = Field(default=PROJECT_ROOT / "data" / "quant.db")
    echo: bool = Field(default=False)
    busy_timeout_ms: int = Field(default=10_000, ge=100)
    journal_mode: Literal["WAL", "DELETE", "TRUNCATE"] = Field(default="WAL")
    pool_size: int = Field(default=5, ge=1, le=50)

    @property
    def url(self) -> str:
        """Return the SQLAlchemy async connection URL."""
        return f"sqlite+aiosqlite:///{self.path}"


class Settings(BaseSettings):
    """Root configuration object injected across every module."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="ai-quant-binance-futures")
    trading_mode: TradingMode = Field(default="paper")
    #: One-run automation.  `python main.py` collects data and trains by itself;
    #: trading is left OFF so the operator arms it deliberately from the panel.
    auto_setup_on_start: bool = Field(
        default=True, description="Run data collection + training automatically at startup."
    )
    auto_train: bool = Field(
        default=True, description="Train missing/stale models as part of auto-setup."
    )
    autostart_paper_trading: bool = Field(
        default=False, description="Arm paper trading as soon as setup finishes."
    )

    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs")
    timezone: str = Field(default="UTC")

    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    qc: QCSettings = Field(default_factory=QCSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    labels: LabelSettings = Field(default_factory=LabelSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    decision: DecisionSettings = Field(default_factory=DecisionSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level: str = value.upper()
        allowed: set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @model_validator(mode="after")
    def _prepare_directories(self) -> "Settings":
        """Create the runtime directories eagerly so no I/O path can fail later."""
        for directory in (self.log_dir, self.ml.model_dir, self.db.path.parent):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def _warn_if_purge_too_short_for_label_horizon(self) -> "Settings":
        """Flag (never raise on) a purge/embargo gap too short to cover the
        label horizon.

        A label simulated from a row at position ``t`` looks forward up to
        ``labels.max_holding_bars`` candles to resolve. The purge/embargo gap
        cut at every split boundary (``ml.purge_bars``) must therefore be at
        least that wide, or a training/validation row just inside the gap
        can have a label that peeks across the boundary into the next
        block - reintroducing exactly the leakage the gap exists to
        prevent. This only warns (rather than rejects the config) so small,
        deliberately-scaled-down configurations in tests/experiments are not
        blocked; production should never actually run with this warning
        active.
        """
        if self.ml.purge_bars < self.labels.max_holding_bars:
            get_logger(__name__).warning(
                "ml.purge_bars=%d is smaller than labels.max_holding_bars=%d - the "
                "train/validation/test embargo may not fully cover the label "
                "horizon, risking leakage across split boundaries. Set "
                "purge_bars >= max_holding_bars before training on real data.",
                self.ml.purge_bars,
                self.labels.max_holding_bars,
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide singleton :class:`Settings` instance."""
    return Settings()
