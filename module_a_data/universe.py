"""Symbol universe discovery, screening and persistence.

The tradeable universe is not a constant in this system.  It is **discovered
from Binance at runtime**, annotated with the metrics that actually decide
whether a perpetual is worth trading with a small account, presented in the web
panel, and then chosen by the operator.  The saved selection lives in SQLite and
drives every downstream stage: ingestion, training and execution.

Screening dimensions
--------------------
Four things disqualify a symbol, and they are checked independently so the panel
can explain *which* one failed:

1. **Liquidity** - 24 h quote volume below ``min_quote_volume_24h``.  Thin books
   turn a 10x position into its own adverse price move.
2. **Spread** - bid/ask wider than ``max_spread_bps``.  On a 5-minute strategy
   the spread is paid on every round trip; a 15 bps spread eats a 2:1 trade
   before the thesis has a chance.
3. **History** - listed for fewer than ``min_history_days``.  Binance publishes
   ``onboardDate`` in the futures exchange info, so this costs no extra request.
4. **Small-capital fit** - this is the screen most universes get wrong.  A coin
   is unusable on a small account when either the exchange's ``minNotional``
   exceeds the smallest position the risk model can open, or one lot-size step
   costs so much that position sizing becomes hopelessly coarse.  A 0.001 BTC
   step is roughly 100 USDT of notional: on a 1 000 USDT account whose smallest
   position is 10 USDT, BTC simply cannot be sized correctly, no matter how
   liquid it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Sequence

from config.settings import Settings, UniverseSettings
from core.exceptions import DataFetchError
from core.logger import get_logger
from core.utils import safe_float, utc_now_ms
from module_a_data.db_handler import DatabaseHandler
from module_a_data.fetcher import BinanceDataFetcher

_LOGGER = get_logger(__name__)

_SELECTION_KEY: Final[str] = "universe_selection"
_MS_PER_DAY: Final[int] = 86_400_000


@dataclass(slots=True)
class SymbolCandidate:
    """One discovered perpetual, with everything the operator needs to judge it."""

    symbol: str
    base: str
    market_id: str
    price: float = 0.0
    quote_volume_24h: float = 0.0
    price_change_pct_24h: float = 0.0
    spread_bps: float = 0.0
    min_notional: float = 0.0
    amount_step: float = 0.0
    max_leverage: int = 0
    listed_days: float = 0.0

    #: Notional cost of a single lot-size increment, in USDT.
    granularity_usdt: float = 0.0
    #: Smallest position the risk model can open at the reference equity.
    smallest_position_usdt: float = 0.0

    passes_liquidity: bool = False
    passes_spread: bool = False
    passes_history: bool = False
    passes_small_capital: bool = False
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        """``True`` when the symbol clears every screen."""
        return (
            self.passes_liquidity
            and self.passes_spread
            and self.passes_history
            and self.passes_small_capital
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for the panel's selection table."""
        return {
            "symbol": self.symbol,
            "base": self.base,
            "price": self.price,
            "quote_volume_24h": self.quote_volume_24h,
            "price_change_pct_24h": self.price_change_pct_24h,
            "spread_bps": self.spread_bps,
            "min_notional": self.min_notional,
            "amount_step": self.amount_step,
            "granularity_usdt": self.granularity_usdt,
            "smallest_position_usdt": self.smallest_position_usdt,
            "max_leverage": self.max_leverage,
            "listed_days": round(self.listed_days, 1),
            "passes_liquidity": self.passes_liquidity,
            "passes_spread": self.passes_spread,
            "passes_history": self.passes_history,
            "passes_small_capital": self.passes_small_capital,
            "eligible": self.eligible,
            "score": round(self.score, 4),
            "reasons": self.reasons,
        }


class UniverseManager:
    """Discovers, screens, persists and serves the tradeable symbol universe."""

    def __init__(
        self,
        settings: Settings,
        fetcher: BinanceDataFetcher,
        database: DatabaseHandler,
    ) -> None:
        self._settings: Settings = settings
        self._config: UniverseSettings = settings.universe
        self._fetcher: BinanceDataFetcher = fetcher
        self._db: DatabaseHandler = database

        self._cache: list[SymbolCandidate] = []
        self._cached_at_ms: int = 0
        self._selection: list[str] = []
        self._selection_loaded: bool = False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    async def discover(self, force_refresh: bool = False) -> list[SymbolCandidate]:
        """Fetch and screen every USDT-M perpetual, best-scoring first.

        Two requests total: ``load_markets`` (cached by ccxt) and one batched
        ``fetch_tickers``.  Results are memoised for
        ``universe.discovery_cache_seconds``.

        Raises:
            DataFetchError: When the exchange metadata cannot be retrieved.
        """
        age_ms: int = utc_now_ms() - self._cached_at_ms
        if self._cache and not force_refresh and age_ms < self._config.discovery_cache_seconds * 1_000:
            return list(self._cache)

        await self._fetcher.load_markets(reload=force_refresh)
        markets: dict[str, Any] = self._fetcher.exchange.markets or {}
        perpetuals: dict[str, Any] = {
            symbol: market
            for symbol, market in markets.items()
            if self._is_tradeable_perpetual(market)
        }
        if not perpetuals:
            raise DataFetchError("no USDT-M perpetual markets returned by the exchange")

        tickers: dict[str, dict[str, Any]] = await self._fetcher.fetch_raw_tickers(
            list(perpetuals)
        )
        _LOGGER.info(
            "Discovered %d USDT-M perpetuals (%d with live tickers)",
            len(perpetuals),
            len(tickers),
        )

        candidates: list[SymbolCandidate] = []
        for symbol, market in perpetuals.items():
            candidate: SymbolCandidate | None = self._build_candidate(
                symbol, market, tickers.get(symbol, {})
            )
            if candidate is not None:
                candidates.append(candidate)

        self._apply_scores(candidates)
        candidates.sort(key=lambda item: (item.eligible, item.score), reverse=True)

        self._cache = candidates
        self._cached_at_ms = utc_now_ms()
        _LOGGER.info(
            "Screened %d candidates: %d eligible",
            len(candidates),
            sum(1 for item in candidates if item.eligible),
        )
        return list(candidates)

    @staticmethod
    def _is_tradeable_perpetual(market: dict[str, Any]) -> bool:
        """``True`` for an active, linear, USDT-settled perpetual swap."""
        return bool(
            market.get("swap")
            and market.get("linear")
            and market.get("active", True)
            and market.get("settle") == "USDT"
            and market.get("quote") == "USDT"
            and not market.get("expiry")
        )

    def _build_candidate(
        self,
        symbol: str,
        market: dict[str, Any],
        ticker: dict[str, Any],
    ) -> SymbolCandidate | None:
        """Merge market metadata with live ticker data and run every screen."""
        info: dict[str, Any] = market.get("info") or {}
        limits: dict[str, Any] = market.get("limits") or {}
        precision: dict[str, Any] = market.get("precision") or {}

        price: float = safe_float(ticker.get("last") or ticker.get("close"), 0.0)
        if price <= 0.0:
            return None

        bid: float = safe_float(ticker.get("bid"), 0.0)
        ask: float = safe_float(ticker.get("ask"), 0.0)
        spread_bps: float = (
            ((ask - bid) / ((ask + bid) / 2.0)) * 10_000.0 if bid > 0.0 and ask > bid else 0.0
        )

        amount_step: float = self._amount_step(precision, limits)
        min_notional: float = safe_float((limits.get("cost") or {}).get("min"), 0.0)
        min_amount: float = safe_float((limits.get("amount") or {}).get("min"), 0.0)
        # Binance enforces both a minimum quantity and a minimum notional; the
        # binding constraint is whichever costs more.
        effective_min_notional: float = max(min_notional, min_amount * price)

        onboard_ms: float = safe_float(info.get("onboardDate"), 0.0)
        listed_days: float = (
            (utc_now_ms() - onboard_ms) / _MS_PER_DAY if onboard_ms > 0.0 else 0.0
        )

        candidate = SymbolCandidate(
            symbol=symbol,
            base=str(market.get("base", "")),
            market_id=str(market.get("id", "")),
            price=price,
            quote_volume_24h=safe_float(
                ticker.get("quoteVolume") or info.get("quoteVolume"), 0.0
            ),
            price_change_pct_24h=safe_float(ticker.get("percentage"), 0.0),
            spread_bps=spread_bps,
            min_notional=effective_min_notional,
            amount_step=amount_step,
            max_leverage=int(safe_float((limits.get("leverage") or {}).get("max"), 0.0)),
            listed_days=listed_days,
            granularity_usdt=amount_step * price,
        )
        self._screen(candidate)
        return candidate

    @staticmethod
    def _amount_step(precision: dict[str, Any], limits: dict[str, Any]) -> float:
        """Resolve the lot-size step across ccxt's two precision conventions.

        ccxt reports ``precision.amount`` either as a step size (0.001) or as a
        number of decimal places (3), depending on the exchange and version.
        Values ``>= 1`` that are whole numbers are treated as decimal places.
        """
        raw: float = safe_float(precision.get("amount"), 0.0)
        if raw <= 0.0:
            return safe_float((limits.get("amount") or {}).get("min"), 0.0)
        if raw >= 1.0 and float(raw).is_integer():
            return float(10.0 ** -int(raw))
        return raw

    def _screen(self, candidate: SymbolCandidate) -> None:
        """Apply the four screens and record a human-readable reason for each."""
        config: UniverseSettings = self._config
        decision = self._settings.decision

        candidate.passes_liquidity = candidate.quote_volume_24h >= config.min_quote_volume_24h
        if not candidate.passes_liquidity:
            candidate.reasons.append(
                f"24h volume {candidate.quote_volume_24h / 1e6:.1f}M < "
                f"{config.min_quote_volume_24h / 1e6:.0f}M USDT"
            )

        # A zero spread means the ticker carried no book; do not fail on missing data.
        candidate.passes_spread = (
            candidate.spread_bps <= config.max_spread_bps or candidate.spread_bps <= 0.0
        )
        if not candidate.passes_spread:
            candidate.reasons.append(
                f"spread {candidate.spread_bps:.1f} bps > {config.max_spread_bps:.1f} bps"
            )

        candidate.passes_history = candidate.listed_days >= config.min_history_days
        if not candidate.passes_history:
            candidate.reasons.append(
                f"listed {candidate.listed_days:.0f}d < {config.min_history_days}d of history"
            )

        # Smallest position the risk model can actually open at the reference equity.
        smallest: float = (
            config.reference_equity
            * decision.min_capital_allocation_pct
            * float(max(1, decision.min_leverage))
        )
        candidate.smallest_position_usdt = smallest

        affordable: bool = candidate.min_notional <= smallest
        if not affordable:
            candidate.reasons.append(
                f"exchange minimum {candidate.min_notional:.2f} USDT > smallest "
                f"position {smallest:.2f} USDT"
            )

        granular: bool = (
            candidate.granularity_usdt <= smallest * config.max_granularity_fraction
            or candidate.granularity_usdt <= 0.0
        )
        if not granular:
            candidate.reasons.append(
                f"lot step costs {candidate.granularity_usdt:.2f} USDT - too coarse to size "
                f"a {smallest:.2f} USDT position"
            )

        candidate.passes_small_capital = affordable and granular

    def _apply_scores(self, candidates: Sequence[SymbolCandidate]) -> None:
        """Rank candidates on liquidity, spread, affordability and history.

        Liquidity and affordability are log-scaled before normalising: raw
        24 h volume spans four orders of magnitude, and without the log the top
        one or two symbols would flatten every other score to zero.
        """
        if not candidates:
            return
        config: UniverseSettings = self._config

        def normalise(values: list[float], invert: bool = False) -> list[float]:
            lowest: float = min(values)
            highest: float = max(values)
            span: float = highest - lowest
            if span <= 0.0:
                return [0.5] * len(values)
            scaled: list[float] = [(value - lowest) / span for value in values]
            return [1.0 - value for value in scaled] if invert else scaled

        liquidity: list[float] = normalise(
            [math.log10(max(1.0, item.quote_volume_24h)) for item in candidates]
        )
        spread: list[float] = normalise(
            [item.spread_bps if item.spread_bps > 0.0 else config.max_spread_bps for item in candidates],
            invert=True,
        )
        affordability: list[float] = normalise(
            [math.log10(max(0.01, item.granularity_usdt)) for item in candidates], invert=True
        )
        history: list[float] = normalise([min(item.listed_days, 1_095.0) for item in candidates])

        total_weight: float = max(
            1e-9,
            config.weight_liquidity
            + config.weight_spread
            + config.weight_affordability
            + config.weight_history,
        )
        for index, candidate in enumerate(candidates):
            candidate.score = (
                config.weight_liquidity * liquidity[index]
                + config.weight_spread * spread[index]
                + config.weight_affordability * affordability[index]
                + config.weight_history * history[index]
            ) / total_weight

    # ------------------------------------------------------------------
    # Suggestion
    # ------------------------------------------------------------------
    async def suggest(self, limit: int | None = None) -> list[str]:
        """Return the top ``limit`` eligible symbols by score.

        This is what the panel's "suggest" button pre-ticks.  It never returns an
        ineligible symbol: if fewer than ``limit`` pass the screens, the shorter
        list is returned rather than padding it with symbols that failed.
        """
        count: int = limit or self._config.target_count
        candidates: list[SymbolCandidate] = await self.discover()
        eligible: list[SymbolCandidate] = [item for item in candidates if item.eligible]
        eligible.sort(key=lambda item: item.score, reverse=True)
        if len(eligible) < count:
            _LOGGER.warning(
                "Only %d symbol(s) pass every screen (asked for %d)", len(eligible), count
            )
        return [item.symbol for item in eligible[:count]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    async def load_selection(self) -> list[str]:
        """Load the operator's saved selection from SQLite."""
        stored: dict[str, Any] | None = await self._db.get_state(_SELECTION_KEY)
        symbols: list[str] = []
        if stored and isinstance(stored.get("symbols"), list):
            symbols = [str(item) for item in stored["symbols"] if str(item).strip()]
        self._selection = symbols
        self._selection_loaded = True
        if symbols:
            _LOGGER.info("Loaded saved universe: %d symbol(s)", len(symbols))
        else:
            _LOGGER.info("No universe saved yet - awaiting selection from the web panel")
        return list(symbols)

    async def get_selection(self) -> list[str]:
        """Return the saved selection, loading it on first access."""
        if not self._selection_loaded:
            await self.load_selection()
        return list(self._selection)

    async def save_selection(
        self,
        symbols: Sequence[str],
        operator: str = "web-panel",
    ) -> dict[str, Any]:
        """Validate and persist a universe selection.

        Symbols are checked against the discovered market list, so a typo or a
        delisted contract is rejected here rather than surfacing as a stream of
        ``BadSymbol`` errors during the next ingestion cycle.

        Returns:
            ``{"symbols": [...], "accepted": n, "rejected": {...}, "changed": bool}``.
        """
        candidates: list[SymbolCandidate] = await self.discover()
        known: dict[str, SymbolCandidate] = {item.symbol: item for item in candidates}

        accepted: list[str] = []
        rejected: dict[str, str] = {}
        seen: set[str] = set()

        for raw in symbols:
            symbol: str = str(raw).strip()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            candidate: SymbolCandidate | None = known.get(symbol)
            if candidate is None:
                rejected[symbol] = "not an active USDT-M perpetual on Binance"
                continue
            accepted.append(symbol)
            if not candidate.eligible:
                _LOGGER.warning(
                    "Universe includes %s which fails a screen (%s) - honouring the "
                    "operator's explicit choice",
                    symbol,
                    "; ".join(candidate.reasons),
                )

        if not accepted:
            raise ValueError("selection is empty: no valid USDT-M perpetual was supplied")

        previous: list[str] = await self.get_selection()
        changed: bool = sorted(previous) != sorted(accepted)

        await self._db.set_state(
            _SELECTION_KEY,
            {
                "symbols": accepted,
                "updated_ms": utc_now_ms(),
                "updated_by": operator,
                "count": len(accepted),
            },
        )
        self._selection = accepted
        self._selection_loaded = True

        _LOGGER.info(
            "Universe saved by %s: %d symbol(s)%s",
            operator,
            len(accepted),
            f", {len(rejected)} rejected" if rejected else "",
        )
        return {
            "symbols": accepted,
            "accepted": len(accepted),
            "rejected": rejected,
            "changed": changed,
        }

    async def clear_selection(self) -> None:
        """Forget the saved universe (returns the system to the picker)."""
        await self._db.set_state(_SELECTION_KEY, {"symbols": [], "updated_ms": utc_now_ms()})
        self._selection = []
        self._selection_loaded = True

    def cached_candidates(self) -> list[SymbolCandidate]:
        """Last discovery result without triggering a network call."""
        return list(self._cache)
