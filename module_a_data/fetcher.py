"""Asynchronous Binance USDT-M futures data fetcher.

Every network call in this module is wrapped in an exponential-backoff retry
loop that distinguishes between:

* **Transient** failures (``NetworkError``, ``RequestTimeout``,
  ``ExchangeNotAvailable``, ``DDoSProtection``, HTTP 429 ``RateLimitExceeded``)
  which are retried with a jittered exponential delay, and
* **Permanent** failures (``BadSymbol``, ``AuthenticationError``,
  ``PermissionDenied``) which abort immediately - retrying them only burns the
  IP weight budget.

A module-level semaphore bounds in-flight requests so a 30-symbol fan-out never
trips Binance's per-IP weight limit.
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, Final, Sequence

import ccxt.async_support as ccxt

from config.settings import Settings
from core.exceptions import DataFetchError
from core.logger import get_logger
from core.utils import async_retry, backoff_delay, safe_float, utc_now_ms
from module_a_data.models import FuturesMetrics, OHLCVCandle, OrderBookSnapshot

_LOGGER = get_logger(__name__)

#: Errors worth retrying - all of them are transient by nature.
TRANSIENT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
    ccxt.OnMaintenance,
    asyncio.TimeoutError,
)

#: Errors that will never succeed on retry.
PERMANENT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    ccxt.AuthenticationError,
    ccxt.PermissionDenied,
    ccxt.BadSymbol,
    ccxt.ArgumentsRequired,
)


def _safe_int(value: Any) -> int | None:
    """Coerce ``value`` to ``int``, returning ``None`` (never a guessed 0) on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class BinanceDataFetcher:
    """Async gateway to every Binance USDT-M market-data endpoint we consume.

    The instance owns a single ``ccxt.async_support.binance`` client; it must be
    closed via :meth:`close` (or used as an async context manager) so the
    underlying ``aiohttp`` session is released.
    """

    def __init__(self, settings: Settings, exchange: ccxt.binance | None = None) -> None:
        self._settings: Settings = settings
        self._timeframe: str = settings.data.timeframe
        self._timeframe_ms: int = settings.data.timeframe_ms
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(
            settings.exchange.max_concurrent_requests
        )
        self._markets_loaded: bool = False
        self._owns_exchange: bool = exchange is None
        self._exchange: ccxt.binance = exchange if exchange is not None else self._build_exchange()
        #: Counter consumed by the Risk Guard to detect API degradation.
        self.consecutive_errors: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _build_exchange(self) -> ccxt.binance:
        """Instantiate the ccxt client from settings."""
        config: dict[str, Any] = {
            "apiKey": self._settings.exchange.api_key or None,
            "secret": self._settings.exchange.api_secret or None,
            "enableRateLimit": self._settings.exchange.enable_rate_limit,
            "timeout": self._settings.exchange.request_timeout_ms,
            "options": {
                "defaultType": self._settings.exchange.default_type,
                "adjustForTimeDifference": True,
                "recvWindow": 10_000,
            },
        }
        exchange: ccxt.binance = ccxt.binance(config)
        self._restrict_to_linear_markets(exchange)
        # Deliberately never `set_sandbox_mode(True)`. This one client serves
        # every read in the system - the candles training learns from, the
        # funding and open-interest history behind the derivatives features, and
        # the prices the backtest and paper trader replay. Binance's testnet
        # publishes a different, largely synthetic book, and its `fapiData`
        # endpoints do not exist there at all: a full run against it rejected
        # 162 requests and left open_interest_change, long_short_ratio and
        # taker_buy_sell_ratio constant for the entire history, with nothing in
        # the pipeline treating that as a failure.
        self._apply_rate_scale(exchange)
        return exchange

    def _apply_rate_scale(self, exchange: ccxt.binance) -> None:
        """Slow ccxt's built-in request pacing to ``request_rate_scale`` of default.

        ``enableRateLimit`` makes ccxt sleep ``exchange.rateLimit`` milliseconds
        between weight-1 requests regardless of how many coroutines are waiting
        on the semaphore, so this is the one lever that actually controls the
        rate at which requests leave the process - inflating it is what caps
        real throughput at a fraction of the exchange's default speed.
        """
        scale: float = self._settings.exchange.request_rate_scale
        if scale >= 1.0:
            return
        original: float = float(exchange.rateLimit)
        exchange.rateLimit = int(round(original / scale))
        _LOGGER.info(
            "API request rate capped at %.0f%% of default (ccxt rateLimit %d -> %d ms)",
            scale * 100.0,
            int(original),
            exchange.rateLimit,
        )

    @staticmethod
    def _restrict_to_linear_markets(exchange: ccxt.binance) -> None:
        """Load only USDT-M (linear) markets.

        By default ccxt loads spot, linear *and* inverse markets, which costs
        several megabytes and a chunk of request weight this system never uses.
        Worse, in sandbox mode the spot leg is served by ``testnet.binance.vision``
        - an entirely separate testnet from the futures one - so a futures-only
        bot fails to start for a reason that has nothing to do with futures.

        The option's shape changed across ccxt releases (a bare list in 4.2, a
        dict with a ``types`` key in 4.5), so both are handled.
        """
        current: Any = (exchange.options or {}).get("fetchMarkets")
        if isinstance(current, dict):
            exchange.options["fetchMarkets"] = {**current, "types": ["linear"]}
        else:
            exchange.options["fetchMarkets"] = ["linear"]

    @property
    def exchange(self) -> ccxt.binance:
        """Expose the raw ccxt client (used by the execution engine)."""
        return self._exchange

    async def __aenter__(self) -> "BinanceDataFetcher":
        await self.load_markets()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Release the underlying aiohttp session (idempotent)."""
        if not self._owns_exchange:
            return
        try:
            await self._exchange.close()
        except Exception as error:  # pragma: no cover - shutdown best effort
            _LOGGER.warning("Error while closing the ccxt client: %s", error)

    async def load_markets(self, reload: bool = False) -> None:
        """Load (or reload) the market metadata table."""
        if self._markets_loaded and not reload:
            return
        await self._call(
            "load_markets",
            lambda: self._exchange.load_markets(reload),
        )
        self._markets_loaded = True
        _LOGGER.info("Loaded %d Binance markets", len(self._exchange.markets or {}))

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------
    async def _call(self, label: str, operation: Any) -> Any:
        """Execute ``operation`` under the semaphore with retry/backoff.

        Args:
            label: Human-readable operation name used in log lines.
            operation: Zero-argument callable returning an awaitable.

        Raises:
            DataFetchError: Once the retry budget is exhausted, or immediately
                for permanent errors.
        """

        def _log_retry(attempt: int, error: BaseException, delay: float) -> None:
            _LOGGER.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.2fs",
                label,
                attempt + 1,
                self._settings.exchange.max_retries,
                error,
                delay,
            )

        async def _guarded() -> Any:
            async with self._semaphore:
                return await operation()

        try:
            result: Any = await async_retry(
                _guarded,
                attempts=self._settings.exchange.max_retries,
                base_seconds=self._settings.exchange.backoff_base_seconds,
                max_seconds=self._settings.exchange.backoff_max_seconds,
                jitter=self._settings.exchange.backoff_jitter,
                retry_on=TRANSIENT_ERRORS,
                give_up_on=PERMANENT_ERRORS,
                on_error=_log_retry,
            )
        except PERMANENT_ERRORS as error:  # type: ignore[misc]
            self.consecutive_errors += 1
            raise DataFetchError(f"{label} failed permanently", reason=str(error)) from error
        except TRANSIENT_ERRORS as error:  # type: ignore[misc]
            self.consecutive_errors += 1
            raise DataFetchError(f"{label} exhausted retries", reason=str(error)) from error
        except ccxt.ExchangeError as error:
            self.consecutive_errors += 1
            raise DataFetchError(f"{label} rejected by exchange", reason=str(error)) from error

        self.consecutive_errors = 0
        return result

    def _market_id(self, symbol: str) -> str:
        """Translate a ccxt unified symbol into the raw Binance market id."""
        markets: dict[str, Any] = self._exchange.markets or {}
        market: dict[str, Any] | None = markets.get(symbol)
        if market is None:
            # Fall back to a mechanical conversion, e.g. "BTC/USDT:USDT" -> "BTCUSDT".
            return symbol.split(":")[0].replace("/", "")
        return str(market["id"])

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------
    async def fetch_ohlcv(
        self,
        symbol: str,
        limit: int | None = None,
        since_ms: int | None = None,
    ) -> list[OHLCVCandle]:
        """Fetch a single page of 5-minute candles.

        Args:
            symbol: ccxt unified symbol, e.g. ``"BTC/USDT:USDT"``.
            limit: Number of candles requested (defaults to ``data.ohlcv_limit``).
            since_ms: Optional inclusive lower bound (candle open time, ms).

        Returns:
            Validated candles sorted ascending by open time.  The still-forming
            candle is dropped when ``data.drop_unclosed_candle`` is enabled, so
            every returned candle is guaranteed to be closed.
        """
        _raw_count, _malformed, candles = await self._fetch_ohlcv_page(
            symbol, limit=limit, since_ms=since_ms
        )
        return candles

    async def _fetch_ohlcv_page(
        self,
        symbol: str,
        limit: int | None = None,
        since_ms: int | None = None,
    ) -> tuple[int, int, list[OHLCVCandle]]:
        """Fetch and validate one page, exposing row-count detail beyond the list.

        Returns ``(raw_count, malformed_count, candles)``.  Three outcomes look
        identical if you only inspect ``candles`` (it's simply empty), but they
        mean very different things to :meth:`fetch_ohlcv_range`:

        * ``raw_count == 0`` - the exchange truly has no more candles here.
        * ``raw_count > 0``, ``malformed_count == raw_count`` - every row failed
          structural validation; this is a genuine data problem worth retrying.
        * ``raw_count > 0``, ``malformed_count == 0`` - every row parsed fine but
          was filtered out as the still-forming candle (``drop_unclosed_candle``).
          That is the live edge, not corruption - there is nothing more to fetch
          by paging further, and it must never be mistaken for "malformed".
        """
        await self.load_markets()
        page_limit: int = limit if limit is not None else self._settings.data.ohlcv_limit

        raw: Sequence[Sequence[Any]] = await self._call(
            f"fetch_ohlcv[{symbol}]",
            lambda: self._exchange.fetch_ohlcv(
                symbol,
                timeframe=self._timeframe,
                since=since_ms,
                limit=page_limit,
            ),
        )

        cutoff_ms: int = utc_now_ms()
        candles: list[OHLCVCandle] = []
        malformed: int = 0
        for row in raw:
            try:
                candle: OHLCVCandle = OHLCVCandle.from_ccxt(row, symbol, self._timeframe)
            except (ValueError, TypeError) as error:
                # A structurally broken row is dropped here; the QC validator will
                # observe the resulting gap and trigger a targeted heal.
                _LOGGER.warning("Dropping malformed candle for %s: %s", symbol, error)
                malformed += 1
                continue
            if self._settings.data.drop_unclosed_candle:
                if candle.timestamp + self._timeframe_ms > cutoff_ms:
                    continue
            candles.append(candle)

        candles.sort(key=lambda item: item.timestamp)
        return len(raw), malformed, candles

    async def fetch_ohlcv_range(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        page_limit: int | None = None,
    ) -> list[OHLCVCandle]:
        """Fetch every closed candle in ``[start_ms, end_ms]`` using pagination.

        Binance caps a single ``klines`` response at 1500 rows, so long ranges
        are walked forward page by page.  The loop distinguishes four outcomes
        per page:

        * **Genuine end of history** (the exchange returned zero rows) - a clean
          stop, nothing more to fetch.
        * **The live edge** (every row parsed fine but was filtered out as the
          still-forming candle) - also a clean stop: there is nothing more to
          fetch by paging further, and this must never be mistaken for
          malformed data just because the validated list happens to be empty.
        * **A page of rows that all failed validation** - retried a bounded
          number of times with backoff rather than silently accepted as "no more
          data", so a burst of malformed rows cannot punch a silent hole at the
          tail of the range.
        * **The page-count safety guard tripping before ``end_ms`` is reached** -
          raised as an error rather than returned as a quietly truncated result,
          because a caller that only inspects the returned list has no way to
          tell "complete" from "silently cut short".
        """
        await self.load_markets()
        limit: int = page_limit if page_limit is not None else self._settings.data.ohlcv_limit
        collected: dict[int, OHLCVCandle] = {}
        cursor: int = start_ms
        guard: int = 0
        # Generous on purpose: a real gap in the exchange's history (a halt, a
        # delisting window) makes the cursor jump *forward* faster than a naive
        # per-page estimate assumes, but retried empty-validation pages consume
        # guard budget without advancing the cursor at all.
        max_pages: int = max(
            10, 2 * ((end_ms - start_ms) // (self._timeframe_ms * limit) + 1) + 20
        )
        empty_validation_retries: int = 0
        max_empty_validation_retries: int = 3

        while cursor <= end_ms and guard < max_pages:
            guard += 1
            raw_count, malformed_count, page = await self._fetch_ohlcv_page(
                symbol, limit=limit, since_ms=cursor
            )

            if raw_count == 0:
                # The exchange itself reports nothing more from `cursor` onward.
                break

            if not page:
                if malformed_count == 0:
                    # Every row parsed fine but was filtered out as the
                    # still-forming candle (the live edge) - not corruption,
                    # and paging further will not produce anything either.
                    break
                empty_validation_retries += 1
                if empty_validation_retries > max_empty_validation_retries:
                    raise DataFetchError(
                        f"fetch_ohlcv_range[{symbol}] received only malformed rows",
                        window_start_ms=cursor,
                        window_end_ms=end_ms,
                    )
                _LOGGER.warning(
                    "%s: page at %d returned %d row(s), %d malformed, none valid "
                    "- retry %d/%d",
                    symbol,
                    cursor,
                    raw_count,
                    malformed_count,
                    empty_validation_retries,
                    max_empty_validation_retries,
                )
                await asyncio.sleep(
                    backoff_delay(
                        empty_validation_retries - 1,
                        base_seconds=self._settings.exchange.backoff_base_seconds,
                        max_seconds=self._settings.exchange.backoff_max_seconds,
                        jitter=self._settings.exchange.backoff_jitter,
                    )
                )
                continue
            empty_validation_retries = 0

            fresh: int = 0
            for candle in page:
                if start_ms <= candle.timestamp <= end_ms and candle.timestamp not in collected:
                    collected[candle.timestamp] = candle
                    fresh += 1

            next_cursor: int = page[-1].timestamp + self._timeframe_ms
            if next_cursor <= cursor and fresh == 0:
                # The exchange is not advancing; stop rather than loop forever.
                break
            cursor = next_cursor

        if cursor <= end_ms and guard >= max_pages:
            raise DataFetchError(
                f"fetch_ohlcv_range[{symbol}] hit the {max_pages}-page safety guard "
                "before covering the requested window - refusing to return a "
                "silently truncated result",
                window_start_ms=start_ms,
                window_end_ms=end_ms,
                reached_ms=cursor,
                collected=len(collected),
            )

        return [collected[key] for key in sorted(collected)]

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------
    async def fetch_order_book(self, symbol: str) -> OrderBookSnapshot | None:
        """Fetch an L2 snapshot and reduce it to micro-structure statistics.

        Returns ``None`` when the book is empty or crossed - both indicate a
        transient exchange glitch that must not poison the feature set.
        """
        await self.load_markets()
        depth: int = self._settings.data.orderbook_depth
        levels: int = min(self._settings.data.orderbook_levels_for_imbalance, depth)

        book: dict[str, Any] = await self._call(
            f"fetch_order_book[{symbol}]",
            lambda: self._exchange.fetch_order_book(symbol, limit=depth),
        )

        bids: list[list[float]] = list(book.get("bids") or [])
        asks: list[list[float]] = list(book.get("asks") or [])
        if not bids or not asks:
            _LOGGER.warning("Empty order book received for %s", symbol)
            return None

        best_bid: float = safe_float(bids[0][0])
        best_ask: float = safe_float(asks[0][0])
        if best_bid <= 0.0 or best_ask <= 0.0 or best_bid >= best_ask:
            _LOGGER.warning(
                "Crossed/invalid book for %s (bid=%s ask=%s)", symbol, best_bid, best_ask
            )
            return None

        bid_volume: float = sum(safe_float(level[1]) for level in bids[:levels])
        ask_volume: float = sum(safe_float(level[1]) for level in asks[:levels])
        timestamp: int = int(book.get("timestamp") or utc_now_ms())

        try:
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=timestamp,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                levels=levels,
            )
        except ValueError as error:
            _LOGGER.warning("Order book snapshot rejected for %s: %s", symbol, error)
            return None

    # ------------------------------------------------------------------
    # Futures-specific metrics
    # ------------------------------------------------------------------
    async def fetch_futures_metrics(self, symbol: str) -> FuturesMetrics:
        """Aggregate funding, open interest, positioning ratios and liquidations.

        Each sub-request is independently guarded: a failure in one endpoint
        degrades that single field to its neutral default instead of aborting the
        whole cycle.  Binance's ``futures/data`` endpoints are notoriously flaky
        and are *not* worth halting trading over.
        """
        await self.load_markets()
        market_id: str = self._market_id(symbol)
        now_ms: int = utc_now_ms()

        results: list[Any] = await asyncio.gather(
            self._safe_funding_rate(symbol),
            self._safe_open_interest(symbol),
            self._safe_ratio(market_id, "fapiDataGetGlobalLongShortAccountRatio", "longShortRatio"),
            self._safe_ratio(market_id, "fapiDataGetTopLongShortAccountRatio", "longShortRatio"),
            self._safe_ratio(market_id, "fapiDataGetTakerlongshortRatio", "buySellRatio"),
            self._safe_liquidations(symbol),
            return_exceptions=False,
        )

        funding: dict[str, float | int | None] = results[0]
        open_interest: dict[str, float] = results[1]
        global_ls: float = results[2]
        top_ls: float = results[3]
        taker_ratio: float = results[4]
        liquidations: tuple[float, float] = results[5]

        return FuturesMetrics(
            symbol=symbol,
            timestamp=now_ms,
            funding_rate=float(funding.get("rate") or 0.0),
            next_funding_time=(
                int(funding["next_time"]) if funding.get("next_time") is not None else None
            ),
            open_interest=open_interest.get("amount", 0.0),
            open_interest_value=open_interest.get("value", 0.0),
            long_short_ratio=global_ls,
            top_trader_long_short_ratio=top_ls,
            taker_buy_sell_ratio=taker_ratio,
            liquidation_buy_volume=liquidations[0],
            liquidation_sell_volume=liquidations[1],
            mark_price=float(funding.get("mark_price") or 0.0),
            index_price=float(funding.get("index_price") or 0.0),
        )

    async def _safe_funding_rate(self, symbol: str) -> dict[str, float | int | None]:
        """Fetch the current funding rate plus mark/index price."""
        try:
            payload: dict[str, Any] = await self._call(
                f"fetch_funding_rate[{symbol}]",
                lambda: self._exchange.fetch_funding_rate(symbol),
            )
        except DataFetchError as error:
            _LOGGER.warning("Funding rate unavailable for %s: %s", symbol, error)
            return {"rate": 0.0, "next_time": None, "mark_price": 0.0, "index_price": 0.0}

        rate: float = safe_float(payload.get("fundingRate"), 0.0)
        # Guard against the odd malformed payload: the Pydantic model rejects
        # anything above 5 %, which would abort the entire bundle.
        if abs(rate) > 0.05:
            _LOGGER.warning("Implausible funding rate %.6f for %s - zeroed", rate, symbol)
            rate = 0.0

        return {
            "rate": rate,
            "next_time": payload.get("fundingTimestamp") or payload.get("nextFundingTime"),
            "mark_price": max(0.0, safe_float(payload.get("markPrice"), 0.0)),
            "index_price": max(0.0, safe_float(payload.get("indexPrice"), 0.0)),
        }

    async def _safe_open_interest(self, symbol: str) -> dict[str, float]:
        """Fetch open interest in contracts and notional USDT value."""
        try:
            payload: dict[str, Any] = await self._call(
                f"fetch_open_interest[{symbol}]",
                lambda: self._exchange.fetch_open_interest(symbol),
            )
        except DataFetchError as error:
            _LOGGER.warning("Open interest unavailable for %s: %s", symbol, error)
            return {"amount": 0.0, "value": 0.0}

        info: dict[str, Any] = payload.get("info") or {}
        amount: float = safe_float(
            payload.get("openInterestAmount") or info.get("openInterest"), 0.0
        )
        value: float = safe_float(
            payload.get("openInterestValue") or info.get("sumOpenInterestValue"), 0.0
        )
        return {"amount": max(0.0, amount), "value": max(0.0, value)}

    async def _safe_ratio(self, market_id: str, endpoint: str, field: str) -> float:
        """Fetch a positioning ratio from the ``futures/data`` implicit endpoints.

        Returns ``1.0`` (perfectly balanced) whenever the endpoint is missing,
        rate-limited or returns an empty series.
        """
        method: Any = getattr(self._exchange, endpoint, None)
        if method is None:
            return 1.0

        params: dict[str, Any] = {"symbol": market_id, "period": self._timeframe, "limit": 1}
        try:
            payload: Any = await self._call(f"{endpoint}[{market_id}]", lambda: method(params))
        except DataFetchError as error:
            _LOGGER.debug("%s unavailable for %s: %s", endpoint, market_id, error)
            return 1.0

        if not isinstance(payload, list) or not payload:
            return 1.0
        value: float = safe_float(payload[-1].get(field), 1.0)
        return max(0.0, value)

    async def _safe_liquidations(self, symbol: str) -> tuple[float, float]:
        """Fetch recent liquidation flow as ``(buy_volume, sell_volume)``.

        Binance restricts the public force-order feed, so this degrades to
        ``(0.0, 0.0)`` whenever the endpoint is not exposed by the installed ccxt
        build.  The features derived from it are additive, never load-bearing.
        """
        if not bool((self._exchange.has or {}).get("fetchLiquidations")):
            return (0.0, 0.0)

        since_ms: int = utc_now_ms() - self._timeframe_ms
        try:
            payload: Any = await self._call(
                f"fetch_liquidations[{symbol}]",
                lambda: self._exchange.fetch_liquidations(symbol, since=since_ms, limit=100),
            )
        except DataFetchError as error:
            _LOGGER.debug("Liquidations unavailable for %s: %s", symbol, error)
            return (0.0, 0.0)

        buy_volume: float = 0.0
        sell_volume: float = 0.0
        for entry in payload if isinstance(payload, list) else []:
            quote: float = safe_float(entry.get("quoteValue"), 0.0)
            if quote <= 0.0:
                quote = safe_float(entry.get("baseValue"), 0.0) * safe_float(entry.get("price"), 0.0)
            side: str = str(entry.get("side") or "").lower()
            if side == "buy":
                buy_volume += quote
            else:
                sell_volume += quote
        return (buy_volume, sell_volume)

    # ------------------------------------------------------------------
    # Historical futures-metrics backfill
    # ------------------------------------------------------------------
    # ``fetch_futures_metrics`` above only ever captures "now" - it is called
    # once per live 5-minute cycle, so a freshly-bootstrapped deployment has
    # essentially no real history for funding/OI/positioning under a long
    # training window and every micro-structure feature derived from them
    # sits at its neutral default for nearly every historical row. Order-book
    # depth has no historical endpoint at all on Binance (a snapshot is only
    # ever "now"), but funding rate, open interest and the positioning ratios
    # *do* have dedicated history endpoints and can be backfilled - this
    # section does that, loudly logging when a source cannot deliver data
    # instead of silently leaving the caller with defaults.

    async def fetch_funding_rate_history(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> list[tuple[int, float]]:
        """Backfill funding-rate history over ``[start_ms, end_ms]``.

        Binance retains funding-rate history for the entire life of a
        contract via ``/fapi/v1/fundingRate`` (unlike open interest and the
        positioning ratios below, which are capped to a recent rolling
        window) - this is the one micro-structure/derivatives source that can
        genuinely be recovered across a multi-month training window.
        """
        await self.load_markets()
        market_id: str = self._market_id(symbol)
        collected: list[tuple[int, float]] = []
        cursor: int = start_ms
        limit: int = 1_000
        guard: int = 0
        max_pages: int = 500
        while cursor <= end_ms and guard < max_pages:
            guard += 1
            params: dict[str, Any] = {
                "symbol": market_id,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit,
            }
            try:
                payload: Any = await self._call(
                    f"fetch_funding_rate_history[{symbol}]",
                    lambda: self._exchange.fapiPublicGetFundingRate(params),
                )
            except DataFetchError as error:
                _LOGGER.error(
                    "Funding-rate history backfill aborted for %s at cursor %d: %s",
                    symbol,
                    cursor,
                    error,
                )
                break
            if not isinstance(payload, list) or not payload:
                break
            for row in payload:
                timestamp: int | None = _safe_int(row.get("fundingTime"))
                if timestamp is None:
                    continue
                rate: float = safe_float(row.get("fundingRate"), 0.0)
                if abs(rate) > 0.05:
                    # Same implausibility guard as the live path - drop, don't poison.
                    continue
                collected.append((timestamp, rate))
            last_timestamp: int | None = _safe_int(payload[-1].get("fundingTime"))
            if last_timestamp is None or last_timestamp < cursor:
                break
            next_cursor: int = last_timestamp + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(payload) < limit:
                break
        if not collected:
            _LOGGER.warning(
                "No funding-rate history recovered for %s in [%d, %d] - funding_rate "
                "will stay at its neutral default for this window",
                symbol,
                start_ms,
                end_ms,
            )
        return collected

    async def fetch_open_interest_history(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> list[tuple[int, float]]:
        """Backfill open interest via ``futures/data/openInterestHist``.

        Binance only retains ~30 days of history for this endpoint - a
        request reaching further back simply returns fewer rows than the
        window implies, which is logged, not swallowed.
        """
        return await self._fetch_futures_data_series(
            symbol,
            endpoint_name="fapiDataGetOpenInterestHist",
            field="sumOpenInterest",
            start_ms=start_ms,
            end_ms=end_ms,
        )

    async def fetch_long_short_ratio_history(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> list[tuple[int, float]]:
        """Backfill the global long/short account ratio (~30-day retention)."""
        return await self._fetch_futures_data_series(
            symbol,
            endpoint_name="fapiDataGetGlobalLongShortAccountRatio",
            field="longShortRatio",
            start_ms=start_ms,
            end_ms=end_ms,
        )

    async def fetch_taker_ratio_history(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> list[tuple[int, float]]:
        """Backfill the taker buy/sell volume ratio (~30-day retention)."""
        return await self._fetch_futures_data_series(
            symbol,
            endpoint_name="fapiDataGetTakerlongshortRatio",
            field="buySellRatio",
            start_ms=start_ms,
            end_ms=end_ms,
        )

    async def _fetch_futures_data_series(
        self,
        symbol: str,
        endpoint_name: str,
        field: str,
        start_ms: int,
        end_ms: int,
    ) -> list[tuple[int, float]]:
        """Paginate one ``futures/data/*`` aggregated series over a time range.

        These endpoints (open interest, long/short ratios, taker ratio) are
        documented by Binance as retaining only the most recent ~30 days -
        requesting further back is not a bug, it genuinely has nothing to
        return, so an empty page ends pagination cleanly rather than raising.
        A missing/unsupported endpoint (older ccxt build) *is* logged loudly,
        since that would otherwise look identical to "no data in range".
        """
        await self.load_markets()
        method: Any = getattr(self._exchange, endpoint_name, None)
        if method is None:
            _LOGGER.error(
                "%s is not exposed by the installed ccxt build - cannot backfill "
                "%s history for %s; it will stay at its neutral default",
                endpoint_name,
                field,
                symbol,
            )
            return []

        market_id: str = self._market_id(symbol)
        collected: list[tuple[int, float]] = []
        cursor: int = start_ms
        limit: int = 500
        guard: int = 0
        max_pages: int = 500
        while cursor <= end_ms and guard < max_pages:
            guard += 1
            params: dict[str, Any] = {
                "symbol": market_id,
                "period": self._timeframe,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit,
            }
            try:
                payload: Any = await self._call(
                    f"{endpoint_name}[{market_id}]", lambda: method(params)
                )
            except DataFetchError as error:
                _LOGGER.warning(
                    "%s backfill interrupted for %s at cursor %d: %s",
                    endpoint_name,
                    symbol,
                    cursor,
                    error,
                )
                break
            if not isinstance(payload, list) or not payload:
                break
            for row in payload:
                timestamp: int | None = _safe_int(row.get("timestamp"))
                if timestamp is None:
                    continue
                value: float = safe_float(row.get(field), 0.0)
                collected.append((timestamp, value))
            last_timestamp: int | None = _safe_int(payload[-1].get("timestamp"))
            if last_timestamp is None or last_timestamp < cursor:
                break
            next_cursor: int = last_timestamp + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(payload) < limit:
                break
        if not collected:
            _LOGGER.warning(
                "%s returned no rows for %s in [%d, %d] - likely outside Binance's "
                "retention window for this endpoint (~30 days)",
                endpoint_name,
                symbol,
                start_ms,
                end_ms,
            )
        return collected

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    async def fetch_last_price(self, symbol: str) -> float:
        """Return the last traded price, used by the paper trader and monitors."""
        await self.load_markets()
        ticker: dict[str, Any] = await self._call(
            f"fetch_ticker[{symbol}]",
            lambda: self._exchange.fetch_ticker(symbol),
        )
        price: float = safe_float(ticker.get("last") or ticker.get("close"), 0.0)
        if price <= 0.0:
            raise DataFetchError("ticker returned a non-positive price", symbol=symbol)
        return price

    async def fetch_raw_tickers(
        self,
        symbols: Sequence[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Batch-fetch complete ticker payloads (price, bid/ask, 24 h volume).

        Used by universe discovery, which needs the spread and quote volume that
        :meth:`fetch_tickers` discards.  Passing ``None`` fetches the whole
        market in a single weighted request.
        """
        await self.load_markets()
        requested: list[str] | None = list(symbols) if symbols else None
        payload: dict[str, Any] = await self._call(
            "fetch_raw_tickers",
            lambda: self._exchange.fetch_tickers(requested),
        )
        return {
            str(symbol): dict(ticker)
            for symbol, ticker in payload.items()
            if isinstance(ticker, dict)
        }

    async def fetch_tickers(self, symbols: Sequence[str]) -> dict[str, float]:
        """Batch-fetch last prices for many symbols in a single request."""
        await self.load_markets()
        payload: dict[str, Any] = await self._call(
            "fetch_tickers",
            lambda: self._exchange.fetch_tickers(list(symbols)),
        )
        prices: dict[str, float] = {}
        for symbol, ticker in payload.items():
            price: float = safe_float(ticker.get("last") or ticker.get("close"), 0.0)
            if price > 0.0:
                prices[symbol] = price
        return prices
