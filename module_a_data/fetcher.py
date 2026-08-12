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
from core.utils import async_retry, safe_float, utc_now_ms
from module_a_data.models import AggTradeFlow, FuturesMetrics, OHLCVCandle, OrderBookSnapshot

_LOGGER = get_logger(__name__)

#: Binance rejects an ``aggTrades`` window wider than one hour.
_AGG_TRADE_MAX_SPAN_MS: Final[int] = 60 * 60 * 1_000

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
        if self._settings.exchange.testnet:
            exchange.set_sandbox_mode(True)
        return exchange

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
        for row in raw:
            try:
                candle: OHLCVCandle = OHLCVCandle.from_ccxt(row, symbol, self._timeframe)
            except (ValueError, TypeError) as error:
                # A structurally broken row is dropped here; the QC validator will
                # observe the resulting gap and trigger a targeted heal.
                _LOGGER.warning("Dropping malformed candle for %s: %s", symbol, error)
                continue
            if self._settings.data.drop_unclosed_candle:
                if candle.timestamp + self._timeframe_ms > cutoff_ms:
                    continue
            candles.append(candle)

        candles.sort(key=lambda item: item.timestamp)
        return candles

    async def fetch_ohlcv_range(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        page_limit: int | None = None,
    ) -> list[OHLCVCandle]:
        """Fetch every closed candle in ``[start_ms, end_ms]`` using pagination.

        Binance caps a single ``klines`` response at 1500 rows, so long ranges
        are walked forward page by page.  The loop is defensive against an
        exchange that returns an empty or non-advancing page (it breaks instead
        of spinning forever).
        """
        await self.load_markets()
        limit: int = page_limit if page_limit is not None else self._settings.data.ohlcv_limit
        collected: dict[int, OHLCVCandle] = {}
        cursor: int = start_ms
        guard: int = 0
        max_pages: int = max(1, (end_ms - start_ms) // (self._timeframe_ms * limit) + 4)

        while cursor <= end_ms and guard < max_pages:
            guard += 1
            page: list[OHLCVCandle] = await self.fetch_ohlcv(
                symbol, limit=limit, since_ms=cursor
            )
            if not page:
                break

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
        now_ms: int = utc_now_ms()

        results: list[Any] = await asyncio.gather(
            self._safe_funding_rate(symbol),
            self._safe_open_interest(symbol),
            self._safe_liquidations(symbol),
            return_exceptions=False,
        )

        funding: dict[str, float | int | None] = results[0]
        open_interest: dict[str, float] = results[1]
        liquidations: tuple[float, float] = results[2]

        return FuturesMetrics(
            symbol=symbol,
            timestamp=now_ms,
            funding_rate=float(funding.get("rate") or 0.0),
            next_funding_time=(
                int(funding["next_time"]) if funding.get("next_time") is not None else None
            ),
            open_interest=open_interest.get("amount", 0.0),
            open_interest_value=open_interest.get("value", 0.0),
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
    # Aggregated trades -> 5-minute order-flow buckets
    # ------------------------------------------------------------------
    async def fetch_agg_trade_flow(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[AggTradeFlow]:
        """Fold Binance Futures ``aggTrades`` into closed 5-minute flow buckets.

        The endpoint is walked forward with ``startTime``/``endTime`` paging
        (1000 rows per page).  Every aggregated trade carries ``m``
        (``isBuyerMaker``), which is the *only* trustworthy source of aggressor
        side: ``m == false`` means the buyer lifted the offer, so the quantity is
        aggressive buy volume; ``m == true`` means the seller hit the bid.

        Args:
            symbol: ccxt unified symbol.
            start_ms: Inclusive lower bound; snapped down to a bucket boundary.
            end_ms: Exclusive upper bound on trade time.  Buckets that would
                extend past it are dropped, so a partially observed bucket can
                never reach the feature stack.

        Returns:
            Buckets sorted ascending by open time.  Intervals in which nothing
            traded are simply absent - the feature layer treats a missing bucket
            as genuinely zero flow, which is what it is.
        """
        await self.load_markets()
        bucket_ms: int = self._timeframe_ms
        if end_ms <= start_ms:
            return []

        first_bucket: int = (start_ms // bucket_ms) * bucket_ms
        # Only fully closed buckets may be emitted.
        last_bucket_exclusive: int = (end_ms // bucket_ms) * bucket_ms
        if last_bucket_exclusive <= first_bucket:
            return []

        market_id: str = self._market_id(symbol)
        method: Any = getattr(self._exchange, "fapiPublicGetAggTrades", None)
        page_limit: int = self._settings.data.agg_trade_page_limit

        buckets: dict[int, dict[str, float]] = {}
        cursor: int = first_bucket
        pages: int = 0

        while cursor < last_bucket_exclusive and pages < self._settings.data.agg_trade_max_pages:
            pages += 1
            rows: list[dict[str, Any]] = await self._fetch_agg_trade_page(
                symbol, market_id, method, cursor, last_bucket_exclusive, page_limit
            )
            if not rows:
                break

            newest: int = cursor
            for row in rows:
                traded_at: int = int(safe_float(row.get("T") or row.get("timestamp"), 0.0))
                if traded_at <= 0 or traded_at >= last_bucket_exclusive:
                    continue
                newest = max(newest, traded_at)

                quantity: float = safe_float(row.get("q") or row.get("amount"), 0.0)
                price: float = safe_float(row.get("p") or row.get("price"), 0.0)
                if quantity <= 0.0:
                    continue

                bucket_key: int = (traded_at // bucket_ms) * bucket_ms
                bucket: dict[str, float] = buckets.setdefault(
                    bucket_key,
                    {
                        "buy_volume": 0.0,
                        "sell_volume": 0.0,
                        "buy_quote_volume": 0.0,
                        "sell_quote_volume": 0.0,
                        "trades": 0.0,
                    },
                )
                buyer_is_maker: bool = self._is_buyer_maker(row)
                side: str = "sell" if buyer_is_maker else "buy"
                bucket[f"{side}_volume"] += quantity
                bucket[f"{side}_quote_volume"] += quantity * price
                bucket["trades"] += 1.0

            if len(rows) < page_limit:
                # The exchange returned a partial page: the range is exhausted.
                break
            advanced: int = newest + 1
            if advanced <= cursor:
                break  # No forward progress; stop rather than spin.
            cursor = advanced

        return [
            AggTradeFlow(
                symbol=symbol,
                timestamp=key,
                buy_volume=values["buy_volume"],
                sell_volume=values["sell_volume"],
                buy_quote_volume=values["buy_quote_volume"],
                sell_quote_volume=values["sell_quote_volume"],
                trades=int(values["trades"]),
            )
            for key, values in sorted(buckets.items())
        ]

    async def _fetch_agg_trade_page(
        self,
        symbol: str,
        market_id: str,
        method: Any,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch one ``aggTrades`` page, preferring the raw Binance endpoint.

        The implicit ``fapiPublicGetAggTrades`` call is used when the installed
        ccxt build exposes it, because it returns ``m`` (``isBuyerMaker``)
        verbatim.  Otherwise the call falls back to unified ``fetch_trades``,
        whose parsed ``side`` ccxt derives from exactly the same flag.
        """
        if method is not None:
            params: dict[str, Any] = {
                "symbol": market_id,
                "startTime": int(start_ms),
                # Binance requires endTime - startTime <= 1 h on this endpoint.
                "endTime": int(min(end_ms, start_ms + _AGG_TRADE_MAX_SPAN_MS)),
                "limit": limit,
            }
            try:
                payload: Any = await self._call(
                    f"agg_trades[{market_id}]", lambda: method(params)
                )
            except DataFetchError as error:
                _LOGGER.warning("aggTrades unavailable for %s: %s", symbol, error)
                return []
            return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []

        try:
            trades: Any = await self._call(
                f"fetch_trades[{symbol}]",
                lambda: self._exchange.fetch_trades(symbol, since=int(start_ms), limit=limit),
            )
        except DataFetchError as error:
            _LOGGER.warning("Trade history unavailable for %s: %s", symbol, error)
            return []
        return [row for row in trades if isinstance(row, dict)] if isinstance(trades, list) else []

    @staticmethod
    def _is_buyer_maker(row: dict[str, Any]) -> bool:
        """Read the aggressor side out of a raw or ccxt-parsed trade row.

        Raw Binance rows carry ``m``; ccxt-parsed rows carry ``side`` (``"sell"``
        when the buyer was the maker) and often ``info.m``.  Anything else is
        treated as a maker-buy, i.e. the *conservative* reading that keeps an
        unclassifiable trade out of the aggressive-buy bucket.
        """
        if "m" in row:
            return bool(row["m"])
        info: Any = row.get("info")
        if isinstance(info, dict) and "m" in info:
            return bool(info["m"])
        side: str = str(row.get("side") or "").lower()
        if side in {"buy", "sell"}:
            return side == "sell"
        return True

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
