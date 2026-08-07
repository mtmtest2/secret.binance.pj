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
from typing import Any, Final, Literal, Sequence

import ccxt.async_support as ccxt

from config.settings import Settings
from core.exceptions import DataFetchError
from core.logger import get_logger
from core.utils import async_retry, safe_float, utc_now_ms
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


#: Which of the two connections an instance represents.
FetcherRole = Literal["market_data", "execution"]


class BinanceDataFetcher:
    """Async gateway to Binance USDT-M.

    Two roles, deliberately kept as separate instances:

    ``market_data``
        Mainnet, unauthenticated, never sandboxed.  Supplies every candle,
        order book, funding rate and ticker in *all* modes - backtest, paper and
        live - so the models are only ever trained and run on real production
        market data.  The API credentials are never attached to this client, so
        a bug in the data path cannot touch the account.

    ``execution``
        Carries the personal API key/secret and honours
        ``exchange.testnet``.  Only the live executor constructs one.

    The instance owns a single ``ccxt.async_support.binance`` client; close it
    via :meth:`close` (or use it as an async context manager) so the underlying
    ``aiohttp`` session is released.
    """

    def __init__(
        self,
        settings: Settings,
        exchange: ccxt.binance | None = None,
        *,
        role: FetcherRole = "market_data",
    ) -> None:
        self._settings: Settings = settings
        self._role: FetcherRole = role
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
    @property
    def role(self) -> FetcherRole:
        """Which connection this instance represents."""
        return self._role

    @property
    def is_authenticated(self) -> bool:
        """``True`` when this client carries API credentials."""
        return bool(getattr(self._exchange, "apiKey", None))

    @property
    def is_sandbox(self) -> bool:
        """``True`` when this client is pointed at the futures testnet."""
        return self._role == "execution" and self._settings.exchange.testnet

    def _build_exchange(self) -> ccxt.binance:
        """Instantiate the ccxt client for this instance's role.

        The market-data client is built without credentials and is never put
        into sandbox mode: real market data is a hard requirement of the design,
        not a configurable preference.
        """
        execution: bool = self._role == "execution"
        config: dict[str, Any] = {
            "enableRateLimit": self._settings.exchange.enable_rate_limit,
            "timeout": self._settings.exchange.request_timeout_ms,
            "options": {
                "defaultType": self._settings.exchange.default_type,
                "adjustForTimeDifference": True,
                "recvWindow": 10_000,
            },
        }
        if execution:
            config["apiKey"] = self._settings.exchange.api_key.strip() or None
            config["secret"] = self._settings.exchange.api_secret.strip() or None

        exchange: ccxt.binance = ccxt.binance(config)
        self._restrict_to_linear_markets(exchange)

        if execution and self._settings.exchange.testnet:
            exchange.set_sandbox_mode(True)
            _LOGGER.warning("Execution client is on the Binance futures TESTNET")
        elif execution:
            _LOGGER.info("Execution client is on Binance MAINNET (real funds)")
        else:
            _LOGGER.info("Market-data client is on Binance MAINNET (public, unauthenticated)")
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

    async def _call_interactive(self, label: str, operation: Any) -> Any:
        """Like :meth:`_call` but bounded to a few seconds.

        Used by the pair-selection page and the credential check, where a human
        is waiting.  The patient retry chain used for background ingestion can
        take minutes to give up, which in a browser is indistinguishable from a
        hang - so interactive paths fail fast and surface the real reason.
        """
        exchange_settings = self._settings.exchange
        original_timeout: Any = self._exchange.timeout
        self._exchange.timeout = exchange_settings.interactive_timeout_ms
        try:
            return await async_retry(
                operation,
                attempts=exchange_settings.interactive_max_retries,
                base_seconds=0.5,
                max_seconds=3.0,
                jitter=0.2,
                retry_on=TRANSIENT_ERRORS,
                give_up_on=PERMANENT_ERRORS,
                on_error=lambda attempt, error, delay: _LOGGER.warning(
                    "%s failed (attempt %d): %s", label, attempt + 1, error
                ),
            )
        except PERMANENT_ERRORS as error:  # type: ignore[misc]
            raise DataFetchError(f"{label} rejected", reason=str(error)) from error
        except TRANSIENT_ERRORS as error:  # type: ignore[misc]
            raise DataFetchError(
                f"{label} could not reach Binance", reason=str(error)
            ) from error
        except ccxt.ExchangeError as error:
            raise DataFetchError(f"{label} failed", reason=str(error)) from error
        finally:
            self._exchange.timeout = original_timeout

    async def verify_credentials(self) -> dict[str, Any]:
        """Check the configured API key against the account endpoint.

        Performs a real signed request, so it proves the key exists, the secret
        matches, futures trading is permitted and the server clock is close
        enough for the signature to validate.  Never raises and never returns the
        secret - the key is masked before it leaves this method.

        Returns:
            ``{"configured", "valid", "masked_key", "environment", "balance",
            "error"}``.
        """
        settings = self._settings.exchange
        result: dict[str, Any] = {
            "configured": settings.has_credentials,
            "valid": False,
            "masked_key": settings.masked_api_key,
            "environment": "testnet" if self.is_sandbox else "mainnet",
            "balance": 0.0,
            "error": "",
        }
        if not settings.has_credentials:
            result["error"] = "EXCHANGE__API_KEY / EXCHANGE__API_SECRET are not set"
            return result

        try:
            await self.load_markets()
            payload: dict[str, Any] = await self._call_interactive(
                "verify_credentials", lambda: self._exchange.fetch_balance()
            )
        except DataFetchError as error:
            reason: str = str(error).lower()
            if "invalid api" in reason or "signature" in reason or "-2015" in reason:
                result["error"] = (
                    "Binance rejected the key. Check that it is a FUTURES-enabled key, "
                    "that the secret matches, and that this server's IP is on the key's "
                    "IP allow-list."
                )
            elif "timestamp" in reason or "recvwindow" in reason:
                result["error"] = (
                    "Signature timestamp rejected - this server's clock is out of sync. "
                    "Run an NTP sync."
                )
            else:
                result["error"] = str(error)
            return result

        usdt: dict[str, Any] = payload.get("USDT") or {}
        result["valid"] = True
        result["balance"] = safe_float(usdt.get("total"), 0.0)
        _LOGGER.info(
            "API credentials verified on %s (key %s, USDT balance %.2f)",
            result["environment"],
            result["masked_key"],
            result["balance"],
        )
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
        timeframe: str | None = None,
    ) -> list[OHLCVCandle]:
        """Fetch a single page of candles.

        Args:
            symbol: ccxt unified symbol, e.g. ``"BTC/USDT:USDT"``.
            limit: Number of candles requested (defaults to ``data.ohlcv_limit``).
            since_ms: Optional inclusive lower bound (candle open time, ms).
            timeframe: Override the instance's configured timeframe (``"5m"``)
                for this call, e.g. ``"1m"`` for the labeler's intra-candle
                refinement. Defaults to the instance's timeframe.

        Returns:
            Validated candles sorted ascending by open time.  The still-forming
            candle is dropped when ``data.drop_unclosed_candle`` is enabled, so
            every returned candle is guaranteed to be closed.
        """
        await self.load_markets()
        page_limit: int = limit if limit is not None else self._settings.data.ohlcv_limit
        tf: str = timeframe if timeframe is not None else self._timeframe
        tf_ms: int = self._timeframe_ms_for(tf)

        raw: Sequence[Sequence[Any]] = await self._call(
            f"fetch_ohlcv[{symbol}:{tf}]",
            lambda: self._exchange.fetch_ohlcv(
                symbol,
                timeframe=tf,
                since=since_ms,
                limit=page_limit,
            ),
        )

        cutoff_ms: int = utc_now_ms()
        candles: list[OHLCVCandle] = []
        for row in raw:
            try:
                candle: OHLCVCandle = OHLCVCandle.from_ccxt(row, symbol, tf)
            except (ValueError, TypeError) as error:
                # A structurally broken row is dropped here; the QC validator will
                # observe the resulting gap and trigger a targeted heal.
                _LOGGER.warning("Dropping malformed candle for %s: %s", symbol, error)
                continue
            if self._settings.data.drop_unclosed_candle:
                if candle.timestamp + tf_ms > cutoff_ms:
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
        timeframe: str | None = None,
    ) -> list[OHLCVCandle]:
        """Fetch every closed candle in ``[start_ms, end_ms]`` using pagination.

        Binance caps a single ``klines`` response at 1500 rows, so long ranges
        are walked forward page by page.  The loop is defensive against an
        exchange that returns an empty or non-advancing page (it breaks instead
        of spinning forever).

        Args:
            timeframe: Override the instance's configured timeframe for this
                call (see :meth:`fetch_ohlcv`).
        """
        await self.load_markets()
        limit: int = page_limit if page_limit is not None else self._settings.data.ohlcv_limit
        tf: str = timeframe if timeframe is not None else self._timeframe
        tf_ms: int = self._timeframe_ms_for(tf)
        collected: dict[int, OHLCVCandle] = {}
        cursor: int = start_ms
        guard: int = 0
        max_pages: int = max(1, (end_ms - start_ms) // (tf_ms * limit) + 4)

        while cursor <= end_ms and guard < max_pages:
            guard += 1
            page: list[OHLCVCandle] = await self.fetch_ohlcv(
                symbol, limit=limit, since_ms=cursor, timeframe=tf
            )
            if not page:
                break

            fresh: int = 0
            for candle in page:
                if start_ms <= candle.timestamp <= end_ms and candle.timestamp not in collected:
                    collected[candle.timestamp] = candle
                    fresh += 1

            next_cursor: int = page[-1].timestamp + tf_ms
            if next_cursor <= cursor and fresh == 0:
                # The exchange is not advancing; stop rather than loop forever.
                break
            cursor = next_cursor

        return [collected[key] for key in sorted(collected)]

    def _timeframe_ms_for(self, timeframe: str) -> int:
        """Resolve a timeframe string to its duration in milliseconds.

        Uses the instance's own precomputed value for its configured
        timeframe (avoids a redundant parse on the hot path); any other
        timeframe is resolved through ccxt's parser.
        """
        if timeframe == self._timeframe:
            return self._timeframe_ms
        return int(self._exchange.parse_timeframe(timeframe) * 1000)

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
        # Always request the *whole* board and filter locally.  Handing ccxt a
        # 500-symbol list makes it validate every one against the market table
        # and can fan out into per-symbol requests on some builds; the unfiltered
        # call is a single request with a fixed weight.
        payload: dict[str, Any] = await self._call_interactive(
            "fetch_raw_tickers",
            lambda: self._exchange.fetch_tickers(),
        )
        wanted: set[str] | None = set(symbols) if symbols else None
        return {
            str(symbol): dict(ticker)
            for symbol, ticker in payload.items()
            if isinstance(ticker, dict) and (wanted is None or symbol in wanted)
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
