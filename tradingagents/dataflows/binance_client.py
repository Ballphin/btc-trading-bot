"""Binance Futures public API — derivatives data. No auth required."""

import logging
from datetime import datetime

import pandas as pd

from tradingagents.dataflows.base_client import BaseDataClient

logger = logging.getLogger(__name__)


class BinanceClient(BaseDataClient):
    """Fetch derivatives data from Binance Futures public REST API."""

    BASE_URL = "https://fapi.binance.com"

    def __init__(self, cache_ttl: int = 3600):
        super().__init__("binance", cache_ttl)

    @staticmethod
    def _to_ms(date_str: str) -> int:
        """Convert yyyy-mm-dd to milliseconds since epoch."""
        return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)

    def get_funding_rates(
        self,
        symbol: str = "BTCUSDT",
        start: str = None,
        end: str = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        GET /fapi/v1/fundingRate

        Returns DataFrame with: fundingTime, symbol, fundingRate, markPrice
        """
        params = {"symbol": symbol, "limit": limit}
        if start:
            params["startTime"] = self._to_ms(start)
        if end:
            params["endTime"] = self._to_ms(end)

        try:
            data = self._request(
                f"{self.BASE_URL}/fapi/v1/fundingRate",
                params,
                cache_prefix=f"bn-funding-{symbol}",
            )
        except Exception as e:
            logger.warning(f"Binance funding rate fetch failed: {e}")
            return pd.DataFrame(columns=["fundingTime", "symbol", "fundingRate"])

        df = pd.DataFrame(data)
        if not df.empty and "fundingTime" in df.columns:
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
            df["fundingRate"] = df["fundingRate"].astype(float)
        return df

    def get_open_interest_hist(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1d",
        start: str = None,
        end: str = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        GET /futures/data/openInterestHist

        Returns DataFrame with: timestamp, symbol, sumOpenInterest, sumOpenInterestValue
        """
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start:
            params["startTime"] = self._to_ms(start)
        if end:
            params["endTime"] = self._to_ms(end)

        try:
            data = self._request(
                f"{self.BASE_URL}/futures/data/openInterestHist",
                params,
                cache_prefix=f"bn-oi-{symbol}-{period}",
            )
        except Exception as e:
            logger.warning(f"Binance OI fetch failed: {e}")
            return pd.DataFrame(columns=["timestamp", "sumOpenInterest", "sumOpenInterestValue"])

        df = pd.DataFrame(data)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
            df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
        return df

    def get_taker_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1d",
        start: str = None,
        end: str = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        GET /futures/data/takerlongshortRatio

        Returns DataFrame with: buySellRatio, buyVol, sellVol, timestamp
        """
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start:
            params["startTime"] = self._to_ms(start)
        if end:
            params["endTime"] = self._to_ms(end)

        try:
            data = self._request(
                f"{self.BASE_URL}/futures/data/takerlongshortRatio",
                params,
                cache_prefix=f"bn-taker-{symbol}-{period}",
            )
        except Exception as e:
            logger.warning(f"Binance taker ratio fetch failed: {e}")
            return pd.DataFrame(columns=["buySellRatio", "buyVol", "sellVol", "timestamp"])

        df = pd.DataFrame(data)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["buySellRatio"] = pd.to_numeric(df["buySellRatio"], errors="coerce")
            df["buyVol"] = pd.to_numeric(df["buyVol"], errors="coerce")
            df["sellVol"] = pd.to_numeric(df["sellVol"], errors="coerce")
        return df

    # ── Historical OHLCV (klines) — for pre-Hyperliquid bear backfill ─

    #: Binance `/fapi/v1/klines` caps each response at this many bars.
    _KLINES_MAX_PER_REQUEST = 1500

    #: Map of our canonical interval strings → milliseconds per bar.
    #: Used both to page the fetch and to decide cache TTL.
    _INTERVAL_MS = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "2h": 2 * 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "6h": 6 * 60 * 60_000,
        "12h": 12 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Paginated historical perp OHLCV from Binance Futures.

        ``GET /fapi/v1/klines`` — public, no auth, no rate-limit headers
        required for <=2400 weight/min. Each request returns at most
        :attr:`_KLINES_MAX_PER_REQUEST` bars, so we loop until we cover
        the requested range.

        Args:
            symbol: Binance perp symbol, e.g. ``"BTCUSDT"``. Callers that
                use our internal ``BTC-USD`` convention should map via
                :class:`HistoricalDataRouter`.
            interval: Candle interval — one of :attr:`_INTERVAL_MS` keys.
            start: ``yyyy-mm-dd`` (inclusive). Defaults to earliest-available.
            end: ``yyyy-mm-dd`` (exclusive, like Binance semantics).

        Returns:
            DataFrame sorted ascending with columns:
                timestamp (tz-naive UTC), open, high, low, close,
                volume (base asset), quote_volume, n_trades
            Empty DataFrame on total failure (no partial results).

        Notes:
            * Each page is cached via ``BaseDataClient._request`` so a
              re-fetch of an already-materialized bear-era window is free.
            * We only advance the pagination cursor if the page returned
              data, otherwise we break to avoid infinite loops when
              Binance returns [] at the edge of the history.
        """
        if interval not in self._INTERVAL_MS:
            raise ValueError(
                f"Unsupported interval {interval!r}; "
                f"expected one of {sorted(self._INTERVAL_MS)}"
            )
        start_ms = self._to_ms(start) if start else 0
        # Binance `endTime` is inclusive on fapi; callers pass an
        # exclusive yyyy-mm-dd, so we do NOT subtract 1ms here and let
        # the loop terminator handle the trailing partial page.
        end_ms = self._to_ms(end) if end else int(pd.Timestamp.utcnow().timestamp() * 1000)

        all_rows: list[list] = []
        cursor = start_ms
        step_ms = self._INTERVAL_MS[interval] * self._KLINES_MAX_PER_REQUEST
        pages = 0
        # Absolute safety cap: a 2-year 1m fetch is ~700 pages; 2000 is
        # far above any legitimate request and prevents unbounded loops
        # if Binance starts returning duplicate cursors.
        MAX_PAGES = 2000

        while cursor < end_ms and pages < MAX_PAGES:
            page_end = min(cursor + step_ms, end_ms)
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": page_end,
                "limit": self._KLINES_MAX_PER_REQUEST,
            }
            try:
                data = self._request(
                    f"{self.BASE_URL}/fapi/v1/klines",
                    params,
                    cache_prefix=f"bn-klines-{symbol}-{interval}",
                )
            except Exception as e:
                logger.warning(
                    f"[BinanceClient.get_klines] page={pages} "
                    f"symbol={symbol} interval={interval} "
                    f"cursor={cursor} failed: {e}"
                )
                break

            if not isinstance(data, list) or not data:
                # Either gap in history or end of coverage — advance the
                # cursor by one page so we don't spin forever on a sparse
                # era, but stop if two consecutive pages yield nothing.
                if not all_rows or (all_rows and pages > 0 and not data):
                    break
                cursor = page_end
                pages += 1
                continue

            all_rows.extend(data)
            # Advance past the last bar's open time so we don't re-fetch it.
            last_open = int(data[-1][0])
            next_cursor = last_open + self._INTERVAL_MS[interval]
            if next_cursor <= cursor:
                # Protect against Binance returning a non-advancing page.
                break
            cursor = next_cursor
            pages += 1

        if pages >= MAX_PAGES:
            logger.warning(
                f"[BinanceClient.get_klines] MAX_PAGES hit for "
                f"{symbol}/{interval} {start}→{end}; results may be truncated"
            )

        if not all_rows:
            return pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close",
                "volume", "quote_volume", "n_trades",
            ])

        # Binance kline row layout (indices documented here rather than
        # using dict keys — the REST response is a list-of-lists):
        #   0 openTime(ms), 1 open, 2 high, 3 low, 4 close,
        #   5 volume(base), 6 closeTime(ms), 7 quoteAssetVolume,
        #   8 numberOfTrades, 9 takerBuyBaseVolume, 10 takerBuyQuoteVolume
        df = pd.DataFrame(all_rows, columns=[
            "openTime", "open", "high", "low", "close", "volume",
            "closeTime", "quote_volume", "n_trades",
            "taker_buy_base", "taker_buy_quote", "_ignore",
        ])
        # De-dup on openTime in case two overlapping pages returned
        # the same bar (happens near ``start`` when caller passes mid-bar).
        df = df.drop_duplicates(subset=["openTime"], keep="last")
        df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms")
        for col in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce").astype("Int64")
        df = df[[
            "timestamp", "open", "high", "low", "close",
            "volume", "quote_volume", "n_trades",
        ]].sort_values("timestamp").reset_index(drop=True)
        return df
