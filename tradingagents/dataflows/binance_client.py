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
