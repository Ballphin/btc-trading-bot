"""Deribit public API — perp funding rates, OHLCV, options context. No auth required."""

import logging
from datetime import datetime

import pandas as pd

from tradingagents.dataflows.base_client import BaseDataClient

logger = logging.getLogger(__name__)


class DeribitClient(BaseDataClient):
    """Fetch derivatives data from the Deribit public REST API."""

    BASE_URL = "https://www.deribit.com/api/v2/public"

    def __init__(self, cache_ttl: int = 3600):
        super().__init__("deribit", cache_ttl)

    @staticmethod
    def _to_ms(date_str: str) -> int:
        """Convert yyyy-mm-dd to milliseconds since epoch."""
        return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)

    def get_funding_rate_history(
        self,
        instrument: str = "BTC-PERPETUAL",
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        GET /public/get_funding_rate_history

        Returns hourly DataFrame with: timestamp, index_price, prev_index_price,
        interest_8h, interest_1h
        """
        params = {"instrument_name": instrument}
        if start:
            params["start_timestamp"] = self._to_ms(start)
        if end:
            params["end_timestamp"] = self._to_ms(end)

        try:
            data = self._request(
                f"{self.BASE_URL}/get_funding_rate_history",
                params,
                cache_prefix=f"dr-funding-{instrument}",
            )
        except Exception as e:
            logger.warning(f"Deribit funding rate fetch failed: {e}")
            return pd.DataFrame(columns=["timestamp", "index_price", "interest_8h", "interest_1h"])

        result = data.get("result", [])
        df = pd.DataFrame(result)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def get_ohlcv(
        self,
        instrument: str = "BTC-PERPETUAL",
        resolution: str = "1D",
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        GET /public/get_tradingview_chart_data

        Args:
            instrument: e.g. "BTC-PERPETUAL"
            resolution: "1D", "60", "1", etc.
            start: yyyy-mm-dd
            end: yyyy-mm-dd

        Returns:
            DataFrame[timestamp, open, high, low, close, volume]
        """
        params = {"instrument_name": instrument, "resolution": resolution}
        if start:
            params["start_timestamp"] = self._to_ms(start)
        if end:
            params["end_timestamp"] = self._to_ms(end)

        try:
            data = self._request(
                f"{self.BASE_URL}/get_tradingview_chart_data",
                params,
                cache_prefix=f"dr-ohlcv-{instrument}-{resolution}",
            )
        except Exception as e:
            logger.warning(f"Deribit OHLCV fetch failed: {e}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        result = data.get("result", {})
        if not result or "ticks" not in result:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            {
                "timestamp": result.get("ticks", []),
                "open": result.get("open", []),
                "high": result.get("high", []),
                "low": result.get("low", []),
                "close": result.get("close", []),
                "volume": result.get("volume", []),
            }
        )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def get_ticker(self, instrument: str = "BTC-PERPETUAL") -> dict:
        """
        GET /public/ticker — mark price, open interest, funding rate context.

        Returns raw ticker dict.
        """
        params = {"instrument_name": instrument}
        try:
            data = self._request(
                f"{self.BASE_URL}/ticker",
                params,
                cache_prefix=f"dr-ticker-{instrument}",
            )
            return data.get("result", {})
        except Exception as e:
            logger.warning(f"Deribit ticker fetch failed: {e}")
            return {}

    def get_option_summary(self, currency: str = "BTC") -> pd.DataFrame:
        """
        GET /public/get_book_summary_by_currency — option put/call ratio context.

        Returns DataFrame of option book summaries.
        """
        params = {"currency": currency, "kind": "option"}
        try:
            data = self._request(
                f"{self.BASE_URL}/get_book_summary_by_currency",
                params,
                cache_prefix=f"dr-options-{currency}",
            )
        except Exception as e:
            logger.warning(f"Deribit options fetch failed: {e}")
            return pd.DataFrame()

        result = data.get("result", [])
        if not result:
            return pd.DataFrame()
            
        df = pd.DataFrame(result)
        
        # Calculate Implied Probability of expiring ITM using Delta as a proxy
        if "instrument_name" in df.columns:
            def extract_prob(row):
                # Deribit book summary doesn't always expose greeks without a ticker call,
                # but if we have delta available, we extract it. 
                # Proxy: Delta of Call = Prob(ITM), |Delta| of Put = Prob(ITM)
                prob = None
                try:
                    if isinstance(row.get("greeks"), dict) and "delta" in row["greeks"]:
                        delta = row["greeks"]["delta"]
                        if "-C" in row.get("instrument_name", ""):
                            prob = delta * 100
                        elif "-P" in row.get("instrument_name", ""):
                            prob = abs(delta) * 100
                except Exception:
                    pass
                return prob
                
            df["implied_probability"] = df.apply(extract_prob, axis=1)
            
        return df
