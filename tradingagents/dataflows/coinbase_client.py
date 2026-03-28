"""Coinbase Exchange public API — OHLCV candles. No auth required."""

import logging
from datetime import datetime, timedelta

import pandas as pd

from tradingagents.dataflows.base_client import BaseDataClient

logger = logging.getLogger(__name__)

# Granularity constants (seconds)
GRANULARITY_1D = 86400
GRANULARITY_6H = 21600
GRANULARITY_1H = 3600
GRANULARITY_15M = 900
GRANULARITY_5M = 300

MAX_CANDLES_PER_REQUEST = 300


class CoinbaseClient(BaseDataClient):
    """Fetch OHLCV candle data from the Coinbase Exchange public REST API."""

    BASE_URL = "https://api.exchange.coinbase.com"

    def __init__(self, cache_ttl: int = 3600):
        super().__init__("coinbase", cache_ttl)

    def get_ohlcv(
        self,
        product_id: str,
        granularity: int = GRANULARITY_1D,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles with automatic pagination.

        Args:
            product_id: Trading pair, e.g. "BTC-USD"
            granularity: Candle width in seconds (86400, 21600, 3600, 900, 300)
            start: Start date yyyy-mm-dd
            end: End date yyyy-mm-dd

        Returns:
            DataFrame[timestamp, open, high, low, close, volume] sorted ascending.
        """
        url = f"{self.BASE_URL}/products/{product_id}/candles"

        if not start or not end:
            # Default to last 30 days
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=30)
        else:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")

        all_candles = []
        current_end = end_dt

        while current_end > start_dt:
            chunk_start = max(
                start_dt,
                current_end - timedelta(seconds=granularity * MAX_CANDLES_PER_REQUEST),
            )
            params = {
                "start": chunk_start.isoformat(),
                "end": current_end.isoformat(),
                "granularity": granularity,
            }
            try:
                data = self._request(
                    url, params, cache_prefix=f"cb-ohlcv-{product_id}-{granularity}-{chunk_start.date()}"
                )
                if isinstance(data, list):
                    all_candles.extend(data)
            except Exception as e:
                logger.warning(f"Coinbase OHLCV fetch failed for chunk ending {current_end}: {e}")
                break

            current_end = chunk_start - timedelta(seconds=1)

        if not all_candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Coinbase returns: [timestamp, low, high, open, close, volume]
        df = pd.DataFrame(
            all_candles, columns=["timestamp", "low", "high", "open", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_spot_price(self, pair: str = "BTC-USD") -> float:
        """Fetch current spot price."""
        url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
        data = self._request(url, cache_prefix=f"cb-spot-{pair}")
        return float(data.get("data", {}).get("amount", 0))
