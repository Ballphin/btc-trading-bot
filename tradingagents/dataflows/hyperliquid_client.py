"""Hyperliquid public API — OHLCV candles, spot prices, and funding rates.

No authentication required. No rate limits documented.
Uses POST requests to https://api.hyperliquid.xyz/info.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from tradingagents.dataflows.base_client import BaseDataClient

logger = logging.getLogger(__name__)

# Hyperliquid interval strings
INTERVAL_1H = "1h"
INTERVAL_4H = "4h"
INTERVAL_1D = "1d"

# Max candles per request (API undocumented, stay under 5000)
MAX_CANDLES = 5000

# Map interval string → seconds for pagination
_INTERVAL_SECONDS = {
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class HyperliquidClient(BaseDataClient):
    """Fetch OHLCV candles, spot prices, and funding rates from Hyperliquid."""

    BASE_URL = "https://api.hyperliquid.xyz/info"

    def __init__(self, cache_ttl: int = 3600):
        super().__init__("hyperliquid", cache_ttl)

    def _post_request(
        self,
        payload: dict,
        cache_prefix: str = None,
        max_age_override: int = None,
    ) -> list | dict:
        """POST to Hyperliquid info endpoint with caching and retries.

        Args:
            payload: JSON body for the POST request.
            cache_prefix: Cache key prefix.
            max_age_override: If set, override cache TTL. 0 = never read cache.
        """
        import time
        import json
        import hashlib

        # Build cache path
        raw = f"{cache_prefix or 'hl'}:{json.dumps(payload, sort_keys=True)}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / f"{digest}.json"

        # Read cache (respect max_age_override)
        if max_age_override != 0:
            ttl = max_age_override if max_age_override is not None else self.cache_ttl
            if cache_path.exists():
                age = time.time() - cache_path.stat().st_mtime
                if age < ttl:
                    try:
                        import json as _json
                        return _json.loads(cache_path.read_text())
                    except Exception:
                        pass

        # POST with retries
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self.session.post(
                    self.BASE_URL, json=payload, timeout=30
                )
                if resp.status_code == 429:
                    wait = self.BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"Hyperliquid rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()

                # Write cache (skip if max_age_override=0 — ephemeral request)
                if max_age_override != 0:
                    self._write_cache(cache_path, data)

                return data
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(
                        f"Hyperliquid request failed after {self.MAX_RETRIES} retries: {e}"
                    )
                    raise
                import time as _t
                _t.sleep(self.BACKOFF_BASE ** (attempt + 1))

        raise RuntimeError(
            f"Hyperliquid request failed after {self.MAX_RETRIES} retries"
        )

    # ── OHLCV Candles ─────────────────────────────────────────────────

    def get_ohlcv(
        self,
        coin: str,
        interval: str = INTERVAL_1D,
        start: str = None,
        end: str = None,
        max_age_override: int = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles with automatic pagination.

        Args:
            coin: Base asset, e.g. "BTC" (not "BTC-USD").
            interval: "1h", "4h", or "1d".
            start: Start date yyyy-mm-dd.
            end: End date yyyy-mm-dd.
            max_age_override: Override cache TTL (0 = no cache).

        Returns:
            DataFrame[timestamp, open, high, low, close, volume] sorted ascending.
        """
        if not start or not end:
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=30)
        else:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")

        interval_sec = _INTERVAL_SECONDS.get(interval, 86400)
        all_candles = []
        current_start = start_dt

        while current_start < end_dt:
            chunk_end = min(
                end_dt,
                current_start + timedelta(seconds=interval_sec * MAX_CANDLES),
            )
            start_ms = int(current_start.timestamp() * 1000)
            end_ms = int(chunk_end.timestamp() * 1000)

            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
            }
            try:
                data = self._post_request(
                    payload,
                    cache_prefix=f"hl-ohlcv-{coin}-{interval}-{current_start.date()}",
                    max_age_override=max_age_override,
                )
                if isinstance(data, list):
                    all_candles.extend(data)
            except Exception as e:
                logger.warning(
                    f"Hyperliquid OHLCV fetch failed for {coin} chunk "
                    f"ending {chunk_end}: {e}"
                )
                break

            current_start = chunk_end + timedelta(seconds=1)

        if not all_candles:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(all_candles)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df["open"] = df["o"].astype(float)
        df["high"] = df["h"].astype(float)
        df["low"] = df["l"].astype(float)
        df["close"] = df["c"].astype(float)
        df["volume"] = df["v"].astype(float)
        df = (
            df[["timestamp", "open", "high", "low", "close", "volume"]]
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )
        return df

    # ── Spot Price ────────────────────────────────────────────────────

    def get_spot_price(
        self, coin: str, max_age_override: int = None
    ) -> Optional[float]:
        """Fetch current mid price for a coin.

        Args:
            coin: Base asset, e.g. "BTC".
            max_age_override: Override cache TTL (0 = no cache).
        """
        payload = {"type": "allMids"}
        try:
            data = self._post_request(
                payload,
                cache_prefix=f"hl-mids",
                max_age_override=max_age_override,
            )
            if isinstance(data, dict) and coin in data:
                return float(data[coin])
            return None
        except Exception as e:
            logger.warning(f"Hyperliquid spot price failed for {coin}: {e}")
            return None

    # ── Asset Context (full snapshot) ─────────────────────────────────

    def get_asset_context(
        self, coin: str, max_age_override: int = None
    ) -> Optional[dict]:
        """Fetch full asset context from metaAndAssetCtxs.

        Returns dict with: funding, premium, openInterest, dayNtlVlm,
        markPx, oraclePx, prevDayPx, or None on failure.
        """
        payload = {"type": "metaAndAssetCtxs"}
        try:
            data = self._post_request(
                payload,
                cache_prefix="hl-meta",
                max_age_override=max_age_override,
            )
            if isinstance(data, list) and len(data) >= 2:
                meta = data[0]
                ctxs = data[1]
                universe = meta.get("universe", [])
                for i, asset in enumerate(universe):
                    if asset.get("name", "").upper() == coin.upper():
                        if i < len(ctxs):
                            ctx = ctxs[i]
                            return {
                                "coin": coin,
                                "funding": float(ctx.get("funding", 0)),
                                "premium": float(ctx.get("premium", 0)),
                                "openInterest": float(ctx.get("openInterest", 0)),
                                "dayNtlVlm": float(ctx.get("dayNtlVlm", 0)),
                                "markPx": float(ctx.get("markPx", 0)),
                                "oraclePx": float(ctx.get("oraclePx", 0)),
                                "prevDayPx": float(ctx.get("prevDayPx", 0)),
                            }
            return None
        except Exception as e:
            logger.warning(f"Hyperliquid asset context failed for {coin}: {e}")
            return None

    # ── Funding Rates ─────────────────────────────────────────────────

    def get_predicted_funding(
        self, coin: str, max_age_override: int = None
    ) -> Optional[dict]:
        """Fetch predicted (next) funding rate.

        Returns:
            {"coin": str, "predicted_rate": float, "premium": float} or None.
        """
        payload = {"type": "metaAndAssetCtxs"}
        try:
            data = self._post_request(
                payload,
                cache_prefix="hl-meta",
                max_age_override=max_age_override,
            )
            if isinstance(data, list) and len(data) >= 2:
                meta = data[0]
                ctxs = data[1]
                # Find coin index
                universe = meta.get("universe", [])
                for i, asset in enumerate(universe):
                    if asset.get("name", "").upper() == coin.upper():
                        if i < len(ctxs):
                            ctx = ctxs[i]
                            return {
                                "coin": coin,
                                "predicted_rate": float(ctx.get("funding", 0)),
                                "premium": float(ctx.get("premium", 0)),
                                "open_interest": float(ctx.get("openInterest", 0)),
                            }
            return None
        except Exception as e:
            logger.warning(f"Hyperliquid predicted funding failed for {coin}: {e}")
            return None

    def get_funding_history(
        self,
        coin: str,
        start: str = None,
        end: str = None,
        max_age_override: int = None,
    ) -> pd.DataFrame:
        """Fetch historical (realized) funding rate settlements.

        Args:
            coin: Base asset.
            start: Start date yyyy-mm-dd (default: 30 days ago).
            end: End date yyyy-mm-dd (default: now).

        Returns:
            DataFrame[timestamp, funding_rate] sorted ascending.
        """
        if not start or not end:
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=30)
        else:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        payload = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": start_ms,
            "endTime": end_ms,
        }
        try:
            data = self._post_request(
                payload,
                cache_prefix=f"hl-funding-{coin}-{start_dt.date()}",
                max_age_override=max_age_override,
            )
            if not isinstance(data, list) or not data:
                return pd.DataFrame(columns=["timestamp", "funding_rate"])

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
            df["funding_rate"] = df["fundingRate"].astype(float)
            return (
                df[["timestamp", "funding_rate"]]
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )
        except Exception as e:
            logger.warning(f"Hyperliquid funding history failed for {coin}: {e}")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

    def get_realized_funding(
        self, coin: str, max_age_override: int = None
    ) -> Optional[float]:
        """Get the most recent realized funding rate.

        Returns:
            The last settled funding rate as a float, or None.
        """
        df = self.get_funding_history(
            coin,
            start=(datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d"),
            end=datetime.utcnow().strftime("%Y-%m-%d"),
            max_age_override=max_age_override,
        )
        if df.empty:
            return None
        return float(df.iloc[-1]["funding_rate"])
