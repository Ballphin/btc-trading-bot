"""Hyperliquid public API — OHLCV candles, spot prices, and funding rates.

No authentication required. No rate limits documented.
Uses POST requests to https://api.hyperliquid.xyz/info.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from tradingagents.dataflows.base_client import BaseDataClient

logger = logging.getLogger(__name__)

# Hyperliquid interval strings
INTERVAL_1M = "1m"
INTERVAL_3M = "3m"
INTERVAL_5M = "5m"
INTERVAL_15M = "15m"
INTERVAL_30M = "30m"
INTERVAL_1H = "1h"
INTERVAL_4H = "4h"
INTERVAL_1D = "1d"

# Max candles per request (API undocumented, stay under 5000)
MAX_CANDLES = 5000

# Map interval string → seconds for pagination
_INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Per-interval cache TTL for Pulse — each callsite MUST pass via max_age_override.
# Without this, BaseDataClient.cache_ttl (3600s) makes sub-hourly data stale.
PULSE_CACHE_TTL = {
    "1m": 60,
    "3m": 120,
    "5m": 120,
    "15m": 300,
    "30m": 600,
    "1h": 600,
    "4h": 1800,
    "1d": 3600,
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
            interval: "1m", "5m", "15m", "1h", "4h", or "1d".
            start: Start date yyyy-mm-dd.
            end: End date yyyy-mm-dd.
            max_age_override: Override cache TTL (0 = no cache).

        Returns:
            DataFrame[timestamp, open, high, low, close, volume] sorted ascending.
        """
        if not start or not end:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=30)
        else:
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

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
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
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
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=30)
        else:
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

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
            start=(datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d"),
            end=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            max_age_override=max_age_override,
        )
        if df.empty:
            return None
        return float(df.iloc[-1]["funding_rate"])

    # ── L2 Book Snapshot (book imbalance) ─────────────────────────────

    def get_l2_snapshot(
        self, coin: str, max_age_override: int = None
    ) -> Optional[dict]:
        """Fetch L2 order book snapshot.

        Returns:
            {"coin": str, "bids": [(px, sz), ...], "asks": [(px, sz), ...]} or None.
            Levels are sorted best → worst.
        """
        payload = {"type": "l2Book", "coin": coin}
        try:
            data = self._post_request(
                payload,
                cache_prefix=f"hl-l2-{coin}",
                max_age_override=max_age_override,
            )
            if not isinstance(data, dict) or "levels" not in data:
                return None
            levels = data["levels"]
            if not isinstance(levels, list) or len(levels) < 2:
                return None
            bids_raw = levels[0] if isinstance(levels[0], list) else []
            asks_raw = levels[1] if isinstance(levels[1], list) else []

            def _parse(lst: list):
                out = []
                for lvl in lst:
                    if isinstance(lvl, dict):
                        px = float(lvl.get("px", 0))
                        sz = float(lvl.get("sz", 0))
                        if px > 0 and sz > 0:
                            out.append((px, sz))
                return out

            return {
                "coin": coin,
                "bids": _parse(bids_raw),
                "asks": _parse(asks_raw),
            }
        except Exception as e:
            logger.warning(f"Hyperliquid L2 snapshot failed for {coin}: {e}")
            return None

    def compute_book_imbalance(
        self, coin: str, levels: int = 1, max_age_override: int = 60
    ) -> Optional[float]:
        """(bid_size - ask_size) / (bid_size + ask_size) over top `levels` rungs.

        Returns:
            Float in [-1, +1] or None if snapshot unavailable.
        """
        snap = self.get_l2_snapshot(coin, max_age_override=max_age_override)
        if snap is None:
            return None
        bids = snap["bids"][:levels]
        asks = snap["asks"][:levels]
        if not bids or not asks:
            return None
        bid_sz = sum(sz for _, sz in bids)
        ask_sz = sum(sz for _, sz in asks)
        denom = bid_sz + ask_sz
        if denom < 1e-12:
            return None
        return float((bid_sz - ask_sz) / denom)

    # ── Recent trades / liquidation cascade detection ─────────────────

    def get_recent_trades(
        self, coin: str, max_age_override: int = 60
    ) -> list:
        """Fetch recent trades for a coin. Returns list of dicts with at least
        {px, sz, time, side}. Empty list on failure."""
        # Note: Hyperliquid public endpoint is "recentTrades"; exact field
        # names are px, sz, time, side, users.
        payload = {"type": "recentTrades", "coin": coin}
        try:
            data = self._post_request(
                payload,
                cache_prefix=f"hl-trades-{coin}",
                max_age_override=max_age_override,
            )
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.warning(f"Hyperliquid recentTrades failed for {coin}: {e}")
            return []

    def liquidation_cluster_score(
        self,
        coin: str,
        window_minutes: int = 15,
        size_normalizer_usd: float = 1e6,
        max_age_override: int = 60,
    ) -> Optional[float]:
        """Log-normalized liquidation cluster magnitude in the trailing window.

        The Hyperliquid public API does not expose a dedicated liquidations
        endpoint. We approximate via recent trades filtered for the
        "liquidation" user (Hyperliquid tags liq fills; when absent we return
        None so callers can skip the factor).

        Returns:
            signed log1p(notional / size_normalizer_usd) with sign matching
            the net direction of liquidated side (positive = shorts liquidated
            on an up-move → bearish unwind; negative = longs liquidated on a
            down-move → bullish snapback opportunity if vol falling).
            None if the feed is unavailable or no liq tag is present.
        """
        import math as _math
        trades = self.get_recent_trades(coin, max_age_override=max_age_override)
        if not trades:
            return None
        cutoff_ms = int((datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).timestamp() * 1000)
        liq_notional_signed = 0.0
        any_liq_tag = False
        for t in trades:
            try:
                ts_ms = int(t.get("time", 0))
                if ts_ms < cutoff_ms:
                    continue
                users = t.get("users") or []
                is_liq = any(
                    isinstance(u, str) and "liq" in u.lower() for u in users
                )
                if not is_liq:
                    continue
                any_liq_tag = True
                px = float(t.get("px", 0))
                sz = float(t.get("sz", 0))
                side = t.get("side", "")
                notional = px * sz
                # side "B" on HL = buy-aggressor (shorts being liquidated)
                # side "A" = ask-aggressor (longs being liquidated)
                sign = 1.0 if side == "B" else -1.0 if side == "A" else 0.0
                liq_notional_signed += sign * notional
            except (ValueError, TypeError):
                continue
        if not any_liq_tag:
            return None
        mag = _math.log1p(abs(liq_notional_signed) / max(size_normalizer_usd, 1.0))
        return float(mag if liq_notional_signed >= 0 else -mag)
