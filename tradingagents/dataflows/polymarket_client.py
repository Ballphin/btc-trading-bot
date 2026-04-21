"""Polymarket Gamma API client — Stage 2 Commit K.2.

Replaces Kalshi's thin crypto-market coverage (typical OI $5-30k) with
Polymarket's USDC-denominated prediction markets (typical $50k-$8M per
market). No auth required for read-only Gamma API endpoints.

Endpoint reference: https://docs.polymarket.com/#gamma-api

Design parallels kalshi_client.py:
    * Source-level BACKTEST_MODE guard.
    * W-Z γ=0.91 multiplicative correction (Polymarket is a CLOB, less
      longshot bias than Kalshi's YES/NO book, but the formula still
      provides a uniform bias correction across vendors).
    * Rate-limit tolerant: catches 429 / 5xx and surfaces as explicit
      [DATA UNAVAILABLE] dashboard lines so agents never confuse an
      outage with a legitimate absence of markets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from tradingagents.backtesting.context import BACKTEST_MODE

logger = logging.getLogger(__name__)

POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"

# Crypto-tag filters used by Polymarket's tag taxonomy. Extend as needed
# when new assets trade on the platform.
CRYPTO_TAGS = ["crypto", "bitcoin", "ethereum", "solana", "memecoin"]

# Minimum per-market liquidity (USDC) for inclusion in the crypto
# prediction-markets narrative. Below this, the spread usually dwarfs
# the correction and the signal is pure noise.
MIN_LIQUIDITY_USDC = 10_000


def _fetch_markets(tag: str, *, limit: int = 20, timeout: float = 10.0) -> Dict[str, Any]:
    """Fetch open markets for a tag from the Gamma API.

    Returns a dict with ``markets`` (list) and ``error`` (str | None).
    The ``error`` key is non-None on any non-200 response or exception
    so the dashboard can render a DATA UNAVAILABLE line.
    """
    url = f"{POLYMARKET_GAMMA_BASE}/markets"
    params = {"tag_slug": tag, "active": "true", "closed": "false", "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            return {"markets": [], "error": "HTTP 429 (rate limit)"}
        if r.status_code != 200:
            return {"markets": [], "error": f"HTTP {r.status_code}"}
        data = r.json()
        return {"markets": data if isinstance(data, list) else data.get("data", []),
                "error": None}
    except Exception as e:
        logger.warning("Polymarket fetch failed for tag=%s: %s", tag, e)
        return {"markets": [], "error": type(e).__name__}


def _wz_correct(p: float, gamma: float = 0.91) -> float:
    """Apply Wolfers-Zitzewitz γ=0.91 multiplicative correction."""
    p = max(min(float(p), 0.999), 0.001)
    num = p ** gamma
    return num / (num + (1.0 - p) ** gamma)


def _summarise_market(m: Dict[str, Any]) -> Optional[str]:
    """Render a single market line or return None if it should be skipped."""
    question = m.get("question") or m.get("title") or "unknown"
    liquidity = float(m.get("liquidity") or m.get("liquidityNum") or 0.0)
    last_price = m.get("lastTradePrice") or m.get("bestBid") or m.get("outcomePrices")
    # Polymarket can surface price as a scalar or a list (YES/NO outcomes)
    if isinstance(last_price, list) and last_price:
        try:
            last_price = float(last_price[0])
        except (TypeError, ValueError):
            last_price = None
    try:
        last_price = float(last_price) if last_price is not None else None
    except (TypeError, ValueError):
        last_price = None

    if last_price is None:
        return f"- *{question}*: [NO PRICE]"
    if liquidity < MIN_LIQUIDITY_USDC:
        return (
            f"- *{question}*: raw {last_price:.2f} "
            f"[LOW LIQUIDITY: ${liquidity:,.0f} < ${MIN_LIQUIDITY_USDC:,}]"
        )
    corrected = _wz_correct(last_price)
    return (
        f"- *{question}*: Implied Probability {round(corrected * 100)}% "
        f"(liq ${liquidity:,.0f})"
    )


def get_polymarket_crypto_context(tags: Optional[List[str]] = None) -> str:
    """Return a markdown crypto prediction-market dashboard.

    Backtest-mode aware at the source (never emits live data in a
    historical replay). Safe to call even without network — errors
    surface as explicit DATA UNAVAILABLE lines.
    """
    if BACKTEST_MODE.get():
        return "[POLYMARKET: DISABLED IN BACKTEST MODE]"

    tags = tags or CRYPTO_TAGS
    lines: List[str] = ["### Polymarket Crypto Prediction Markets\n"]

    for tag in tags:
        payload = _fetch_markets(tag)
        if payload["error"]:
            lines.append(f"**{tag}**: [DATA UNAVAILABLE: {payload['error']}]\n")
            continue
        markets = payload["markets"] or []
        if not markets:
            continue
        # Sort newest-close first so time-sensitive markets surface first.
        markets.sort(
            key=lambda m: float(m.get("liquidity") or m.get("liquidityNum") or 0),
            reverse=True,
        )
        lines.append(f"**{tag.capitalize()}**")
        for m in markets[:5]:
            s = _summarise_market(m)
            if s:
                lines.append(s)
        lines.append("")  # spacer

    if len(lines) == 1:
        return "Polymarket crypto markets unavailable (no active markets or connection failure)."
    return "\n".join(lines)
