"""Market regime detection for regime-aware backtest lesson retrieval.

Detects the current market regime so that only lessons from similar
market conditions are injected into agent prompts.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


REGIMES = ["trending_up", "trending_down", "ranging", "volatile"]


def detect_regime_context(ticker: str, date: str) -> dict:
    """Detect market regime and return full price context in a single yfinance call.

    Args:
        ticker: Ticker symbol
        date: Date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)

    Returns:
        Dict with keys: regime (str), current_price (float|None),
        daily_vol (float), above_sma20 (bool)
    """
    fallback = {
        "regime": "unknown",
        "current_price": None,
        "daily_vol": 0.0,
        "above_sma20": True,
    }
    try:
        import yfinance as yf

        date_only = date.split(" ")[0]
        dt = datetime.strptime(date_only, "%Y-%m-%d")

        fetch_start = (dt - timedelta(days=70)).strftime("%Y-%m-%d")
        fetch_end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

        data = yf.download(ticker, start=fetch_start, end=fetch_end,
                           progress=False, auto_adjust=True)
        if data.empty or len(data) < 20:
            return fallback

        closes = data["Close"].squeeze()

        sma20 = closes.rolling(20).mean().iloc[-1]
        sma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else sma20

        current = float(closes.iloc[-1])
        sma20_val = float(sma20)
        sma50_val = float(sma50)

        returns = closes.pct_change().dropna()
        vol_20 = float(returns.tail(20).std()) if len(returns) >= 20 else 0.0

        high_vol_threshold = 0.04 if _is_crypto(ticker) else 0.02
        above_sma20 = current > sma20_val

        if vol_20 > high_vol_threshold:
            regime = "volatile"
        elif current > sma20_val > sma50_val:
            regime = "trending_up"
        elif current < sma20_val < sma50_val:
            regime = "trending_down"
        else:
            regime = "ranging"

        return {
            "regime": regime,
            "current_price": current,
            "daily_vol": vol_20,
            "above_sma20": above_sma20,
        }

    except Exception as e:
        logger.debug(f"Regime context detection failed for {ticker} on {date}: {e}")
        return fallback


def detect_regime(ticker: str, date: str, results_dir: str = "./eval_results") -> str:
    """Detect the market regime for a ticker on a given date.

    Uses price data from cache if available, otherwise falls back to 'unknown'.

    Args:
        ticker: Ticker symbol
        date: Date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        results_dir: Results directory (unused, kept for API consistency)

    Returns:
        Regime string: one of trending_up, trending_down, ranging, volatile, unknown
    """
    return detect_regime_context(ticker, date)["regime"]


def _is_crypto(ticker: str) -> bool:
    """Simple check for crypto tickers."""
    return ticker.upper().endswith("-USD") or ticker.upper().endswith("USDT")


def tag_backtest_with_regime(decisions: list, ticker: str) -> str:
    """Determine the dominant regime across a backtest's decisions.

    Args:
        decisions: List of decision dicts from backtest
        ticker: Ticker symbol

    Returns:
        Dominant regime string
    """
    if not decisions:
        return "unknown"

    # Sample up to 5 evenly-spaced dates
    step = max(1, len(decisions) // 5)
    sampled = decisions[::step][:5]

    regime_counts: dict = {}
    for d in sampled:
        date = d.get("date", "")
        if not date:
            continue
        regime = detect_regime(ticker, str(date))
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    if not regime_counts:
        return "unknown"

    return max(regime_counts, key=regime_counts.get)
