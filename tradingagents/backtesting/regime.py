"""Market regime detection for regime-aware backtest lesson retrieval.

Detects the current market regime so that only lessons from similar
market conditions are injected into agent prompts.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Sub-daily intervals supported for crypto only
_SUBDAILY_INTERVALS = {"1h", "4h"}


REGIMES = ["trending_up", "trending_down", "ranging", "volatile"]


def detect_regime_context(ticker: str, date: str, interval: str = "1d") -> dict:
    """Detect market regime and return full price context.

    Args:
        ticker: Ticker symbol
        date: Date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        interval: Candle interval ("1d", "1h", "4h"). Sub-daily only for crypto.

    Returns:
        Dict with keys: regime (str), current_price (float|None),
        daily_vol (float), above_sma20 (bool), interval (str)
    """
    fallback = {
        "regime": "unknown",
        "current_price": None,
        "daily_vol": 0.0,
        "above_sma20": True,
        "interval": interval,
    }

    # Sub-daily only supported for crypto
    is_crypto = _is_crypto(ticker)
    if interval in _SUBDAILY_INTERVALS and not is_crypto:
        logger.debug(f"Sub-daily regime detection only supported for crypto; {ticker} is not crypto")
        return fallback

    try:
        if interval in _SUBDAILY_INTERVALS and is_crypto:
            # Use binance client for sub-daily crypto data
            return _detect_regime_context_binance(ticker, date, interval)

        # Default: daily data via yfinance
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
            "interval": interval,
        }

    except Exception as e:
        logger.debug(f"Regime context detection failed for {ticker} on {date}: {e}")
        return fallback


def _detect_regime_context_binance(ticker: str, date: str, interval: str) -> dict:
    """Detect regime using Binance sub-daily data for crypto.

    Args:
        ticker: Ticker symbol (e.g., "BTC-USD")
        date: Date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        interval: "1h" or "4h"

    Returns:
        Regime context dict
    """
    from tradingagents.dataflows.binance_client import BinanceClient

    fallback = {
        "regime": "unknown",
        "current_price": None,
        "daily_vol": 0.0,
        "above_sma20": True,
        "interval": interval,
    }

    try:
        client = BinanceClient()

        # Convert ticker to Binance format (BTC-USD -> BTCUSDT)
        symbol = ticker.replace("-", "").replace("USD", "USDT")
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"

        # Parse date
        date_only = date.split(" ")[0]
        dt = datetime.strptime(date_only, "%Y-%m-%d")

        # Compute lookback: 70 days of daily = 70 candles
        # For 4h: 70 * 6 = 420 candles; for 1h: 70 * 24 = 1680 candles
        # We fetch slightly more to ensure we have enough after filtering
        if interval == "4h":
            lookback_days = 120  # ~720 4h candles
        else:  # 1h
            lookback_days = 80  # ~1920 1h candles

        fetch_start = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        fetch_end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

        data = client.get_klines(symbol=symbol, interval=interval, start=fetch_start, end=fetch_end)

        if data.empty or len(data) < 20:
            return fallback

        closes = data["close"].squeeze()

        sma20 = closes.rolling(20).mean().iloc[-1]
        sma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else sma20

        current = float(closes.iloc[-1])
        sma20_val = float(sma20)
        sma50_val = float(sma50)

        returns = closes.pct_change().dropna()
        vol_20 = float(returns.tail(20).std()) if len(returns) >= 20 else 0.0

        # Use higher threshold for sub-daily (annualized vol is higher at shorter intervals)
        # For 4h: threshold roughly 2x daily; for 1h: roughly 4x daily
        multiplier = 2.0 if interval == "4h" else 4.0
        high_vol_threshold = 0.04 * multiplier
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
            "interval": interval,
        }

    except Exception as e:
        logger.debug(f"Binance regime detection failed for {ticker} on {date}: {e}")
        return fallback


def detect_regime(ticker: str, date: str, results_dir: str = "./eval_results", interval: str = "1d") -> str:
    """Detect the market regime for a ticker on a given date.

    Uses price data from cache if available, otherwise falls back to 'unknown'.

    Args:
        ticker: Ticker symbol
        date: Date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        results_dir: Results directory (unused, kept for API consistency)
        interval: Candle interval ("1d", "1h", "4h"). Sub-daily only for crypto.

    Returns:
        Regime string: one of trending_up, trending_down, ranging, volatile, unknown
    """
    return detect_regime_context(ticker, date, interval=interval)["regime"]


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
