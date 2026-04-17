"""Unified crypto data layer — wraps Coinbase, Binance, Deribit, BGeometrics,
FRED, and Fear & Greed clients for vendor routing.

Each function returns a formatted string matching the pattern used by y_finance.py,
so the vendor routing system in interface.py can swap them in transparently.
"""

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.coinbase_client import CoinbaseClient
from tradingagents.dataflows.deribit_client import DeribitClient
from tradingagents.dataflows.bgeometrics_client import BGeometricsClient
from tradingagents.dataflows.fred_client import FREDClient
from tradingagents.dataflows.fear_greed_client import FearGreedClient
from tradingagents.dataflows.hyperliquid_client import HyperliquidClient

logger = logging.getLogger(__name__)


def _get_display_tz() -> ZoneInfo:
    """Return the configured display timezone (default US/Eastern)."""
    try:
        from tradingagents.dataflows.config import get_config
        tz_name = get_config().get("display_timezone", "US/Eastern")
    except Exception:
        tz_name = "US/Eastern"
    return ZoneInfo(tz_name)


def _fmt_ts_local(utc_dt: datetime, fmt: str = "%Y-%m-%d %H:%M %Z") -> str:
    """Format a UTC datetime into the display timezone."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(_get_display_tz()).strftime(fmt)


def _is_current_date(date_str: str) -> bool:
    """Check if a date string refers to today or the future (live query)."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return d >= datetime.now(timezone.utc).date()
    except ValueError:
        return False

# Lazy-initialized singletons
_clients = {}


def _get(cls, key):
    if key not in _clients:
        _clients[key] = cls()
    return _clients[key]


def _normalize_symbol(ticker: str) -> str:
    """BTC-USD -> BTC, ETH-USD -> ETH, BTCUSDT -> BTC"""
    return ticker.replace("-USD", "").replace("USDT", "").upper()


def _to_coinbase_product(ticker: str) -> str:
    """Ensure ticker is in Coinbase format: BTC-USD"""
    base = _normalize_symbol(ticker)
    return f"{base}-USD"


def _to_deribit_instrument(ticker: str) -> str:
    """Ensure ticker is in Deribit format: BTC-PERPETUAL"""
    base = _normalize_symbol(ticker)
    return f"{base}-PERPETUAL"


# ---------------------------------------------------------------------------
# Vendor routing implementations
# ---------------------------------------------------------------------------

# Timeframe → granularity mapping
_TIMEFRAME_GRANULARITY = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 21600,
    "1d": 86400,
}


def get_crypto_price_data(
    symbol: str, start_date: str, end_date: str, timeframe: str = "1d"
) -> str:
    """
    OHLCV from Hyperliquid (primary) / Coinbase (fallback).

    Args:
        symbol: Ticker symbol (e.g. BTC-USD).
        start_date: Start date yyyy-mm-dd.
        end_date: End date yyyy-mm-dd.
        timeframe: "1h", "4h", or "1d" (default).

    Returns formatted string of OHLCV data.
    """
    base_asset = _normalize_symbol(symbol)
    granularity = _TIMEFRAME_GRANULARITY.get(timeframe, 86400)

    # Use short cache TTL for live (current-day) queries
    cache_override = 120 if _is_current_date(end_date) else None

    # Try Hyperliquid first (better altcoin coverage, no rate limits)
    df = pd.DataFrame()
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        df = hl.get_ohlcv(base_asset, timeframe, start_date, end_date,
                          max_age_override=cache_override)
    except Exception as e:
        logger.warning(f"Hyperliquid OHLCV failed for {symbol}, falling back to Coinbase: {e}")

    # Fallback to Coinbase
    if df.empty:
        cb = _get(CoinbaseClient, "coinbase")
        product = _to_coinbase_product(symbol)
        df = cb.get_ohlcv(product, granularity, start_date, end_date)

    if df.empty:
        return f"No OHLCV data available for {symbol} ({start_date} to {end_date}, {timeframe})"

    # Format to match yfinance output style, converting to display timezone
    df_display = df.copy()
    display_tz = _get_display_tz()
    if timeframe == "1d":
        df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d")
    else:
        df_display["timestamp"] = df_display["timestamp"].dt.tz_localize("UTC").dt.tz_convert(display_tz).dt.strftime("%Y-%m-%d %H:%M %Z")
    df_display = df_display.rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    tz_label = datetime.now(timezone.utc).astimezone(display_tz).strftime("%Z")
    header = f"# {symbol} OHLCV ({timeframe} bars) — {start_date} to {end_date} (times in {tz_label})\n"
    return header + df_display.to_string(index=False)


def get_crypto_indicators(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
    timeframe: str = "1d",
) -> str:
    """
    Technical indicators computed from Hyperliquid/Coinbase OHLCV via stockstats.

    Args:
        timeframe: "1h", "4h", or "1d" (default). Labels output accordingly.
    """
    base_asset = _normalize_symbol(symbol)
    granularity = _TIMEFRAME_GRANULARITY.get(timeframe, 86400)

    # Fetch enough history for indicator calculation
    start = (
        datetime.strptime(curr_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        - timedelta(days=look_back_days + 60)
    ).strftime("%Y-%m-%d")

    # Use short cache TTL for live (current-day) queries
    cache_override = 120 if _is_current_date(curr_date) else None

    # Try Hyperliquid first, fall back to Coinbase
    df = pd.DataFrame()
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        df = hl.get_ohlcv(base_asset, timeframe, start, curr_date,
                          max_age_override=cache_override)
    except Exception as e:
        logger.warning(f"Hyperliquid indicators failed for {symbol}, falling back: {e}")

    if df.empty:
        cb = _get(CoinbaseClient, "coinbase")
        product = _to_coinbase_product(symbol)
        df = cb.get_ohlcv(product, granularity, start, curr_date)
    if df.empty:
        return f"No data available for {symbol} to compute {indicator}"

    # Prepare DataFrame for stockstats (expects Date, Open, High, Low, Close, Volume)
    df = df.rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])

    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    df[price_cols] = df[price_cols].ffill().bfill()

    ss = wrap(df)
    ss["Date"] = ss["Date"].dt.strftime("%Y-%m-%d")

    try:
        ss[indicator]  # trigger stockstats calculation
    except Exception as e:
        return f"Could not compute indicator '{indicator}' for {symbol}: {e}"

    tf_label = f"({timeframe} bars)" if timeframe != "1d" else "(daily bars)"
    curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")
    matching = ss[ss["Date"].str.startswith(curr_date_str)]

    if not matching.empty:
        value = matching[indicator].values[-1]  # last bar on target date
        return f"{indicator} {tf_label} for {symbol} on {curr_date}: {value}"
    else:
        return f"{indicator} {tf_label} for {symbol}: N/A (not a trading day or insufficient data)"


def get_derivatives_data(symbol: str, curr_date: str) -> str:
    """
    Hyperliquid funding rates + open interest + premium (directional aggression),
    cross-referenced with Deribit perpetual funding rates.

    Returns a structured markdown report.
    """
    hl = _get(HyperliquidClient, "hyperliquid")
    dr = _get(DeribitClient, "deribit")

    base_asset = _normalize_symbol(symbol)
    dr_instrument = _to_deribit_instrument(symbol)

    start = (
        datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=30)
    ).strftime("%Y-%m-%d")

    # Fetch Hyperliquid data
    hl_funding = hl.get_funding_history(base_asset, start, curr_date)
    hl_ctx = hl.get_asset_context(base_asset)
    dr_funding = dr.get_funding_rate_history(dr_instrument, start, curr_date)

    lines = [
        f"# Derivatives Analysis: {symbol}",
        f"**Period**: {start} to {curr_date}",
        "",
    ]

    # Hyperliquid Funding Rates (historical)
    lines.append("## Funding Rates — Hyperliquid (8h settlements)")
    if not hl_funding.empty and "funding_rate" in hl_funding.columns:
        latest_rate = float(hl_funding.iloc[-1]["funding_rate"])
        avg_rate = hl_funding["funding_rate"].mean()
        max_rate = hl_funding["funding_rate"].max()
        min_rate = hl_funding["funding_rate"].min()
        lines.append(f"- **Latest realized**: {latest_rate:.6f}")
        lines.append(f"- **30d average**: {avg_rate:.6f}")
        lines.append(f"- **30d range**: [{min_rate:.6f}, {max_rate:.6f}]")
        if latest_rate > 0.0003:
            lines.append("- ⚠️ Elevated positive funding — longs paying shorts, potential long squeeze risk")
        elif latest_rate < -0.0003:
            lines.append("- ⚠️ Elevated negative funding — shorts paying longs, potential short squeeze risk")

        # Funding Rate Momentum: predicted vs realized + direction
        if len(hl_funding) >= 2:
            prev_rate = float(hl_funding.iloc[-2]["funding_rate"])
            funding_8h_change = latest_rate - prev_rate
            if funding_8h_change > 0.0001:
                direction = "rising"
            elif funding_8h_change < -0.0001:
                direction = "falling"
            else:
                direction = "stable"
            lines.append(f"- **8h change**: {funding_8h_change:+.6f} ({direction})")
            lines.append(f"- **Previous settlement**: {prev_rate:.6f}")

        # Predicted funding from asset context
        if hl_ctx:
            pred_rate = hl_ctx["funding"]
            lines.append(f"- **Predicted (next settlement)**: {pred_rate:.6f}")
            pred_vs_realized = pred_rate - latest_rate
            pred_dir = "rising" if pred_vs_realized > 0.0001 else ("falling" if pred_vs_realized < -0.0001 else "stable")
            lines.append(f"- **Predicted vs realized**: {pred_vs_realized:+.6f} ({pred_dir})")
    else:
        lines.append("- Data unavailable")
    lines.append("")

    # Open Interest + Volume (from asset context snapshot)
    lines.append("## Open Interest & Volume — Hyperliquid")
    if hl_ctx:
        oi = hl_ctx["openInterest"]
        mark_px = hl_ctx["markPx"]
        oi_value = oi * mark_px if mark_px > 0 else 0
        day_vol = hl_ctx["dayNtlVlm"]
        lines.append(f"- **Open Interest**: {oi:,.2f} contracts")
        lines.append(f"- **OI Notional Value**: ${oi_value:,.0f}")
        lines.append(f"- **24h Notional Volume**: ${day_vol:,.0f}")
        if day_vol > 0 and oi_value > 0:
            turnover = day_vol / oi_value
            lines.append(f"- **OI Turnover (vol/OI)**: {turnover:.2f}x")
    else:
        lines.append("- Data unavailable")
    lines.append("")

    # Directional Aggression (premium — replaces Binance taker ratio)
    lines.append("## Directional Aggression — Hyperliquid Premium")
    if hl_ctx:
        premium = hl_ctx["premium"]
        mark_px = hl_ctx["markPx"]
        oracle_px = hl_ctx["oraclePx"]
        prev_day_px = hl_ctx["prevDayPx"]
        premium_bps = premium * 10000
        lines.append(f"- **Mark Price**: ${mark_px:,.2f}")
        lines.append(f"- **Oracle Price**: ${oracle_px:,.2f}")
        lines.append(f"- **Premium**: {premium_bps:+.1f} bps ({premium:.6f})")
        if premium > 0.0005:
            lines.append("- Perp trading above index — **buyers aggressing**, bullish pressure")
        elif premium < -0.0005:
            lines.append("- Perp trading below index — **sellers aggressing**, bearish pressure")
        else:
            lines.append("- Balanced — perp near index price")
        if prev_day_px > 0:
            day_change = (mark_px - prev_day_px) / prev_day_px * 100
            lines.append(f"- **24h price change**: {day_change:+.2f}%")
    else:
        lines.append("- Data unavailable")
    lines.append("")

    # Deribit Perp Funding (cross-reference)
    lines.append("## Deribit Perpetual Funding (hourly, cross-reference)")
    if not dr_funding.empty and "interest_8h" in dr_funding.columns:
        latest_8h = dr_funding.iloc[-1].get("interest_8h", "N/A")
        lines.append(f"- **Latest 8h rate**: {latest_8h}")
        avg_8h = dr_funding["interest_8h"].mean()
        lines.append(f"- **Period average 8h**: {avg_8h:.8f}")
    else:
        lines.append("- Data unavailable")

    return "\n".join(lines)


def get_onchain_data(symbol: str, curr_date: str) -> str:
    """
    BGeometrics on-chain metrics: MVRV, SOPR, exchange netflows, reserves, NUPL.

    NOTE: These metrics are Bitcoin-specific. For non-BTC assets,
    returns a message indicating on-chain data is not available.

    Returns formatted markdown report.
    """
    base = _normalize_symbol(symbol)

    # BGeometrics only provides Bitcoin on-chain metrics
    if base != "BTC":
        return (
            f"# On-Chain Analysis for {symbol}\n\n"
            f"On-chain metrics (MVRV, SOPR, exchange flows, NUPL) are only "
            f"available for Bitcoin. These metrics are derived from Bitcoin's "
            f"UTXO-based blockchain and do not apply to {symbol}.\n\n"
            f"For {symbol} analysis, focus on:\n"
            f"- Technical indicators and price action\n"
            f"- Derivatives data (funding rates, open interest)\n"
            f"- Market sentiment and macro context\n"
        )

    bg = _get(BGeometricsClient, "bgeometrics")
    return bg.get_onchain_summary(curr_date, look_back_days=60)


def get_macro_context(curr_date: str) -> str:
    """
    FRED macro indicators: M2 money supply, DXY, 2Y/10Y treasury yields.

    Returns formatted markdown table.
    """
    fred = _get(FREDClient, "fred")
    return fred.get_macro_summary(curr_date, look_back_days=90)


def get_crypto_sentiment(curr_date: str) -> str:
    """
    Alternative.me Fear & Greed index with historical trend.

    Returns formatted markdown report.
    """
    fg = _get(FearGreedClient, "fear_greed")
    return fg.get_sentiment_report(look_back_days=30)


# ---------------------------------------------------------------------------
# Intraday summary (BLOCKER 2)
# ---------------------------------------------------------------------------

def get_intraday_summary(symbol: str, curr_date: str) -> str:
    """
    Fetch last 24h of 1H candles and compute intraday context metrics.

    Returns formatted markdown with:
    - high_low_range_pct
    - approx_vwap (Typical Price VWAP from hourly bars)
    - max_drawdown_intraday_pct (high-to-low worst case)
    - binance_taker_ratio_latest
    - funding_rate_8h_change (predicted - last realized)

    For non-crypto assets, returns a graceful "not available" message.
    """
    from tradingagents.dataflows.asset_detection import is_crypto as _is_crypto

    if not _is_crypto(symbol):
        return f"Intraday summary not available for {symbol} (equities not supported in v1). Use daily bars."

    base_asset = _normalize_symbol(symbol)

    # Date range: 24h ending at curr_date EOD (UTC)
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    start_dt = end_dt - timedelta(hours=24)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # Always bypass cache for intraday summary (live context)
    df = pd.DataFrame()
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        df = hl.get_ohlcv(base_asset, "1h", start_str, end_str, max_age_override=0)
    except Exception as e:
        logger.warning(f"Hyperliquid 1H failed for {symbol}: {e}")

    if df.empty:
        try:
            cb = _get(CoinbaseClient, "coinbase")
            product = _to_coinbase_product(symbol)
            df = cb.get_ohlcv(product, 3600, start_str, end_str)
        except Exception as e:
            logger.warning(f"Coinbase 1H fallback failed for {symbol}: {e}")

    if df.empty:
        return f"Intraday data unavailable for {symbol} on {curr_date}"

    # Display timezone for report headers
    display_tz = _get_display_tz()
    end_local = end_dt.astimezone(display_tz).strftime("%Y-%m-%d %H:%M %Z")

    lines = [
        f"# Intraday Summary: {symbol} (24h ending {end_local})",
        f"**Bars**: {len(df)} hourly candles",
        "",
    ]

    # High-Low range
    range_pct = (df["high"].max() - df["low"].min()) / df["open"].iloc[0] * 100
    lines.append(f"## Price Range")
    lines.append(f"- **24h High**: {df['high'].max():,.2f}")
    lines.append(f"- **24h Low**: {df['low'].min():,.2f}")
    lines.append(f"- **Range**: {range_pct:.2f}%")
    lines.append("")

    # Approx VWAP (Typical Price method)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    total_vol = df["volume"].sum()
    if total_vol > 0:
        vwap = (tp * df["volume"]).sum() / total_vol
        lines.append(f"## Approx VWAP (hourly)")
        lines.append(f"- **VWAP**: {vwap:,.2f}")
        last_close = df["close"].iloc[-1]
        vwap_diff_pct = (last_close - vwap) / vwap * 100
        lines.append(f"- **Price vs VWAP**: {vwap_diff_pct:+.2f}% ({'above' if vwap_diff_pct > 0 else 'below'})")
        lines.append("")

    # Max drawdown intraday (high for peak, low for trough — worst case)
    cummax_high = df["high"].cummax()
    drawdown = (df["low"] - cummax_high) / cummax_high * 100
    max_dd = drawdown.min()
    lines.append(f"## Max Intraday Drawdown")
    lines.append(f"- **Worst-case drawdown from peak**: {max_dd:.2f}%")
    lines.append("")

    # Directional aggression (Hyperliquid premium)
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        hl_ctx = hl.get_asset_context(base_asset)
        if hl_ctx:
            premium = hl_ctx["premium"]
            premium_bps = premium * 10000
            lines.append(f"## Directional Aggression — Hyperliquid Premium")
            lines.append(f"- **Premium**: {premium_bps:+.1f} bps ({premium:.6f})")
            if premium > 0.0005:
                lines.append("- Perp above index — **buyers aggressing**, bullish pressure")
            elif premium < -0.0005:
                lines.append("- Perp below index — **sellers aggressing**, bearish pressure")
            else:
                lines.append("- Balanced — perp near index price")
            lines.append("")
    except Exception as e:
        logger.warning(f"Hyperliquid premium failed for {symbol}: {e}")

    # Funding rate change (predicted - last realized)
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        predicted = hl.get_predicted_funding(base_asset)
        realized = hl.get_realized_funding(base_asset)
        if predicted and realized is not None:
            pred_rate = predicted["predicted_rate"]
            change = pred_rate - realized
            direction = "rising" if change > 0.0001 else ("falling" if change < -0.0001 else "stable")
            lines.append(f"## Funding Rate Momentum")
            lines.append(f"- **Predicted (next settlement)**: {pred_rate:.6f}")
            lines.append(f"- **Last realized**: {realized:.6f}")
            lines.append(f"- **8h change**: {change:+.6f} ({direction})")
            if abs(pred_rate) > 0.0003:
                lines.append(f"- ⚠️ Elevated funding — potential squeeze risk")
            lines.append("")
    except Exception as e:
        logger.warning(f"Funding rate summary failed for {symbol}: {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Realtime snapshot (BLOCKER 3)
# ---------------------------------------------------------------------------

def get_realtime_snapshot(symbol: str) -> str:
    """
    Zero-cache realtime snapshot: spot price, 1h candle, predicted + realized funding.

    Designed for same-day re-analysis — always fetches live data.
    NOT safe for backtests (would inject future data).
    """
    from tradingagents.dataflows.asset_detection import is_crypto as _is_crypto

    if not _is_crypto(symbol):
        return f"Realtime snapshot not available for {symbol} (equities not supported). Use get_stock_data."

    base_asset = _normalize_symbol(symbol)
    now = datetime.now(timezone.utc)
    now_local = _fmt_ts_local(now, "%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"# Realtime Snapshot: {symbol}",
        f"**Timestamp**: {now_local}",
        "",
    ]

    # Spot price (zero cache)
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        price = hl.get_spot_price(base_asset, max_age_override=0)
        if price:
            lines.append(f"## Current Price")
            lines.append(f"- **Mid price**: ${price:,.2f}")
            lines.append("")
    except Exception as e:
        logger.warning(f"Realtime spot failed for {symbol}: {e}")

    # Latest 1H candle (zero cache)
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        start_1h = (now - timedelta(hours=2)).strftime("%Y-%m-%d")
        end_1h = (now + timedelta(hours=1)).strftime("%Y-%m-%d")
        df = hl.get_ohlcv(base_asset, "1h", start_1h, end_1h, max_age_override=0)
        if not df.empty:
            last = df.iloc[-1]
            change_pct = (last["close"] - last["open"]) / last["open"] * 100
            lines.append(f"## Latest 1H Candle")
            lines.append(f"- **Open**: {last['open']:,.2f}  **Close**: {last['close']:,.2f}")
            lines.append(f"- **High**: {last['high']:,.2f}  **Low**: {last['low']:,.2f}")
            lines.append(f"- **1H change**: {change_pct:+.2f}%")
            lines.append(f"- **Volume**: {last['volume']:,.2f}")
            lines.append("")
    except Exception as e:
        logger.warning(f"Realtime 1H candle failed for {symbol}: {e}")

    # Funding rates (zero cache)
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        predicted = hl.get_predicted_funding(base_asset, max_age_override=0)
        realized = hl.get_realized_funding(base_asset, max_age_override=0)
        lines.append(f"## Funding Rates (Live)")
        if predicted:
            lines.append(f"- **Predicted (next settlement)**: {predicted['predicted_rate']:.6f}")
            lines.append(f"- **Open interest**: {predicted['open_interest']:,.2f}")
        if realized is not None:
            lines.append(f"- **Last realized**: {realized:.6f}")
        if predicted and realized is not None:
            change = predicted["predicted_rate"] - realized
            direction = "rising" if change > 0.0001 else ("falling" if change < -0.0001 else "stable")
            lines.append(f"- **Direction**: {direction} ({change:+.6f})")
        lines.append("")
    except Exception as e:
        logger.warning(f"Realtime funding failed for {symbol}: {e}")

    # Directional aggression (Hyperliquid premium, zero cache)
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        hl_ctx = hl.get_asset_context(base_asset, max_age_override=0)
        if hl_ctx:
            premium = hl_ctx["premium"]
            premium_bps = premium * 10000
            day_vol = hl_ctx["dayNtlVlm"]
            lines.append(f"## Market Aggression")
            lines.append(f"- **Premium**: {premium_bps:+.1f} bps ({premium:.6f})")
            if premium > 0.0005:
                lines.append("- Perp above index — **buyers aggressing**")
            elif premium < -0.0005:
                lines.append("- Perp below index — **sellers aggressing**")
            else:
                lines.append("- Balanced — perp near index")
            lines.append(f"- **24h Notional Volume**: ${day_vol:,.0f}")
            lines.append("")
    except Exception as e:
        logger.warning(f"Realtime premium failed for {symbol}: {e}")

    return "\n".join(lines)
