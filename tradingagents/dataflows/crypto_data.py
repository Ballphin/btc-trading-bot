"""Unified crypto data layer — wraps Coinbase, Binance, Deribit, BGeometrics,
FRED, and Fear & Greed clients for vendor routing.

Each function returns a formatted string matching the pattern used by y_finance.py,
so the vendor routing system in interface.py can swap them in transparently.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.coinbase_client import CoinbaseClient
from tradingagents.dataflows.binance_client import BinanceClient
from tradingagents.dataflows.deribit_client import DeribitClient
from tradingagents.dataflows.bgeometrics_client import BGeometricsClient
from tradingagents.dataflows.fred_client import FREDClient
from tradingagents.dataflows.fear_greed_client import FearGreedClient
from tradingagents.dataflows.hyperliquid_client import HyperliquidClient

logger = logging.getLogger(__name__)

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


def _to_binance_symbol(ticker: str) -> str:
    """Ensure ticker is in Binance format: BTCUSDT"""
    base = _normalize_symbol(ticker)
    return f"{base}USDT"


def _to_deribit_instrument(ticker: str) -> str:
    """Ensure ticker is in Deribit format: BTC-PERPETUAL"""
    base = _normalize_symbol(ticker)
    return f"{base}-PERPETUAL"


# ---------------------------------------------------------------------------
# Vendor routing implementations
# ---------------------------------------------------------------------------

# Timeframe → granularity mapping
_TIMEFRAME_GRANULARITY = {
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

    # Try Hyperliquid first (better altcoin coverage, no rate limits)
    df = pd.DataFrame()
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        df = hl.get_ohlcv(base_asset, timeframe, start_date, end_date)
    except Exception as e:
        logger.warning(f"Hyperliquid OHLCV failed for {symbol}, falling back to Coinbase: {e}")

    # Fallback to Coinbase
    if df.empty:
        cb = _get(CoinbaseClient, "coinbase")
        product = _to_coinbase_product(symbol)
        df = cb.get_ohlcv(product, granularity, start_date, end_date)

    if df.empty:
        return f"No OHLCV data available for {symbol} ({start_date} to {end_date}, {timeframe})"

    # Format to match yfinance output style
    df_display = df.copy()
    if timeframe == "1d":
        df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d")
    else:
        df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df_display = df_display.rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    header = f"# {symbol} OHLCV ({timeframe} bars) — {start_date} to {end_date}\n"
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
        datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days + 60)
    ).strftime("%Y-%m-%d")

    # Try Hyperliquid first, fall back to Coinbase
    df = pd.DataFrame()
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        df = hl.get_ohlcv(base_asset, timeframe, start, curr_date)
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
    Binance funding rates + open interest + taker ratio,
    cross-referenced with Deribit perpetual funding rates.

    Returns a structured markdown report.
    """
    bn = _get(BinanceClient, "binance")
    dr = _get(DeribitClient, "deribit")

    bn_symbol = _to_binance_symbol(symbol)
    dr_instrument = _to_deribit_instrument(symbol)

    start = (
        datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=30)
    ).strftime("%Y-%m-%d")

    # Fetch all derivatives data
    bn_funding = bn.get_funding_rates(bn_symbol, start, curr_date)
    bn_oi = bn.get_open_interest_hist(bn_symbol, "1d", start, curr_date)
    bn_taker = bn.get_taker_ratio(bn_symbol, "1d", start, curr_date)
    dr_funding = dr.get_funding_rate_history(dr_instrument, start, curr_date)

    lines = [
        f"# Derivatives Analysis: {symbol}",
        f"**Period**: {start} to {curr_date}",
        "",
    ]

    # Binance Funding Rates
    lines.append("## Binance Funding Rates (8h)")
    if not bn_funding.empty and "fundingRate" in bn_funding.columns:
        latest_rate = float(bn_funding.iloc[-1]["fundingRate"])
        avg_rate = bn_funding["fundingRate"].astype(float).mean()
        max_rate = bn_funding["fundingRate"].astype(float).max()
        min_rate = bn_funding["fundingRate"].astype(float).min()
        lines.append(f"- **Latest realized**: {latest_rate:.6f}")
        lines.append(f"- **30d average**: {avg_rate:.6f}")
        lines.append(f"- **30d range**: [{min_rate:.6f}, {max_rate:.6f}]")
        if latest_rate > 0.0003:
            lines.append("- ⚠️ Elevated positive funding — longs paying shorts, potential long squeeze risk")
        elif latest_rate < -0.0003:
            lines.append("- ⚠️ Elevated negative funding — shorts paying longs, potential short squeeze risk")

        # Funding Rate Momentum (HIGH 7): predicted vs realized + direction
        if len(bn_funding) >= 2:
            prev_rate = float(bn_funding.iloc[-2]["fundingRate"])
            funding_8h_change = latest_rate - prev_rate
            if funding_8h_change > 0.0001:
                direction = "rising"
            elif funding_8h_change < -0.0001:
                direction = "falling"
            else:
                direction = "stable"
            lines.append(f"- **8h change**: {funding_8h_change:+.6f} ({direction})")
            lines.append(f"- **Previous settlement**: {prev_rate:.6f}")

        # Add Hyperliquid predicted funding if available
        try:
            base_asset = _normalize_symbol(symbol)
            hl = _get(HyperliquidClient, "hyperliquid")
            predicted = hl.get_predicted_funding(base_asset)
            if predicted:
                pred_rate = predicted["predicted_rate"]
                lines.append(f"- **Predicted (next settlement, Hyperliquid)**: {pred_rate:.6f}")
                pred_vs_realized = pred_rate - latest_rate
                pred_dir = "rising" if pred_vs_realized > 0.0001 else ("falling" if pred_vs_realized < -0.0001 else "stable")
                lines.append(f"- **Predicted vs realized**: {pred_vs_realized:+.6f} ({pred_dir})")
        except Exception as e:
            logger.debug(f"Hyperliquid predicted funding unavailable: {e}")
    else:
        lines.append("- Data unavailable")
    lines.append("")

    # Open Interest
    lines.append("## Open Interest (Binance)")
    if not bn_oi.empty and "sumOpenInterest" in bn_oi.columns:
        latest_oi = bn_oi.iloc[-1]
        lines.append(f"- **Latest OI**: {float(latest_oi['sumOpenInterest']):,.2f} BTC")
        lines.append(f"- **Latest OI Value**: ${float(latest_oi['sumOpenInterestValue']):,.0f}")
        if len(bn_oi) >= 7:
            oi_7d_ago = bn_oi.iloc[-7]["sumOpenInterest"]
            oi_change = ((float(latest_oi["sumOpenInterest"]) - float(oi_7d_ago)) / float(oi_7d_ago) * 100)
            lines.append(f"- **7d change**: {oi_change:+.1f}%")
    else:
        lines.append("- Data unavailable")
    lines.append("")

    # Taker Buy/Sell Ratio
    lines.append("## Taker Buy/Sell Ratio (Binance)")
    if not bn_taker.empty and "buySellRatio" in bn_taker.columns:
        latest_ratio = float(bn_taker.iloc[-1]["buySellRatio"])
        avg_ratio = bn_taker["buySellRatio"].astype(float).mean()
        lines.append(f"- **Latest**: {latest_ratio:.4f}")
        lines.append(f"- **30d average**: {avg_ratio:.4f}")
        if latest_ratio > 1.1:
            lines.append("- Buyers dominating — bullish pressure")
        elif latest_ratio < 0.9:
            lines.append("- Sellers dominating — bearish pressure")
        else:
            lines.append("- Balanced buy/sell pressure")
    else:
        lines.append("- Data unavailable")
    lines.append("")

    # Deribit Perp Funding
    lines.append("## Deribit Perpetual Funding (hourly)")
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

    Returns formatted markdown report.
    """
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

    # Date range: 24h ending at curr_date EOD
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d") + timedelta(days=1)
    start_dt = end_dt - timedelta(hours=24)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # Fetch 1H candles — Hyperliquid primary, Coinbase fallback
    df = pd.DataFrame()
    try:
        hl = _get(HyperliquidClient, "hyperliquid")
        df = hl.get_ohlcv(base_asset, "1h", start_str, end_str)
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

    lines = [
        f"# Intraday Summary: {symbol} (24h ending {curr_date})",
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

    # Binance taker buy/sell ratio (latest)
    try:
        bn = _get(BinanceClient, "binance")
        bn_symbol = _to_binance_symbol(symbol)
        taker_df = bn.get_taker_ratio(bn_symbol, "1h", start_str, end_str)
        if not taker_df.empty and "buySellRatio" in taker_df.columns:
            latest_ratio = float(taker_df.iloc[-1]["buySellRatio"])
            lines.append(f"## Cross-Venue Volume Signal")
            lines.append(f"- **Binance taker buy/sell ratio (latest 1h)**: {latest_ratio:.4f}")
            if latest_ratio > 1.1:
                lines.append("- Buyers aggressing — bullish pressure")
            elif latest_ratio < 0.9:
                lines.append("- Sellers aggressing — bearish pressure")
            else:
                lines.append("- Balanced aggression")
            lines.append(f"- *(Coinbase volume is ~5-10% of total; Binance taker ratio provides cross-venue context)*")
            lines.append("")
    except Exception as e:
        logger.warning(f"Binance taker ratio failed for {symbol}: {e}")

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
    now = datetime.utcnow()
    lines = [
        f"# Realtime Snapshot: {symbol}",
        f"**Timestamp (UTC)**: {now.strftime('%Y-%m-%d %H:%M:%S')}",
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

    # 24h change from Binance taker ratio
    try:
        bn = _get(BinanceClient, "binance")
        bn_symbol = _to_binance_symbol(symbol)
        start_24h = (now - timedelta(hours=24)).strftime("%Y-%m-%d")
        end_24h = now.strftime("%Y-%m-%d")
        taker_df = bn.get_taker_ratio(bn_symbol, "1h", start_24h, end_24h)
        if not taker_df.empty and "buySellRatio" in taker_df.columns:
            latest = float(taker_df.iloc[-1]["buySellRatio"])
            lines.append(f"## Market Aggression")
            lines.append(f"- **Binance taker ratio (latest 1h)**: {latest:.4f}")
            lines.append("")
    except Exception as e:
        logger.warning(f"Realtime taker ratio failed for {symbol}: {e}")

    return "\n".join(lines)
