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

def get_crypto_price_data(symbol: str, start_date: str, end_date: str) -> str:
    """
    OHLCV from Coinbase Exchange — drop-in replacement for get_YFin_data_online.

    Returns formatted string of daily OHLCV data.
    """
    cb = _get(CoinbaseClient, "coinbase")
    product = _to_coinbase_product(symbol)

    df = cb.get_ohlcv(product, 86400, start_date, end_date)
    if df.empty:
        return f"No OHLCV data available for {symbol} ({start_date} to {end_date})"

    # Format to match yfinance output style
    df_display = df.copy()
    df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d")
    df_display = df_display.rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    return df_display.to_string(index=False)


def get_crypto_indicators(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """
    Technical indicators computed from Coinbase OHLCV data via stockstats.

    Reuses the same stockstats library the equity path uses.
    """
    cb = _get(CoinbaseClient, "coinbase")
    product = _to_coinbase_product(symbol)

    # Fetch enough history for indicator calculation
    start = (
        datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days + 60)
    ).strftime("%Y-%m-%d")

    df = cb.get_ohlcv(product, 86400, start, curr_date)
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

    curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")
    matching = ss[ss["Date"].str.startswith(curr_date_str)]

    if not matching.empty:
        value = matching[indicator].values[0]
        return f"{indicator} for {symbol} on {curr_date}: {value}"
    else:
        return f"{indicator} for {symbol}: N/A (not a trading day or insufficient data)"


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
        lines.append(f"- **Latest**: {latest_rate:.6f}")
        lines.append(f"- **30d average**: {avg_rate:.6f}")
        lines.append(f"- **30d range**: [{min_rate:.6f}, {max_rate:.6f}]")
        if latest_rate > 0.0003:
            lines.append("- ⚠️ Elevated positive funding — longs paying shorts, potential long squeeze risk")
        elif latest_rate < -0.0003:
            lines.append("- ⚠️ Elevated negative funding — shorts paying longs, potential short squeeze risk")
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
