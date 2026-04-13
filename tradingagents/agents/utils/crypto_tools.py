"""LangChain @tool wrappers for crypto-specific data sources.

These tools are used by crypto-aware analysts and route through
the vendor routing system in interface.py.
"""

import re
import logging
from langchain_core.tools import tool
from typing import Annotated

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.crypto_news_scraper import (
    get_crypto_news as _scraper_news,
    get_crypto_global_news as _scraper_global,
)

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{4}-\d{2}-\d{2}$")

def _sanitize_date(raw: str) -> str:
    """Fix common LLM date-formatting errors (e.g. '2026-2026-04-13' → '2026-04-13')."""
    if raw and _DATE_RE.match(raw):
        fixed = raw[5:]
        logger.warning("Sanitized malformed date %r → %r", raw, fixed)
        # #region agent log
        import json as _j, time as _t
        try:
            with open("/Users/daniel/Desktop/TradingAgents/.cursor/debug-f18c74.log", "a") as _f:
                _f.write(_j.dumps({"sessionId":"f18c74","hypothesisId":"H1","location":"crypto_tools.py:_sanitize_date","message":"Date sanitized","data":{"raw":raw,"fixed":fixed},"timestamp":int(_t.time()*1000)}) + "\n")
        except Exception:
            pass
        # #endregion
        return fixed
    return raw


@tool
def get_derivatives_data(
    symbol: Annotated[str, "ticker symbol, e.g. BTC-USD"],
    curr_date: Annotated[str, "current trading date, yyyy-mm-dd"],
) -> str:
    """
    Retrieve crypto derivatives data: funding rates, open interest, taker long/short ratio.
    Data sourced from Binance Futures and Deribit public APIs.

    Args:
        symbol: Ticker symbol (e.g. BTC-USD, ETH-USD)
        curr_date: Current trading date in yyyy-mm-dd format

    Returns:
        Formatted report of derivatives market data
    """
    return route_to_vendor("get_derivatives_data", symbol, _sanitize_date(curr_date))


@tool
def get_macro_indicators(
    curr_date: Annotated[str, "current trading date, yyyy-mm-dd"],
) -> str:
    """
    Retrieve macro indicators: FRED M2 money supply, DXY dollar index, 2Y/10Y treasury yields.

    Args:
        curr_date: Current trading date in yyyy-mm-dd format

    Returns:
        Formatted report of macroeconomic indicators
    """
    return route_to_vendor("get_macro_indicators", _sanitize_date(curr_date))


@tool
def get_sentiment_data(
    curr_date: Annotated[str, "current trading date, yyyy-mm-dd"],
) -> str:
    """
    Retrieve crypto sentiment data: Alternative.me Fear & Greed index with historical trend.

    Args:
        curr_date: Current trading date in yyyy-mm-dd format

    Returns:
        Formatted sentiment report with current reading and trend
    """
    return route_to_vendor("get_sentiment_data", _sanitize_date(curr_date))


@tool
def get_crypto_news(
    ticker: Annotated[str, "ticker symbol, e.g. BTC-USD"],
    start_date: Annotated[str, "start date, yyyy-mm-dd"],
    end_date: Annotated[str, "end date, yyyy-mm-dd"],
) -> str:
    """
    Retrieve crypto-specific news from Cointelegraph, CoinDesk, and BeInCrypto RSS feeds.
    Filters articles by ticker keyword and date range.

    Args:
        ticker: Ticker symbol (e.g. BTC-USD, ETH-USD)
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted markdown report of crypto news articles
    """
    return _scraper_news(ticker, _sanitize_date(start_date), _sanitize_date(end_date))


@tool
def get_onchain_data(
    symbol: Annotated[str, "ticker symbol, e.g. BTC-USD"],
    curr_date: Annotated[str, "current trading date, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive on-chain metrics for cryptocurrency analysis.
    
    Returns hash rate, active addresses, exchange flows, MVRV ratio, 
    supply distribution, and other blockchain network health indicators.
    Data sourced from BGeometrics and other on-chain analytics providers.

    Args:
        symbol: Ticker symbol (e.g. BTC-USD, ETH-USD)
        curr_date: Current trading date in yyyy-mm-dd format

    Returns:
        Formatted report of on-chain metrics and network fundamentals
    """
    return route_to_vendor("get_fundamentals", symbol, _sanitize_date(curr_date))


@tool
def get_crypto_global_news(
    curr_date: Annotated[str, "current trading date, yyyy-mm-dd"],
    look_back_days: Annotated[int, "number of days to look back"] = 7,
    limit: Annotated[int, "max articles to return"] = 15,
) -> str:
    """
    Retrieve global crypto market news from Cointelegraph, CoinDesk, and BeInCrypto RSS feeds.
    Returns broad crypto market and macro news without ticker filtering.

    Args:
        curr_date: Current trading date in yyyy-mm-dd format
        look_back_days: Number of days to look back (default 7)
        limit: Maximum number of articles (default 15)

    Returns:
        Formatted markdown report of global crypto news
    """
    return _scraper_global(_sanitize_date(curr_date), look_back_days, limit)
