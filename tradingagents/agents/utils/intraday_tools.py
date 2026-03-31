"""LangChain @tool wrappers for intraday and realtime data sources.

These tools provide multi-timeframe candle data and zero-cache realtime
snapshots for crypto analysis. Routed through interface.py vendor system.
"""

from langchain_core.tools import tool
from typing import Annotated

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_intraday_data(
    symbol: Annotated[str, "ticker symbol, e.g. BTC-USD"],
    curr_date: Annotated[str, "current trading date, yyyy-mm-dd"],
) -> str:
    """
    Retrieve intraday summary from the last 24 hours of 1H candle data.

    Returns:
    - 24h price range (high/low)
    - Approximate VWAP (hourly typical price method)
    - Max intraday drawdown from peak (worst-case, high-to-low)
    - Binance taker buy/sell ratio (cross-venue volume signal)
    - Funding rate momentum (predicted vs last realized, direction)

    For equities, returns a message that intraday data is not available in v1.

    Args:
        symbol: Ticker symbol (e.g. BTC-USD, ETH-USD)
        curr_date: Current trading date in yyyy-mm-dd format

    Returns:
        Formatted markdown report of intraday market context
    """
    return route_to_vendor("get_intraday_data", symbol, curr_date)


@tool
def get_realtime_snapshot(
    symbol: Annotated[str, "ticker symbol, e.g. BTC-USD"],
) -> str:
    """
    Retrieve a zero-cache realtime snapshot of current market conditions.

    Always fetches live data (never cached). Returns:
    - Current spot price
    - Latest 1H candle (open/high/low/close/volume)
    - Predicted and last realized funding rates with direction
    - Binance taker ratio (latest hour)

    Use this when re-running analysis on the same day to get fresh data
    that reflects current market conditions.

    NOT available during backtests (would inject future data).

    Args:
        symbol: Ticker symbol (e.g. BTC-USD, ETH-USD)

    Returns:
        Formatted markdown report of current market state
    """
    return route_to_vendor("get_realtime_snapshot", symbol)
