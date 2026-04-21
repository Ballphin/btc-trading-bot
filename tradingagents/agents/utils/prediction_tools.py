from langchain_core.tools import tool
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.backtesting.context import BACKTEST_MODE


@tool
def get_prediction_markets() -> str:
    """
    Returns the Kalshi Macro Risk Dashboard.
    Provides batch-polled, highly liquid predictive odds on macroeconomic events:
    - Target Federal Funds Rate decisions (FOMC).
    - Inflation (CPI) target brackets.
    - US Recession Probabilities.
    
    Use this tool to gauge "smart money" outlooks for quantitative macro variables.
    NOTE: For crypto-specific predictions (BTC/ETH price action), use get_crypto_prediction_markets (Polymarket) instead.
    """
    if BACKTEST_MODE.get():
        return "[PREDICTION MARKETS: DISABLED IN BACKTEST MODE — USE ONLY HISTORICAL DATA]"
    return route_to_vendor(
        "get_kalshi_macro_context"
    )


@tool
def get_crypto_prediction_markets() -> str:
    """
    Returns the Polymarket crypto prediction-market dashboard.

    Polymarket is USDC-denominated and has materially deeper liquidity
    for crypto-specific questions (per-market liquidity typically
    $50k-$8M versus Kalshi's $5-30k for comparable questions).

    Output is a markdown summary of active, high-liquidity markets
    grouped by tag (crypto, bitcoin, ethereum, solana, memecoin) with
    Wolfers-Zitzewitz γ=0.91 corrected implied probabilities.

    Disabled in backtest mode.
    """
    if BACKTEST_MODE.get():
        return "[CRYPTO PREDICTION MARKETS: DISABLED IN BACKTEST MODE]"
    # Import lazily so the tool module stays cheap to import in backtest.
    from tradingagents.dataflows.polymarket_client import get_polymarket_crypto_context
    return get_polymarket_crypto_context()
