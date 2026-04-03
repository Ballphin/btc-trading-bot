from langchain_core.tools import tool
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_prediction_markets() -> str:
    """
    Returns the Kalshi Macro Risk Dashboard.
    Provides batch-polled, highly liquid predictive odds on macroeconomic events:
    - Target Federal Funds Rate decisions (FOMC).
    - Inflation (CPI) target brackets.
    - US Recession Probabilities.
    
    Use this tool to gauge "smart money" outlooks for quantitative macro variables.
    NOTE: For crypto-specific predictions (BTC/ETH price action), use Derivatives options Delta instead of Kalshi.
    """
    return route_to_vendor(
        "get_kalshi_macro_context"
    )
