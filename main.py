import argparse
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from datetime import date

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run TradingAgents analysis on a single ticker.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. BTC-USD, NVDA")
    parser.add_argument("--date", default=date.today().strftime("%Y-%m-%d"),
                        help="Analysis date in YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    # Create a custom config
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "deepseek"  # Use DeepSeek provider
    config["deep_think_llm"] = "deepseek-reasoner"  # DeepSeek reasoning model
    config["quick_think_llm"] = "deepseek-chat"  # DeepSeek chat model
    config["max_debate_rounds"] = 1  # Debate rounds

    # Configure data vendors (default uses yfinance, no extra API keys needed)
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",           # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",      # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",          # Options: alpha_vantage, yfinance
        "news_data": "yfinance",                 # Options: alpha_vantage, yfinance
    }

    # ── Run analysis ──────────────────────────────────────────────────
    ta = TradingAgentsGraph(debug=True, config=config)

    ticker = args.ticker.upper()
    trade_date = args.date

    print(f"\n{'='*60}")
    print(f"  Analyzing {ticker} on {trade_date}")
    print(f"{'='*60}\n")
    _, decision = ta.propagate(ticker, trade_date)
    print(f"\n>>> {ticker} Decision: {decision}\n")

if __name__ == "__main__":
    main()
