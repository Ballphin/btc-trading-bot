from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from datetime import date

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Always use the most recent date
today = date.today().strftime("%Y-%m-%d")

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "deepseek"
config["deep_think_llm"] = "deepseek-v4-flash"
config["quick_think_llm"] = "deepseek-v4-flash"
config["max_debate_rounds"] = 1

config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

# ── Run BTC-USD and NVDA analysis ─────────────────────────────────
ta = TradingAgentsGraph(debug=True, config=config)

for ticker in ["BTC-USD", "NVDA"]:
    print(f"\n{'='*60}")
    print(f"  Analyzing {ticker} on {today}")
    print(f"{'='*60}\n")
    _, decision = ta.propagate(ticker, today)
    print(f"\n>>> {ticker} Decision: {decision}\n")
