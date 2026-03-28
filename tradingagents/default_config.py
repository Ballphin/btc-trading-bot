import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.2",
    "quick_think_llm": "gpt-5-mini",
    "backend_url": "https://api.openai.com/v1",
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Asset type: "auto" detects from ticker, or force "equity" / "crypto"
    "asset_type": "auto",
    # Per-asset-type vendor configuration for crypto assets
    # "crypto" = Coinbase OHLCV + Binance derivatives + Deribit funding + BGeometrics on-chain
    "crypto_vendors": {
        "core_stock_apis": "crypto",             # Coinbase OHLCV
        "technical_indicators": "crypto",        # stockstats on Coinbase OHLCV
        "fundamental_data": "crypto",            # BGeometrics on-chain metrics
        "derivatives_data": "crypto",            # Binance Futures + Deribit
        "macro_data": "fred",                    # FRED API (M2, DXY, yields)
        "sentiment_data": "crypto",              # Alternative.me Fear & Greed
        "news_data": "yfinance",                 # yfinance news (works for crypto)
    },
}
