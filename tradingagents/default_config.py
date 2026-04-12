import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings - OpenRouter primary, DeepSeek fallback
    "llm_provider": "openrouter",
    "deep_think_llm": "qwen/qwen3.6-plus",
    "quick_think_llm": "qwen/qwen3.6-plus",
    "backend_url": "https://openrouter.ai/api/v1",
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    "llm_temperature": 0.4,             # 0.0=deterministic, 0.4=modest diversity
    
    # Ensemble analysis settings
    "enable_ensemble": True,             # Auto-enabled for OpenRouter
    "ensemble_runs": 3,                  # Number of parallel analyses
    "ensemble_max_retries": 2,           # Re-run attempts on divergence
    "ensemble_divergence_range_threshold": 0.20,  # HIGH FIX: Range threshold (not std)
    "ensemble_timeout_per_run": 600,     # 10 minutes per run (free-tier models are slower)
    "ensemble_max_total_time": 30,       # 30s before stale warning (BLOCKER FIX)
    "ensemble_temperature_variation": 0.05,  # +/- temp spread across runs
    "ensemble_enabled_providers": ["openrouter"],  # Only these get ensemble
    "ensemble_disabled_providers": ["deepseek"],  # Never ensemble these
    "openrouter_fallback_model": "google/gemma-4-31b-it:free",  # BLOCKER FIX: Retry diversity (free tier)
    "openrouter_max_tokens": 4096,       # Avoid oversized token budgets on limited OpenRouter credits
    
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
        "prediction_market_data": "kalshi",  # Prediction Markets
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Asset type: "auto" detects from ticker, or force "equity" / "crypto"
    "asset_type": "auto",
    # Backtest mode: disables realtime/zero-cache tools to prevent look-ahead bias
    "backtest_mode": False,
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
        "prediction_market_data": "kalshi",      # Overwrites generic macro odds
    },
}
