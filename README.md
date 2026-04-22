<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-TauricResearch-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/Join_GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>

<div align="center">
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

# TradingAgents: Multi-Agents LLM Financial Trading Framework

A production-grade, multi-agent LLM framework for quantitative trading research. TradingAgents deploys specialized AI agents—market analysts, sentiment experts, news analysts, fundamental researchers, and risk managers—who engage in structured debates to reach consensus on trading decisions. Built with LangGraph for orchestration, FastAPI for the backend, and React for the dashboard.

> 🎉 **TradingAgents** is now fully open-source! Join our community of quantitative researchers and algorithmic traders.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [What is TradingAgents?](#what-is-tradingagents)
- [System Architecture](#system-architecture)
- [Codebase Structure](#codebase-structure)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Key Concepts](#key-concepts)
- [Testing](#testing)
- [Development Guide](#development-guide)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## News
- [2026-03] **TradingAgents v0.2.2** — GPT-5.4/Gemini 3.1/Claude 4.6 support, five-tier rating scale, OpenAI Responses API, Anthropic effort control
- [2026-02] **TradingAgents v0.2.0** — Multi-provider LLM support (GPT-5.x, Gemini 3.x, Claude 4.x, Grok 4.x)
- [2026-01] **Trading-R1** [Technical Report](https://arxiv.org/abs/2509.11420) released

<div align="center">
<a href="https://www.star-history.com/#TauricResearch/TradingAgents&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" />
   <img alt="TradingAgents Star History" src="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" style="width: 80%; height: auto;" />
 </picture>
</a>
</div>

## Quick Start

Get TradingAgents running locally in under 5 minutes:

```bash
# 1. Clone and setup
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or other LLM provider keys

# 3. Start the backend
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 4. Start the frontend (in a new terminal)
cd frontend && npm install && npm run dev

# 5. Open http://localhost:5173 and analyze any ticker!
```

## What is TradingAgents?

TradingAgents is a **multi-agent LLM framework** that simulates the organizational structure of a professional trading firm. Instead of relying on a single LLM to make trading decisions (which suffers from reasoning limitations and bias), TradingAgents deploys specialized agents that collaborate through structured debate:

**Why Multi-Agent?**
- **Specialization**: Each agent focuses on one domain (technicals, sentiment, fundamentals, news)
- **Debate & Consensus**: Bull/Bear researchers challenge each other's theses, surfacing risks a single model might miss
- **Risk Management**: Independent risk team evaluates position sizing, stop-losses, and portfolio exposure
- **Audit Trail**: Every decision includes full debate history for post-hoc analysis

### ⚡ How It Works

1. **You enter a ticker** — any stock (`NVDA`, `AAPL`) or crypto (`BTC-USD`, `ETH-USD`)
2. **4 AI analysts research it in parallel** — Market (technicals), Sentiment (social), News, and Fundamentals
3. **Bull & Bear researchers debate** — arguing for and against the trade, surfacing hidden risks
4. **A Trader agent creates a structured signal** — entry, exit, and position sizing
5. **3 Risk Managers debate sizing** — aggressive, neutral, and conservative perspectives
6. **Portfolio Manager makes the final call** → you get: **BUY/SELL/HOLD** with confidence %, stop-loss, take-profit, and position size

### 📊 Understanding the Output

Every analysis produces a structured signal:

| Field | What It Means |
|-------|--------------|
| **Signal** | `BUY` (open long), `SELL` (close long), `SHORT` (open short), `COVER` (close short), `HOLD` (no action) |
| **Confidence** | 0–100% — how strongly the agents agree. 80%+ = high conviction, below 60% = low conviction |
| **Stop-Loss** | Price where you should exit to limit losses |
| **Take-Profit** | Price target for taking gains |
| **Position Size** | Suggested % of capital to allocate |
| **R:R Ratio** | Reward-to-risk ratio (higher = better) |

**Key Differentiators:**
| Feature | Description |
|---------|-------------|
| **Multi-Provider LLM** | OpenAI, Anthropic, Google, xAI, OpenRouter, Ollama — swap models without changing code |
| **Asset-Agnostic** | Equities via yfinance/Alpha Vantage; Crypto via Hyperliquid/Coinbase/Deribit |
| **Production Backtesting** | Replay, simulation, and hybrid modes with funding rates, slippage, and walk-forward validation |
| **Shadow Trading** | Paper-trade mode for forward-testing with Brier score calibration |
| **4H Scheduler** | Automated crypto analysis every 4 hours, synced to UTC candle closes |
| **React Dashboard** | Real-time SSE streaming, interactive charts, backtest visualization |

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> **Disclaimer**: TradingAgents is for research purposes. Performance varies by model, temperature, data quality, and market conditions. [Not financial advice.](https://tauric.ai/disclaimer/)

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                                   │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌─────────────┐   │
│  │  Home   │  │ Analyze  │  │ History │  │ Backtest │  │ Scorecard   │   │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘  └─────┬───────┘   │
└───────┼────────────┼─────────────┼────────────┼──────────────┼───────────┘
        │            │             │            │              │
        └────────────┴─────────────┴────────────┴──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (server.py)                           │
│  • REST API endpoints (/api/analyze, /api/backtest, /api/history)        │
│  • SSE streaming for real-time agent progress                            │
│  • Job management (analysis jobs, backtest jobs)                         │
│  • 4H scheduler (automated BTC-USD analysis)                             │
│  • Shadow trading endpoints (paper-trade scoring)                        │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRADINGAGENTSGRAPH (LangGraph)                        │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    PARALLEL ANALYST PHASE                          │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │   │
│  │  │   Market    │ │  Sentiment  │ │    News     │ │ Fundamentals│  │   │
│  │  │  Analyst    │ │  Analyst    │ │  Analyst    │ │   Analyst   │  │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘  │   │
│  └─────────┼──────────────┼──────────────┼──────────────┼──────────┘   │
│            │              │              │              │              │
│            └──────────────┴──────────────┴──────────────┘              │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    RESEARCHER DEBATE PHASE                         │   │
│  │         ┌─────────────┐         ┌─────────────┐                 │   │
│  │         │ Bull Researcher│  ←→   │ Bear Researcher│                │   │
│  │         └──────┬──────┘         └──────┬──────┘                 │   │
│  │                └──────────┬────────────┘                        │   │
│  │                           ▼                                      │   │
│  │                    Judge Decision                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    TRADER AGENT                                  │   │
│  │           Composes structured signal (signal, confidence,      │   │
│  │           stop_loss, take_profit, position_size)                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                 RISK MANAGEMENT DEBATE                           │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │   │
│  │  │  Aggressive  │ │  Conservative │ │   Neutral    │            │   │
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │   │
│  │         └────────────────┼────────────────┘                      │   │
│  │                          ▼                                       │   │
│  │                   Risk Assessment                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              PORTFOLIO MANAGER (Final Gate)                      │   │
│  │              • Kelly position sizing                             │   │
│  │              • DSR (Deflated Sharpe) validation                  │   │
│  │              • Final approval/rejection                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│                    ┌──────────────────┐                                  │
│                    │  FINAL DECISION  │                                  │
│                    │  (BUY/SELL/HOLD/ │                                  │
│                    │   SHORT/COVER)   │                                  │
│                    └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        │     ┌─────────────────────────────────────────────────────────────┐
        │     │                     DATA LAYER                               │
        │     │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐  │
        └────►│  │  Hyperliquid│ │   Coinbase  │ │   yfinance  │ │ Kalshi │  │
              │  │  (Crypto)   │ │   (Crypto)  │ │  (Equities) │ │(Events)│  │
              │  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘  │
              │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
              │  │   Deribit   │ │  BGeometrics│ │    FRED     │              │
              │  │  (Funding)  │ │ (On-chain)  │ │  (Macro)    │              │
              │  └─────────────┘ └─────────────┘ └─────────────┘              │
              └─────────────────────────────────────────────────────────────┘
        │
        │     ┌─────────────────────────────────────────────────────────────┐
        └────►│                     LLM PROVIDERS                            │
              │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────┐ │
              │  │  OpenAI  │ │ Anthropic│ │  Google  │ │  xAI   │ │Ollama│ │
              │  │ (GPT-5.x)│ │ (Claude) │ │ (Gemini) │ │ (Grok) │ │Local │ │
              │  └──────────┘ └──────────┘ └──────────┘ └────────┘ └──────┘ │
              └─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Location | Key Files | Purpose |
|-----------|----------|-----------|---------|
| **Frontend Dashboard** | `frontend/` | `src/pages/`, `src/components/`, `src/lib/api.ts` | React + Vite + Tailwind UI for analysis, history, backtesting |
| **FastAPI Server** | `server.py` | ~2,300 lines | REST API + SSE streaming, job management, scheduler |
| **TradingAgentsGraph** | `tradingagents/graph/` | `trading_graph.py`, `signal_processing.py`, `conditional_logic.py` | LangGraph orchestration, routing, state management |
| **Agents** | `tradingagents/agents/` | `analysts/`, `researchers/`, `trader/`, `risk_mgmt/`, `managers/` | Specialized LLM agents with prompts |
| **LLM Clients** | `tradingagents/llm_clients/` | `base_client.py`, `openai_client.py`, `anthropic_client.py`, ... | Unified interface for all providers |
| **Dataflows** | `tradingagents/dataflows/` | `hyperliquid_client.py`, `y_finance.py`, `crypto_data.py`, `interface.py` | Vendor abstraction + routing |
| **Backtesting** | `tradingagents/backtesting/` | `engine.py`, `portfolio.py`, `metrics.py`, `walk_forward.py` | Simulation, metrics, validation |
| **Configuration** | `tradingagents/` | `default_config.py` | All tunable parameters |

## Codebase Structure

```
TradingAgents/
├── README.md                          # This comprehensive guide
├── server.py                          # FastAPI backend (~2,300 lines)
├── main.py                            # CLI entry point
├── backtest.py                        # Standalone backtest script
├── pyproject.toml                     # Python dependencies
├── .env.example                       # Environment template
│
├── frontend/                          # React Dashboard
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.tsx              # Ticker search, quick access
│   │   │   ├── Analyze.tsx           # Live analysis with SSE
│   │   │   ├── History.tsx           # Browse past analyses
│   │   │   ├── AnalysisDetail.tsx    # Full decision breakdown
│   │   │   ├── Backtest.tsx          # Configure & run backtests
│   │   │   ├── BacktestResults.tsx   # Results, charts, lessons
│   │   │   ├── Scorecard.tsx         # Shadow trading scorecard
│   │   │   └── RecentBacktests.tsx   # Historical backtest list
│   │   ├── components/
│   │   │   ├── SignalBadge.tsx       # BUY/SELL/HOLD indicators
│   │   │   ├── PriceChart.tsx        # Recharts OHLCV visualization
│   │   │   ├── DebatePanel.tsx       # Bull/Bear debate display
│   │   │   ├── FinalDecisionCard.tsx # Structured signal output
│   │   │   └── AgentReportCard.tsx   # Collapsible agent reports
│   │   ├── lib/
│   │   │   └── api.ts                # API client + TypeScript types
│   │   ├── hooks/
│   │   │   └── useDocumentTitle.ts   # Page title management
│   │   └── App.tsx                   # Router + lazy loading
│   ├── package.json                   # Node dependencies
│   └── vite.config.ts                 # Vite + Vitest configuration
│
├── tradingagents/                     # Core Framework
│   ├── __init__.py                    # Package exports
│   ├── default_config.py              # ALL configuration options
│   │
│   ├── graph/                         # LangGraph Orchestration
│   │   ├── trading_graph.py           # Main graph construction
│   │   ├── signal_processing.py       # Parse LLM outputs → structured signals
│   │   ├── conditional_logic.py       # Routing: market→sentiment→news→fundamentals
│   │   ├── confidence.py              # Kelly sizing, DSR gate, position sizing
│   │   ├── propagation.py             # Graph execution entry point
│   │   └── reflection.py              # Agent self-reflection
│   │
│   ├── agents/                        # Multi-Agent System
│   │   ├── analysts/
│   │   │   ├── market_analyst.py      # Price, technicals, regime
│   │   │   ├── sentiment_analyst.py   # Social sentiment, fear-greed
│   │   │   ├── news_analyst.py        # News analysis + RSS
│   │   │   └── fundamentals_analyst.py # Financial metrics
│   │   ├── researchers/
│   │   │   ├── bull_researcher.py     # Bullish thesis defender
│   │   │   └── bear_researcher.py     # Bearish thesis challenger
│   │   ├── trader/
│   │   │   └── trading_agent.py       # Decision composer
│   │   ├── risk_mgmt/
│   │   │   ├── aggressive_risk.py     # Risk-tolerant evaluator
│   │   │   ├── conservative_risk.py   # Risk-averse evaluator
│   │   │   └── neutral_risk.py        # Balanced evaluator
│   │   ├── managers/
│   │   │   ├── portfolio_manager.py   # Final approval gate
│   │   │   └── research_manager.py    # Research coordination
│   │   ├── prompts/                   # System prompts for all agents
│   │   └── utils/                     # LangChain tool wrappers
│   │
│   ├── llm_clients/                   # Unified LLM Interface
│   │   ├── base_client.py             # Abstract base + normalize_content
│   │   ├── openai_client.py           # GPT-5.x, reasoning_effort
│   │   ├── anthropic_client.py        # Claude 4.x, effort control
│   │   ├── google_client.py           # Gemini 3.x, thinking_level
│   │   ├── xai_client.py              # Grok support
│   │   ├── openrouter_client.py       # Multi-provider routing
│   │   └── ollama_client.py           # Local model support
│   │
│   ├── dataflows/                     # Data Vendor Abstraction
│   │   ├── interface.py               # VENDOR_METHODS registry
│   │   ├── base_client.py             # BaseDataClient (OHLCV, funding, etc.)
│   │   ├── hyperliquid_client.py      # Primary crypto source (spot + perps)
│   │   ├── coinbase_client.py         # Secondary crypto OHLCV
│   │   ├── deribit_client.py          # Options + funding rates
│   │   ├── y_finance.py               # Equity data (yfinance)
│   │   ├── crypto_data.py               # Aggregated crypto data utilities
│   │   ├── crypto_news_scraper.py     # Cointelegraph, CoinDesk RSS
│   │   ├── alpha_vantage_*.py         # Alpha Vantage integrations
│   │   ├── kalshi_client.py           # Prediction markets
│   │   ├── fred_client.py             # Macroeconomic data (M2, DXY, yields)
│   │   ├── bgeometrics_client.py    # On-chain Bitcoin metrics
│   │   └── fear_greed_client.py       # Alternative.me Fear & Greed
│   │
│   └── backtesting/                   # Simulation & Validation
│       ├── engine.py                  # Backtest orchestration
│       ├── portfolio.py               # Position tracking, PnL, funding, slippage
│       ├── metrics.py                 # Sharpe, Sortino, Calmar, DSR, omega
│       ├── walk_forward.py            # Walk-forward validation (Bailey-López de Prado)
│       ├── scorecard.py               # Shadow/paper trading evaluation
│       ├── knowledge_store.py         # Historical trade memory for Kelly sizing
│       ├── regime.py                  # Bull/bear/ranging classification
│       ├── feedback.py                # Post-trade analysis & lessons
│       └── report.py                  # HTML/JSON report generation
│
├── tests/                             # Comprehensive Test Suite (336 tests)
│   ├── conftest.py                    # Shared fixtures (mock data, temp dirs)
│   ├── test_backtesting/              # 94 tests: portfolio, metrics, engine
│   │   ├── test_engine.py
│   │   ├── test_portfolio_extended.py
│   │   ├── test_metrics_extended.py
│   │   ├── test_walk_forward.py
│   │   └── test_confidence.py
│   ├── test_dataflows/                # Data vendor tests
│   │   ├── test_hyperliquid_client.py
│   │   └── test_coinbase_client.py
│   ├── test_graph/                    # Graph layer tests
│   │   ├── test_signal_processing.py
│   │   ├── test_conditional_logic.py
│   │   └── test_propagation.py
│   ├── test_llm_clients/              # LLM client tests
│   │   └── test_base_llm.py
│   └── integration/                   # API integration tests
│       ├── test_server_api.py
│       └── test_data_routing.py
│
├── cli/                               # Interactive CLI
│   └── main.py                        # Rich-based TUI
│
├── results/                           # Analysis output storage
│   └── {ticker}/{date}/{timestamp}.json
│
└── eval_results/                      # Backtest output storage
    └── {ticker}/{job_id}/result.json
```

## Features

### Core Analysis Pipeline

**Live Multi-Agent Analysis**
- **Endpoint**: `POST /api/analyze` → `GET /api/stream/{job_id}` (SSE)
- **Process**: 4 parallel analysts → Researcher debate → Trader decision → Risk management → Final signal
- **Output**: Structured JSON with `signal`, `confidence`, `stop_loss_price`, `take_profit_price`, `position_size_pct`, `reasoning`

**History & Persistence**
- All analyses saved to `results/{ticker}/{date}/{timestamp}.json`
- Browse via History page (`/history/:ticker`)
- Full decision breakdown including agent reports and debate transcripts

### Frontend Dashboard

| Page | Route | Description |
|------|-------|-------------|
| **Home** | `/` | Ticker search, quick-access tickers, recent analyses |
| **Analyze** | `/analyze/:ticker` | Live analysis with real-time SSE progress stream |
| **History** | `/history`, `/history/:ticker` | Browse past analyses by ticker and date |
| **Analysis Detail** | `/history/:ticker/:date` | Full decision breakdown with debate visualization |
| **Backtest** | `/backtest` | Configure and run backtests (replay/simulation/hybrid) |
| **Backtest Results** | `/backtest/results/:id` | Equity curve, trade history, metrics, lessons learned |
| **Scorecard** | `/scorecard` | Shadow trading forward-test scoring and calibration |
| **Pulse** | `/pulse` | Automated 4H analysis signals and ensemble agreement |
| **Auto-Tune** | `/autotune` | AI model/parameter optimization |

**What each page does (for new users):**
- **Home** → Type a ticker and click Analyze. That's your starting point.
- **Analyze** → Watch the AI work in real-time: 4 analysts research → bull/bear debate → risk review → final signal.
- **History** → Browse all your past analyses. Click any entry to see the full decision breakdown.
- **Backtest** → Test how past AI decisions would have performed as real trades.
- **Scorecard** → "Practice mode" — track AI accuracy over time without risking real money.
- **Pulse** → Automated signals every 4 hours. Enable the scheduler and check back for results.
- **Auto-Tune** → Find the best AI model/settings for your target ticker.

### Backtesting System

**Three Backtest Modes:**
1. **Replay**: Use saved historical analyses (fastest, deterministic)
2. **Simulation**: Regenerate decisions with current LLM (tests prompt/model changes)
3. **Hybrid**: Replay available dates, simulate missing dates (backfill strategy)

**Crypto-Specific Features:**
- Leverage support (isolated margin simulation)
- Funding rate accumulation (Hyperliquid/Deribit)
- Maker/taker fee modeling
- Position sizing: Fixed, Kelly Criterion, or ATR-based
- Liquidation simulation

**Metrics & Validation:**
- Sharpe, Sortino, Calmar, Omega ratios
- Deflated Sharpe Ratio (DSR) for multiple testing correction
- Walk-forward validation (train/test splits to prevent overfitting)
- Regime tagging (bull/bear/ranging)
- Kelly-optimal position sizing with contamination guard

### Shadow (Paper) Trading

Forward-test decisions without capital at risk:
- **Record**: `POST /api/shadow/record` stores decisions at decision time
- **Score**: `GET /api/shadow/score/{ticker}` calculates Brier score, win rate by signal/regime
- **Calibrate**: `POST /api/shadow/calibrate` assesses confidence calibration
- **Walk-Forward**: `POST /api/shadow/walk-forward` runs temporal validation

### 4H Auto-Scheduler

Automated crypto analysis:
- **Ticker**: BTC-USD (configurable)
- **Interval**: Every 4 hours synced to UTC candle closes (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
- **Data Delay**: 5-minute offset to ensure closed candle availability
- **Control**: Enable/disable via Scorecard page or `/api/scheduler/toggle`

## Installation

### Prerequisites

- Python ≥3.12
- Node.js ≥20 (for frontend)
- API keys for at least one LLM provider

### Full Development Setup

```bash
# 1. Clone repository
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents

# 2. Create Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install Python dependencies (editable mode)
pip install -e ".[dev]"

# 4. Setup frontend
cd frontend
npm install
cd ..

# 5. Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

Configure at minimum one LLM provider:

```bash
# LLM Providers (choose one or more)
export OPENAI_API_KEY="sk-..."           # OpenAI GPT-5.x
export ANTHROPIC_API_KEY="sk-ant-..."    # Anthropic Claude 4.x
export GOOGLE_API_KEY="..."              # Google Gemini 3.x
export XAI_API_KEY="..."                   # xAI Grok
export OPENROUTER_API_KEY="..."            # OpenRouter (multi-provider)

# Data Providers (optional but recommended)
export ALPHA_VANTAGE_API_KEY="..."         # Equity fundamentals
export FRED_API_KEY="..."                  # Macroeconomic data
export KALSHI_API_KEY="..."                # Prediction markets
export BGEOMETRICS_API_KEY="..."         # On-chain metrics
```

For local models, install Ollama and set `llm_provider: "ollama"` in config.

### Running the Full Stack

```bash
# Terminal 1: Backend
.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend && npm run dev

# Access dashboard at http://localhost:5173
```

### CLI Usage

Interactive TUI for single analysis:
```bash
# Installed command
tradingagents

# Or run from source
python -m cli.main

# Command-line args
python -m cli.main --ticker BTC-USD --date 2026-04-10
```

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

## Configuration

All configuration lives in `tradingagents/default_config.py`. Copy and modify for your use case:

### LLM Configuration

```python
config = {
    "llm_provider": "openai",           # openai, google, anthropic, xai, openrouter, ollama
    "deep_think_llm": "gpt-5.2",        # Complex reasoning tasks
    "quick_think_llm": "gpt-5-mini",    # Fast/simple tasks
    "llm_temperature": 0.4,               # 0.0=deterministic, higher=more creative
    "openai_reasoning_effort": "medium", # low, medium, high (for o1/o3 models)
    "anthropic_effort": None,           # high, medium, low
    "google_thinking_level": None,      # high, minimal
}
```

### Data Vendor Routing

```python
config["asset_type"] = "auto"  # auto, equity, crypto

config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
    "prediction_market_data": "kalshi",
}

config["crypto_vendors"] = {
    "core_stock_apis": "crypto",
    "derivatives_data": "crypto",
    "fundamental_data": "crypto",
}
```

### Debate Settings

```python
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1
config["backtest_mode"] = False  # Disables real-time tools during backtest
```

## API Reference

### Analysis Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/analyze` | POST | Start analysis. Body: `{"ticker": "BTC-USD", "date?": "2026-04-10"}` |
| `GET /api/stream/{job_id}` | SSE | Real-time progress: agent_start, agent_report, decision, done |
| `GET /api/history` | GET | List all tickers with analyses |
| `GET /api/history/{ticker}` | GET | List analysis dates for ticker |
| `GET /api/history/{ticker}/{date}` | GET | Full analysis JSON with all reports |
| `GET /api/price/{ticker}?days=90` | GET | OHLCV + SMA data |

### Backtest Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/backtest` | POST | Start backtest. Body includes mode, dates, capital, sizing |
| `GET /api/backtest/stream/{job_id}` | SSE | Progress: status, progress, decision, complete |
| `GET /api/backtest/{job_id}` | GET | Full results with metrics, trades, equity curve |
| `GET /api/backtest/results` | GET | List historical backtest jobs |

### Shadow Trading Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/shadow/record` | POST | Record decision for forward-test |
| `GET /api/shadow/decisions/{ticker}` | GET | List shadow decisions |
| `GET /api/shadow/score/{ticker}` | GET | Scorecard with win rates, Brier scores |
| `POST /api/shadow/calibrate/{ticker}` | POST | Confidence calibration analysis |
| `POST /api/shadow/walk-forward/{ticker}` | POST | Temporal validation |

### Scheduler Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/scheduler/status` | GET | Current state: enabled, next_run, last_run |
| `POST /api/scheduler/toggle` | POST | Enable/disable scheduler |
| `POST /api/scheduler/run-now` | POST | Trigger immediate run |

## Key Concepts

### Kelly Criterion Position Sizing

Position size = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win

**Contamination Guard**: Requires ≥30 historical trades in knowledge store before using learned win rates. Falls back to fixed 15% below threshold to prevent overfitting on small samples.

**Implementation**: `tradingagents/graph/confidence.py::score()`

### Deflated Sharpe Ratio (DSR)

Adjusts Sharpe ratio for multiple testing bias and non-normal returns (skewness, kurtosis). Based on Bailey-López de Prado (2012).

**Gating**: If DSR < threshold, position size is reduced proportionally.

**Fallback**: Without `scipy`, DSR reverts to basic Sharpe (tests marked xfail).

### Walk-Forward Validation

Prevents overfitting by testing on unseen data:
1. Train on [t-n to t]
2. Test on [t+1]
3. Roll forward, repeat
4. Aggregate test period results

**Entry**: Scorecard page or `POST /api/shadow/walk-forward/{ticker}`

### Shadow Mode

Forward-testing without capital:
1. Record decision with timestamp, price, signal, confidence
2. Wait N days (actual outcome known)
3. Score: Brier = (confidence - outcome)^2, calibration plots
4. Compare predicted vs actual win rates

## Testing

### Test Suite Structure (336 tests)

```bash
tests/
├── test_backtesting/     # 94 tests: portfolio, metrics, walk-forward
├── test_dataflows/       # Data vendor clients
├── test_graph/           # Signal processing, conditional logic
├── test_llm_clients/     # Provider client unit tests
└── integration/            # API endpoint integration tests
```

### Running Tests

```bash
# Full suite
.venv/bin/python -m pytest

# Specific module
pytest tests/test_backtesting/test_portfolio_extended.py -v

# With coverage
pytest --cov=tradingagents --cov-report=html

# Live tests (require API keys)
export OPENAI_API_KEY=...
pytest -m live
```

### Frontend Tests

```bash
cd frontend

# Unit tests (Vitest)
npm run test

# E2E tests (Playwright)
npm run e2e

# With coverage
npm run test:coverage
```

## Development Guide

### Adding a New LLM Provider

1. Create `tradingagents/llm_clients/{provider}_client.py`
2. Inherit from `BaseLLMClient`
3. Implement `chat()` and `validate_model()`
4. Handle provider-specific kwargs (temperature, effort, etc.)
5. Add tests in `tests/test_llm_clients/`

### Adding a New Data Vendor

1. Create client in `tradingagents/dataflows/`
2. Inherit from `BaseDataClient`
3. Implement required methods: `get_ohlcv()`, `get_price()`, etc.
4. Register in `interface.py::VENDOR_METHODS`
5. Add asset detection in `asset_detection.py` if needed
6. Add tests in `tests/test_dataflows/`

### Debugging the Graph

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-5-mini"  # Cheaper for debugging

ta = TradingAgentsGraph(debug=True, config=config)
state, decision = ta.propagate("BTC-USD", "2026-04-10")

# Inspect full state
print(state.agent_results)
print(state.decision_history)
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'tradingagents'` | Not installed in editable mode | Run `pip install -e .` |
| Frontend can't connect to backend | CORS or port mismatch | Backend must be on port 8000; check `vite.config.ts` proxy |
| Tests fail with "scipy not installed" | Optional dependency missing | Install with `pip install scipy` or ignore (expected xfail) |
| LLM timeout | Large model + complex prompt | Use `quick_think_llm` for testing; reduce `max_debate_rounds` |
| Kalshi auth error | Missing/incorrect key | Check `KALSHI_API_KEY` and `kalshi_private_key.pem` |
| Hyperliquid rate limits | Too many requests | Enable caching; reduce concurrent backtest jobs |

## Contributing

We welcome contributions! Areas of particular interest:

- **New LLM providers** (DeepSeek, Mistral, etc.)
- **Additional data vendors** (CoinMarketCap, DeFiLlama, etc.)
- **Strategy variations** (swing trading, options, futures)
- **Performance optimizations** (caching, async, batching)
- **Documentation improvements**

### Development Workflow

1. Fork and create feature branch
2. Write/update tests (336 tests must pass)
3. Run `npm run lint` in frontend
4. Run `npm run build` to verify production build
5. Submit PR with clear description and test results

### Code Style

- Python: PEP 8, type hints encouraged, docstrings for public APIs
- Frontend: ESLint + Prettier enforced
- Commits: Conventional commits preferred (`feat:`, `fix:`, `docs:`)

Join our research community at [Tauric Research](https://tauric.ai/).

## Citation

Please reference our work if you find *TradingAgents* provides you with some help:

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```

---

**Disclaimer**: TradingAgents is designed for research purposes. Trading performance varies based on model selection, temperature settings, market conditions, and data quality. [Not financial advice.](https://tauric.ai/disclaimer/)
