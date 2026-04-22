# TradingAgents — Frontend Dashboard

The React-based dashboard for the TradingAgents multi-agent trading framework. This is what you see when you open `http://localhost:5173` in your browser.

## What the Dashboard Does

The dashboard lets you run AI-powered analysis on any stock or crypto ticker, view real-time progress as multiple AI agents research and debate, and receive structured trading signals — all from your browser.

## Pages & Features

### 🏠 Home (`/`)
**Your starting point.** Type any ticker symbol (e.g., `BTC-USD`, `NVDA`, `AAPL`) into the search bar and click **Analyze**. You'll also see quick-access buttons for popular tickers and a list of your most recent analyses.

**How to use it:** Just type a ticker and hit Enter or click Analyze. That's it.

### 📊 Live Analysis (`/analyze/:ticker`)
**Watch AI agents work in real-time.** After you start an analysis, this page shows a live progress stream:

1. **4 Analyst Agents** run in parallel — Market, Sentiment, News, and Fundamentals
2. **Bull & Bear Researchers** debate the findings, arguing for and against the trade
3. **A Trader Agent** composes a structured trading signal
4. **3 Risk Managers** (Aggressive, Neutral, Conservative) debate position sizing
5. **Portfolio Manager** makes the final call

You'll see each step update live via Server-Sent Events (SSE). The final output is a structured signal with: **action** (BUY/SELL/HOLD/SHORT/COVER), **confidence** (0–100%), **stop-loss price**, **take-profit price**, and **position size**.

### 📜 History (`/history`, `/history/:ticker`)
**Browse all your past analyses.** See every ticker you've analyzed, grouped by date. Click any entry to see the full decision breakdown, including all agent reports and debate transcripts.

### 🔍 Analysis Detail (`/history/:ticker/:date`)
**Deep-dive into a single analysis.** See the complete reasoning chain: what each analyst found, how the bull/bear debate went, what risk parameters were set, and why the final decision was made.

### 📈 Backtest (`/backtest`)
**Test strategies against historical data.** Three modes available:

| Mode | What It Does | When to Use |
|------|-------------|-------------|
| **Replay** | Re-run past analyses exactly as computed | Verify historical performance quickly |
| **Simulation** | Run fresh AI analysis on past dates | Test new model/prompt changes |
| **Hybrid** | Use saved results where available, simulate the rest | Backfill gaps in your history |

Configure ticker, date range, initial capital, position sizing method (Fixed, Kelly, ATR), leverage, and fees.

### 📊 Backtest Results (`/backtest/results/:id`)
**Visualize backtest performance.** See equity curves, trade-by-trade history, performance metrics (Sharpe, Sortino, max drawdown), and AI-generated lessons learned from the backtest.

### 📋 Backtest History (`/backtests`)
**View all past backtests.** Compare performance across different configurations and time periods.

### 🎯 Scorecard (`/scorecard`)
**Forward-test your AI without real money.** Shadow (paper) trading lets you:
- Record AI decisions with timestamps and prices
- Wait for the market to move
- Score the AI's accuracy using **Brier score** (lower = better calibrated)
- See win rates by signal type and market regime
- Run **confidence calibration** — is the AI overconfident or underconfident?

### ⚡ Pulse (`/pulse`)
**Automated 4-hour analysis.** The Pulse system runs AI analysis on a schedule (every 4 hours, synced to UTC candle closes). This page shows:
- Recent pulse signals with timestamps
- Ensemble agreement across multiple model configurations
- Signal history and trends

### 🔧 Auto-Tune (`/autotune`)
**Optimize your configuration.** Auto-Tune tests different model and parameter combinations to find the best-performing setup for your target ticker.

### 🔗 Model Selector
**Choose your AI model.** Available on the Home page — select from OpenRouter, DeepSeek, OpenAI, Anthropic, and other providers. Pick the model that balances cost, speed, and quality for your needs.

## Development Setup

```bash
# Install dependencies
npm install

# Start development server (hot-reload)
npm run dev

# Run tests
npm run test

# Build for production
npm run build

# Lint code
npm run lint
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `VITE_API_BASE_URL` | Backend API URL (default: `http://localhost:8000/api`) |

For production deployments (e.g., Render), set this to your backend's public URL.

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast builds and HMR
- **TailwindCSS** for styling
- **Recharts** for data visualization
- **Lucide React** for icons
- **React Router** for navigation
- **Vitest** for testing

## Project Structure

```
src/
├── pages/           # Route-level page components
│   ├── Home.tsx           # Ticker search + recent analyses
│   ├── Analyze.tsx        # Live SSE analysis stream
│   ├── History.tsx        # Browse past analyses
│   ├── AnalysisDetail.tsx # Full decision breakdown
│   ├── Backtest.tsx       # Configure backtests
│   ├── BacktestResults.tsx# Equity curves + metrics
│   ├── Scorecard.tsx      # Shadow trading scorecard
│   ├── Pulse.tsx          # 4H automated signals
│   ├── PulseExplain.tsx   # Pulse signal deep-dive
│   └── AutoTune.tsx       # Configuration optimizer
├── components/      # Reusable UI components
│   ├── Navbar.tsx         # Top navigation bar
│   ├── SignalBadge.tsx    # BUY/SELL/HOLD colored badges
│   ├── PriceChart.tsx     # OHLCV candlestick charts
│   ├── DebatePanel.tsx    # Bull vs Bear debate display
│   └── FinalDecisionCard.tsx  # Structured signal output
├── lib/
│   └── api.ts             # API client + TypeScript types
├── hooks/
│   └── useDocumentTitle.ts# Page title management
├── index.css              # Global styles + design tokens
└── App.tsx                # Router configuration
```
