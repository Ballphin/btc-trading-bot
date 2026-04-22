# TradingAgents Documentation

A self-contained, single-file HTML documentation site that works locally without any external dependencies.

## 📖 Start Here

**Not sure where to begin?** Pick your role:

| I am a... | Start with | What you'll learn |
|-----------|-----------|-------------------|
| **Trader / End User** | [User Guide](#user-guide) → [Features Guide](#features-guide) | How to run analyses, read signals, use backtesting |
| **Quant Researcher** | [Quant Reference](#quant) | DSR, Kelly Criterion, walk-forward validation, calibration |
| **Developer** | [Developer Guide](#developer) → [API Reference](#api) | Adding agents, data sources, extending the framework |

## Viewing the Documentation

Simply open the HTML file in any modern web browser:

```bash
# macOS
open docs/tradingagents-docs.html

# Linux
xdg-open docs/tradingagents-docs.html

# Windows
start docs/tradingagents-docs.html
```

## What's Inside

### 1. 🏠 Welcome
Overview of what TradingAgents does, who it's for, and how to get started in under 5 minutes.

### 2. 🏗️ Architecture
System diagrams showing how data flows from your ticker input through AI agents to a final trading signal. Includes Mermaid flowcharts and module structure trees.

### 3. 📘 User Guide
Step-by-step instructions for running your first analysis, understanding the output, interpreting signals, and using backtesting and shadow trading.

### 4. 🗺️ Features Guide
**Page-by-page walkthrough** of every dashboard feature — what it does, how to use it, and what the output means. Written in plain language for end users.

### 5. 💻 Developer Guide
How to add new analyst agents, data sources, and LLM providers. Includes code examples, testing guide, and module reference.

### 6. 📡 API Reference
Complete reference for all 38+ FastAPI endpoints, key classes, and data client hierarchy.

### 7. 📐 Quant Reference
Mathematical foundations: Deflated Sharpe Ratio, Sharpe Standard Error, regime detection, walk-forward validation, Brier score calibration.

### 8. 📖 Glossary
Plain-language definitions of all technical terms used throughout the documentation — DSR, Brier Score, Kelly Criterion, SSE, and more.

## Other Documentation

| Document | What It Covers |
|----------|---------------|
| [PERSISTENCE.md](PERSISTENCE.md) | How to persist your analysis history across restarts (Gist sync, external directories) |
| [RENDER_FREE_TIER.md](../RENDER_FREE_TIER.md) | Step-by-step guide for deploying on Render.com's free tier |
| [Frontend README](../frontend/README.md) | Dashboard page-by-page guide, development setup, tech stack |

## Regenerating

To regenerate the documentation after code changes:

```bash
cd /Users/daniel/Desktop/TradingAgents
python scripts/generate_docs.py
```

## Statistics

- **File Size**: ~75 KB
- **Modules Documented**: 112+
- **API Endpoints**: 38+
- **Diagrams**: 5 Mermaid flowcharts + 2 ASCII trees
- **Sections**: 8 main sections (including Features Guide and Glossary)

---

Generated on: 2026-04-22
