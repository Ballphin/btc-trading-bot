# TradingAgents Documentation

A self-contained, single-file HTML documentation that works locally without any external dependencies.

## Viewing the Documentation

Simply open the HTML file in any modern web browser:

```bash
# macOS
open tradingagents-docs.html

# Linux
xdg-open tradingagents-docs.html

# Windows
start tradingagents-docs.html
```

Or directly:
```
file:///Users/daniel/Desktop/TradingAgents/docs/tradingagents-docs.html
```

## Features

- **Single File**: Everything (CSS, JavaScript, diagrams) is embedded - no external dependencies
- **Multi-Page SPA**: 6 sections with sidebar navigation
- **Dark/Light Mode**: Toggle in top-right corner
- **Search**: Quick search in sidebar
- **Diagrams**: Mermaid flowcharts and ASCII trees
- **Auto-Generated**: API reference and module structure extracted from codebase

## Sections

1. **Welcome** - Overview, quick start, key features
2. **Architecture** - System diagrams, data flow, module structure
3. **User Guide** - Running analysis, understanding outputs, backtesting
4. **Developer Guide** - Adding agents, adding data sources, testing
5. **API Reference** - 38 FastAPI endpoints, key classes, data client hierarchy
6. **Quant Reference** - DSR, Sharpe SE, regime detection, walk-forward validation

## Regenerating

To regenerate the documentation after code changes:

```bash
cd /Users/daniel/Desktop/TradingAgents
python scripts/generate_docs.py
```

This will:
- Parse all 112+ Python modules
- Extract 38+ FastAPI endpoints
- Generate fresh HTML output

## Statistics

- **File Size**: ~51 KB
- **Modules Documented**: 112
- **API Endpoints**: 38
- **Diagrams**: 5 Mermaid flowcharts + 2 ASCII trees
- **Sections**: 6 main sections

---

Generated on: 2026-04-21
