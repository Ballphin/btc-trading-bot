"""Decision-replay backtesting engine for TradingAgents."""

from tradingagents.backtesting.engine import BacktestEngine
from tradingagents.backtesting.portfolio import Portfolio, Position
from tradingagents.backtesting.metrics import compute_metrics
from tradingagents.backtesting.report import generate_report
from tradingagents.backtesting.feedback import BacktestFeedbackGenerator
from tradingagents.backtesting.knowledge_store import BacktestKnowledgeStore
from tradingagents.backtesting.regime import detect_regime, tag_backtest_with_regime

__all__ = [
    "BacktestEngine",
    "Portfolio",
    "Position",
    "compute_metrics",
    "generate_report",
    "BacktestFeedbackGenerator",
    "BacktestKnowledgeStore",
    "detect_regime",
    "tag_backtest_with_regime",
]
