"""Backtest knowledge store for retrieving relevant lessons during live trading.

Manages loading and retrieval of backtest lessons for agent decision-making.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from tradingagents.backtesting.feedback import BacktestFeedbackGenerator

logger = logging.getLogger(__name__)


class BacktestKnowledgeStore:
    """Central knowledge store for backtest lessons that agents can query."""
    
    def __init__(self, results_dir: str = "./eval_results"):
        """Initialize the knowledge store.
        
        Args:
            results_dir: Directory containing backtest results
        """
        self.results_dir = results_dir
        self.feedback_generator = BacktestFeedbackGenerator(results_dir)
        self._lesson_cache = {}  # ticker -> lessons
    
    def get_relevant_lessons(
        self,
        ticker: str,
        categories: Optional[List[str]] = None,
        max_lessons: int = 3,
        min_confidence: str = "medium",
        current_regime: Optional[str] = None,
    ) -> str:
        """Get relevant lessons formatted for agent prompts.

        Args:
            ticker: Ticker symbol
            categories: Filter by categories (e.g., ["signal_accuracy", "risk_management"])
            max_lessons: Maximum number of lessons to return
            min_confidence: Minimum confidence level ("low", "medium", "high")
            current_regime: If provided, prefer lessons matching this regime

        Returns:
            Formatted string of lessons for injection into agent prompts
        """
        # Load lessons (use cache if available)
        if ticker not in self._lesson_cache:
            lessons = self.feedback_generator.load_latest_lessons(ticker)
            if not lessons:
                lessons = self.feedback_generator.generate_lessons(ticker, min_trades=5)
                if lessons:
                    self.feedback_generator.save_lessons(ticker, lessons)
            self._lesson_cache[ticker] = lessons

        lessons = list(self._lesson_cache.get(ticker, []))

        if not lessons:
            return "No backtest lessons available yet for this ticker."

        # Filter by categories if specified
        if categories:
            lessons = [l for l in lessons if l.get("category") in categories]

        # Filter by confidence
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        min_conf_level = confidence_order.get(min_confidence, 1)
        lessons = [
            l for l in lessons
            if confidence_order.get(l.get("confidence", "low"), 0) >= min_conf_level
        ]

        if not lessons:
            return "No high-confidence backtest lessons available for the current criteria."

        # Regime-aware sorting: exact match first, then "unknown"/untagged, then others
        def regime_sort_key(lesson):
            conf_score = confidence_order.get(lesson.get("confidence", "low"), 0)
            lesson_regime = lesson.get("regime", "unknown")
            if current_regime and lesson_regime != "unknown":
                regime_match = 2 if lesson_regime == current_regime else 0
            else:
                regime_match = 1  # neutral when no current regime
            return (regime_match, conf_score)

        lessons.sort(key=regime_sort_key, reverse=True)
        lessons = lessons[:max_lessons]

        # Format for prompt injection
        formatted_lessons = []
        for i, lesson in enumerate(lessons, 1):
            category = lesson.get("category", "general").replace("_", " ").title()
            confidence = lesson.get("confidence", "medium").upper()
            lesson_text = lesson.get("lesson", "")
            formatted_lessons.append(f"{i}. [{category} - {confidence} CONFIDENCE] {lesson_text}")

        return "\n".join(formatted_lessons)
    
    def get_signal_lessons(self, ticker: str, max_lessons: int = 2) -> str:
        """Get signal accuracy lessons specifically.
        
        Args:
            ticker: Ticker symbol
            max_lessons: Maximum lessons to return
            
        Returns:
            Formatted signal lessons
        """
        return self.get_relevant_lessons(
            ticker,
            categories=["signal_accuracy"],
            max_lessons=max_lessons,
            min_confidence="medium"
        )
    
    def get_signal_win_rate(
        self,
        ticker: str,
        signal: str,
        regime: Optional[str] = None,
    ) -> Optional[dict]:
        """Get historical win rate for a signal type, optionally filtered by regime.

        Args:
            ticker: Ticker symbol
            signal: Signal type (BUY, SELL, SHORT, etc.)
            regime: If provided, prefer lessons from this regime; falls back to all regimes

        Returns:
            Dict with 'win_rate' (float) and 'sample_size' (int), or None if no data
        """
        # Ensure cache is loaded (mirrors get_relevant_lessons load pattern)
        if ticker not in self._lesson_cache:
            lessons = self.feedback_generator.load_latest_lessons(ticker)
            if not lessons:
                lessons = self.feedback_generator.generate_lessons(ticker, min_trades=5)
                if lessons:
                    self.feedback_generator.save_lessons(ticker, lessons)
            self._lesson_cache[ticker] = lessons or []

        lessons = self._lesson_cache.get(ticker, [])

        # Filter to signal_accuracy lessons matching this signal type
        matching = [
            l for l in lessons
            if l.get("category") == "signal_accuracy"
            and l.get("signal_type", "").upper() == signal.upper()
        ]

        if not matching:
            return None

        # Prefer exact regime match if regime provided; fall back to all matches
        if regime:
            exact = [l for l in matching if l.get("regime") == regime]
            if exact:
                matching = exact

        # Compute sample-size-weighted average win rate
        total_samples = sum(l.get("sample_size", 1) for l in matching)
        if total_samples == 0:
            return None

        weighted_win_rate = sum(
            l.get("win_rate", 0.5) * l.get("sample_size", 1) for l in matching
        ) / total_samples

        return {"win_rate": weighted_win_rate, "sample_size": total_samples}

    def get_risk_lessons(self, ticker: str, max_lessons: int = 2) -> str:
        """Get risk management lessons specifically.
        
        Args:
            ticker: Ticker symbol
            max_lessons: Maximum lessons to return
            
        Returns:
            Formatted risk lessons
        """
        return self.get_relevant_lessons(
            ticker,
            categories=["risk_management"],
            max_lessons=max_lessons,
            min_confidence="medium"
        )
    
    def get_sizing_lessons(self, ticker: str, max_lessons: int = 2) -> str:
        """Get position sizing lessons specifically.
        
        Args:
            ticker: Ticker symbol
            max_lessons: Maximum lessons to return
            
        Returns:
            Formatted sizing lessons
        """
        return self.get_relevant_lessons(
            ticker,
            categories=["position_sizing"],
            max_lessons=max_lessons,
            min_confidence="medium"
        )
    
    def refresh_lessons(self, ticker: str):
        """Force regeneration of lessons for a ticker.
        
        Args:
            ticker: Ticker symbol
        """
        lessons = self.feedback_generator.generate_lessons(ticker, min_trades=5)
        if lessons:
            self.feedback_generator.save_lessons(ticker, lessons)
            self._lesson_cache[ticker] = lessons
            logger.info(f"Refreshed {len(lessons)} lessons for {ticker}")
        else:
            logger.info(f"No lessons generated for {ticker}")
    
    def clear_cache(self):
        """Clear the lesson cache."""
        self._lesson_cache.clear()
