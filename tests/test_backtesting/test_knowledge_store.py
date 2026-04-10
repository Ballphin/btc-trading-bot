"""Tests for knowledge_store.py — lesson retrieval, caching, regime-aware sorting."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tradingagents.backtesting.knowledge_store import BacktestKnowledgeStore


@pytest.fixture
def store(tmp_path):
    return BacktestKnowledgeStore(results_dir=str(tmp_path))


@pytest.fixture
def sample_lessons():
    return [
        {
            "category": "signal_accuracy",
            "signal_type": "BUY",
            "confidence": "high",
            "regime": "trending_up",
            "win_rate": 0.65,
            "sample_size": 30,
            "lesson": "BUY signals in uptrend have 65% win rate.",
        },
        {
            "category": "signal_accuracy",
            "signal_type": "BUY",
            "confidence": "medium",
            "regime": "ranging",
            "win_rate": 0.45,
            "sample_size": 20,
            "lesson": "BUY signals in ranging market have 45% win rate.",
        },
        {
            "category": "risk_management",
            "confidence": "high",
            "regime": "volatile",
            "lesson": "Reduce position size in volatile regimes.",
        },
        {
            "category": "position_sizing",
            "confidence": "low",
            "lesson": "Consider larger positions in confirmed trends.",
        },
    ]


class TestGetRelevantLessons:
    def test_no_lessons_returns_message(self, store):
        result = store.get_relevant_lessons("FAKE-TICKER")
        assert "No backtest lessons available" in result

    def test_filter_by_category(self, store, sample_lessons):
        store._lesson_cache["BTC-USD"] = sample_lessons
        result = store.get_relevant_lessons("BTC-USD", categories=["risk_management"])
        assert "Reduce position size" in result

    def test_filter_by_confidence(self, store, sample_lessons):
        store._lesson_cache["BTC-USD"] = sample_lessons
        result = store.get_relevant_lessons("BTC-USD", min_confidence="high")
        assert "65% win rate" in result

    def test_max_lessons(self, store, sample_lessons):
        store._lesson_cache["BTC-USD"] = sample_lessons
        result = store.get_relevant_lessons("BTC-USD", max_lessons=1)
        # Should only have 1 numbered lesson
        assert result.count(". [") == 1

    def test_regime_aware_sorting(self, store, sample_lessons):
        store._lesson_cache["BTC-USD"] = sample_lessons
        result = store.get_relevant_lessons(
            "BTC-USD",
            categories=["signal_accuracy"],
            current_regime="trending_up",
            max_lessons=1,
        )
        assert "uptrend" in result.lower() or "65%" in result


class TestGetSignalWinRate:
    def test_returns_win_rate(self, store, sample_lessons):
        store._lesson_cache["BTC-USD"] = sample_lessons
        result = store.get_signal_win_rate("BTC-USD", "BUY")
        assert result is not None
        assert "win_rate" in result
        assert "sample_size" in result

    def test_no_matching_signal_returns_none(self, store, sample_lessons):
        store._lesson_cache["BTC-USD"] = sample_lessons
        result = store.get_signal_win_rate("BTC-USD", "SHORT")
        assert result is None

    def test_regime_filter(self, store, sample_lessons):
        store._lesson_cache["BTC-USD"] = sample_lessons
        result = store.get_signal_win_rate("BTC-USD", "BUY", regime="trending_up")
        assert result is not None
        assert result["win_rate"] == pytest.approx(0.65)

    def test_weighted_average_across_regimes(self, store, sample_lessons):
        store._lesson_cache["BTC-USD"] = sample_lessons
        result = store.get_signal_win_rate("BTC-USD", "BUY")
        # Weighted: (0.65*30 + 0.45*20) / (30+20) = (19.5+9.0)/50 = 0.57
        assert result["win_rate"] == pytest.approx(0.57, rel=0.01)
        assert result["sample_size"] == 50


class TestCacheOperations:
    def test_clear_cache(self, store, sample_lessons):
        store._lesson_cache["BTC-USD"] = sample_lessons
        store.clear_cache()
        assert len(store._lesson_cache) == 0

    def test_auto_populates_cache(self, store, tmp_path):
        # Create fake lessons file
        lessons_dir = tmp_path / "backtests" / "BTC-USD"
        lessons_dir.mkdir(parents=True)
        # No lessons file → should try generate and return empty
        result = store.get_relevant_lessons("BTC-USD")
        assert "No backtest lessons" in result
