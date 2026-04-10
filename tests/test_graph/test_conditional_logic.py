"""Tests for conditional_logic.py — debate and risk round routing."""

import pytest
from unittest.mock import MagicMock

from tradingagents.graph.conditional_logic import ConditionalLogic


@pytest.fixture
def logic():
    return ConditionalLogic(max_debate_rounds=2, max_risk_discuss_rounds=2)


def _make_state(messages=None, debate_state=None, risk_state=None):
    """Create a minimal AgentState dict."""
    if messages is None:
        msg = MagicMock()
        msg.tool_calls = []
        messages = [msg]
    return {
        "messages": messages,
        "investment_debate_state": debate_state or {
            "count": 0,
            "current_response": "",
            "bull_history": "",
            "bear_history": "",
            "judge_decision": "",
        },
        "risk_debate_state": risk_state or {
            "count": 0,
            "latest_speaker": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": "",
        },
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
    }


# ── Analyst tool routing ─────────────────────────────────────────────

class TestAnalystRouting:
    def test_market_has_tool_calls(self, logic):
        msg = MagicMock()
        msg.tool_calls = [{"name": "get_price"}]
        state = _make_state(messages=[msg])
        assert logic.should_continue_market(state) == "tools_market"

    def test_market_no_tool_calls(self, logic):
        state = _make_state()
        assert logic.should_continue_market(state) == "Msg Clear Market"

    def test_social_routing(self, logic):
        state = _make_state()
        assert logic.should_continue_social(state) == "Msg Clear Social"

    def test_news_routing(self, logic):
        state = _make_state()
        assert logic.should_continue_news(state) == "Msg Clear News"

    def test_fundamentals_routing(self, logic):
        state = _make_state()
        assert logic.should_continue_fundamentals(state) == "Msg Clear Fundamentals"


# ── Investment debate routing ────────────────────────────────────────

class TestDebateRouting:
    def test_bull_then_bear(self, logic):
        state = _make_state(debate_state={
            "count": 1,
            "current_response": "Bull: I believe...",
            "bull_history": "", "bear_history": "", "judge_decision": "",
        })
        assert logic.should_continue_debate(state) == "Bear Researcher"

    def test_bear_then_bull(self, logic):
        state = _make_state(debate_state={
            "count": 1,
            "current_response": "Bear: I disagree...",
            "bull_history": "", "bear_history": "", "judge_decision": "",
        })
        assert logic.should_continue_debate(state) == "Bull Researcher"

    def test_debate_terminates_at_max(self, logic):
        """max_debate_rounds=2 → terminates at count=4 (2*2)."""
        state = _make_state(debate_state={
            "count": 4,
            "current_response": "Bull: final word",
            "bull_history": "", "bear_history": "", "judge_decision": "",
        })
        assert logic.should_continue_debate(state) == "Research Manager"

    def test_debate_continues_below_max(self, logic):
        state = _make_state(debate_state={
            "count": 3,
            "current_response": "Bull: next round",
            "bull_history": "", "bear_history": "", "judge_decision": "",
        })
        result = logic.should_continue_debate(state)
        assert result in ("Bull Researcher", "Bear Researcher")


# ── Risk analysis routing ────────────────────────────────────────────

class TestRiskRouting:
    def test_aggressive_then_conservative(self, logic):
        state = _make_state(risk_state={
            "count": 1,
            "latest_speaker": "Aggressive: push harder",
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "judge_decision": "",
        })
        assert logic.should_continue_risk_analysis(state) == "Conservative Analyst"

    def test_conservative_then_neutral(self, logic):
        state = _make_state(risk_state={
            "count": 1,
            "latest_speaker": "Conservative: be careful",
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "judge_decision": "",
        })
        assert logic.should_continue_risk_analysis(state) == "Neutral Analyst"

    def test_neutral_then_aggressive(self, logic):
        state = _make_state(risk_state={
            "count": 1,
            "latest_speaker": "Neutral: balanced view",
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "judge_decision": "",
        })
        assert logic.should_continue_risk_analysis(state) == "Aggressive Analyst"

    def test_risk_terminates_at_max(self, logic):
        """max_risk_discuss_rounds=2 → terminates at count=6 (3*2)."""
        state = _make_state(risk_state={
            "count": 6,
            "latest_speaker": "Aggressive: done",
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "judge_decision": "",
        })
        assert logic.should_continue_risk_analysis(state) == "Portfolio Manager"
