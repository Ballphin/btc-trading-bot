"""Map LangGraph stream chunks to pipeline step labels for SSE progress."""

from typing import Any, Dict, Optional, Set


def detect_step_from_chunk(chunk: Dict[str, Any], seen_steps: Set[str]) -> Optional[Dict[str, Any]]:
    """Detect which step just completed based on state fields in the chunk."""
    report_fields = [
        ("market_report", "market", "Market Analyst", 1),
        ("sentiment_report", "social", "Social Media Analyst", 2),
        ("news_report", "news", "News Analyst", 3),
        ("fundamentals_report", "fundamentals", "Fundamentals Analyst", 4),
    ]

    for report_key, step_key, label, step_num in report_fields:
        if chunk.get(report_key) and step_key not in seen_steps:
            return {"key": step_key, "label": label, "step": step_num}

    investment_debate = chunk.get("investment_debate_state", {})
    if (
        investment_debate.get("bull_history")
        and investment_debate.get("bear_history")
        and "bull_bear" not in seen_steps
    ):
        return {"key": "bull_bear", "label": "Bull vs Bear Debate", "step": 5}

    risk_debate = chunk.get("risk_debate_state", {})
    if risk_debate.get("aggressive_history") and "risk_debate" not in seen_steps:
        return {"key": "risk_debate", "label": "Risk Debate", "step": 8}

    if chunk.get("final_trade_decision") and "portfolio_manager" not in seen_steps:
        return {"key": "portfolio_manager", "label": "Portfolio Manager", "step": 9}

    if chunk.get("trader_investment_plan") and "trader" not in seen_steps:
        return {"key": "trader", "label": "Trader", "step": 7}

    if chunk.get("investment_plan") and "research_manager" not in seen_steps:
        return {"key": "research_manager", "label": "Research Manager", "step": 6}

    return None
