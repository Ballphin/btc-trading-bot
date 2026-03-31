from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_portfolio_manager(llm, memory, backtest_knowledge_store=None):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # Get backtest lessons if available (regime-aware)
        backtest_lessons = ""
        if backtest_knowledge_store:
            try:
                ticker = state["company_of_interest"]
                trade_date = state.get("trade_date", "")
                current_regime = None
                if trade_date:
                    from tradingagents.backtesting.regime import detect_regime
                    current_regime = detect_regime(ticker, str(trade_date))
                lessons = backtest_knowledge_store.get_relevant_lessons(
                    ticker, max_lessons=3, current_regime=current_regime
                )
                if lessons and "No backtest lessons" not in lessons:
                    backtest_lessons = f"\n\n**Lessons from Past Backtests:**\n{lessons}\n"
            except Exception:
                pass  # Silently skip if lessons unavailable

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **BUY**: Strong conviction to enter or add to position
- **OVERWEIGHT**: Favorable outlook, gradually increase exposure
- **HOLD**: Maintain current position, no action needed
- **UNDERWEIGHT**: Reduce exposure, take partial profits
- **SELL**: Exit position or avoid entry
- **SHORT**: Strong conviction to enter a short position. Supported by derivatives over-leverage, on-chain weakness, or deteriorating macro liquidity.
- **COVER**: Close an existing short position. The risk/reward no longer favors the short thesis, or a trend reversal is confirmed.

**Context:**
- Trader's proposed plan: **{trader_plan}**
- Lessons from past decisions: **{past_memory_str}**{backtest_lessons}

**Risk Analysts Debate History:**
{history}

---

**CRITICAL: You MUST output your decision as valid JSON in this EXACT format:**

```json
{{
  "signal": "BUY|SELL|SHORT|COVER|HOLD|OVERWEIGHT|UNDERWEIGHT",
  "stop_loss_price": <number>,
  "take_profit_price": <number>,
  "confidence": <0.0-1.0>,
  "max_hold_days": <integer>,
  "reasoning": "<1-2 sentence executive summary>"
}}
```

**Field Requirements:**
- **signal**: Must be one of the 7 ratings above (uppercase)
- **stop_loss_price**: Specific price level to exit if trade goes against you (based on technical support/resistance, not a percentage)
- **take_profit_price**: Specific price level to take profits (based on technical resistance/support targets)
- **confidence**: Rate your conviction using THESE EXACT TIERS (do NOT default to 0.65):
  - Below 0.45: Analysts strongly disagree OR critical data is missing
  - 0.45–0.60: Mixed signals, some alignment but major counterarguments. Use this tier if your reasoning mentions "volatile", "uncertain", or "high risk".
  - 0.60–0.75: Clear direction, moderate consensus, known risks accounted for
  - 0.75–0.88: Strong multi-factor alignment, specific technical levels confirm the thesis
  - Above 0.88: Exceptional confluence across ALL signals — rare, textbook setup only
  RULE: If your reasoning contains "but", "however", "uncertain", or "volatile", confidence MUST be ≤ 0.60.
- **max_hold_days**: Maximum days to hold before auto-exit — choose based on trade type:
  Breakout/momentum: 3–5 days | Trend following: 7–14 days | Mean reversion: 2–4 days | Macro/fundamental thesis: 21–45 days
- **reasoning**: Brief 1-2 sentence summary of the key thesis

**Important Notes:**
- For HOLD signals, set stop_loss_price and take_profit_price to 0
- For SHORT positions, stop_loss_price should be ABOVE entry, take_profit_price BELOW entry
- For LONG positions, stop_loss_price should be BELOW entry, take_profit_price ABOVE entry
- Base stop/take levels on actual chart structure (support/resistance, Fibonacci levels, volume nodes), NOT arbitrary percentages
- Consider current volatility (ATR) when setting stop distance — wider stops in high volatility

Be decisive and ground every conclusion in specific evidence from the analysts' debate."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
