from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
)
from tradingagents.agents.utils.crypto_tools import (
    get_onchain_data,
    get_macro_indicators,
    get_derivatives_data,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.asset_detection import is_crypto


_EQUITY_SYSTEM_MESSAGE = (
    "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
    " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
    " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
)

_CRYPTO_SYSTEM_MESSAGE = (
    "You are a researcher tasked with analyzing fundamental information about a cryptocurrency asset. "
    "Cryptocurrencies do NOT have traditional financial statements (balance sheets, income statements, cash flow). "
    "Instead, analyze using the available crypto-specific tools:\n"
    "- `get_fundamentals`: Returns on-chain metrics for BTC (MVRV, SOPR, exchange flows, NUPL). For non-BTC assets, this will indicate on-chain data is unavailable — that is expected.\n"
    "- `get_onchain_data`: Same as get_fundamentals for crypto — returns BTC on-chain metrics or a guidance message for non-BTC assets.\n"
    "- `get_macro_indicators`: FRED macro data (M2 money supply, DXY dollar index, 2Y/10Y treasury yields) — critical for understanding macro liquidity conditions affecting crypto.\n"
    "- `get_derivatives_data`: Funding rates, open interest, premium/discount, directional aggression from Hyperliquid and Deribit.\n\n"
    "For non-BTC crypto assets, focus your analysis on derivatives data and macro indicators since on-chain metrics are BTC-only. "
    "Write a comprehensive report with specific, actionable insights. "
    "Make sure to append a Markdown table at the end of the report to organize key points."
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        if is_crypto(ticker):
            tools = [
                get_fundamentals,
                get_onchain_data,
                get_macro_indicators,
                get_derivatives_data,
            ]
            system_message = _CRYPTO_SYSTEM_MESSAGE
        else:
            tools = [
                get_fundamentals,
                get_balance_sheet,
                get_cashflow,
                get_income_statement,
            ]
            system_message = _EQUITY_SYSTEM_MESSAGE

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
