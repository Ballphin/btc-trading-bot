"""Cross-vendor macro narrative composer — Stage 2 Commit M.

Composes Kalshi macro + Polymarket crypto + FRED dashboard + Fear-Greed
+ Deribit DVol into a single deterministic markdown briefing (~400
words). Used by ``news_analyst.py`` as a pre-pulse macro injection so
agents share a common macro context.

The composer is deterministic (no LLM call) — inputs are concatenated
in a fixed order with clear section headers so the output is diffable
across runs. Per the WCT rebuttal during the debate, this is strictly a
narrative aggregator: Pulse scoring remains pure price / order-flow.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.tools import tool

from tradingagents.backtesting.context import BACKTEST_MODE

logger = logging.getLogger(__name__)


def _safe_call(fn, label: str) -> str:
    """Invoke a macro source; swallow errors so one bad vendor never
    blanks the whole briefing."""
    try:
        out = fn()
        if not out or not isinstance(out, str):
            return f"### {label}\n[no content]"
        return out
    except Exception as e:
        logger.warning("macro composer: %s failed: %s", label, e)
        return f"### {label}\n[DATA UNAVAILABLE: {type(e).__name__}]"


def compose_macro_narrative(
    *,
    include_kalshi: bool = True,
    include_polymarket: bool = True,
    include_fred: bool = True,
    include_fear_greed: bool = True,
    include_deribit_dvol: bool = False,
) -> str:
    """Return a deterministic multi-vendor macro briefing.

    Each section header is stable so downstream agents can pattern-match
    on them. In backtest mode a banner is emitted and live-data sections
    are replaced with ``[DISABLED IN BACKTEST MODE]`` stubs (honouring
    the M.2 mode discipline).
    """
    parts: List[str] = ["# Cross-Vendor Macro Briefing\n"]
    if BACKTEST_MODE.get():
        parts.append(
            "> **Mode:** backtest — live vendor calls disabled; sections "
            "below are stubs.\n"
        )

    if include_kalshi:
        from tradingagents.dataflows.kalshi_client import get_kalshi_macro_context
        parts.append(_safe_call(get_kalshi_macro_context, "Kalshi Macro"))

    if include_polymarket:
        from tradingagents.dataflows.polymarket_client import (
            get_polymarket_crypto_context,
        )
        parts.append(_safe_call(get_polymarket_crypto_context, "Polymarket Crypto"))

    if include_fred:
        from tradingagents.dataflows.fred_client import get_fred_macro_dashboard
        parts.append(_safe_call(get_fred_macro_dashboard, "FRED Macro"))

    if include_fear_greed:
        # Optional — only wire if the client is importable at call time.
        try:
            from tradingagents.dataflows.fear_greed_client import FearGreedClient
            fg = FearGreedClient()
            parts.append(_safe_call(
                lambda: getattr(fg, "get_dashboard", lambda: "[no dashboard method]")(),
                "Crypto Fear & Greed",
            ))
        except Exception as e:
            parts.append(f"### Crypto Fear & Greed\n[DATA UNAVAILABLE: {type(e).__name__}]")

    if include_deribit_dvol:
        try:
            from tradingagents.dataflows.deribit_client import DeribitClient
            dc = DeribitClient()
            parts.append(_safe_call(
                lambda: getattr(dc, "get_dvol_summary", lambda: "[no dvol method]")(),
                "Deribit DVol",
            ))
        except Exception as e:
            parts.append(f"### Deribit DVol\n[DATA UNAVAILABLE: {type(e).__name__}]")

    return "\n\n".join(parts).rstrip() + "\n"


@tool
def get_macro_narrative() -> str:
    """
    Returns a cross-vendor macro briefing (Kalshi + Polymarket + FRED
    + Crypto Fear & Greed) with deterministic section ordering. Use as
    a pre-pulse macro context injection for directional bias.
    """
    return compose_macro_narrative()
