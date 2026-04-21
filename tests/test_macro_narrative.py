"""Stage 2 Commit M — cross-vendor narrative composer tests."""

from __future__ import annotations

from unittest.mock import patch

from tradingagents.agents.utils.macro_narrative import compose_macro_narrative
from tradingagents.backtesting.context import BACKTEST_MODE


@patch("tradingagents.dataflows.kalshi_client.get_kalshi_macro_context",
       return_value="### Kalshi Macro Risk Dashboard\nhello")
@patch("tradingagents.dataflows.polymarket_client.get_polymarket_crypto_context",
       return_value="### Polymarket Crypto Prediction Markets\nhi")
@patch("tradingagents.dataflows.fred_client.get_fred_macro_dashboard",
       return_value="### FRED Macro Dashboard\nmacro")
def test_composes_all_enabled_sources(mk, pm, fred):
    out = compose_macro_narrative(
        include_kalshi=True, include_polymarket=True,
        include_fred=True, include_fear_greed=False,
    )
    assert "# Cross-Vendor Macro Briefing" in out
    assert "Kalshi Macro" in out
    assert "Polymarket" in out
    assert "FRED Macro" in out


def test_sections_are_ordered_deterministically():
    """Kalshi → Polymarket → FRED ordering must remain stable."""
    with patch("tradingagents.dataflows.kalshi_client.get_kalshi_macro_context",
               return_value="### Kalshi Macro\nK"), \
         patch("tradingagents.dataflows.polymarket_client.get_polymarket_crypto_context",
               return_value="### Polymarket\nP"), \
         patch("tradingagents.dataflows.fred_client.get_fred_macro_dashboard",
               return_value="### FRED Macro Dashboard\nF"):
        out = compose_macro_narrative(include_fear_greed=False)
    k_idx = out.find("Kalshi")
    p_idx = out.find("Polymarket")
    f_idx = out.find("FRED")
    assert k_idx < p_idx < f_idx


def test_single_vendor_failure_doesnt_blank_output():
    def boom():
        raise RuntimeError("boom")
    with patch("tradingagents.dataflows.kalshi_client.get_kalshi_macro_context",
               side_effect=boom), \
         patch("tradingagents.dataflows.polymarket_client.get_polymarket_crypto_context",
               return_value="### PM\ngood"), \
         patch("tradingagents.dataflows.fred_client.get_fred_macro_dashboard",
               return_value="### FRED\ngood"):
        out = compose_macro_narrative(include_fear_greed=False)
    assert "DATA UNAVAILABLE" in out
    assert "good" in out  # other sections still rendered


def test_backtest_mode_banner():
    token = BACKTEST_MODE.set(True)
    try:
        out = compose_macro_narrative(
            include_kalshi=True, include_polymarket=True,
            include_fred=False, include_fear_greed=False,
        )
        assert "Mode:** backtest" in out
        # Kalshi + Polymarket should self-stub (their source guards).
        assert "DISABLED IN BACKTEST MODE" in out
    finally:
        BACKTEST_MODE.reset(token)
