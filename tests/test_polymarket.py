"""Tests for the Polymarket Gamma API client (Stage 2 Commit K.2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.backtesting.context import BACKTEST_MODE
from tradingagents.dataflows import polymarket_client as pm


def _fake_market(question, price=0.7, liquidity=50_000):
    return {
        "question": question,
        "lastTradePrice": price,
        "liquidity": liquidity,
        "active": True, "closed": False,
    }


class TestWolfersZitzewitz:
    @pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_symmetric_around_half(self, p):
        assert abs(pm._wz_correct(0.5) - 0.5) < 1e-9

    def test_monotonic(self):
        vals = [pm._wz_correct(p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        assert all(a < b for a, b in zip(vals, vals[1:]))


@patch("tradingagents.dataflows.polymarket_client.requests.get")
def test_live_market_parse(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value=[_fake_market("Will BTC hit $100k in 2026?", 0.65, 200_000)]),
    )
    out = pm.get_polymarket_crypto_context(tags=["bitcoin"])
    assert "Will BTC hit $100k in 2026?" in out
    assert "Implied Probability" in out


@patch("tradingagents.dataflows.polymarket_client.requests.get")
def test_illiquid_market_flagged(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value=[_fake_market("Tiny market", 0.5, 1_000)]),
    )
    out = pm.get_polymarket_crypto_context(tags=["bitcoin"])
    assert "LOW LIQUIDITY" in out
    assert "Implied Probability" not in out.split("Tiny market")[1].split("\n")[0]


@patch("tradingagents.dataflows.polymarket_client.requests.get")
def test_rate_limit_surfaces(mock_get):
    mock_get.return_value = MagicMock(status_code=429)
    out = pm.get_polymarket_crypto_context(tags=["bitcoin"])
    assert "DATA UNAVAILABLE" in out
    assert "429" in out


def test_backtest_mode_source_guard():
    token = BACKTEST_MODE.set(True)
    try:
        with patch("tradingagents.dataflows.polymarket_client.requests.get") as mg:
            out = pm.get_polymarket_crypto_context()
            assert "DISABLED IN BACKTEST MODE" in out
            mg.assert_not_called()
    finally:
        BACKTEST_MODE.reset(token)


@patch("tradingagents.dataflows.polymarket_client.requests.get")
def test_crypto_tag_filter(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value=[_fake_market("Will ETH flip SOL?", 0.45, 150_000)]),
    )
    out = pm.get_polymarket_crypto_context(tags=["ethereum"])
    # Tag name should appear as a header capitalised.
    assert "Ethereum" in out


@patch("tradingagents.dataflows.polymarket_client.requests.get")
def test_wz_applied_in_dashboard(mock_get):
    # Raw 0.75 → W-Z γ=0.91 → ~0.73. Confirm we actually correct it.
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value=[_fake_market("Test", 0.75, 200_000)]),
    )
    out = pm.get_polymarket_crypto_context(tags=["bitcoin"])
    assert "Implied Probability 73%" in out
