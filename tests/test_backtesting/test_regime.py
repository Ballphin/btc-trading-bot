"""Tests for regime.py — detection, tagging, crypto vs equity thresholds."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from tradingagents.backtesting.regime import (
    detect_regime_context,
    detect_regime,
    tag_backtest_with_regime,
    _is_crypto,
)


def _make_mock_data(closes, start_date="2024-01-01"):
    """Create a mock yfinance DataFrame from a list of close prices."""
    dates = pd.date_range(start=start_date, periods=len(closes), freq="D")
    df = pd.DataFrame({
        "Close": closes,
        "Open": closes,
        "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes],
        "Volume": [1000] * len(closes),
    }, index=dates)
    return df


class TestIsCrypto:
    def test_btc_usd(self):
        assert _is_crypto("BTC-USD")

    def test_ethusdt(self):
        assert _is_crypto("ETHUSDT")

    def test_aapl(self):
        assert not _is_crypto("AAPL")

    def test_case_insensitive(self):
        assert _is_crypto("btc-usd")


class TestDetectRegimeContext:
    @patch("yfinance.download")
    def test_trending_up(self, mock_dl):
        # Steadily rising prices → trending_up
        closes = [50000 + i * 100 for i in range(60)]
        mock_dl.return_value = _make_mock_data(closes)
        ctx = detect_regime_context("BTC-USD", "2024-03-01")
        assert ctx["regime"] == "trending_up"
        assert ctx["above_sma20"] is True
        assert ctx["current_price"] > 0

    @patch("yfinance.download")
    def test_trending_down(self, mock_dl):
        closes = [60000 - i * 100 for i in range(60)]
        mock_dl.return_value = _make_mock_data(closes)
        ctx = detect_regime_context("BTC-USD", "2024-03-01")
        assert ctx["regime"] == "trending_down"
        assert ctx["above_sma20"] is False

    @patch("yfinance.download")
    def test_volatile(self, mock_dl):
        # Wild swings → volatile regime
        import random
        random.seed(42)
        closes = [60000]
        for _ in range(59):
            closes.append(closes[-1] * (1 + random.uniform(-0.08, 0.08)))
        mock_dl.return_value = _make_mock_data(closes)
        ctx = detect_regime_context("BTC-USD", "2024-03-01")
        assert ctx["regime"] == "volatile"

    @patch("yfinance.download")
    def test_empty_data_returns_unknown(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        ctx = detect_regime_context("BTC-USD", "2024-03-01")
        assert ctx["regime"] == "unknown"

    @patch("yfinance.download")
    def test_insufficient_data_returns_unknown(self, mock_dl):
        closes = [60000, 61000, 62000]  # Only 3 points
        mock_dl.return_value = _make_mock_data(closes)
        ctx = detect_regime_context("BTC-USD", "2024-03-01")
        assert ctx["regime"] == "unknown"

    @patch("yfinance.download")
    def test_yfinance_exception_returns_fallback(self, mock_dl):
        mock_dl.side_effect = Exception("Network error")
        ctx = detect_regime_context("BTC-USD", "2024-03-01")
        assert ctx["regime"] == "unknown"
        assert ctx["current_price"] is None


class TestDetectRegime:
    @patch("yfinance.download")
    def test_returns_string(self, mock_dl):
        closes = [50000 + i * 100 for i in range(60)]
        mock_dl.return_value = _make_mock_data(closes)
        regime = detect_regime("BTC-USD", "2024-03-01")
        assert isinstance(regime, str)
        assert regime in ("trending_up", "trending_down", "ranging", "volatile", "unknown")


class TestTagBacktestWithRegime:
    @patch("tradingagents.backtesting.regime.detect_regime")
    def test_dominant_regime(self, mock_detect):
        mock_detect.return_value = "trending_up"
        decisions = [{"date": f"2024-01-{i:02d}"} for i in range(1, 11)]
        regime = tag_backtest_with_regime(decisions, "BTC-USD")
        assert regime == "trending_up"

    def test_empty_decisions(self):
        assert tag_backtest_with_regime([], "BTC-USD") == "unknown"

    @patch("tradingagents.backtesting.regime.detect_regime")
    def test_samples_evenly(self, mock_detect):
        mock_detect.return_value = "volatile"
        decisions = [{"date": f"2024-01-{i:02d}"} for i in range(1, 21)]
        regime = tag_backtest_with_regime(decisions, "BTC-USD")
        # Should sample ≤5 dates
        assert mock_detect.call_count <= 5

    @pytest.mark.xfail(reason="Sub-daily regime transition not supported — uses daily yfinance")
    def test_sub_daily_regime_transition(self):
        """4H data: 6 candles trending_up then 2 candles with >10% crash → should be volatile."""
        # This documents a known gap: regime detection uses daily data
        assert False
