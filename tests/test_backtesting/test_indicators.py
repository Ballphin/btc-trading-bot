"""Tests for indicators.py — ATR, volatility calculations."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from tradingagents.backtesting.indicators import calculate_atr, calculate_volatility


class TestCalculateATR:
    @patch("tradingagents.backtesting.indicators.yf.download")
    def test_returns_float(self, mock_dl):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        mock_dl.return_value = pd.DataFrame({
            "High": [100 + i * 0.5 for i in range(30)],
            "Low": [99 + i * 0.5 for i in range(30)],
            "Close": [99.5 + i * 0.5 for i in range(30)],
        }, index=dates)
        atr = calculate_atr("AAPL", "2024-01-30")
        assert atr is not None
        assert isinstance(atr, float)
        assert atr > 0

    @patch("tradingagents.backtesting.indicators.yf.download")
    def test_insufficient_data_returns_none(self, mock_dl):
        mock_dl.return_value = pd.DataFrame({
            "High": [100], "Low": [99], "Close": [99.5],
        }, index=pd.DatetimeIndex(["2024-01-01"]))
        atr = calculate_atr("AAPL", "2024-01-01")
        assert atr is None

    @patch("tradingagents.backtesting.indicators.yf.download")
    def test_empty_data_returns_none(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        atr = calculate_atr("AAPL", "2024-01-01")
        assert atr is None

    @patch("tradingagents.backtesting.indicators.yf.download")
    def test_exception_returns_none(self, mock_dl):
        mock_dl.side_effect = Exception("Network error")
        atr = calculate_atr("AAPL", "2024-01-01")
        assert atr is None


class TestCalculateVolatility:
    @patch("tradingagents.backtesting.indicators.yf.download")
    def test_returns_float(self, mock_dl):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        mock_dl.return_value = pd.DataFrame({
            "Close": [100 + i * 0.3 for i in range(30)],
        }, index=dates)
        vol = calculate_volatility("AAPL", "2024-01-30")
        assert vol is not None
        assert isinstance(vol, float)
        assert vol > 0

    @patch("tradingagents.backtesting.indicators.yf.download")
    def test_insufficient_data_returns_none(self, mock_dl):
        mock_dl.return_value = pd.DataFrame({"Close": [100]}, index=pd.DatetimeIndex(["2024-01-01"]))
        vol = calculate_volatility("AAPL", "2024-01-01")
        assert vol is None
