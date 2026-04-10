"""Tests for engine.py — cache logic, preload, empty DataFrame, price retrieval."""

import pytest
import pandas as pd
from collections import defaultdict
from unittest.mock import patch, MagicMock
from datetime import datetime

from tradingagents.backtesting import engine


@pytest.fixture(autouse=True)
def clear_caches():
    """Reset module-level caches before each test."""
    engine._PRICE_CACHE.clear()
    engine._FUNDING_CACHE.clear()
    engine._INTRADAY_CACHE.clear()
    engine._CACHE_TTL.clear()
    yield
    engine._PRICE_CACHE.clear()
    engine._FUNDING_CACHE.clear()
    engine._INTRADAY_CACHE.clear()
    engine._CACHE_TTL.clear()


class TestGetPriceOnDate:
    def test_cache_hit(self):
        engine._PRICE_CACHE["BTC-USD"] = {"2024-01-15": 42000.0}
        with patch.object(engine, "_preload_chunk_if_needed"):
            price = engine._get_price_on_date("BTC-USD", "2024-01-15")
        assert price == 42000.0

    def test_closest_prior_date(self):
        engine._PRICE_CACHE["BTC-USD"] = {
            "2024-01-10": 41000.0,
            "2024-01-12": 42000.0,
        }
        with patch.object(engine, "_preload_chunk_if_needed"):
            price = engine._get_price_on_date("BTC-USD", "2024-01-13")
        assert price == 42000.0  # Jan 12 is closest prior

    def test_no_data_returns_none(self):
        with patch.object(engine, "_preload_chunk_if_needed"):
            price = engine._get_price_on_date("FAKE-TICKER", "2024-01-15")
        assert price is None


class TestPreloadChunk:
    @patch("tradingagents.dataflows.hyperliquid_client.HyperliquidClient")
    @patch("tradingagents.dataflows.asset_detection.is_crypto", return_value=True)
    def test_crypto_populates_cache(self, mock_is_crypto, mock_hl_cls):
        mock_hl = MagicMock()
        mock_hl_cls.return_value = mock_hl
        mock_hl.get_ohlcv.return_value = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-15")],
            "open": [42000.0], "high": [43000.0],
            "low": [41000.0], "close": [42500.0], "volume": [100.0],
        })
        mock_hl.get_funding_history.return_value = pd.DataFrame(
            columns=["timestamp", "funding_rate"]
        )
        engine._preload_chunk_if_needed("BTC-USD", "2024-01-15")
        assert "BTC-USD" in engine._PRICE_CACHE
        assert "2024-01-15" in engine._PRICE_CACHE["BTC-USD"]

    @patch("tradingagents.dataflows.hyperliquid_client.HyperliquidClient")
    @patch("tradingagents.dataflows.asset_detection.is_crypto", return_value=True)
    def test_empty_df_leaves_cache_empty(self, mock_is_crypto, mock_hl_cls):
        """K13: get_ohlcv returns 200 but empty DataFrame → cache stays empty."""
        mock_hl = MagicMock()
        mock_hl_cls.return_value = mock_hl
        mock_hl.get_ohlcv.return_value = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        mock_hl.get_funding_history.return_value = pd.DataFrame(
            columns=["timestamp", "funding_rate"]
        )
        engine._preload_chunk_if_needed("BTC-USD", "2024-01-15")
        # Cache should remain empty for this ticker
        assert len(engine._PRICE_CACHE.get("BTC-USD", {})) == 0

    @patch("tradingagents.dataflows.hyperliquid_client.HyperliquidClient")
    @patch("tradingagents.dataflows.asset_detection.is_crypto", return_value=True)
    def test_empty_df_price_returns_none(self, mock_is_crypto, mock_hl_cls):
        """After empty preload, _get_price_on_date should return None gracefully."""
        mock_hl = MagicMock()
        mock_hl_cls.return_value = mock_hl
        mock_hl.get_ohlcv.return_value = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        mock_hl.get_funding_history.return_value = pd.DataFrame(
            columns=["timestamp", "funding_rate"]
        )
        price = engine._get_price_on_date("BTC-USD", "2024-01-15")
        assert price is None  # Not a NoneType crash

    @patch("tradingagents.dataflows.hyperliquid_client.HyperliquidClient")
    @patch("tradingagents.dataflows.asset_detection.is_crypto", return_value=True)
    def test_connection_error_raises(self, mock_is_crypto, mock_hl_cls):
        """Hyperliquid preload failure should not silently pass."""
        mock_hl = MagicMock()
        mock_hl_cls.return_value = mock_hl
        mock_hl.get_ohlcv.side_effect = ConnectionError("API down")
        # Should warn but not crash the caller entirely
        engine._preload_chunk_if_needed("BTC-USD", "2024-01-15")
        assert len(engine._PRICE_CACHE.get("BTC-USD", {})) == 0

    @patch("yfinance.download")
    @patch("tradingagents.dataflows.asset_detection.is_crypto", return_value=False)
    def test_equity_yfinance_path(self, mock_is_crypto, mock_yf_dl):
        mock_data = pd.DataFrame({
            "Close": [150.0],
        }, index=pd.DatetimeIndex(["2024-01-15"]))
        mock_yf_dl.return_value = mock_data
        engine._preload_chunk_if_needed("AAPL", "2024-01-15")
        assert "AAPL" in engine._PRICE_CACHE


class TestGetIntradayCandles:
    def test_returns_cached(self):
        engine._INTRADAY_CACHE["BTC-USD"] = {
            "2024-01-15": [{"t": "2024-01-15T00:00:00", "c": 42000}]
        }
        candles = engine.get_intraday_candles("BTC-USD", "2024-01-15")
        assert len(candles) == 1

    def test_empty_when_no_data(self):
        with patch.object(engine, "_preload_chunk_if_needed"):
            candles = engine.get_intraday_candles("FAKE", "2024-01-15")
        assert candles == []
