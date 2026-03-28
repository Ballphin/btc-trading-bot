"""Tests for CoinbaseClient with mocked API responses."""

import json
import pytest
from unittest.mock import patch, MagicMock
from tradingagents.dataflows.coinbase_client import CoinbaseClient


@pytest.fixture
def client(tmp_path):
    """Create a CoinbaseClient with a temp cache dir."""
    c = CoinbaseClient(cache_ttl=60)
    c.cache_dir = tmp_path / "coinbase"
    c.cache_dir.mkdir()
    return c


class TestCoinbaseClient:

    def test_get_ohlcv_parses_response(self, client):
        """Test OHLCV response parsing (Coinbase format: [time, low, high, open, close, volume])."""
        mock_data = [
            [1704067200, 42000, 43000, 42500, 42800, 1500],
            [1704153600, 42800, 44000, 42900, 43500, 2000],
        ]

        with patch.object(client, "_request", return_value=mock_data):
            df = client.get_ohlcv("BTC-USD", 86400, "2024-01-01", "2024-01-02")

        assert not df.empty
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(df) == 2

    def test_get_ohlcv_empty_response(self, client):
        with patch.object(client, "_request", return_value=[]):
            df = client.get_ohlcv("BTC-USD", 86400, "2024-01-01", "2024-01-02")
        assert df.empty

    def test_get_spot_price(self, client):
        mock_data = {"data": {"amount": "67500.00", "currency": "USD"}}
        with patch.object(client, "_request", return_value=mock_data):
            price = client.get_spot_price("BTC-USD")
        assert price == 67500.0

    def test_get_ohlcv_deduplicates(self, client):
        """Verify duplicate timestamps are removed."""
        mock_data = [
            [1704067200, 42000, 43000, 42500, 42800, 1500],
            [1704067200, 42000, 43000, 42500, 42800, 1500],  # duplicate
        ]
        with patch.object(client, "_request", return_value=mock_data):
            df = client.get_ohlcv("BTC-USD", 86400, "2024-01-01", "2024-01-02")
        assert len(df) == 1
