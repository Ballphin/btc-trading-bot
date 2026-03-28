"""Tests for BinanceClient with mocked API responses."""

import pytest
from unittest.mock import patch
from tradingagents.dataflows.binance_client import BinanceClient


@pytest.fixture
def client(tmp_path):
    c = BinanceClient(cache_ttl=60)
    c.cache_dir = tmp_path / "binance"
    c.cache_dir.mkdir()
    return c


class TestBinanceClient:

    def test_get_funding_rates(self, client):
        mock_data = [
            {"symbol": "BTCUSDT", "fundingTime": 1704067200000, "fundingRate": "0.0001", "markPrice": "42500"},
            {"symbol": "BTCUSDT", "fundingTime": 1704096000000, "fundingRate": "0.00015", "markPrice": "42800"},
        ]
        with patch.object(client, "_request", return_value=mock_data):
            df = client.get_funding_rates("BTCUSDT", "2024-01-01", "2024-01-02")
        assert not df.empty
        assert "fundingRate" in df.columns
        assert len(df) == 2

    def test_get_open_interest_hist(self, client):
        mock_data = [
            {"symbol": "BTCUSDT", "sumOpenInterest": "50000", "sumOpenInterestValue": "2125000000", "timestamp": 1704067200000},
        ]
        with patch.object(client, "_request", return_value=mock_data):
            df = client.get_open_interest_hist("BTCUSDT")
        assert not df.empty
        assert df.iloc[0]["sumOpenInterest"] == 50000

    def test_get_taker_ratio(self, client):
        mock_data = [
            {"buySellRatio": "1.05", "buyVol": "1000", "sellVol": "952", "timestamp": 1704067200000},
        ]
        with patch.object(client, "_request", return_value=mock_data):
            df = client.get_taker_ratio("BTCUSDT")
        assert not df.empty
        assert df.iloc[0]["buySellRatio"] == pytest.approx(1.05)

    def test_empty_funding_response(self, client):
        with patch.object(client, "_request", return_value=[]):
            df = client.get_funding_rates("BTCUSDT")
        assert df.empty

    def test_to_ms_conversion(self):
        from datetime import datetime
        expected = int(datetime.strptime("2024-01-01", "%Y-%m-%d").timestamp() * 1000)
        ms = BinanceClient._to_ms("2024-01-01")
        assert ms == expected
