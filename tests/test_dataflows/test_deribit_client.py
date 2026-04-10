"""Tests for DeribitClient — funding history, OHLCV, ticker, options."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from tradingagents.dataflows.deribit_client import DeribitClient


@pytest.fixture
def dr(tmp_path):
    with patch("tradingagents.dataflows.base_client.BaseDataClient.__init__") as mock_init:
        mock_init.return_value = None
        client = DeribitClient()
        client.cache_dir = tmp_path
        client.cache_ttl = 3600
        client.session = MagicMock()
        client.MAX_RETRIES = 3
        client.BACKOFF_BASE = 2
        return client


class TestToMs:
    def test_conversion(self):
        ms = DeribitClient._to_ms("2024-01-01")
        assert isinstance(ms, int)
        assert ms > 0

    def test_format_error(self):
        with pytest.raises(ValueError):
            DeribitClient._to_ms("not-a-date")


class TestGetFundingRateHistory:
    def test_parses_result(self, dr):
        mock_data = {
            "result": [
                {"timestamp": 1700000000000, "index_price": 37000, "interest_8h": 0.0001, "interest_1h": 0.0000125},
                {"timestamp": 1700003600000, "index_price": 37050, "interest_8h": 0.00012, "interest_1h": 0.000015},
            ]
        }
        with patch.object(dr, "_request", return_value=mock_data):
            df = dr.get_funding_rate_history("BTC-PERPETUAL", "2023-11-14", "2023-11-16")
        assert len(df) == 2
        assert "timestamp" in df.columns

    def test_empty_result(self, dr):
        with patch.object(dr, "_request", return_value={"result": []}):
            df = dr.get_funding_rate_history()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_api_error_returns_empty(self, dr):
        with patch.object(dr, "_request", side_effect=ConnectionError("down")):
            df = dr.get_funding_rate_history()
        assert df.empty


class TestGetOHLCV:
    def test_parses_chart_data(self, dr):
        mock_data = {
            "result": {
                "ticks": [1700000000000, 1700086400000],
                "open": [37000, 37100],
                "high": [37500, 37600],
                "low": [36800, 36900],
                "close": [37200, 37400],
                "volume": [100.5, 200.3],
            }
        }
        with patch.object(dr, "_request", return_value=mock_data):
            df = dr.get_ohlcv("BTC-PERPETUAL", "1D", "2023-11-14", "2023-11-16")
        assert len(df) == 2
        assert df.iloc[0]["close"] == 37200

    def test_no_ticks_returns_empty(self, dr):
        with patch.object(dr, "_request", return_value={"result": {}}):
            df = dr.get_ohlcv()
        assert df.empty

    def test_api_error_returns_empty(self, dr):
        with patch.object(dr, "_request", side_effect=ConnectionError("down")):
            df = dr.get_ohlcv()
        assert df.empty


class TestGetTicker:
    def test_returns_result(self, dr):
        mock_data = {"result": {"mark_price": 67000, "open_interest": 1500000}}
        with patch.object(dr, "_request", return_value=mock_data):
            t = dr.get_ticker("BTC-PERPETUAL")
        assert t["mark_price"] == 67000

    def test_api_error_returns_empty_dict(self, dr):
        with patch.object(dr, "_request", side_effect=ConnectionError("down")):
            t = dr.get_ticker()
        assert t == {}


class TestGetOptionSummary:
    def test_parses_options(self, dr):
        mock_data = {
            "result": [
                {"instrument_name": "BTC-30DEC24-100000-C", "volume": 50},
                {"instrument_name": "BTC-30DEC24-50000-P", "volume": 30},
            ]
        }
        with patch.object(dr, "_request", return_value=mock_data):
            df = dr.get_option_summary("BTC")
        assert len(df) == 2

    def test_empty_returns_empty_df(self, dr):
        with patch.object(dr, "_request", return_value={"result": []}):
            df = dr.get_option_summary()
        assert df.empty

    def test_api_error_returns_empty_df(self, dr):
        with patch.object(dr, "_request", side_effect=ConnectionError("down")):
            df = dr.get_option_summary()
        assert df.empty
