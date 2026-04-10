"""Tests for HyperliquidClient — OHLCV, spot, funding, asset context."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

from tradingagents.dataflows.hyperliquid_client import HyperliquidClient


@pytest.fixture
def hl(tmp_path):
    """HyperliquidClient with patched cache dir."""
    with patch("tradingagents.dataflows.base_client.BaseDataClient.__init__") as mock_init:
        mock_init.return_value = None
        client = HyperliquidClient()
        client.cache_dir = tmp_path
        client.cache_ttl = 3600
        client.session = MagicMock()
        client.MAX_RETRIES = 3
        client.BACKOFF_BASE = 2
        return client


# ── OHLCV ────────────────────────────────────────────────────────────

class TestGetOHLCV:
    def test_parses_candles(self, hl):
        raw_candles = [
            {"t": 1700000000000, "o": "100", "h": "105", "l": "99", "c": "103", "v": "500"},
            {"t": 1700086400000, "o": "103", "h": "108", "l": "101", "c": "107", "v": "600"},
        ]
        with patch.object(hl, "_post_request", return_value=raw_candles):
            df = hl.get_ohlcv("BTC", "1d", "2023-11-14", "2023-11-16")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert df.iloc[0]["close"] == 103.0

    def test_empty_response_returns_empty_df(self, hl):
        with patch.object(hl, "_post_request", return_value=[]):
            df = hl.get_ohlcv("BTC", "1d", "2023-11-14", "2023-11-16")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert "close" in df.columns

    def test_deduplicates_by_timestamp(self, hl):
        dup_candles = [
            {"t": 1700000000000, "o": "100", "h": "105", "l": "99", "c": "103", "v": "500"},
            {"t": 1700000000000, "o": "100", "h": "105", "l": "99", "c": "103", "v": "500"},
        ]
        with patch.object(hl, "_post_request", return_value=dup_candles):
            df = hl.get_ohlcv("BTC", "1d", "2023-11-14", "2023-11-16")
        assert len(df) == 1

    def test_no_dates_defaults_30_days(self, hl):
        with patch.object(hl, "_post_request", return_value=[]) as mock_post:
            hl.get_ohlcv("BTC", "1d")
        assert mock_post.called

    def test_api_error_returns_empty_df(self, hl):
        with patch.object(hl, "_post_request", side_effect=ConnectionError("down")):
            df = hl.get_ohlcv("BTC", "1d", "2023-11-14", "2023-11-16")
        assert df.empty


# ── Spot Price ───────────────────────────────────────────────────────

class TestGetSpotPrice:
    def test_returns_float(self, hl):
        with patch.object(hl, "_post_request", return_value={"BTC": "67123.45"}):
            price = hl.get_spot_price("BTC")
        assert price == pytest.approx(67123.45)

    def test_unknown_coin_returns_none(self, hl):
        with patch.object(hl, "_post_request", return_value={"ETH": "3500"}):
            price = hl.get_spot_price("XYZ_FAKE")
        assert price is None

    def test_api_error_returns_none(self, hl):
        with patch.object(hl, "_post_request", side_effect=ConnectionError("down")):
            price = hl.get_spot_price("BTC")
        assert price is None


# ── Asset Context ────────────────────────────────────────────────────

class TestGetAssetContext:
    def _mock_meta_response(self):
        return [
            {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
            [
                {
                    "funding": "0.0001",
                    "premium": "0.0005",
                    "openInterest": "1500000",
                    "dayNtlVlm": "800000000",
                    "markPx": "67000",
                    "oraclePx": "67050",
                    "prevDayPx": "66500",
                },
                {
                    "funding": "0.0002",
                    "premium": "0.0003",
                    "openInterest": "500000",
                    "dayNtlVlm": "300000000",
                    "markPx": "3500",
                    "oraclePx": "3505",
                    "prevDayPx": "3480",
                },
            ],
        ]

    def test_returns_context(self, hl):
        with patch.object(hl, "_post_request", return_value=self._mock_meta_response()):
            ctx = hl.get_asset_context("BTC")
        assert ctx is not None
        assert ctx["coin"] == "BTC"
        assert ctx["funding"] == pytest.approx(0.0001)
        assert ctx["markPx"] == pytest.approx(67000.0)

    def test_unknown_coin(self, hl):
        with patch.object(hl, "_post_request", return_value=self._mock_meta_response()):
            ctx = hl.get_asset_context("DOGE")
        assert ctx is None

    def test_malformed_response(self, hl):
        with patch.object(hl, "_post_request", return_value={"unexpected": True}):
            ctx = hl.get_asset_context("BTC")
        assert ctx is None


# ── Funding History ──────────────────────────────────────────────────

class TestGetFundingHistory:
    def test_parses_history(self, hl):
        raw = [
            {"time": 1700000000000, "coin": "BTC", "fundingRate": "0.0001"},
            {"time": 1700028800000, "coin": "BTC", "fundingRate": "-0.00005"},
        ]
        with patch.object(hl, "_post_request", return_value=raw):
            df = hl.get_funding_history("BTC", "2023-11-14", "2023-11-16")
        assert len(df) == 2
        assert "funding_rate" in df.columns
        assert df.iloc[0]["funding_rate"] == pytest.approx(0.0001)

    def test_empty_returns_empty_df(self, hl):
        with patch.object(hl, "_post_request", return_value=[]):
            df = hl.get_funding_history("BTC", "2023-11-14", "2023-11-16")
        assert df.empty

    def test_api_error_returns_empty_df(self, hl):
        with patch.object(hl, "_post_request", side_effect=ConnectionError("fail")):
            df = hl.get_funding_history("BTC", "2023-11-14", "2023-11-16")
        assert df.empty


# ── Predicted Funding ────────────────────────────────────────────────

class TestGetPredictedFunding:
    def test_returns_rate(self, hl):
        meta_resp = [
            {"universe": [{"name": "BTC"}]},
            [{"funding": "0.0003", "premium": "0.001", "openInterest": "1000000"}],
        ]
        with patch.object(hl, "_post_request", return_value=meta_resp):
            result = hl.get_predicted_funding("BTC")
        assert result is not None
        assert result["predicted_rate"] == pytest.approx(0.0003)

    def test_unknown_coin(self, hl):
        meta_resp = [{"universe": [{"name": "ETH"}]}, [{"funding": "0.0001"}]]
        with patch.object(hl, "_post_request", return_value=meta_resp):
            result = hl.get_predicted_funding("BTC")
        assert result is None


# ── Realized Funding ─────────────────────────────────────────────────

class TestGetRealizedFunding:
    def test_returns_latest_rate(self, hl):
        raw = [
            {"time": 1700000000000, "coin": "BTC", "fundingRate": "0.0001"},
            {"time": 1700028800000, "coin": "BTC", "fundingRate": "0.00015"},
        ]
        with patch.object(hl, "_post_request", return_value=raw):
            rate = hl.get_realized_funding("BTC")
        assert rate == pytest.approx(0.00015)

    def test_empty_history_returns_none(self, hl):
        with patch.object(hl, "_post_request", return_value=[]):
            rate = hl.get_realized_funding("BTC")
        assert rate is None
