"""Tests for FearGreedClient with mocked API responses."""

import pytest
from unittest.mock import patch
from tradingagents.dataflows.fear_greed_client import FearGreedClient


@pytest.fixture
def client(tmp_path):
    c = FearGreedClient(cache_ttl=60)
    c.cache_dir = tmp_path / "fear_greed"
    c.cache_dir.mkdir()
    return c


class TestFearGreedClient:

    def test_get_index(self, client):
        mock_data = {
            "data": [
                {"value": "25", "value_classification": "Extreme Fear", "timestamp": "1704067200"},
                {"value": "30", "value_classification": "Fear", "timestamp": "1704153600"},
                {"value": "55", "value_classification": "Greed", "timestamp": "1704240000"},
            ]
        }
        with patch.object(client, "_request", return_value=mock_data):
            df = client.get_index(limit=3)
        assert len(df) == 3
        assert df.iloc[0]["value"] == 25

    def test_get_sentiment_report_extreme_fear(self, client):
        mock_data = {
            "data": [
                {"value": "15", "value_classification": "Extreme Fear", "timestamp": str(1704067200 + i*86400)}
                for i in range(30)
            ]
        }
        with patch.object(client, "_request", return_value=mock_data):
            report = client.get_sentiment_report(30)
        assert "Extreme Fear" in report
        assert "historically precedes rallies" in report

    def test_get_sentiment_report_extreme_greed(self, client):
        mock_data = {
            "data": [
                {"value": "85", "value_classification": "Extreme Greed", "timestamp": str(1704067200 + i*86400)}
                for i in range(30)
            ]
        }
        with patch.object(client, "_request", return_value=mock_data):
            report = client.get_sentiment_report(30)
        assert "Extreme Greed" in report
        assert "historically precedes corrections" in report

    def test_empty_response(self, client):
        with patch.object(client, "_request", return_value={"data": []}):
            df = client.get_index(limit=30)
        assert df.empty
