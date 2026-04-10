"""Tests for BaseDataClient — caching, retries, rate limiting."""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tradingagents.dataflows.base_client import BaseDataClient


@pytest.fixture
def client(tmp_path):
    """BaseDataClient that uses a tmp directory for cache."""
    with patch.object(BaseDataClient, "__init__", lambda self, *a, **k: None):
        c = BaseDataClient.__new__(BaseDataClient)
        c.cache_dir = tmp_path
        c.cache_ttl = 60
        c.session = MagicMock()
        c.MAX_RETRIES = 3
        c.BACKOFF_BASE = 2
        return c


class TestCacheKey:
    def test_deterministic(self, client):
        k1 = client._cache_key("prefix", {"a": 1})
        k2 = client._cache_key("prefix", {"a": 1})
        assert k1 == k2

    def test_different_params(self, client):
        k1 = client._cache_key("prefix", {"a": 1})
        k2 = client._cache_key("prefix", {"a": 2})
        assert k1 != k2


class TestReadCache:
    def test_hit(self, client, tmp_path):
        path = tmp_path / "test.json"
        path.write_text(json.dumps({"key": "value"}))
        result = client._read_cache(path)
        assert result == {"key": "value"}

    def test_miss_nonexistent(self, client, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert client._read_cache(path) is None

    def test_expired(self, client, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"key": "value"}))
        client.cache_ttl = 0  # Expire immediately
        import os
        os.utime(path, (time.time() - 100, time.time() - 100))
        assert client._read_cache(path) is None

    def test_corrupt_json(self, client, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json{{{")
        assert client._read_cache(path) is None


class TestWriteCache:
    def test_writes_json(self, client, tmp_path):
        path = tmp_path / "out.json"
        client._write_cache(path, {"foo": "bar"})
        assert json.loads(path.read_text()) == {"foo": "bar"}

    def test_write_failure_graceful(self, client, tmp_path, caplog):
        path = tmp_path / "no_dir" / "sub" / "deep.json"
        # Parent doesn't exist — should warn, not crash
        import logging
        with caplog.at_level(logging.WARNING):
            client._write_cache(path, {"a": 1})
        assert "Cache write failed" in caplog.text


class TestRequest:
    def test_cache_hit_returns_cached(self, client, tmp_path):
        # Pre-populate cache
        key = client._cache_key("http://test.com", {})
        key.write_text(json.dumps({"cached": True}))
        result = client._request("http://test.com")
        assert result == {"cached": True}
        client.session.get.assert_not_called()

    def test_network_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"live": True}
        mock_resp.raise_for_status = MagicMock()
        client.session.get.return_value = mock_resp
        # Ensure no cache
        client.cache_ttl = 0
        result = client._request("http://test.com", cache_prefix="test")
        assert result == {"live": True}

    @patch("tradingagents.dataflows.base_client.time.sleep")
    def test_rate_limit_retry(self, mock_sleep, client):
        rate_resp = MagicMock()
        rate_resp.status_code = 429

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}
        ok_resp.raise_for_status = MagicMock()

        client.session.get.side_effect = [rate_resp, ok_resp]
        client.cache_ttl = 0
        result = client._request("http://test.com", cache_prefix="rl")
        assert result == {"ok": True}
        assert mock_sleep.called

    @patch("tradingagents.dataflows.base_client.time.sleep")
    def test_all_retries_exhausted(self, mock_sleep, client):
        import requests
        client.session.get.side_effect = requests.ConnectionError("fail")
        client.cache_ttl = 0
        with pytest.raises(requests.ConnectionError):
            client._request("http://test.com", cache_prefix="fail")
        assert client.session.get.call_count == 3
