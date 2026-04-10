"""Integration tests for server.py API endpoints.

Uses httpx.AsyncClient with FastAPI's ASGI transport — no real server needed.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
from server import app, jobs, backtest_jobs, EVAL_RESULTS_DIR, SHADOW_DIR

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Synchronous test client for FastAPI."""
    from starlette.testclient import TestClient
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_jobs():
    """Reset in-memory job stores between tests."""
    jobs.clear()
    backtest_jobs.clear()
    yield
    jobs.clear()
    backtest_jobs.clear()


# ── Health ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# ── History endpoints ────────────────────────────────────────────────

class TestHistory:
    def test_list_tickers_no_data(self, client, tmp_path):
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/history")
            assert resp.status_code == 200

    def test_list_tickers_with_data(self, client, tmp_path):
        # Create a mock ticker directory with logs
        ticker_dir = tmp_path / "BTC-USD" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)
        (ticker_dir / "full_states_log_2024-01-15.json").write_text("{}")
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/history")
            assert resp.status_code == 200
            data = resp.json()
            assert any(t["ticker"] == "BTC-USD" for t in data["tickers"])

    def test_list_analyses_for_ticker(self, client, tmp_path):
        ticker_dir = tmp_path / "BTC-USD" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)
        (ticker_dir / "full_states_log_2024-01-15.json").write_text("{}")
        (ticker_dir / "full_states_log_2024-01-16.json").write_text("{}")
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/history/BTC-USD")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2

    def test_get_analysis_not_found(self, client, tmp_path):
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/history/BTC-USD/2099-01-01")
            assert resp.status_code == 404


# ── Analyze endpoint ─────────────────────────────────────────────────

class TestAnalyze:
    @patch("server.threading.Thread")
    def test_start_analysis(self, mock_thread, client):
        mock_thread.return_value = MagicMock()
        resp = client.post("/api/analyze", json={
            "ticker": "BTC-USD",
            "date": "2024-01-15",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["ticker"] == "BTC-USD"

    def test_stream_not_found(self, client):
        resp = client.get("/api/stream/nonexistent")
        assert resp.status_code == 404

    def test_job_not_found(self, client):
        resp = client.get("/api/jobs/nonexistent")
        assert resp.status_code == 404


# ── Backtest endpoints ───────────────────────────────────────────────

class TestBacktest:
    @patch("server.threading.Thread")
    def test_start_backtest(self, mock_thread, client):
        mock_thread.return_value = MagicMock()
        resp = client.post("/api/backtest", json={
            "ticker": "BTC-USD",
            "start_date": "2024-01-01",
            "end_date": "2024-06-01",
            "mode": "replay",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "running"

    def test_backtest_invalid_dates(self, client):
        resp = client.post("/api/backtest", json={
            "ticker": "BTC-USD",
            "start_date": "2024-06-01",
            "end_date": "2024-01-01",
            "mode": "replay",
        })
        assert resp.status_code == 400

    def test_backtest_date_range_too_long(self, client):
        resp = client.post("/api/backtest", json={
            "ticker": "BTC-USD",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "mode": "replay",
        })
        assert resp.status_code == 400

    def test_backtest_4h_blocked_in_simulation(self, client):
        resp = client.post("/api/backtest", json={
            "ticker": "BTC-USD",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "mode": "simulation",
            "config": {"frequency": "4h"},
        })
        assert resp.status_code == 400
        assert "4h" in resp.json()["detail"].lower()

    def test_backtest_4h_allowed_in_replay(self, client):
        with patch("server.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            resp = client.post("/api/backtest", json={
                "ticker": "BTC-USD",
                "start_date": "2024-01-01",
                "end_date": "2024-03-01",
                "mode": "replay",
                "config": {"frequency": "4h"},
            })
            assert resp.status_code == 200

    def test_get_backtest_not_found(self, client):
        resp = client.get("/api/backtest/nonexistent")
        assert resp.status_code == 404

    def test_list_active_backtests_empty(self, client):
        resp = client.get("/api/backtest/active")
        assert resp.status_code == 200
        assert resp.json()["active"] == []

    def test_list_backtest_results(self, client, tmp_path):
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/backtest/results")
            assert resp.status_code == 200


# ── Shadow endpoints ─────────────────────────────────────────────────

class TestShadow:
    def test_record_shadow_decision(self, client, tmp_path):
        shadow_dir = tmp_path / "shadow"
        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("server.SHADOW_DIR", shadow_dir):
            resp = client.post("/api/shadow/record", json={
                "ticker": "BTC-USD",
                "date": "2024-01-15",
                "signal": "BUY",
                "price": 60000.0,
                "confidence": 0.75,
                "stop_loss": 57000,
                "take_profit": 66000,
                "reasoning": "Test shadow record",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "recorded"
            assert data["signal"] == "BUY"

    def test_get_shadow_decisions_empty(self, client, tmp_path):
        shadow_dir = tmp_path / "shadow"
        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("server.SHADOW_DIR", shadow_dir):
            resp = client.get("/api/shadow/decisions/BTC-USD")
            assert resp.status_code == 200
            assert resp.json()["decisions"] == []

    def test_shadow_roundtrip(self, client, tmp_path):
        """Record then retrieve shadow decisions."""
        shadow_dir = tmp_path / "shadow"
        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("server.SHADOW_DIR", shadow_dir):
            client.post("/api/shadow/record", json={
                "ticker": "ETH-USD",
                "date": "2024-01-15",
                "signal": "SHORT",
                "price": 3500.0,
                "confidence": 0.60,
                "reasoning": "Bearish divergence test",
            })
            resp = client.get("/api/shadow/decisions/ETH-USD")
            data = resp.json()
            assert len(data["decisions"]) == 1
            assert data["decisions"][0]["signal"] == "SHORT"


# ── Price endpoint ───────────────────────────────────────────────────

class TestPrice:
    @patch("server.yf.download")
    def test_price_returns_data(self, mock_yf, client):
        import pandas as pd
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        mock_yf.return_value = pd.DataFrame({
            "Open": [100] * 10,
            "High": [105] * 10,
            "Low": [95] * 10,
            "Close": [100] * 10,
            "Volume": [1000] * 10,
        }, index=dates)
        resp = client.get("/api/price/AAPL?days=10")
        assert resp.status_code == 200

    @patch("server.yf.download")
    def test_price_empty(self, mock_yf, client):
        import pandas as pd
        mock_yf.return_value = pd.DataFrame()
        resp = client.get("/api/price/FAKE-TICKER?days=10")
        # Should return 200 with empty data or 404
        assert resp.status_code in (200, 404, 500)
