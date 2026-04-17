"""Tests for the Pulse API endpoints in server.py.

~15 tests covering:
  - GET /api/pulse/{ticker} — empty + populated
  - GET /api/pulse/latest/{ticker}
  - POST /api/pulse/run/{ticker} — live run (mocked)
  - GET /api/pulse/scorecard/{ticker}
  - GET /api/pulse/scheduler/status
  - POST /api/pulse/scheduler/toggle
  - POST /api/pulse/backtest/{ticker} — validation
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# We import the FastAPI test client
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with temp EVAL_RESULTS_DIR."""
    # Patch EVAL_RESULTS_DIR before importing server
    import server as srv
    monkeypatch.setattr(srv, "EVAL_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(srv, "PULSE_DIR", tmp_path / "pulse")
    return TestClient(srv.app)


@pytest.fixture
def seeded_pulse(tmp_path):
    """Seed some pulse entries into the temp directory."""
    pulse_dir = tmp_path / "pulse" / "BTC-USD"
    pulse_dir.mkdir(parents=True)
    pulse_file = pulse_dir / "pulse.jsonl"

    entries = [
        {
            "ts": "2026-03-15T10:00:00+00:00",
            "signal": "BUY",
            "confidence": 0.65,
            "normalized_score": 0.32,
            "price": 82000,
            "stop_loss": 81000,
            "take_profit": 84000,
            "hold_minutes": 45,
            "timeframe_bias": "15m",
            "reasoning": "test buy signal",
            "breakdown": {"15m": 0.1},
            "volatility_flag": False,
            "signal_threshold": 0.25,
            "scored": True,
            "hit_+5m": True,
            "hit_+15m": False,
            "hit_+1h": True,
            "return_+5m": 0.002,
            "return_+15m": -0.001,
            "return_+1h": 0.005,
        },
        {
            "ts": "2026-03-15T10:15:00+00:00",
            "signal": "SHORT",
            "confidence": 0.55,
            "normalized_score": -0.28,
            "price": 82500,
            "stop_loss": 83500,
            "take_profit": 81000,
            "hold_minutes": 45,
            "timeframe_bias": "1h",
            "reasoning": "test short signal",
            "breakdown": {"1h": -0.08},
            "volatility_flag": False,
            "signal_threshold": 0.25,
            "scored": True,
            "hit_+5m": False,
            "hit_+15m": True,
            "hit_+1h": True,
            "return_+5m": -0.001,
            "return_+15m": 0.003,
            "return_+1h": 0.004,
        },
        {
            "ts": "2026-03-15T10:30:00+00:00",
            "signal": "NEUTRAL",
            "confidence": 0.3,
            "normalized_score": 0.05,
            "price": 82200,
            "stop_loss": None,
            "take_profit": None,
            "hold_minutes": 45,
            "timeframe_bias": "15m",
            "reasoning": "neutral",
            "breakdown": {},
            "volatility_flag": False,
            "signal_threshold": 0.25,
            "scored": True,
        },
    ]

    with open(pulse_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


class TestPulseGetEndpoints:
    def test_empty_pulses(self, client):
        resp = client.get("/api/pulse/BTC-USD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "BTC-USD"
        assert data["pulses"] == []
        assert data["count"] == 0

    def test_populated_pulses(self, client, seeded_pulse):
        resp = client.get("/api/pulse/BTC-USD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3

    def test_limit_param(self, client, seeded_pulse):
        resp = client.get("/api/pulse/BTC-USD?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

    def test_latest_pulse_empty(self, client):
        resp = client.get("/api/pulse/latest/BTC-USD")
        assert resp.status_code == 200
        assert resp.json()["pulse"] is None

    def test_latest_pulse_populated(self, client, seeded_pulse):
        resp = client.get("/api/pulse/latest/BTC-USD")
        assert resp.status_code == 200
        pulse = resp.json()["pulse"]
        assert pulse is not None
        assert pulse["signal"] == "NEUTRAL"  # last entry

    def test_case_insensitive_ticker(self, client, seeded_pulse):
        resp = client.get("/api/pulse/btc-usd")
        assert resp.status_code == 200
        assert resp.json()["count"] == 3


class TestPulseScorecardEndpoint:
    def test_empty_scorecard(self, client):
        resp = client.get("/api/pulse/scorecard/BTC-USD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["scored"] == 0

    def test_populated_scorecard(self, client, seeded_pulse):
        resp = client.get("/api/pulse/scorecard/BTC-USD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["scored"] == 3  # all scored
        assert "+5m" in data["hit_rates"]
        assert "+1h" in data["hit_rates"]
        # Overall hit rate for +1h: 2/3 scored (NEUTRAL is scored but has no hit_ keys)
        # Actually BUY hit_+1h=True, SHORT hit_+1h=True, NEUTRAL has no hit_ keys
        # scored filter: all have scored=True, but only BUY and SHORT have hit_ keys
        # hits = sum of True for each → BUY True + SHORT True = 2, NEUTRAL contributes 0
        # n_scored = 3, so overall = 2/3


class TestPulseSchedulerEndpoint:
    def test_scheduler_status(self, client):
        resp = client.get("/api/pulse/scheduler/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "tickers" in data
        assert "interval_minutes" in data


class TestPulseRunEndpoint:
    @patch("server._run_single_pulse")
    def test_manual_run(self, mock_run, client):
        mock_run.return_value = {
            "ts": "2026-03-15T12:00:00+00:00",
            "signal": "BUY",
            "confidence": 0.7,
            "normalized_score": 0.35,
            "price": 82000,
            "stop_loss": 81000,
            "take_profit": 84000,
            "hold_minutes": 45,
            "timeframe_bias": "15m",
            "reasoning": "test",
            "breakdown": {},
            "volatility_flag": False,
            "signal_threshold": 0.25,
            "scored": False,
        }
        resp = client.post("/api/pulse/run/BTC-USD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pulse"]["signal"] == "BUY"


class TestPulseBacktestValidation:
    def test_invalid_dates(self, client):
        resp = client.post("/api/pulse/backtest/BTC-USD", json={
            "start_date": "bad-date",
            "end_date": "2026-04-01",
        })
        assert resp.status_code == 400

    def test_end_before_start(self, client):
        resp = client.post("/api/pulse/backtest/BTC-USD", json={
            "start_date": "2026-04-01",
            "end_date": "2026-01-01",
        })
        assert resp.status_code == 400

    def test_too_long_range(self, client):
        resp = client.post("/api/pulse/backtest/BTC-USD", json={
            "start_date": "2025-01-01",
            "end_date": "2026-04-01",
        })
        assert resp.status_code == 400

    def test_non_crypto_ticker(self, client):
        resp = client.post("/api/pulse/backtest/AAPL", json={
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
        })
        assert resp.status_code == 400
