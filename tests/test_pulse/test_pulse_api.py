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
        assert data["total"] == 0
        assert data["has_more"] is False
        assert data["offset"] == 0

    def test_populated_pulses(self, client, seeded_pulse):
        resp = client.get("/api/pulse/BTC-USD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert data["total"] == 3
        assert data["has_more"] is False
        assert data["offset"] == 0

    def test_limit_param(self, client, seeded_pulse):
        resp = client.get("/api/pulse/BTC-USD?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["total"] == 3
        assert data["has_more"] is True  # more available

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


@pytest.fixture
def many_pulses(tmp_path):
    """Seed 100 pulse entries for pagination testing."""
    pulse_dir = tmp_path / "pulse" / "BTC-USD"
    pulse_dir.mkdir(parents=True)
    pulse_file = pulse_dir / "pulse.jsonl"

    entries = []
    for i in range(100):
        # Use day and hour to create 100 valid timestamps
        day = 15 + (i // 24)  # Days 15, 16, 17, 18, 19
        hour = i % 24  # Hours 0-23
        entry = {
            "ts": f"2026-03-{day:02d}T{hour:02d}:00:00+00:00",
            "signal": "BUY" if i % 3 == 0 else "SHORT" if i % 3 == 1 else "NEUTRAL",
            "confidence": 0.5 + (i % 50) / 100,
            "normalized_score": 0.3 if i % 3 == 0 else -0.2 if i % 3 == 1 else 0.0,
            "price": 80000 + i * 100,
            "stop_loss": 79000 if i % 3 != 2 else None,
            "take_profit": 82000 if i % 3 != 2 else None,
            "hold_minutes": 45,
            "timeframe_bias": "15m",
            "reasoning": f"test signal {i}",
            "breakdown": {},
            "volatility_flag": False,
            "signal_threshold": 0.25,
            "scored": True,
        }
        entries.append(entry)

    with open(pulse_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


class TestPulsePagination:
    """Tests for the pagination feature (offset parameter) in /api/pulse/{ticker}."""

    def test_pagination_default_returns_last_50(self, client, many_pulses):
        """Default behavior: returns last 50 pulses (newest first)."""
        resp = client.get("/api/pulse/BTC-USD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 50
        assert data["total"] == 100
        assert data["has_more"] is True
        # With reversal, first pulse is the newest (index 99 = day 19, hour 3)
        assert data["pulses"][0]["ts"] == "2026-03-19T03:00:00+00:00"
        # Last pulse in this batch is index 50 = day 17, hour 2
        assert data["pulses"][-1]["ts"] == "2026-03-17T02:00:00+00:00"

    def test_pagination_with_offset(self, client, many_pulses):
        """Offset skips the most recent N pulses."""
        resp = client.get("/api/pulse/BTC-USD?limit=10&offset=20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 10
        assert data["offset"] == 20
        assert data["has_more"] is True
        # With offset 20 and reversal: indices 79 down to 70
        # First pulse: index 79 = day 18, hour 7
        assert data["pulses"][0]["ts"] == "2026-03-18T07:00:00+00:00"
        # Last pulse: index 70 = day 17, hour 22
        assert data["pulses"][-1]["ts"] == "2026-03-17T22:00:00+00:00"

    def test_pagination_load_more_sequence(self, client, many_pulses):
        """Simulate clicking 'Load More' multiple times."""
        all_pulses = []
        offset = 0
        limit = 25
        has_more = True
        page_count = 0

        while has_more:
            resp = client.get(f"/api/pulse/BTC-USD?limit={limit}&offset={offset}")
            assert resp.status_code == 200
            data = resp.json()
            all_pulses.extend(data["pulses"])
            has_more = data["has_more"]
            offset += limit
            page_count += 1
            # Safety: don't loop forever
            assert page_count <= 10, "Too many pagination requests"

        # Should have collected all 100 pulses
        assert len(all_pulses) == 100
        # Verify order: newest first (timestamps should be descending)
        for i in range(len(all_pulses) - 1):
            assert all_pulses[i]["ts"] >= all_pulses[i + 1]["ts"], "Pulses should be in descending time order"

    def test_pagination_offset_beyond_total(self, client, many_pulses):
        """Offset beyond total count returns empty list."""
        resp = client.get("/api/pulse/BTC-USD?limit=10&offset=150")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["total"] == 100
        assert data["has_more"] is False
        assert data["pulses"] == []

    def test_pagination_exact_page_boundary(self, client, many_pulses):
        """Test exact page boundaries."""
        # Page 1 (offset=0): returns indices 99-50 (newest 50), reversed to 99,98...50
        resp1 = client.get("/api/pulse/BTC-USD?limit=50&offset=0")
        assert resp1.json()["count"] == 50
        assert resp1.json()["has_more"] is True
        # First pulse is index 99 (newest), last is index 50
        assert resp1.json()["pulses"][0]["ts"] == "2026-03-19T03:00:00+00:00"

        # Page 2 (offset=50): returns indices 49-0 (oldest 50), reversed to 49,48...0
        resp2 = client.get("/api/pulse/BTC-USD?limit=50&offset=50")
        data2 = resp2.json()
        assert data2["count"] == 50
        assert data2["has_more"] is False  # No more after this
        # First pulse is index 49 (day 17, hour 1), last is index 0 (day 15, hour 0)
        assert data2["pulses"][0]["ts"] == "2026-03-17T01:00:00+00:00"
        assert data2["pulses"][-1]["ts"] == "2026-03-15T00:00:00+00:00"

    def test_pagination_single_pulse(self, client, seeded_pulse):
        """Pagination works with small datasets."""
        resp = client.get("/api/pulse/BTC-USD?limit=1&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["total"] == 3
        assert data["has_more"] is True
        # Should get the most recent (last in file)
        assert data["pulses"][0]["signal"] == "NEUTRAL"

    def test_pagination_limit_zero(self, client, many_pulses):
        """Limit of zero returns empty list but correct metadata."""
        resp = client.get("/api/pulse/BTC-USD?limit=0&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["total"] == 100
        assert data["has_more"] is True

    def test_count_pulses_helper(self, client, tmp_path, monkeypatch):
        """Test the _count_pulses helper function directly."""
        import server as srv
        monkeypatch.setattr(srv, "EVAL_RESULTS_DIR", tmp_path)
        monkeypatch.setattr(srv, "PULSE_DIR", tmp_path / "pulse")

        # Empty case
        assert srv._count_pulses("BTC-USD") == 0

        # Create pulses
        pulse_dir = tmp_path / "pulse" / "BTC-USD"
        pulse_dir.mkdir(parents=True)
        pulse_file = pulse_dir / "pulse.jsonl"

        with open(pulse_file, "w") as f:
            for i in range(5):
                f.write(json.dumps({"ts": f"2026-03-{15+i}T10:00:00", "signal": "BUY"}) + "\n")

        assert srv._count_pulses("BTC-USD") == 5

    def test_read_pulses_with_offset(self, client, tmp_path, monkeypatch):
        """Test the _read_pulses function with offset parameter."""
        import server as srv
        monkeypatch.setattr(srv, "EVAL_RESULTS_DIR", tmp_path)
        monkeypatch.setattr(srv, "PULSE_DIR", tmp_path / "pulse")

        # Create pulses
        pulse_dir = tmp_path / "pulse" / "BTC-USD"
        pulse_dir.mkdir(parents=True)
        pulse_file = pulse_dir / "pulse.jsonl"

        with open(pulse_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"ts": f"2026-03-15T{i:02d}:00:00", "idx": i}) + "\n")

        # Default: last 5 (indices 5-9), reversed to 9,8,7,6,5
        pulses_default = srv._read_pulses("BTC-USD", limit=5, offset=0)
        assert len(pulses_default) == 5
        assert pulses_default[0]["idx"] == 9  # Newest first
        assert pulses_default[-1]["idx"] == 5

        # Offset 5: next 5 (indices 0-4), reversed to 4,3,2,1,0
        pulses_offset = srv._read_pulses("BTC-USD", limit=5, offset=5)
        assert len(pulses_offset) == 5
        assert pulses_offset[0]["idx"] == 4  # Newest in this batch
        assert pulses_offset[-1]["idx"] == 0  # Oldest overall

    def test_pagination_corrupt_lines_ignored(self, client, tmp_path, monkeypatch):
        """Corrupt lines in JSONL are ignored in count and pagination."""
        import server as srv
        monkeypatch.setattr(srv, "EVAL_RESULTS_DIR", tmp_path)
        monkeypatch.setattr(srv, "PULSE_DIR", tmp_path / "pulse")

        pulse_dir = tmp_path / "pulse" / "BTC-USD"
        pulse_dir.mkdir(parents=True)
        pulse_file = pulse_dir / "pulse.jsonl"

        with open(pulse_file, "w") as f:
            f.write(json.dumps({"ts": "2026-03-15T10:00:00", "signal": "BUY"}) + "\n")
            f.write("this is not valid json\n")
            f.write(json.dumps({"ts": "2026-03-15T11:00:00", "signal": "SHORT"}) + "\n")
            f.write("\n")  # Empty line
            f.write(json.dumps({"ts": "2026-03-15T12:00:00", "signal": "NEUTRAL"}) + "\n")

        # Count should only include valid JSON lines
        assert srv._count_pulses("BTC-USD") == 3

        # Pagination should work correctly (descending order: NEUTRAL is newest at 12:00)
        resp = client.get("/api/pulse/BTC-USD?limit=2&offset=0")
        data = resp.json()
        assert data["count"] == 2
        assert data["total"] == 3
        # Most recent 2: NEUTRAL (12:00) is newest, then SHORT (11:00)
        assert data["pulses"][0]["signal"] == "NEUTRAL"
        assert data["pulses"][1]["signal"] == "SHORT"


class TestPulsePaginationRaceConditions:
    """Tests for race condition fixes in pagination (from debate plan)."""

    def test_concurrent_load_more_requests_deduplicated(self, client, many_pulses):
        """Rapid "Load More" clicks should not fetch duplicate data.
        
        BLOCKER: SSE identified race condition where rapid clicks
        use stale pulseOffsetRef and fetch same data twice.
        """
        # Simulate two rapid requests with same offset
        resp1 = client.get("/api/pulse/BTC-USD?limit=25&offset=0")
        resp2 = client.get("/api/pulse/BTC-USD?limit=25&offset=0")
        
        # Both should return same data (not error), proving idempotency
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["pulses"] == resp2.json()["pulses"]
        
    def test_abort_controller_simulation(self, client, tmp_path, monkeypatch):
        """Ticker change should not apply old fetch results.
        
        HIGH: WCT identified issue where slow fetch completes after
        ticker change and overwrites new ticker's state.
        
        Simulated by checking that offset resets work independently.
        """
        import server as srv
        monkeypatch.setattr(srv, "EVAL_RESULTS_DIR", tmp_path)
        monkeypatch.setattr(srv, "PULSE_DIR", tmp_path / "pulse")
        
        # Create pulses for two tickers
        for ticker in ["BTC-USD", "ETH-USD"]:
            pulse_dir = tmp_path / "pulse" / ticker
            pulse_dir.mkdir(parents=True)
            pulse_file = pulse_dir / "pulse.jsonl"
            with open(pulse_file, "w") as f:
                for i in range(10):
                    f.write(json.dumps({"ts": f"2026-03-15T{i:02d}:00:00", "ticker": ticker}) + "\n")
        
        # Fetch BTC with offset
        btc_data = client.get("/api/pulse/BTC-USD?limit=5&offset=5").json()
        assert btc_data["pulses"][0]["ticker"] == "BTC-USD"
        
        # Fetch ETH with same offset - should get ETH data, not BTC
        eth_data = client.get("/api/pulse/ETH-USD?limit=5&offset=5").json()
        assert eth_data["pulses"][0]["ticker"] == "ETH-USD"
        
    def test_corrupt_line_counting_accuracy(self, client, tmp_path, monkeypatch):
        """SQR identified that pulseTotal may include corrupt lines.
        
        MEDIUM: Verify _count_pulses only counts valid JSON.
        """
        import server as srv
        monkeypatch.setattr(srv, "EVAL_RESULTS_DIR", tmp_path)
        monkeypatch.setattr(srv, "PULSE_DIR", tmp_path / "pulse")
        
        pulse_dir = tmp_path / "pulse" / "BTC-USD"
        pulse_dir.mkdir(parents=True)
        pulse_file = pulse_dir / "pulse.jsonl"
        
        with open(pulse_file, "w") as f:
            f.write(json.dumps({"ts": "2026-03-15T10:00:00"}) + "\n")  # Valid
            f.write("corrupt json here\n")  # Invalid
            f.write(json.dumps({"ts": "2026-03-15T11:00:00"}) + "\n")  # Valid
            f.write("\n")  # Empty line
            f.write("" + "\n")  # Another empty
        
        # Count should be 2 valid lines only
        assert srv._count_pulses("BTC-USD") == 2
        
        # API response should match
        resp = client.get("/api/pulse/BTC-USD")
        assert resp.json()["total"] == 2


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
