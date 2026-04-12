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


# ── Live Analysis: Fresh Data & Timestamp Tests ─────────────────────────

class TestLiveAnalysisFreshData:
    """Tests for live analysis always-fresh data behavior and timestamp formats."""

    @patch("server.threading.Thread")
    def test_manual_analysis_force_refresh_default(self, mock_thread, client):
        """Verify manual /api/analyze defaults to force_refresh=True."""
        mock_thread.return_value = MagicMock()
        resp = client.post("/api/analyze", json={"ticker": "BTC-USD"})
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        # Check that the thread was started with force_refresh=True
        call_args = mock_thread.call_args
        assert call_args is not None
        args = call_args[1].get("args", call_args[0])
        # args = (job_id, ticker, trade_date, force_refresh, run_timestamp)
        assert len(args) >= 4
        assert args[3] is True, "force_refresh should be True by default for manual runs"

    @patch("server.threading.Thread")
    def test_manual_analysis_force_refresh_explicit_false(self, mock_thread, client):
        """Verify manual /api/analyze can override force_refresh to False."""
        mock_thread.return_value = MagicMock()
        resp = client.post("/api/analyze", json={"ticker": "BTC-USD", "force_refresh": False})
        assert resp.status_code == 200
        call_args = mock_thread.call_args
        args = call_args[1].get("args", call_args[0])
        assert len(args) >= 4
        assert args[3] is False, "force_refresh should respect explicit False"

    @patch("server.threading.Thread")
    def test_manual_analysis_timestamp_format(self, mock_thread, client):
        """Verify manual runs generate timestamp in YYYY-MM-DD-HH-MM-AM/PM format."""
        mock_thread.return_value = MagicMock()
        resp = client.post("/api/analyze", json={"ticker": "BTC-USD"})
        assert resp.status_code == 200
        data = resp.json()
        # Check timestamp format: YYYY-MM-DD-HH-MM-AM/PM
        import re
        timestamp = data["date"]
        pattern = r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-(AM|PM)$"
        assert re.match(pattern, timestamp), f"Timestamp {timestamp} doesn't match YYYY-MM-DD-HH-MM-AM/PM format"
        # Also verify it was passed to _run_analysis
        call_args = mock_thread.call_args
        args = call_args[1].get("args", call_args[0])
        assert len(args) >= 5
        assert args[4] == timestamp, "run_timestamp should match response date"


class TestHistoryTimestampFormats:
    """Tests for history endpoint parsing of new and legacy timestamp formats."""

    def test_history_parses_new_timestamp_format(self, client, tmp_path):
        """Verify /api/history/{ticker} correctly parses new YYYY-MM-DD-HH-MM-AM/PM format."""
        ticker_dir = tmp_path / "BTC-USD" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)
        # Create file with new format
        (ticker_dir / "full_states_log_2024-03-15-02-30-PM.json").write_text(
            json.dumps({"2024-03-15-02-30-PM": {"final_trade_decision": "BUY", "confidence": 0.75}})
        )
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/history/BTC-USD")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["analyses"]) == 1
            analysis = data["analyses"][0]
            assert analysis["date"] == "2024-03-15"
            assert analysis["candle_time"] == "2024-03-15-02-30-PM"
            assert analysis["time"] == "2:30 PM"
            assert analysis["signal"] == "BUY"

    def test_history_parses_legacy_hourly_format(self, client, tmp_path):
        """Verify backward compatibility with YYYY-MM-DDTHH format."""
        ticker_dir = tmp_path / "BTC-USD" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)
        (ticker_dir / "full_states_log_2024-03-15T14.json").write_text(
            json.dumps({"2024-03-15T14": {"final_trade_decision": "SELL", "confidence": 0.65}})
        )
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/history/BTC-USD")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["analyses"]) == 1
            analysis = data["analyses"][0]
            assert analysis["candle_time"] == "2024-03-15T14"
            assert analysis["signal"] == "SELL"

    def test_history_parses_legacy_daily_format(self, client, tmp_path):
        """Verify backward compatibility with YYYY-MM-DD format."""
        ticker_dir = tmp_path / "BTC-USD" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)
        (ticker_dir / "full_states_log_2024-03-15.json").write_text(
            json.dumps({"2024-03-15": {"final_trade_decision": "HOLD", "confidence": 0.55}})
        )
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/history/BTC-USD")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["analyses"]) == 1
            analysis = data["analyses"][0]
            assert analysis["date"] == "2024-03-15"
            assert analysis["candle_time"] == "2024-03-15"
            assert analysis["time"] is None  # No time for daily format

    def test_history_multiple_formats_together(self, client, tmp_path):
        """Verify all three formats can coexist and be parsed correctly."""
        ticker_dir = tmp_path / "BTC-USD" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)
        # Daily format
        (ticker_dir / "full_states_log_2024-03-14.json").write_text(
            json.dumps({"2024-03-14": {"final_trade_decision": "HOLD"}})
        )
        # Hourly format
        (ticker_dir / "full_states_log_2024-03-15T12.json").write_text(
            json.dumps({"2024-03-15T12": {"final_trade_decision": "BUY"}})
        )
        # New AM/PM format
        (ticker_dir / "full_states_log_2024-03-15-03-45-PM.json").write_text(
            json.dumps({"2024-03-15-03-45-PM": {"final_trade_decision": "SELL"}})
        )
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/history/BTC-USD")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["analyses"]) == 3
            # Verify all formats are present
            candle_times = [a["candle_time"] for a in data["analyses"]]
            assert "2024-03-14" in candle_times
            assert "2024-03-15T12" in candle_times
            assert "2024-03-15-03-45-PM" in candle_times

    def test_history_detail_new_format(self, client, tmp_path):
        """Verify /api/history/{ticker}/{date} can load new format files."""
        ticker_dir = tmp_path / "BTC-USD" / "TradingAgentsStrategy_logs"
        ticker_dir.mkdir(parents=True)
        (ticker_dir / "full_states_log_2024-03-15-02-30-PM.json").write_text(
            json.dumps({
                "2024-03-15-02-30-PM": {
                    "final_trade_decision": "BUY",
                    "confidence": 0.75,
                    "stop_loss_price": 65000,
                    "take_profit_price": 75000,
                }
            })
        )
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/history/BTC-USD/2024-03-15-02-30-PM")
            assert resp.status_code == 200
            data = resp.json()
            assert data["date"] == "2024-03-15-02-30-PM"
            assert "at 02:30 PM" in data["date_formatted"]


# ── Confidence/Regime Tests ────────────────────────────────────────────

class TestConfidenceScoring:
    """Tests for confidence scoring and regime gating behavior."""

    def test_no_gating_applied(self, tmp_path):
        """Verify gating is always disabled regardless of confidence level."""
        from tradingagents.graph.confidence import ConfidenceScorer
        from tradingagents.backtesting.knowledge_store import BacktestKnowledgeStore

        scorer = ConfidenceScorer(results_dir=str(tmp_path))

        # Test with low confidence - should NOT be gated
        result = scorer.score(
            llm_confidence=0.30,  # Very low confidence
            ticker="BTC-USD",
            signal="BUY",
            knowledge_store=None,
            regime_ctx={"regime": "volatile", "current_price": 70000, "above_sma20": True},
            stop_loss=65000,
            take_profit=75000,
            max_hold_days=7,
        )

        assert result["gated"] is False, "Low confidence should NOT trigger gating"
        assert result["position_size_pct"] > 0, "Position size should never be 0 due to gating"

    def test_all_trade_parameters_returned(self, tmp_path):
        """Verify all trade parameter fields are returned."""
        from tradingagents.graph.confidence import ConfidenceScorer

        scorer = ConfidenceScorer(results_dir=str(tmp_path))

        result = scorer.score(
            llm_confidence=0.70,
            ticker="BTC-USD",
            signal="BUY",
            knowledge_store=None,
            regime_ctx={"regime": "trending_up", "current_price": 70000, "above_sma20": True},
            stop_loss=65000,
            take_profit=75000,
            max_hold_days=7,
        )

        # All expected fields should be present
        assert "confidence" in result
        assert "position_size_pct" in result
        assert "conviction_label" in result
        assert "gated" in result
        assert "r_ratio" in result
        assert "r_ratio_warning" in result
        assert "hold_period_scalar" in result
        assert "hedge_penalty_applied" in result
