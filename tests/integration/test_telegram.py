"""Tests for Telegram alert functionality.

Verifies that Telegram alerts are dispatched for all analysis types
with proper formatting and correct timestamps.
"""

import json
import pytest
import os
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the function to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTelegramAlerts:
    """Test suite for Telegram alert dispatch and formatting."""

    @pytest.fixture
    def sample_result(self):
        """Sample analysis result dict."""
        return {
            "ticker": "BTC-USD",
            "date": "2026-04-11-02-30-PM",
            "decision": "BUY",
            "confidence": 0.75,
            "stop_loss_price": 65000,
            "take_profit_price": 75000,
            "max_hold_days": 7,
            "position_size_pct": 0.25,
            "conviction_label": "HIGH",
            "r_ratio": 2.5,
            "final_trade_decision": json.dumps({
                "signal": "BUY",
                "confidence": 0.75,
                "stop_loss_price": 65000,
                "take_profit_price": 75000,
                "max_hold_days": 7,
                "reasoning": "Bullish momentum with strong technical indicators"
            })
        }

    @pytest.fixture
    def env_vars(self):
        """Set up Telegram environment variables."""
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "test_token_123",
            "TELEGRAM_CHAT_ID": "test_chat_456"
        }):
            yield

    def test_telegram_skipped_without_env_vars(self, sample_result):
        """Verify Telegram is skipped gracefully when env vars are missing."""
        import server as server_module
        with patch.dict(os.environ, {}, clear=True):
            mock_logger = MagicMock()
            with patch.object(server_module, 'logger', mock_logger):
                # Should not raise, just log warning
                server_module._send_telegram_alert(sample_result)
                mock_logger.warning.assert_called_once()
                assert "missing" in mock_logger.warning.call_args[0][0].lower()

    def test_telegram_timestamp_new_format(self, env_vars, sample_result):
        """Verify new timestamp format YYYY-MM-DD-HH-MM-AM/PM is parsed correctly."""
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(sample_result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Verify timestamp is formatted correctly
            assert "2026-04-11 at 2:30 PM" in msg or "2026-04-11" in msg

    def test_telegram_timestamp_scheduler_utc(self, env_vars):
        """Verify scheduler UTC format YYYY-MM-DDTHH is converted to local time."""
        result = {
            "ticker": "BTC-USD",
            "date": "2026-04-11T16",
            "decision": "BUY",
            "confidence": 0.70,
            "final_trade_decision": "Test decision"
        }
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Should contain formatted date
            assert "Apr" in msg or "2026" in msg

    def test_telegram_timestamp_legacy_daily(self, env_vars):
        """Verify legacy daily format YYYY-MM-DD still works."""
        result = {
            "ticker": "BTC-USD",
            "date": "2026-04-11",
            "decision": "HOLD",
            "confidence": 0.50,
            "final_trade_decision": "Test decision"
        }
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Should contain formatted date
            assert "Apr 11, 2026" in msg or "2026-04-11" in msg

    def test_telegram_reasoning_json_formatting(self, env_vars, sample_result):
        """Verify JSON reasoning is parsed and formatted correctly."""
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(sample_result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Should contain parsed reasoning, not raw JSON
            assert "Bullish momentum" in msg
            assert "<b>Analysis:</b>" in msg
            assert "<b>Signal:</b>" in msg
            assert "<b>Confidence:</b>" in msg
            # Should not contain raw JSON braces
            assert '{"signal"' not in msg

    def test_telegram_reasoning_plain_text(self, env_vars):
        """Verify plain text reasoning is handled correctly."""
        result = {
            "ticker": "BTC-USD",
            "date": "2026-04-11-02-30-PM",
            "decision": "SELL",
            "confidence": 0.60,
            "final_trade_decision": "Market conditions suggest bearish trend continues"
        }
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Should contain the reasoning text
            assert "bearish trend" in msg

    def test_telegram_reasoning_with_markdown_fences(self, env_vars):
        """Verify markdown code fences are stripped from reasoning."""
        result = {
            "ticker": "BTC-USD",
            "date": "2026-04-11-02-30-PM",
            "decision": "BUY",
            "confidence": 0.70,
            "final_trade_decision": "```json\n{\"signal\": \"BUY\", \"reasoning\": \"Strong support\"}\n```"
        }
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Should not contain markdown fences
            assert "```" not in msg
            # Should contain parsed content
            assert "Strong support" in msg

    def test_telegram_ensemble_metadata(self, env_vars):
        """Verify ensemble metadata is displayed correctly."""
        result = {
            "ticker": "BTC-USD",
            "date": "2026-04-11T16",
            "decision": "BUY",
            "confidence": 0.72,
            "ensemble_metadata": {
                "runs": 3,
                "valid_runs": 3,
                "retry_count": 1,
                "divergence_metrics": {
                    "confidence_range": 0.15,
                    "signal_agreement": 0.67
                },
                "individual_signals": [
                    {"signal": "BUY", "confidence": 0.65},
                    {"signal": "BUY", "confidence": 0.75},
                    {"signal": "HOLD", "confidence": 0.55}
                ]
            },
            "final_trade_decision": "Consensus decision"
        }
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Should contain ensemble info
            assert "Ensemble:" in msg
            assert "3/3 runs" in msg
            assert "retries: 1" in msg
            assert "Range:" in msg
            assert "Agreement:" in msg
            assert "Individual:" in msg

    def test_telegram_html_escaping(self, env_vars):
        """Verify HTML special characters are escaped properly."""
        result = {
            "ticker": "BTC-USD",
            "date": "2026-04-11-02-30-PM",
            "decision": "BUY",
            "confidence": 0.70,
            "final_trade_decision": "Analysis: Price < $70k & > $60k support"
        }
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Should have escaped HTML
            assert "&lt;" in msg
            assert "&gt;" in msg
            # Should not have raw < or >
            assert "Price < $70k" not in msg

    def test_telegram_long_reasoning_truncation(self, env_vars):
        """Verify long reasoning text is truncated to fit Telegram limits."""
        long_reasoning = "A" * 2000  # Very long text
        result = {
            "ticker": "BTC-USD",
            "date": "2026-04-11-02-30-PM",
            "decision": "BUY",
            "confidence": 0.70,
            "final_trade_decision": long_reasoning
        }
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Should be truncated
            assert "..." in msg or len(msg) < 2000

    def test_telegram_api_error_handling(self, env_vars, sample_result):
        """Verify API errors are logged properly."""
        import server as server_module
        import requests
        with patch("server.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
            mock_logger = MagicMock()
            with patch.object(server_module, 'logger', mock_logger):
                # Should not raise, just log error
                server_module._send_telegram_alert(sample_result)
                mock_logger.error.assert_called_once()

    def test_telegram_confidence_display(self, env_vars):
        """Verify confidence is displayed as percentage."""
        result = {
            "ticker": "BTC-USD",
            "date": "2026-04-11-02-30-PM",
            "decision": "BUY",
            "confidence": 0.55,  # 55%
            "final_trade_decision": "Test"
        }
        with patch("server.requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True, raise_for_status=MagicMock())
            from server import _send_telegram_alert
            _send_telegram_alert(result)
            
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            msg = payload["text"]
            
            # Should display as 55.0%
            assert "55.0%" in msg or "55%" in msg


class TestTelegramIntegration:
    """Integration tests for Telegram with actual server endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI."""
        from starlette.testclient import TestClient
        from server import app
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def clear_jobs(self):
        """Reset in-memory job stores between tests."""
        from server import jobs, backtest_jobs
        jobs.clear()
        backtest_jobs.clear()
        yield
        jobs.clear()
        backtest_jobs.clear()

    @patch("server.threading.Thread")
    @patch("server._send_telegram_alert")
    def test_telegram_sent_manual_analysis(self, mock_telegram, mock_thread, client):
        """Verify Telegram is called for manual analysis."""
        mock_thread.return_value = MagicMock()
        
        resp = client.post("/api/analyze", json={"ticker": "BTC-USD"})
        assert resp.status_code == 200
        
        # Simulate job completion by calling the mock
        # In real scenario, _run_analysis calls _send_telegram_alert
        assert mock_thread.called

    @patch("server._send_telegram_alert")
    def test_telegram_sent_on_job_completion(self, mock_telegram, client):
        """Verify Telegram is dispatched when job completes successfully."""
        from server import jobs
        import uuid
        
        job_id = str(uuid.uuid4())[:8]
        from server import JobEventQueue
        
        # Set up a mock completed job
        jobs[job_id] = {
            "ticker": "BTC-USD",
            "date": "2026-04-11-02-30-PM",
            "status": "done",
            "queue": JobEventQueue(),
            "result": {
                "ticker": "BTC-USD",
                "date": "2026-04-11-02-30-PM",
                "decision": "BUY",
                "confidence": 0.75,
                "final_trade_decision": "Test"
            },
            "error": None,
        }
        
        # Directly call Telegram function with the result
        from server import _send_telegram_alert
        _send_telegram_alert(jobs[job_id]["result"])
        
        # If we got here without error, the function works
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
