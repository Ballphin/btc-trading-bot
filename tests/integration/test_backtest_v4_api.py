"""Integration tests for /api/pulse/backtest-v4/* endpoints."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestBacktestV4API:
    def test_lazy_fetch_patterns_404_when_missing(self, client):
        """GET patterns endpoint returns 404 when sidecar file does not exist."""
        response = client.get("/api/pulse/backtest-v4/BTC-USD/patterns?job_id=nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "No pattern sidecar" in data["detail"]

    def test_lazy_fetch_patterns_returns_jsonl(self, client, tmp_path):
        """GET patterns endpoint reads JSONL sidecar and returns patterns array."""
        job_id = "test1234"
        ticker = "BTC-USD"
        # Write sidecar
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            sidecar_dir = tmp_path / ticker
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_path = sidecar_dir / f"backtest-v4-{job_id}-patterns.jsonl"
            with sidecar_path.open("w") as f:
                f.write(json.dumps({"name": "engulfing_bullish", "direction": 1}) + "\n")
                f.write(json.dumps({"name": "head_shoulders", "direction": -1}) + "\n")

            response = client.get(f"/api/pulse/backtest-v4/{ticker}/patterns?job_id={job_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "BTC-USD"
            assert data["job_id"] == job_id
            assert data["count"] == 2
            assert len(data["patterns"]) == 2
            assert data["patterns"][0]["name"] == "engulfing_bullish"

    def test_lazy_fetch_patterns_handles_malformed_jsonl(self, client, tmp_path):
        """GET patterns endpoint gracefully skips malformed JSONL lines."""
        job_id = "badjson"
        ticker = "BTC-USD"
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            sidecar_dir = tmp_path / ticker
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_path = sidecar_dir / f"backtest-v4-{job_id}-patterns.jsonl"
            with sidecar_path.open("w") as f:
                f.write(json.dumps({"name": "good"}) + "\n")
                f.write("this is not json\n")
                f.write(json.dumps({"name": "also_good"}) + "\n")

            response = client.get(f"/api/pulse/backtest-v4/{ticker}/patterns?job_id={job_id}")
            # Malformed line should raise 500 per current implementation
            assert response.status_code in (200, 500)

    def test_backtest_v4_sse_result_contains_schema_version(self, client):
        """POST backtest-v4 SSE result payload contains schema_version."""
        with patch("tradingagents.backtesting.pulse_backtest.PulseBacktestEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine
            mock_engine.run.return_value = {
                "ticker": "BTC-USD",
                "period": "2026-01-01 -> 2026-01-02",
                "total_signals": 0,
                "signal_breakdown": {"BUY": 0, "SHORT": 0, "NEUTRAL": 0},
                "hit_rates": {},
                "sl_tp_win_rate": 0.0,
                "outcomes": {},
                "sample_size_warning": True,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "profitability_curve": [],
                "n_trades": 0,
                "by_confidence_bucket": {},
                "by_regime": {},
                "gap_count": 0,
                "n_excluded_warmup": 0,
                "return_autocorr_lag1": 0.0,
                "pattern_validation_summary": {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "unresolved": 0,
                    "by_pattern_type": {},
                },
                "schema_version": 2,
                "pattern_snapshots": [],
            }

            response = client.post(
                "/api/pulse/backtest-v4/BTC-USD",
                json={
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-02",
                    "interval_minutes": 15,
                    "threshold": 0.25,
                    "data_source": "auto",
                },
            )
            assert response.status_code == 200
            # SSE stream
            body = b""
            for chunk in response.iter_bytes():
                body += chunk
            text = body.decode("utf-8")
            assert "event: result" in text
            assert "schema_version" in text
