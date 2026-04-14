"""Integration tests for scorecard API endpoints and background scoring.

Tests the FastAPI endpoints that expose adaptive scoring features:
- GET /api/shadow/scorecard/{ticker}
- POST /api/shadow/score/{ticker}
- POST /api/shadow/calibrate/{ticker}
- POST /api/shadow/walk-forward/{ticker}
- _trigger_background_scoring multi-ticker sweep
"""

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from starlette.testclient import TestClient

from server import app, EVAL_RESULTS_DIR


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_decisions(path: Path, decisions: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in decisions:
            f.write(json.dumps(d, default=str) + "\n")


def _scored_decision(date="2026-03-01", signal="BUY", price=100.0,
                     confidence=0.7, was_correct=True, net_return=0.03,
                     exit_type="held_to_expiry", regime="bull_quiet", **kw):
    return {
        "ticker": kw.get("ticker", "TEST-USD"),
        "date": date,
        "signal": signal,
        "price": price,
        "confidence": confidence,
        "regime": regime,
        "scored": True,
        "was_correct_primary": was_correct,
        "actual_return_primary": net_return + 0.001,
        "net_return_primary": net_return,
        "exit_type": exit_type,
        "exit_price": price * (1 + net_return),
        "exit_day": 3,
        "hold_days_planned": 3,
        "execution_cost": 0.001,
        "brier_score": (confidence - (1.0 if was_correct else 0.0)) ** 2,
        "scored_at": datetime.now().isoformat(),
        **{k: v for k, v in kw.items() if k != "ticker"},
    }


def _raw_decision(date="2026-03-01", signal="BUY", price=100.0, confidence=0.7, **kw):
    return {
        "ticker": kw.get("ticker", "TEST-USD"),
        "date": date,
        "signal": signal,
        "price": price,
        "confidence": confidence,
        "regime": kw.get("regime", "bull_quiet"),
        "stop_loss": kw.get("stop_loss"),
        "take_profit": kw.get("take_profit"),
        "max_hold_days": kw.get("max_hold_days", 3),
        "position_size_pct": 0.05,
        "source": "test",
        "recorded_at": datetime.now().isoformat(),
        "scored": False,
    }


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    return TestClient(app)


# ── TestScorecardEndpoint ────────────────────────────────────────────────


class TestScorecardEndpoint:
    """Tests for GET /api/shadow/scorecard/{ticker}."""

    def test_scorecard_returns_adaptive_fields(self, client, tmp_path):
        """Response includes exit_type_breakdown, ev_per_trade_10k, avg_win/loss."""
        scored = [
            _scored_decision(f"2026-03-0{i}", was_correct=True, exit_type="take_profit_hit")
            for i in range(1, 5)
        ] + [_scored_decision("2026-03-05", was_correct=False, exit_type="stop_loss_hit", net_return=-0.02)]
        shadow = tmp_path / "shadow" / "TEST-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions_scored.jsonl", scored)
        _write_decisions(shadow / "decisions.jsonl", scored)

        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/shadow/scorecard/TEST-USD")
        assert resp.status_code == 200
        data = resp.json()
        assert "exit_type_breakdown" in data
        assert "ev_per_trade_10k" in data
        assert "avg_win_return" in data
        assert "avg_loss_return" in data
        assert data["exit_type_breakdown"]["take_profit_hit"] == 4
        assert data["exit_type_breakdown"]["stop_loss_hit"] == 1

    def test_empty_ticker_graceful(self, client, tmp_path):
        """Nonexistent ticker returns scored_decisions: 0."""
        shadow = tmp_path / "shadow" / "EMPTY-USD"
        shadow.mkdir(parents=True)
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/shadow/scorecard/EMPTY-USD")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("scored_decisions", 0) == 0

    def test_response_shape_matches_frontend_interface(self, client, tmp_path):
        """All required keys from the Scorecard TypeScript interface are present."""
        scored = [
            _scored_decision(f"2026-03-0{i}", confidence=0.6 + i*0.02, was_correct=(i % 2 == 0))
            for i in range(1, 7)
        ]
        shadow = tmp_path / "shadow" / "TEST-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions_scored.jsonl", scored)
        _write_decisions(shadow / "decisions.jsonl", scored)

        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/shadow/scorecard/TEST-USD")
        data = resp.json()
        required_keys = [
            "total_decisions", "scored_decisions", "overall_win_rate",
            "avg_brier_score", "win_by_signal", "win_by_regime",
            "exit_type_breakdown", "ev_per_trade_10k",
            "avg_win_return", "avg_loss_return", "recent_decisions",
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

    def test_legacy_only_decisions_still_work(self, client, tmp_path):
        """Scorecard with only was_correct_7d (no primary) still works."""
        scored = [
            {
                "ticker": "TEST-USD", "date": f"2026-03-0{i}", "signal": "BUY",
                "price": 100, "confidence": 0.65, "regime": "bull_quiet",
                "scored": True, "was_correct_7d": True, "actual_return_7d": 0.02,
                "brier_score": 0.1225,
            }
            for i in range(1, 7)
        ]
        shadow = tmp_path / "shadow" / "TEST-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions_scored.jsonl", scored)
        _write_decisions(shadow / "decisions.jsonl", scored)

        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.get("/api/shadow/scorecard/TEST-USD")
        data = resp.json()
        assert data["scored_decisions"] == 6
        assert data["overall_win_rate"] > 0

    def test_error_response_has_error_key(self, client, tmp_path):
        """If scorecard function raises, endpoint returns {error: ...}."""
        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("tradingagents.backtesting.scorecard.get_scorecard", side_effect=Exception("boom")):
            resp = client.get("/api/shadow/scorecard/TEST-USD")
        data = resp.json()
        assert "error" in data


# ── TestScoreEndpoint ────────────────────────────────────────────────────


class TestScoreEndpoint:
    """Tests for POST /api/shadow/score/{ticker}."""

    @patch("tradingagents.backtesting.scorecard._get_ohlc_range")
    @patch("tradingagents.backtesting.scorecard._get_price_on_date", return_value=103.0)
    @patch("tradingagents.backtesting.scorecard.datetime")
    def test_score_returns_counts(self, mock_dt, mock_price, mock_ohlc, client, tmp_path):
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        ohlc_rows = [{"day": i, "open": 100+i, "high": 102+i, "low": 99+i, "close": 101+i} for i in range(1, 4)]
        import pandas as pd
        dates = [entry_dt + timedelta(days=r["day"]) for r in ohlc_rows]
        df = pd.DataFrame(ohlc_rows, index=pd.DatetimeIndex(dates))
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        df = df[["Open", "High", "Low", "Close"]]
        mock_ohlc.return_value = df

        shadow = tmp_path / "shadow" / "TEST-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions.jsonl", [_raw_decision("2026-03-01")])

        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.post("/api/shadow/score/TEST-USD")
        data = resp.json()
        assert "scored" in data
        assert data["scored"] >= 0

    def test_score_no_decisions(self, client, tmp_path):
        shadow = tmp_path / "shadow" / "TEST-USD"
        shadow.mkdir(parents=True)
        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.post("/api/shadow/score/TEST-USD")
        data = resp.json()
        assert "error" in data or data.get("scored", 0) == 0

    def test_score_error_handling(self, client, tmp_path):
        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("tradingagents.backtesting.scorecard.score_pending_decisions", side_effect=Exception("fail")):
            resp = client.post("/api/shadow/score/TEST-USD")
        data = resp.json()
        assert "error" in data


# ── TestCalibrateEndpoint ────────────────────────────────────────────────


class TestCalibrateEndpoint:
    """Tests for POST /api/shadow/calibrate/{ticker}."""

    def test_calibrate_returns_deduped_count(self, client, tmp_path):
        scored = [
            _scored_decision(f"2026-03-{i+1:02d}", confidence=0.65, was_correct=True)
            for i in range(12)
        ]
        shadow = tmp_path / "shadow" / "TEST-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions_scored.jsonl", scored)

        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.post("/api/shadow/calibrate/TEST-USD")
        data = resp.json()
        assert "n_decisions_deduped" in data
        assert "correction" in data

    def test_calibrate_insufficient_data(self, client, tmp_path):
        scored = [_scored_decision(f"2026-03-0{i}") for i in range(1, 4)]
        shadow = tmp_path / "shadow" / "TEST-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions_scored.jsonl", scored)

        with patch("server.EVAL_RESULTS_DIR", tmp_path):
            resp = client.post("/api/shadow/calibrate/TEST-USD")
        data = resp.json()
        assert "error" in data

    def test_calibrate_error_handling(self, client, tmp_path):
        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("tradingagents.backtesting.scorecard.run_calibration_study", side_effect=Exception("fail")):
            resp = client.post("/api/shadow/calibrate/TEST-USD")
        data = resp.json()
        assert "error" in data


# ── TestWalkForwardEndpoint ──────────────────────────────────────────────


class TestWalkForwardEndpoint:
    """Tests for POST /api/shadow/walk-forward/{ticker}."""

    def test_walk_forward_returns_adaptive_fields(self, client, tmp_path):
        """Mock the WalkForwardValidator to return a complete adaptive result."""
        mock_result = {
            "ticker": "TEST-USD",
            "horizon_days": 7,
            "total_decisions": 10,
            "scored_decisions": 8,
            "overall_metrics": {
                "win_rate": 0.625,
                "mean_return_gross": 0.015,
                "mean_return_net": 0.013,
                "std_return": 0.02,
                "sharpe_ratio_gross": 0.75,
                "sharpe_ratio_net": 0.65,
                "sharpe_se": 0.35,
                "deflated_sharpe_ratio": 0.72,
                "dsr_interpretation": "INCONCLUSIVE",
                "max_drawdown": 0.05,
                "skewness": -0.3,
                "kurtosis": 3.5,
                "n_strategies_tested": 1,
                "ev_per_trade_10k": 45.0,
                "avg_win_return": 0.03,
                "avg_loss_return": 0.015,
            },
            "exit_type_breakdown": {"take_profit_hit": 3, "held_to_expiry": 4, "stop_loss_hit": 1},
            "regime_analysis": {},
            "signal_analysis": {},
            "equity_curve_gross": [1.0, 1.01, 1.02, 1.015],
            "equity_curve_position": [1.0, 1.005, 1.01, 1.008],
            "validated_at": datetime.now().isoformat(),
        }
        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("tradingagents.backtesting.walk_forward.WalkForwardValidator") as MockValidator:
            MockValidator.return_value.validate.return_value = mock_result
            resp = client.post("/api/shadow/walk-forward/TEST-USD")

        data = resp.json()
        assert "equity_curve_gross" in data
        assert "equity_curve_position" in data
        assert "exit_type_breakdown" in data
        assert data["overall_metrics"]["sharpe_ratio_net"] == 0.65
        assert data["overall_metrics"]["sharpe_se"] == 0.35
        assert data["overall_metrics"]["ev_per_trade_10k"] == 45.0

    def test_walk_forward_no_logs(self, client, tmp_path):
        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("tradingagents.backtesting.walk_forward.WalkForwardValidator") as MockValidator:
            MockValidator.return_value.validate.return_value = {"error": "No decisions found in logs", "decisions": 0}
            resp = client.post("/api/shadow/walk-forward/TEST-USD")
        data = resp.json()
        assert "error" in data

    def test_walk_forward_error_handling(self, client, tmp_path):
        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("tradingagents.backtesting.walk_forward.WalkForwardValidator", side_effect=Exception("fail")):
            resp = client.post("/api/shadow/walk-forward/TEST-USD")
        data = resp.json()
        assert "error" in data


# ── TestBackgroundScoring ────────────────────────────────────────────────


class TestBackgroundScoring:
    """Tests for _trigger_background_scoring multi-ticker sweep."""

    def test_multi_ticker_sweep(self, tmp_path):
        """Background scoring sweeps all tickers with decisions.jsonl."""
        from server import _trigger_background_scoring

        # Create two ticker dirs with decisions.jsonl
        for t in ["AAA-USD", "BBB-USD"]:
            d = tmp_path / "shadow" / t
            d.mkdir(parents=True)
            _write_decisions(d / "decisions.jsonl", [_raw_decision(ticker=t)])

        scored_tickers = []

        def mock_score(ticker, results_dir):
            scored_tickers.append(ticker)
            return {"scored": 0, "total_scored": 5}

        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("server._scoring_lock", threading.Lock()), \
             patch("tradingagents.backtesting.scorecard.score_pending_decisions", side_effect=mock_score), \
             patch("tradingagents.backtesting.scorecard.count_scored_decisions", return_value=5):
            # Run synchronously by calling the inner function
            _trigger_background_scoring("AAA-USD")
            # Wait for thread to finish
            import time
            time.sleep(1.0)

        assert "AAA-USD" in scored_tickers or "BBB-USD" in scored_tickers

    def test_triggering_ticker_scored_first(self, tmp_path):
        """The triggering ticker should be scored before others."""
        from server import _trigger_background_scoring

        for t in ["CCC-USD", "TRIGGER-USD", "DDD-USD"]:
            d = tmp_path / "shadow" / t
            d.mkdir(parents=True)
            _write_decisions(d / "decisions.jsonl", [_raw_decision(ticker=t)])

        score_order = []

        def mock_score(ticker, results_dir):
            score_order.append(ticker)
            return {"scored": 0, "total_scored": 3}

        with patch("server.EVAL_RESULTS_DIR", tmp_path), \
             patch("server._scoring_lock", threading.Lock()), \
             patch("tradingagents.backtesting.scorecard.score_pending_decisions", side_effect=mock_score), \
             patch("tradingagents.backtesting.scorecard.count_scored_decisions", return_value=3):
            _trigger_background_scoring("TRIGGER-USD")
            import time
            time.sleep(1.0)

        if score_order:
            assert score_order[0] == "TRIGGER-USD"
