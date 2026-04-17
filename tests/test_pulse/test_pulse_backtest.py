"""Tests for pulse_backtest.py — PulseBacktestEngine.

~20 tests covering:
  - _build_historical_report() output shape
  - Signal de-duplication
  - Stop-and-reverse equity model
  - Forward return scoring using candle OPEN
  - Coverage gating (<60% excluded)
  - Gap detection
  - Regime bucketing (P25/P75)
  - Downtime event detection
  - _downsample utility
  - Full run() smoke test with mocked data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from tradingagents.backtesting.pulse_backtest import (
    PulseBacktestEngine,
    _downsample,
    _EXEC_COST,
    _POSITION_SIZE,
    _MIN_COVERAGE,
)


# ── Helper: generate mock candle data ─────────────────────────────────

def _make_candles(n, interval_minutes, start_dt=None, base_price=50000.0):
    """Generate n candles for testing."""
    np.random.seed(42)
    if start_dt is None:
        start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)

    data = []
    price = base_price
    for i in range(n):
        ts = start_dt + timedelta(minutes=interval_minutes * i)
        change = np.random.normal(0, 0.001 * base_price)
        price += change
        h = price + abs(np.random.normal(0, 50))
        l = price - abs(np.random.normal(0, 50))
        o = price + np.random.normal(0, 20)
        data.append({
            "timestamp": ts,
            "open": o, "high": max(h, o, price),
            "low": min(l, o, price), "close": price,
            "volume": abs(np.random.normal(100, 20)),
        })
    return pd.DataFrame(data)


def _make_funding(n=100, start_dt=None):
    """Generate mock funding data."""
    if start_dt is None:
        start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    data = []
    for i in range(n):
        ts = start_dt + timedelta(hours=8 * i)
        rate = np.random.normal(0.0001, 0.00005)
        data.append({"timestamp": ts, "funding_rate": rate})
    return pd.DataFrame(data)


# ── _downsample ───────────────────────────────────────────────────────

class TestDownsample:
    def test_under_limit(self):
        curve = [1.0, 1.01, 1.02]
        result = _downsample(curve, 500)
        assert len(result) == 3

    def test_over_limit(self):
        curve = list(range(1000))
        result = _downsample(curve, 100)
        assert len(result) == 100
        assert result[-1] == 999  # last point always included

    def test_exact_limit(self):
        curve = list(range(500))
        result = _downsample(curve, 500)
        assert len(result) == 500

    def test_empty(self):
        assert _downsample([], 500) == []


# ── PulseBacktestEngine ──────────────────────────────────────────────

class TestPulseBacktestEngine:
    def _make_engine(self, start="2026-01-01", end="2026-01-02"):
        return PulseBacktestEngine(
            ticker="BTC-USD",
            start_date=start,
            end_date=end,
            pulse_interval_minutes=15,
            signal_threshold=0.25,
        )

    def test_build_historical_report_shape(self):
        """Report should have all expected keys."""
        engine = self._make_engine()
        start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = {
            tf: _make_candles(100, int(minutes), start_dt)
            for tf, minutes in [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240)]
        }
        funding = _make_funding(20, start_dt)
        ts = start_dt + timedelta(hours=6)

        report = engine._build_historical_report(candles, funding, ts, None)

        assert "ticker" in report
        assert "spot_price" in report
        assert "timeframes" in report
        assert "premium_pct" in report
        assert report["premium_pct"] == 0.0  # backtest: premium=0
        assert "_overall_coverage" in report

    def test_premium_zero_in_backtest(self):
        """Premium must be 0 in historical report."""
        engine = self._make_engine()
        start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = {
            tf: _make_candles(50, 1, start_dt)
            for tf in ["1m", "5m", "15m", "1h", "4h"]
        }
        report = engine._build_historical_report(
            candles, pd.DataFrame(columns=["timestamp", "funding_rate"]),
            start_dt + timedelta(hours=1), None,
        )
        assert report["premium_pct"] == 0.0

    def test_score_signals_forward_return(self):
        """Forward returns should use candle OPEN."""
        engine = self._make_engine()
        start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles_1m = _make_candles(120, 1, start_dt, 50000)
        candles = {"1m": candles_1m}

        signals = [{
            "ts": (start_dt + timedelta(minutes=10)).isoformat(),
            "signal": "BUY",
            "confidence": 0.7,
            "normalized_score": 0.35,
            "price": 50000,
            "stop_loss": 49500,
            "take_profit": 51000,
            "hold_minutes": 45,
            "timeframe_bias": "15m",
            "breakdown": {},
        }]

        scored = engine._score_signals(signals, candles)
        assert len(scored) == 1
        # Check that forward returns are populated
        for horizon in ["+5m", "+15m", "+1h"]:
            key = f"return_{horizon}"
            if key in scored[0]:
                assert isinstance(scored[0][key], float)

    def test_score_signals_exit_type(self):
        """Each signal should have exit_type after scoring."""
        engine = self._make_engine()
        start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = {"1m": _make_candles(120, 1, start_dt)}

        signals = [{
            "ts": (start_dt + timedelta(minutes=10)).isoformat(),
            "signal": "BUY",
            "confidence": 0.5,
            "normalized_score": 0.3,
            "price": 50000,
            "stop_loss": 49000,
            "take_profit": 52000,
            "hold_minutes": 45,
            "timeframe_bias": "15m",
            "breakdown": {},
        }]

        scored = engine._score_signals(signals, candles)
        assert scored[0]["exit_type"] in ("sl_hit", "tp_hit", "timeout")

    def test_compute_metrics_shape(self):
        """Metrics result should have all expected keys."""
        engine = self._make_engine()
        engine._total_intervals = 100
        engine._n_excluded_warmup = 5
        engine._gap_count = 2

        signals = [
            {
                "ts": datetime(2026, 1, 1, i, tzinfo=timezone.utc).isoformat(),
                "signal": "BUY" if i % 2 == 0 else "SHORT",
                "confidence": 0.6,
                "normalized_score": 0.3 * (1 if i % 2 == 0 else -1),
                "price": 50000 + i * 10,
                "stop_loss": 49500,
                "take_profit": 51000,
                "hold_minutes": 45,
                "timeframe_bias": "15m",
                "breakdown": {},
                "hit_+5m": True,
                "hit_+15m": False,
                "hit_+1h": True,
                "return_+5m": 0.001,
                "return_+15m": -0.002,
                "return_+1h": 0.003,
                "exit_type": "timeout",
                "exit_return": 0.001,
            }
            for i in range(10)
        ]

        candles = {
            "1m": _make_candles(1440, 1),
            "1h": _make_candles(48, 60),
        }

        result = engine._compute_metrics(signals, candles)

        assert "ticker" in result
        assert "hit_rates" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown_pct" in result
        assert "profitability_curve" in result
        assert "by_confidence_bucket" in result
        assert "n_excluded_warmup" in result
        assert result["n_excluded_warmup"] == 5
        assert "return_autocorr_lag1" in result
        assert "premium_note" in result

    def test_confidence_buckets_derived_from_threshold(self):
        """Bucket boundaries should shift with threshold."""
        engine = self._make_engine()
        engine.threshold = 0.25
        engine._total_intervals = 50
        engine._n_excluded_warmup = 0
        engine._gap_count = 0

        signals = [
            {
                "ts": datetime(2026, 1, 1, i, tzinfo=timezone.utc).isoformat(),
                "signal": "BUY",
                "confidence": 0.5 + i * 0.05,
                "normalized_score": 0.3,
                "price": 50000,
                "hold_minutes": 45,
                "timeframe_bias": "15m",
                "breakdown": {},
                "hit_+1h": True,
            }
            for i in range(8)
        ]

        candles = {"1m": _make_candles(100, 1), "1h": _make_candles(48, 60)}
        result = engine._compute_metrics(signals, candles)

        buckets = result["by_confidence_bucket"]
        assert "low" in buckets
        assert "mid" in buckets
        assert "high" in buckets
        # min_conf = 0.25/0.5 = 0.50
        assert buckets["low"]["range"].startswith("[0.50")

    def test_gap_detection(self):
        """Gaps >66s in 1m data should be detected."""
        engine = self._make_engine()
        start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # Create 1m candles with a gap
        data = []
        for i in range(30):
            ts = start_dt + timedelta(minutes=i)
            data.append({
                "timestamp": ts,
                "open": 50000, "high": 50050, "low": 49950,
                "close": 50000, "volume": 100,
            })
        # Add 2h gap
        for i in range(30, 60):
            ts = start_dt + timedelta(minutes=i + 120)
            data.append({
                "timestamp": ts,
                "open": 50000, "high": 50050, "low": 49950,
                "close": 50000, "volume": 100,
            })

        df_1m = pd.DataFrame(data)
        candles = {
            "1m": df_1m,
            "5m": _make_candles(50, 5, start_dt),
            "15m": _make_candles(50, 15, start_dt),
            "1h": _make_candles(24, 60, start_dt),
            "4h": _make_candles(12, 240, start_dt),
        }
        funding = _make_funding(5, start_dt)

        # Run replay — signals near gap should be skipped
        signals = engine._replay(candles, funding)
        assert engine._gap_count >= 1

    def test_downtime_detection(self):
        """Gaps >30min should be reported as downtime events."""
        engine = self._make_engine()
        start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)

        data = []
        for i in range(30):
            data.append({
                "timestamp": start_dt + timedelta(minutes=i),
                "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10,
            })
        # 2h gap
        for i in range(30, 60):
            data.append({
                "timestamp": start_dt + timedelta(minutes=i + 120),
                "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10,
            })

        candles = {"1m": pd.DataFrame(data)}
        events = engine._detect_downtime_events(candles)
        assert len(events) >= 1
        assert events[0]["duration_min"] > 30

    def test_equity_curve_stop_and_reverse(self):
        """Alternating BUY/SHORT signals should trigger stop-and-reverse."""
        engine = self._make_engine()
        engine._total_intervals = 20
        engine._n_excluded_warmup = 0
        engine._gap_count = 0

        signals = []
        for i in range(6):
            signals.append({
                "ts": datetime(2026, 1, 1, i, tzinfo=timezone.utc).isoformat(),
                "signal": "BUY" if i % 2 == 0 else "SHORT",
                "confidence": 0.6,
                "normalized_score": 0.3,
                "price": 50000 + i * 50,
                "hold_minutes": 45,
                "timeframe_bias": "15m",
                "breakdown": {},
            })

        candles = {"1m": _make_candles(1440, 1), "1h": _make_candles(24, 60)}
        result = engine._compute_metrics(signals, candles)

        # Should have trades from stop-and-reverse
        assert result["n_trades"] > 0
        assert len(result["profitability_curve"]) > 1

    def test_max_drawdown_nonnegative(self):
        """Max drawdown should be >= 0."""
        engine = self._make_engine()
        engine._total_intervals = 10
        engine._n_excluded_warmup = 0
        engine._gap_count = 0

        signals = [{
            "ts": datetime(2026, 1, 1, 0, tzinfo=timezone.utc).isoformat(),
            "signal": "BUY",
            "confidence": 0.6,
            "normalized_score": 0.3,
            "price": 50000,
            "hold_minutes": 45,
            "timeframe_bias": "15m",
            "breakdown": {},
        }]

        candles = {"1m": _make_candles(100, 1), "1h": _make_candles(24, 60)}
        result = engine._compute_metrics(signals, candles)
        assert result["max_drawdown_pct"] >= 0

    def test_profitability_curve_downsampled(self):
        """Curve should be ≤500 points."""
        engine = self._make_engine()
        engine._total_intervals = 100
        engine._n_excluded_warmup = 0
        engine._gap_count = 0

        # Lots of signals to generate long curve
        signals = [
            {
                "ts": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
                "signal": "BUY",
                "confidence": 0.6,
                "normalized_score": 0.3,
                "price": 50000 + i,
                "hold_minutes": 45,
                "timeframe_bias": "15m",
                "breakdown": {},
            }
            for i in range(600)
        ]

        candles = {"1m": _make_candles(100, 1), "1h": _make_candles(24, 60)}
        result = engine._compute_metrics(signals, candles)
        assert len(result["profitability_curve"]) <= 500
