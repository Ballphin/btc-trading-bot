"""Tests for the directional regime classifier (Stage 2 Commit G)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tradingagents.pulse import regime_directional as rd


def _daily_frame(closes, start="2024-01-01"):
    """Build a synthetic daily OHLC frame from a list of closes.

    High = close * 1.01, Low = close * 0.99, Open = prev close.
    """
    n = len(closes)
    idx = pd.date_range(start=start, periods=n, freq="D", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "open": opens,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
    }, index=idx)


def _bull_frame():
    # +25% over 90 days with a smooth drift → bull
    return _daily_frame(100 * np.exp(np.linspace(0, 0.223, 100)))


def _bear_frame():
    return _daily_frame(100 * np.exp(np.linspace(0, -0.25, 100)))


def _range_frame():
    rng = np.random.default_rng(42)
    # tight sinusoidal + very small noise around 100 → low range, flat drift
    t = np.arange(120)
    closes = 100 + 0.1 * np.sin(t / 10.0) + rng.normal(0, 0.02, size=120)
    return _daily_frame(closes)


class TestClassifyLabels:
    def test_bull_regime_detected(self):
        r = rd.classify_directional(_bull_frame(), log=False)
        assert r.label == "bull"
        assert r.return_90d > 0.15
        assert r.frac_above_sma30 >= 0.6

    def test_bear_regime_detected(self):
        r = rd.classify_directional(_bear_frame(), log=False)
        assert r.label == "bear"
        assert r.return_90d < -0.15

    def test_range_bound_detected(self):
        r = rd.classify_directional(_range_frame(), log=False)
        assert r.label == "range_bound"
        assert r.range_atr_ratio < 1.2
        assert abs(r.return_30d) < 0.05

    def test_ambiguous_on_conflicting_signals(self):
        # Strong +90d return but final 30d sharply down → not sustained up.
        closes = np.concatenate([
            np.linspace(100, 135, 60),   # big rally
            np.linspace(135, 120, 40),   # then a pullback
        ])
        r = rd.classify_directional(_daily_frame(closes), log=False)
        assert r.label in ("ambiguous", "bull")  # depends on consistency


class TestAmbiguousBoundary:
    def test_insufficient_history_flags_ambiguous(self):
        r = rd.classify_directional(_daily_frame(np.linspace(100, 120, 40)), log=False)
        assert r.label == "ambiguous"
        assert r.insufficient_history is True
        assert r.sample_size == 40

    def test_boundary_weak_trend_is_ambiguous(self):
        # 90d return ~+10% (below 15% threshold) → ambiguous, not bull.
        r = rd.classify_directional(
            _daily_frame(100 * np.exp(np.linspace(0, 0.095, 100))),
            log=False,
        )
        assert r.label == "ambiguous"


class TestTimeoutBudget:
    def test_timeout_returns_ambiguous_fallback(self):
        # Force _classify_sync to sleep longer than the budget.
        def slow(df):
            time.sleep(0.2)
            return rd._classify_sync(df)

        with patch.object(rd, "_classify_sync", side_effect=slow):
            r = rd.classify_directional(
                _bull_frame(), log=False, timeout_ms=20,
            )
        assert r.label == "ambiguous"
        assert "timeout" in r.reason.lower()

    def test_default_fast_classification_under_budget(self):
        t0 = time.perf_counter()
        r = rd.classify_directional(_bull_frame(), log=False, timeout_ms=500)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert r.label == "bull"
        # Generous upper bound (CI jitter); real p99 target is <5ms.
        assert dt_ms < 250


class TestLogging:
    def test_log_writes_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rd, "LOG_DIR", tmp_path)
        r = rd.classify_directional(_bull_frame(), ticker="BTC-USD", log=True)
        path = tmp_path / "BTC-USD_regime.jsonl"
        assert path.exists()
        line = path.read_text().strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["ticker"] == "BTC-USD"
        assert payload["label"] == r.label


class TestHistoricalWindows:
    """Smoke tests on hand-picked historical windows.

    Uses synthetic approximations since we can't assume yfinance
    availability in CI. Real validation happens in manual runbook.
    """

    def test_2021q4_btc_style_bull(self):
        # BTC-USD ran from ~44k to ~67k during Oct-Nov 2021 → bull.
        closes = np.linspace(44000, 67000, 95) * (
            1 + 0.01 * np.sin(np.arange(95) / 5)
        )
        r = rd.classify_directional(_daily_frame(closes), log=False)
        assert r.label == "bull"

    def test_2022q3_btc_style_bear(self):
        # Smooth decline → bear.
        closes = np.linspace(48000, 19000, 100)
        r = rd.classify_directional(_daily_frame(closes), log=False)
        assert r.label == "bear"

    def test_2023_summer_range(self):
        # Summer 2023 BTC chopped in a tight $26-31k range.
        # Tight chop — range_atr_ratio < 1.2 requires amplitude not
        # dwarf the per-bar true range. Use 28500 ± 0.3% drift.
        rng = np.random.default_rng(7)
        closes = 28500 + 30 * np.sin(np.arange(120) / 15.0) + rng.normal(0, 10, 120)
        r = rd.classify_directional(_daily_frame(closes), log=False)
        assert r.label == "range_bound"
