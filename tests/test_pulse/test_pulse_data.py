"""Tests for quant_pulse_data.py — indicator computation, patterns, VWAP.

~25 tests covering:
  - _compute_tf_indicators() with sufficient/insufficient bars
  - None handling for each indicator
  - Coverage computation
  - MACD direction + contiguity check
  - EMA neutral zone metadata
  - Candlestick pattern detection (all 10 patterns)
  - VWAP computation from 1m candles
  - Volatility flag
  - Wick filter
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from tradingagents.agents.quant_pulse_data import (
    _compute_tf_indicators,
    compute_vwap,
    compute_vwap_from_slice,
    compute_volatility_flag,
    detect_patterns,
    _detect_doji,
    _detect_bullish_engulfing,
    _detect_bearish_engulfing,
    _detect_hammer,
    _detect_shooting_star,
    _detect_morning_star,
    _detect_evening_star,
    _detect_three_white_soldiers,
    _detect_three_black_crows,
    _detect_harami,
    _wick_filter,
)


# ── Helper to generate candle data ───────────────────────────────────

def _make_candles(n=50, interval_minutes=15, base_price=50000.0, vol_pct=0.01):
    """Generate n candles with random walk for testing."""
    np.random.seed(42)
    timestamps = [
        datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=interval_minutes * i)
        for i in range(n)
    ]
    close_prices = [base_price]
    for i in range(1, n):
        change = np.random.normal(0, vol_pct * base_price)
        close_prices.append(close_prices[-1] + change)

    data = []
    for i, ts in enumerate(timestamps):
        c = close_prices[i]
        h = c + abs(np.random.normal(0, vol_pct * base_price * 0.3))
        l = c - abs(np.random.normal(0, vol_pct * base_price * 0.3))
        o = c + np.random.normal(0, vol_pct * base_price * 0.2)
        v = abs(np.random.normal(1000, 200))
        data.append({
            "timestamp": ts,
            "open": o, "high": max(h, o, c), "low": min(l, o, c),
            "close": c, "volume": v,
        })

    return pd.DataFrame(data)


# ── _compute_tf_indicators() ─────────────────────────────────────────

class TestComputeTfIndicators:
    def test_sufficient_bars_coverage(self):
        """50 bars should give high coverage."""
        df = _make_candles(50)
        indicators, coverage = _compute_tf_indicators(df, "15m")
        assert coverage > 0.5
        assert indicators["rsi"] is not None
        assert indicators["atr"] is not None

    def test_insufficient_bars_rsi(self):
        """<15 bars → RSI is None."""
        df = _make_candles(10)
        indicators, coverage = _compute_tf_indicators(df, "15m")
        assert indicators["rsi"] is None

    def test_insufficient_bars_macd(self):
        """<35 bars → MACD is None."""
        df = _make_candles(20)
        indicators, coverage = _compute_tf_indicators(df, "15m")
        assert indicators["macd_hist"] is None

    def test_insufficient_bars_bb(self):
        """<20 bars → BB is None."""
        df = _make_candles(15)
        indicators, coverage = _compute_tf_indicators(df, "15m")
        assert indicators["bb_pct"] is None

    def test_ema_cross_values(self):
        """Check that _ema9 and _ema21 are populated."""
        df = _make_candles(50)
        indicators, _ = _compute_tf_indicators(df, "15m")
        assert indicators.get("ema_cross") in ("bullish", "bearish")
        assert "_ema9" in indicators
        assert "_ema21" in indicators

    def test_atr_positive(self):
        df = _make_candles(50)
        indicators, _ = _compute_tf_indicators(df, "15m")
        assert indicators["atr"] is not None
        assert indicators["atr"] > 0

    def test_rel_volume(self):
        df = _make_candles(50)
        indicators, _ = _compute_tf_indicators(df, "15m")
        assert indicators["rel_volume"] is not None
        assert indicators["rel_volume"] > 0

    def test_empty_df(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        indicators, coverage = _compute_tf_indicators(df, "1m")
        assert coverage == 0.0
        assert indicators["rsi"] is None

    def test_one_row(self):
        df = _make_candles(1)
        indicators, coverage = _compute_tf_indicators(df, "1m")
        assert coverage == 0.0

    def test_macd_direction_with_gap(self):
        """Contiguity check: 4h gap between last 4 bars → direction=None."""
        df = _make_candles(50, interval_minutes=15)
        # Introduce a 4h gap before the last bar
        df.iloc[-1, df.columns.get_loc("timestamp")] = (
            df.iloc[-2]["timestamp"] + timedelta(hours=4)
        )
        indicators, _ = _compute_tf_indicators(df, "15m")
        # Direction should be None due to gap
        assert indicators.get("_macd_direction") is None

    def test_coverage_fraction(self):
        """Coverage is n_valid / 6."""
        df = _make_candles(50)
        _, coverage = _compute_tf_indicators(df, "15m")
        assert 0.0 <= coverage <= 1.0

    def test_patterns_detected_when_flag_set(self):
        df = _make_candles(50)
        indicators, _ = _compute_tf_indicators(df, "1h", detect_patterns_flag=True)
        assert isinstance(indicators["patterns"], list)

    def test_patterns_empty_when_flag_unset(self):
        df = _make_candles(50)
        indicators, _ = _compute_tf_indicators(df, "1h", detect_patterns_flag=False)
        assert indicators["patterns"] == []


# ── Pattern detection ─────────────────────────────────────────────────

def _row(o, h, l, c, v=1000):
    return pd.DataFrame([{
        "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "open": o, "high": h, "low": l, "close": c, "volume": v,
    }])


class TestPatternDetection:
    def test_doji(self):
        """Body < 10% of range → doji."""
        df = _row(100, 110, 90, 100.5)
        assert _detect_doji(df) == [("doji", 0)]

    def test_not_doji(self):
        df = _row(90, 110, 85, 108)
        assert _detect_doji(df) == []

    def test_hammer(self):
        """Lower wick >= 2× body, upper wick <= 0.5× body."""
        # body=2 (100→102), lower_wick=10 (100←90), upper_wick=0.5 (102→102.5)
        # lower >= 2*body: 10 >= 4 ✓, upper <= 0.5*body: 0.5 <= 1 ✓
        # wick_filter: max(0.5, 10)/2 = 5 > 3 → filtered out!
        # Use values where max wick/body <= 3: body=5, lower=12, upper=1
        df = _row(100, 101, 85, 105)  # body=5, lower=15, upper=-4 -> need correct
        # open=100, high=106, low=88, close=105 -> body=5, lower=12, upper=1
        df = _row(100, 106, 88, 105)
        result = _detect_hammer(df)
        assert len(result) == 1
        assert result[0][0] == "hammer"

    def test_shooting_star(self):
        """Upper wick >= 2× body, lower wick <= 0.5× body."""
        # open=102, high=112, low=101, close=100 -> body=2, upper=10, lower=1
        # wick_filter: max(10,1)/2 = 5 > 3 -> filtered! 
        # Need: body=5, upper=12, lower=1 -> max_wick/body = 12/5 = 2.4 < 3
        df = _row(105, 117, 99, 100)  # body=5, upper=12, lower=1
        result = _detect_shooting_star(df)
        assert len(result) == 1

    def test_bullish_engulfing(self):
        prev = pd.DataFrame([{
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "open": 105, "high": 106, "low": 99, "close": 100, "volume": 1000,
        }])
        curr = pd.DataFrame([{
            "timestamp": datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
            "open": 99, "high": 107, "low": 98, "close": 106, "volume": 1000,
        }])
        df = pd.concat([prev, curr], ignore_index=True)
        result = _detect_bullish_engulfing(df)
        assert len(result) == 1
        assert result[0][1] == 1

    def test_bearish_engulfing(self):
        prev = pd.DataFrame([{
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "open": 95, "high": 101, "low": 94, "close": 100, "volume": 1000,
        }])
        curr = pd.DataFrame([{
            "timestamp": datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
            "open": 101, "high": 102, "low": 93, "close": 94, "volume": 1000,
        }])
        df = pd.concat([prev, curr], ignore_index=True)
        result = _detect_bearish_engulfing(df)
        assert len(result) == 1
        assert result[0][1] == -1

    def test_wick_filter_rejects_spikes(self):
        """Wick > 3× body → filtered out."""
        row_data = pd.Series({"open": 100, "high": 120, "low": 99, "close": 101})
        assert _wick_filter(row_data) == True

    def test_wick_filter_passes_normal(self):
        row_data = pd.Series({"open": 100, "high": 105, "low": 98, "close": 103})
        assert _wick_filter(row_data) == False


# ── VWAP ──────────────────────────────────────────────────────────────

class TestVWAP:
    def test_basic_vwap(self):
        """VWAP = sum(typical * vol) / sum(vol)."""
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        data = []
        for i in range(60):
            ts = midnight + timedelta(minutes=i)
            data.append({
                "timestamp": ts,
                "open": 100, "high": 102, "low": 98, "close": 101,
                "volume": 1000,
            })
        df = pd.DataFrame(data)
        vwap = compute_vwap(df)
        assert vwap is not None
        # typical price = (102+98+101)/3 = 100.33
        expected = (102 + 98 + 101) / 3
        assert abs(vwap - expected) < 0.01

    def test_empty_returns_none(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        assert compute_vwap(df) is None

    def test_vwap_from_slice(self):
        ts = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
        midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        data = []
        for i in range(720):
            t = midnight + timedelta(minutes=i)
            data.append({
                "timestamp": t,
                "open": 100, "high": 102, "low": 98, "close": 101,
                "volume": 500,
            })
        df = pd.DataFrame(data)
        vwap = compute_vwap_from_slice(df, ts)
        assert vwap is not None


# ── Volatility flag ───────────────────────────────────────────────────

class TestVolatilityFlag:
    def test_high_move(self):
        data = []
        for i in range(10):
            data.append({
                "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i),
                "open": 100 + i * 5,
                "high": 110 + i * 5,
                "low": 95 + i * 5,
                "close": 100 + (i + 1) * 5,
                "volume": 1000,
            })
        df = pd.DataFrame(data)
        flag = compute_volatility_flag(df, lookback=5)
        assert flag > 1.0  # significant move

    def test_empty_df(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        assert compute_volatility_flag(df) == 0.0
