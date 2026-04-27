"""Golden-vector tests for strict candlestick detectors (Pulse v4)."""

from __future__ import annotations

import pandas as pd

from tradingagents.pulse.patterns.candles import (
    detect_all,
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_dark_cloud_cover,
    is_hammer,
    is_inverted_harami,
)


def _row(o, h, l, c, v=1.0):
    return pd.Series({"open": o, "high": h, "low": l, "close": c, "volume": v})


# ── Hammer: exact-threshold golden vector ────────────────────────────

class TestHammerWickRatio:
    def test_wick_ratio_0_69_fails(self):
        # total=10, lower_wick=6.9 → ratio 0.69
        row = _row(o=9.0, h=10.0, l=0.0, c=9.1)  # body top=9.1, lower_wick=9.0-0.0=9.0 → too high
        # Construct precisely: want lw/(h-l) = 0.69 with h=10, l=0 → lw=6.9 → min(o,c)=6.9
        row = _row(o=9.0, h=10.0, l=0.0, c=6.9)  # bearish close; min=6.9; lw=6.9; ratio=0.69
        assert is_hammer(row, wick_ratio=0.70) is False

    def test_wick_ratio_0_70_passes(self):
        row = _row(o=9.0, h=10.0, l=0.0, c=7.0)  # lw = min(9,7)-0 = 7; ratio = 0.70
        assert is_hammer(row, wick_ratio=0.70) is True

    def test_custom_threshold(self):
        row = _row(o=9.0, h=10.0, l=0.0, c=6.5)  # lw = 6.5
        assert is_hammer(row, wick_ratio=0.60) is True
        assert is_hammer(row, wick_ratio=0.70) is False


# ── Engulfing: volume gate golden vector ────────────────────────────

class TestEngulfingVolumeGate:
    def test_volume_1_19_fails(self):
        prev = _row(o=100, h=101, l=98, c=99, v=100)
        curr = _row(o=99, h=102, l=99, c=101, v=119)   # 1.19×
        assert is_bullish_engulfing(prev, curr, volume_mul=1.2) is False

    def test_volume_1_20_passes(self):
        prev = _row(o=100, h=101, l=98, c=99, v=100)
        curr = _row(o=99, h=102, l=99, c=101, v=120)   # 1.20×
        assert is_bullish_engulfing(prev, curr, volume_mul=1.2) is True

    def test_bearish_engulfing_strict(self):
        prev = _row(o=99, h=101, l=98, c=100, v=100)  # bullish
        curr = _row(o=100, h=101, l=97, c=98, v=130)  # bearish, engulfs
        assert is_bearish_engulfing(prev, curr, volume_mul=1.2) is True

    def test_engulfing_rejects_non_containment(self):
        prev = _row(o=100, h=101, l=99, c=100.5, v=100)  # bullish
        curr = _row(o=101, h=102, l=100.5, c=100.6, v=150)  # not bearish
        assert is_bullish_engulfing(prev, curr) is False


# ── Dark cloud cover ─────────────────────────────────────────────────

def test_dark_cloud_cover_positive():
    prev = _row(o=100, h=102, l=99, c=102)
    curr = _row(o=103, h=104, l=99, c=100)  # opens above prev.high, closes < midpoint(101)
    assert is_dark_cloud_cover(prev, curr) is True


def test_dark_cloud_cover_rejects_weak_close():
    prev = _row(o=100, h=102, l=99, c=102)
    curr = _row(o=103, h=104, l=101, c=101.5)  # close > midpoint(101)
    assert is_dark_cloud_cover(prev, curr) is False


# ── Inverted harami ──────────────────────────────────────────────────

def test_inverted_harami_bullish():
    prev = _row(o=100, h=100.5, l=99.5, c=99.8)   # small bearish body (0.2)
    curr = _row(o=99.5, h=102, l=99, c=101.5)     # large bullish body (2.0)
    assert is_inverted_harami(prev, curr) is True


def test_inverted_harami_rejects_similar_sizes():
    prev = _row(o=100, h=100.5, l=99.5, c=99.0)   # body 1.0
    curr = _row(o=99.0, h=102, l=98.5, c=101.0)   # body 2.0 — not >> prev
    # prev.body / curr.body = 0.5 → fails (needs < 0.5)
    assert is_inverted_harami(prev, curr) is False


# ── detect_all bundled output ────────────────────────────────────────

def test_detect_all_empty_df():
    assert detect_all(pd.DataFrame()) == []


def test_detect_all_hammer_only():
    df = pd.DataFrame([{"open": 9.0, "high": 10.0, "low": 0.0, "close": 7.0, "volume": 1.0}])
    out = detect_all(df)
    assert "hammer" in out


def test_detect_all_does_not_apply_wick_filter():
    """A wick 5× body must still be detected (was rejected by legacy _wick_filter)."""
    # body=1, lower wick = 5, ratio=5/6 ≈ 0.83
    df = pd.DataFrame([{"open": 5, "high": 6, "low": 0, "close": 6, "volume": 1}])
    out = detect_all(df)
    assert "hammer" in out
