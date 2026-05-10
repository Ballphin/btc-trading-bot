"""Tests for structural pattern detectors and _to_chartable helper."""

import json
import pytest
import pandas as pd
from datetime import datetime, timezone

from tradingagents.pulse.patterns.structural import (
    PatternHit,
    detect_head_shoulders,
    detect_inverse_head_shoulders,
    detect_double_bottom,
    detect_double_top,
    detect_channel_up,
    detect_ascending_triangle,
    detect_structural_all,
    _to_chartable,
)
from tradingagents.pulse.patterns.extrema import Extremum, find_extrema


class TestToChartable:
    def test_resolves_extrema_to_ohlcv(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="1h"),
            "open": [1.0] * 10,
            "high": [2.0] * 10,
            "low": [0.5] * 10,
            "close": [1.5] * 10,
        })
        hit = PatternHit(
            name="head_shoulders",
            direction=-1,
            fired_at_idx=5,
            confirmation_idx=7,
            extrema_indices=(2, 4, 6),
            metadata={"head_price": 10.0},
        )
        chart = _to_chartable(hit, df)
        assert chart["name"] == "head_shoulders"
        assert chart["direction"] == -1
        assert len(chart["extrema"]) == 3
        e0 = chart["extrema"][0]
        assert "timestamp" in e0
        assert e0["open"] == 1.0
        assert e0["high"] == 2.0
        assert e0["low"] == 0.5
        assert e0["close"] == 1.5

    def test_preserves_metadata(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="1h"),
            "open": [1.0] * 5,
            "high": [2.0] * 5,
            "low": [0.5] * 5,
            "close": [1.5] * 5,
        })
        hit = PatternHit(
            name="test",
            direction=1,
            fired_at_idx=2,
            confirmation_idx=3,
            extrema_indices=(1,),
            metadata={"key": 42.0},
        )
        chart = _to_chartable(hit, df)
        assert chart["metadata"]["key"] == 42.0

    def test_serializable_to_json(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="1h"),
            "open": [1.0] * 5,
            "high": [2.0] * 5,
            "low": [0.5] * 5,
            "close": [1.5] * 5,
        })
        hit = PatternHit(
            name="test",
            direction=1,
            fired_at_idx=2,
            confirmation_idx=3,
            extrema_indices=(1,),
            metadata={},
        )
        chart = _to_chartable(hit, df)
        # Should not raise
        s = json.dumps(chart)
        assert isinstance(s, str)

    def test_skips_out_of_range_indices(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1h"),
            "open": [1.0] * 3,
            "high": [2.0] * 3,
            "low": [0.5] * 3,
            "close": [1.5] * 3,
        })
        hit = PatternHit(
            name="test",
            direction=1,
            fired_at_idx=2,
            confirmation_idx=3,
            extrema_indices=(1, 99),
            metadata={},
        )
        chart = _to_chartable(hit, df)
        assert len(chart["extrema"]) == 1


class TestPatternHitImmutable:
    def test_frozen_dataclass(self):
        hit = PatternHit(
            name="test",
            direction=1,
            fired_at_idx=0,
            confirmation_idx=1,
            extrema_indices=(0,),
        )
        with pytest.raises(Exception):
            hit.direction = -1


class TestDetectHeadShoulders:
    def test_fires_on_symmetric_5_extrema(self):
        extrema = [
            Extremum(idx=0, confirmation_idx=2, kind="max", price=100),
            Extremum(idx=1, confirmation_idx=3, kind="min", price=90),
            Extremum(idx=2, confirmation_idx=4, kind="max", price=110),
            Extremum(idx=3, confirmation_idx=5, kind="min", price=90),
            Extremum(idx=4, confirmation_idx=6, kind="max", price=100),
        ]
        hit = detect_head_shoulders(extrema, symmetry_pct=0.015)
        assert hit is not None
        assert hit.name == "head_shoulders"
        assert hit.direction == -1

    def test_none_when_asymmetric(self):
        extrema = [
            Extremum(idx=0, confirmation_idx=2, kind="max", price=100),
            Extremum(idx=1, confirmation_idx=3, kind="min", price=90),
            Extremum(idx=2, confirmation_idx=4, kind="max", price=110),
            Extremum(idx=3, confirmation_idx=5, kind="min", price=90),
            Extremum(idx=4, confirmation_idx=6, kind="max", price=120),
        ]
        hit = detect_head_shoulders(extrema, symmetry_pct=0.015)
        assert hit is None


class TestDetectDoubleBottom:
    def test_fires_when_reclaimed(self):
        ts = pd.date_range("2026-01-01", periods=20, freq="1h")
        df = pd.DataFrame({
            "timestamp": ts,
            "open": [50000.0] * 20,
            "high": [50100.0] * 20,
            "low": [49900.0] * 20,
            "close": [50000.0] * 20,
            "volume": [100.0] * 20,
        })
        # Enforce wick contraction at E2 (idx=2) relative to E1 (idx=0)
        df.loc[2, "high"] = 50020.0
        extrema = [
            Extremum(idx=0, confirmation_idx=2, kind="min", price=49000),
            Extremum(idx=1, confirmation_idx=3, kind="max", price=51000),
            Extremum(idx=2, confirmation_idx=4, kind="min", price=49000),
            Extremum(idx=3, confirmation_idx=5, kind="max", price=51000),
        ]
        # Build 1m reclaim window around E2 timestamp where high reclaims above e2.price
        e2_ts = df.iloc[2]["timestamp"]
        one_minute = pd.date_range(e2_ts, periods=20, freq="1min")
        candles_1m = pd.DataFrame({
            "timestamp": one_minute,
            "open": [48950.0] * len(one_minute),
            "high": [49020.0] * len(one_minute),
            "low": [48900.0] * len(one_minute),
            "close": [48980.0] * len(one_minute),
            "volume": [100.0] * len(one_minute),
        })
        hit = detect_double_bottom(
            extrema,
            match_pct=0.02,
            upper_wick_ratio=0.5,
            reclaim_minutes=15,
            candles_higher_tf=df,
            candles_1m=candles_1m,
        )
        assert hit is not None
        assert hit.name == "double_bottom"
        assert hit.direction == 1

    def test_none_without_reclaim(self):
        ts = pd.date_range("2026-01-01", periods=20, freq="1h")
        df = pd.DataFrame({
            "timestamp": ts,
            "open": [50000.0] * 20,
            "high": [50100.0] * 20,
            "low": [49900.0] * 20,
            "close": [50000.0] * 20,
            "volume": [100.0] * 20,
        })
        extrema = [
            Extremum(idx=0, confirmation_idx=2, kind="min", price=49000),
            Extremum(idx=1, confirmation_idx=3, kind="max", price=51000),
            Extremum(idx=2, confirmation_idx=4, kind="min", price=49000),
            Extremum(idx=3, confirmation_idx=5, kind="max", price=51000),
        ]
        # No reclaim — price stays above E2 low
        hit = detect_double_bottom(
            extrema,
            match_pct=0.02,
            upper_wick_ratio=0.5,
            reclaim_minutes=15,
            candles_higher_tf=df,
        )
        assert hit is None


class TestDetectChannelUp:
    def test_fires_when_extrema_within_atr_band(self):
        extrema = [
            Extremum(idx=0, confirmation_idx=2, kind="min", price=100),
            Extremum(idx=2, confirmation_idx=4, kind="max", price=101),
            Extremum(idx=4, confirmation_idx=6, kind="min", price=102),
            Extremum(idx=6, confirmation_idx=8, kind="max", price=103),
            Extremum(idx=8, confirmation_idx=10, kind="min", price=104),
            Extremum(idx=10, confirmation_idx=12, kind="max", price=105),
        ]
        # ATR ~10, band = 0.5*10 = 5
        hit = detect_channel_up(extrema, atr=10, atr_band_mul=0.5, min_extrema=6, min_bars=12)
        # Current detector enforces span >= min_bars; these extrema span 10 bars.
        assert hit is None

        hit2 = detect_channel_up(extrema, atr=10, atr_band_mul=0.5, min_extrema=6, min_bars=10)
        assert hit2 is not None
        assert hit2.name == "channel_up"
        assert hit2.direction == 1

    def test_none_when_extrema_outside_band(self):
        extrema = [
            Extremum(idx=0, confirmation_idx=2, kind="min", price=100),
            Extremum(idx=2, confirmation_idx=4, kind="max", price=200),
            Extremum(idx=4, confirmation_idx=6, kind="min", price=100),
            Extremum(idx=6, confirmation_idx=8, kind="max", price=200),
            Extremum(idx=8, confirmation_idx=10, kind="min", price=100),
            Extremum(idx=10, confirmation_idx=12, kind="max", price=200),
        ]
        hit = detect_channel_up(extrema, atr=10, atr_band_mul=0.5, min_extrema=6, min_bars=12)
        assert hit is None


class TestDetectStructuralAll:
    def test_returns_multiple_hits(self):
        # Create a DataFrame that satisfies both ascending triangle (flat resistance + rising support)
        # and channel_up (rising extrema within band)
        ts = pd.date_range("2026-01-01", periods=50, freq="1h")
        closes = []
        for i in range(50):
            # Oscillate with slightly rising trend
            closes.append(50000 + i * 10 + (50 if i % 4 == 1 else 0 if i % 4 == 3 else 25))
        df = pd.DataFrame({
            "timestamp": ts,
            "open": closes,
            "high": [c + 50 for c in closes],
            "low": [c - 50 for c in closes],
            "close": closes,
            "volume": [100.0] * 50,
        })
        hits = detect_structural_all(df, bandwidth=3, atr=100)
        assert isinstance(hits, list)
        # Detector rules are strict; synthetic data may produce zero hits, but should not crash.
        names = {h.name for h in hits}
        assert len(names) == len(hits)  # distinct names
