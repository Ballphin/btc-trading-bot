"""Tests for tradingagents.backtesting.pulse_verifier."""

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tradingagents.backtesting.pulse_verifier import (
    VERIFICATION_VERSION,
    HorizonResult,
    VerifiedOutcome,
    cache_key,
    compute_hit_rates,
    dedup_signals,
    forward_hit_threshold,
    pulse_id,
    verify_pulses,
    verify_single_pulse,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_candles_1m(start_ts: datetime, n: int, base_price: float = 100.0,
                     high_offset: float = 0.5, low_offset: float = 0.5) -> pd.DataFrame:
    """Generate n 1-minute candle rows starting at start_ts."""
    rows = []
    for i in range(n):
        ts = start_ts + timedelta(minutes=i)
        o = base_price + i * 0.01
        h = o + high_offset
        l = o - low_offset
        c = o + 0.005
        rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _make_candles_5m(start_ts: datetime, n: int, base_price: float = 100.0) -> pd.DataFrame:
    """Generate n 5-minute candle rows starting at start_ts."""
    rows = []
    for i in range(n):
        ts = start_ts + timedelta(minutes=i * 5)
        o = base_price + i * 0.05
        h = o + 1.0
        l = o - 1.0
        c = o + 0.02
        rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 5000})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _make_pulse(ts: datetime, signal: str = "BUY", price: float = 100.0,
                stop_loss: float = 98.0, take_profit: float = 103.0,
                hold_minutes: int = 45, ticker: str = "BTC-USD") -> dict:
    return {
        "ticker": ticker,
        "ts": ts.isoformat(),
        "signal": signal,
        "price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "hold_minutes": hold_minutes,
        "confidence": 0.85,
        "atr_1h_at_pulse": 1.5,
    }


# ── Test: forward_hit_threshold ───────────────────────────────────────

class TestForwardHitThreshold:
    def test_atr_sqrt_time_formula(self):
        """Unified threshold matches ATR-sqrt-time formula with worked example."""
        atr_1h = 50.0
        price = 100_000.0
        h_min = 60
        atr_mul = 0.5
        expected = atr_mul * atr_1h * math.sqrt(h_min / 60.0) / price
        result = forward_hit_threshold(atr_1h, price, h_min)
        assert abs(result - expected) < 1e-10

    def test_sqrt_time_scaling(self):
        """5m threshold < 15m threshold < 1h threshold."""
        atr_1h = 50.0
        price = 100_000.0
        t5 = forward_hit_threshold(atr_1h, price, 5)
        t15 = forward_hit_threshold(atr_1h, price, 15)
        t60 = forward_hit_threshold(atr_1h, price, 60)
        assert t5 < t15 < t60

    def test_fallback_when_no_atr(self):
        """Falls back to fixed bps when ATR is None."""
        thr = forward_hit_threshold(None, 100.0, 5)
        assert thr == 5 / 10_000.0

    def test_fallback_when_atr_zero(self):
        thr = forward_hit_threshold(0.0, 100.0, 15)
        assert thr == 10 / 10_000.0


# ── Test: pulse_id ─────────────────────────────────────────────────────

class TestPulseId:
    def test_deterministic(self):
        """pulse_id is deterministic and stable across re-verification."""
        a = pulse_id("BTC-USD", "2025-01-01T00:00:00+00:00", "BUY", 100000.0)
        b = pulse_id("BTC-USD", "2025-01-01T00:00:00+00:00", "BUY", 100000.0)
        assert a == b
        assert len(a) == 16

    def test_different_inputs_different_ids(self):
        a = pulse_id("BTC-USD", "2025-01-01T00:00:00+00:00", "BUY", 100000.0)
        b = pulse_id("BTC-USD", "2025-01-01T00:00:00+00:00", "SHORT", 100000.0)
        assert a != b


# ── Test: verify_single_pulse ──────────────────────────────────────────

class TestVerifySinglePulse:
    def test_buy_tp_touched_when_high_exceeds_tp(self):
        """BUY TP touched when 1m window high >= TP."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "BUY", price=100.0, take_profit=100.4, stop_loss=99.0)
        candles_1m = _make_candles_1m(ts, 70, base_price=100.0, high_offset=0.5)
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, candles_1m, candles_5m)
        assert result.fwd_5m is not None
        assert result.fwd_5m.tp_touched is True

    def test_buy_sl_touched_when_low_below_sl(self):
        """BUY SL touched when 1m candle low <= SL."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "BUY", price=100.0, stop_loss=99.6, take_profit=105.0)
        candles_1m = _make_candles_1m(ts, 70, base_price=100.0, high_offset=0.3, low_offset=0.5)
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, candles_1m, candles_5m)
        assert result.fwd_5m is not None
        assert result.fwd_5m.sl_touched is True

    def test_short_tp_touched_when_low_below_tp(self):
        """SHORT TP touched when 1m candle low <= TP."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "SHORT", price=100.0, take_profit=99.4, stop_loss=101.0)
        candles_1m = _make_candles_1m(ts, 70, base_price=100.0, high_offset=0.3, low_offset=0.7)
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, candles_1m, candles_5m)
        assert result.fwd_5m is not None
        assert result.fwd_5m.tp_touched is True

    def test_short_sl_touched_when_high_above_sl(self):
        """SHORT SL touched when 1m candle high >= SL."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "SHORT", price=100.0, stop_loss=100.4, take_profit=95.0)
        candles_1m = _make_candles_1m(ts, 70, base_price=100.0, high_offset=0.5, low_offset=0.3)
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, candles_1m, candles_5m)
        assert result.fwd_5m is not None
        assert result.fwd_5m.sl_touched is True

    def test_forward_return_uses_5m_open(self):
        """Forward return uses first 5m candle open at or after target_ts."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "BUY", price=100.0, stop_loss=98.0, take_profit=105.0)
        candles_1m = _make_candles_1m(ts, 70, base_price=100.0)
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, candles_1m, candles_5m)
        assert result.fwd_5m is not None
        assert result.fwd_5m.exit_price is not None
        assert result.fwd_5m.raw_return is not None

    def test_empty_window_incomplete(self):
        """Window with 0 candles → window_complete=False."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "BUY", price=100.0)
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, empty, candles_5m)
        assert result.fwd_5m is not None
        assert result.fwd_5m.window_complete is False
        assert result.fwd_5m.window_candle_count == 0

    def test_missing_stop_loss_exit_type(self):
        """Missing stop_loss → exit_type='missing_sltp'."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "BUY", price=100.0)
        pulse["stop_loss"] = None
        pulse["take_profit"] = None
        candles_1m = _make_candles_1m(ts, 70, base_price=100.0)
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, candles_1m, candles_5m)
        assert result.exit_type == "missing_sltp"

    def test_hold_period_sl_hit_buy(self):
        """BUY hold-period SL hit when 1m low <= SL."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "BUY", price=100.0, stop_loss=99.6, take_profit=105.0, hold_minutes=45)
        candles_1m = _make_candles_1m(ts, 50, base_price=100.0, high_offset=0.3, low_offset=0.5)
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, candles_1m, candles_5m)
        assert result.exit_type == "sl_hit"
        assert result.exit_price is not None

    def test_hold_period_tp_hit_short(self):
        """SHORT hold-period TP hit when 1m low <= TP."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "SHORT", price=100.0, stop_loss=101.0, take_profit=99.4, hold_minutes=45)
        candles_1m = _make_candles_1m(ts, 50, base_price=100.0, high_offset=0.3, low_offset=0.7)
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, candles_1m, candles_5m)
        assert result.exit_type == "tp_hit"

    def test_serialization_roundtrip(self):
        """VerifiedOutcome survives to_dict / from_dict roundtrip."""
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulse = _make_pulse(ts, "BUY", price=100.0)
        candles_1m = _make_candles_1m(ts, 70, base_price=100.0)
        candles_5m = _make_candles_5m(ts, 15, base_price=100.0)
        result = verify_single_pulse(pulse, candles_1m, candles_5m)
        d = result.to_dict()
        restored = VerifiedOutcome.from_dict(d)
        assert restored.pulse_id == result.pulse_id
        assert restored.fwd_5m.window_high == result.fwd_5m.window_high


# ── Test: cache_key ────────────────────────────────────────────────────

class TestCacheKey:
    def test_changes_when_version_increments(self):
        """Cache key changes when verification_version increments."""
        import tradingagents.backtesting.pulse_verifier as pv
        original_version = pv.VERIFICATION_VERSION
        k1 = cache_key("BTC-USD", "2025-01-01", "2025-02-01")
        try:
            pv.VERIFICATION_VERSION = original_version + 1
            k2 = cache_key("BTC-USD", "2025-01-01", "2025-02-01")
            assert k1 != k2
        finally:
            pv.VERIFICATION_VERSION = original_version

    def test_deterministic(self):
        k1 = cache_key("BTC-USD", "2025-01-01", "2025-02-01", True)
        k2 = cache_key("BTC-USD", "2025-01-01", "2025-02-01", True)
        assert k1 == k2


# ── Test: dedup_signals ────────────────────────────────────────────────

class TestDedupSignals:
    def test_dedup_same_direction_within_hold(self):
        """Same-direction signals within hold_minutes are deduped."""
        ts1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        ts2 = ts1 + timedelta(minutes=10)
        signals = [
            {"ts": ts1.isoformat(), "signal": "BUY", "hold_minutes": 45},
            {"ts": ts2.isoformat(), "signal": "BUY", "hold_minutes": 45},
        ]
        result = dedup_signals(signals)
        assert len(result) == 1
        assert result[0]["ts"] == ts1.isoformat()

    def test_different_direction_not_deduped(self):
        ts1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        ts2 = ts1 + timedelta(minutes=10)
        signals = [
            {"ts": ts1.isoformat(), "signal": "BUY", "hold_minutes": 45},
            {"ts": ts2.isoformat(), "signal": "SHORT", "hold_minutes": 45},
        ]
        result = dedup_signals(signals)
        assert len(result) == 2

    def test_signals_after_hold_window_not_deduped(self):
        ts1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        ts2 = ts1 + timedelta(minutes=50)
        signals = [
            {"ts": ts1.isoformat(), "signal": "BUY", "hold_minutes": 45},
            {"ts": ts2.isoformat(), "signal": "BUY", "hold_minutes": 45},
        ]
        result = dedup_signals(signals)
        assert len(result) == 2


# ── Test: compute_hit_rates ────────────────────────────────────────────

class TestComputeHitRates:
    def test_correct_denominator_excludes_incomplete(self):
        """Incomplete windows are excluded from hit-rate denominator."""
        complete = HorizonResult(hit=True, window_complete=True, window_candle_count=5, window_expected=5)
        incomplete = HorizonResult(hit=False, window_complete=False, window_candle_count=2, window_expected=5)

        outcomes = [
            VerifiedOutcome(pulse_id="a", ticker="BTC-USD", ts="2025-01-01T12:00:00+00:00",
                          signal="BUY", entry_price=100.0, fwd_5m=complete),
            VerifiedOutcome(pulse_id="b", ticker="BTC-USD", ts="2025-01-01T12:30:00+00:00",
                          signal="BUY", entry_price=100.0, fwd_5m=incomplete),
        ]
        rates = compute_hit_rates(outcomes)
        assert rates["+5m"]["n_complete"] == 1
        assert rates["+5m"]["overall"] == 1.0

    def test_ci_95_present(self):
        complete = HorizonResult(hit=True, window_complete=True, window_candle_count=5, window_expected=5)
        outcomes = [
            VerifiedOutcome(pulse_id=f"p{i}", ticker="BTC-USD", ts=f"2025-01-01T{12+i}:00:00+00:00",
                          signal="BUY", entry_price=100.0, fwd_5m=complete)
            for i in range(10)
        ]
        rates = compute_hit_rates(outcomes)
        assert rates["+5m"]["ci_95"] >= 0


# ── Test: verify_pulses batch ──────────────────────────────────────────

class TestVerifyPulses:
    def test_batch_skips_neutral(self):
        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        pulses = [
            _make_pulse(ts, "BUY"),
            {"ts": ts.isoformat(), "signal": "NEUTRAL", "price": 100.0, "ticker": "BTC-USD"},
            _make_pulse(ts + timedelta(hours=1), "SHORT"),
        ]
        candles_1m = _make_candles_1m(ts, 130, base_price=100.0)
        candles_5m = _make_candles_5m(ts, 25, base_price=100.0)
        results = verify_pulses(pulses, candles_1m, candles_5m)
        assert len(results) == 2
        assert all(r.signal in ("BUY", "SHORT") for r in results)
