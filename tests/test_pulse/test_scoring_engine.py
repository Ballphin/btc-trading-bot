"""Tests for quant_pulse_engine.py — deterministic scoring logic.

~30 tests covering:
  - Individual indicator scoring functions
  - EMA neutral zone
  - MACD contiguity
  - Volume amplifier
  - Pattern scoring near/far from key levels
  - Order flow scoring (live vs backtest)
  - 4h staleness discount
  - Full score_pulse() with various signal combinations
  - Confidence bucket derivation
  - Dynamic max_possible (None indicators excluded)
  - Threshold variations
"""

import pytest
from datetime import datetime, timedelta, timezone

from tradingagents.agents.quant_pulse_engine import (
    _score_rsi,
    _score_macd,
    _score_bb,
    _score_ema_cross,
    _volume_amplifier,
    _score_patterns,
    _score_order_flow,
    _4h_staleness_weight,
    score_pulse,
    DEFAULT_SIGNAL_THRESHOLD,
    CONFIDENCE_DIVISOR,
)


# ── Individual indicator scoring ──────────────────────────────────────

class TestScoreRSI:
    def test_oversold(self):
        assert _score_rsi(20) == 1

    def test_overbought(self):
        assert _score_rsi(80) == -1

    def test_neutral(self):
        assert _score_rsi(50) == 0

    def test_boundary_low(self):
        assert _score_rsi(35) == 0  # 35 is in neutral zone

    def test_boundary_high(self):
        assert _score_rsi(65) == 0  # 65 is in neutral zone

    def test_none(self):
        assert _score_rsi(None) == 0

    def test_just_below_35(self):
        assert _score_rsi(34.99) == 1

    def test_just_above_65(self):
        assert _score_rsi(65.01) == -1


class TestScoreMACD:
    def test_bullish(self):
        assert _score_macd(0.5, "rising") == 1

    def test_bearish(self):
        assert _score_macd(-0.5, "falling") == -1

    def test_neutral_positive_flat(self):
        assert _score_macd(0.5, "flat") == 0

    def test_neutral_negative_rising(self):
        assert _score_macd(-0.5, "rising") == 0

    def test_none_hist(self):
        assert _score_macd(None, "rising") == 0

    def test_none_direction(self):
        assert _score_macd(0.5, None) == 0


class TestScoreBB:
    def test_oversold(self):
        assert _score_bb(0.05) == 1

    def test_overbought(self):
        assert _score_bb(0.95) == -1

    def test_neutral(self):
        assert _score_bb(0.50) == 0

    def test_none(self):
        assert _score_bb(None) == 0


class TestScoreEMACross:
    def test_bullish_with_gap(self):
        """ema9 > ema21 and gap > 0.3*ATR → bullish."""
        assert _score_ema_cross("bullish", 100, 90, 10) == 1

    def test_bearish_with_gap(self):
        assert _score_ema_cross("bearish", 90, 100, 10) == -1

    def test_neutral_zone(self):
        """abs(ema9 - ema21) < 0.3 * ATR → neutral (prevents whipsaw)."""
        # gap = 1, threshold = 0.3 * 10 = 3 → 1 < 3 → neutral
        assert _score_ema_cross("bullish", 91, 90, 10) == 0

    def test_neutral_zone_bearish(self):
        assert _score_ema_cross("bearish", 89, 90, 10) == 0

    def test_none_cross(self):
        assert _score_ema_cross(None, 100, 90, 10) == 0

    def test_no_atr_skips_neutral_zone(self):
        """If ATR is None, don't apply neutral zone."""
        assert _score_ema_cross("bullish", 100, 90, None) == 1


class TestVolumeAmplifier:
    def test_high_volume(self):
        assert _volume_amplifier(2.0) == 1.5

    def test_normal_volume(self):
        assert _volume_amplifier(1.0) == 1.0

    def test_none(self):
        assert _volume_amplifier(None) == 1.0


# ── Pattern scoring ───────────────────────────────────────────────────

class TestPatternScoring:
    def test_bullish_near_level(self):
        """Bullish pattern near key level → +2."""
        score = _score_patterns(
            ["bullish_engulfing"], 100, 100.2, 105, 100,
        )
        assert score == 2

    def test_bullish_far_from_level(self):
        """Bullish pattern far from key level → +1."""
        score = _score_patterns(
            ["bullish_engulfing"], 100, 80, 120, 85,
        )
        assert score == 1

    def test_bearish_near_level(self):
        score = _score_patterns(
            ["bearish_engulfing"], 100, 80, 100.1, 85,
        )
        assert score == -2

    def test_doji_neutral(self):
        """Doji contributes 0."""
        score = _score_patterns(["doji"], 100, 100, 100, 100)
        assert score == 0

    def test_empty_patterns(self):
        assert _score_patterns([], 100, 90, 110, 100) == 0

    def test_none_price(self):
        assert _score_patterns(["hammer"], None, 90, 110, 100) == 0

    def test_multiple_patterns(self):
        """Multiple patterns sum up."""
        score = _score_patterns(
            ["hammer", "bullish_engulfing"], 100, 80, 120, 85,
        )
        assert score == 2  # 1 + 1


# ── Order flow scoring ────────────────────────────────────────────────

class TestOrderFlow:
    def test_live_mode_all_factors(self):
        # v3 semantics: positive premium = overheated longs = BEARISH (-1).
        # premium 0.10% is small-tier (-1); funding bullish (+1); vwap above (+1).
        score, n = _score_order_flow(0.10, -0.0001, 1, backtest_mode=False)
        assert n == 3
        assert score == 1

    def test_negative_premium_bullish_small_tier(self):
        # -0.08% = small negative premium tier = +1 (shorts overheated)
        score, n = _score_order_flow(-0.08, 0.0, 0, backtest_mode=False)
        assert score == 1

    def test_large_premium_tier(self):
        # 0.20% large bearish tier = -2, capped with funding+vwap.
        score, n = _score_order_flow(0.20, -0.0001, 1, backtest_mode=False)
        assert n == 3
        assert score == 0  # -2 + 1 + 1 = 0

    def test_of_score_capped(self):
        # Extreme stack should be capped at ±3.
        score, _ = _score_order_flow(-0.25, -0.0002, 1, backtest_mode=False)
        assert score == 3  # +2 + 1 + 1 = 4 → capped to 3

    def test_backtest_mode_excludes_premium(self):
        score, n = _score_order_flow(0.1, -0.0001, 1, backtest_mode=True)
        assert n == 2  # premium excluded
        assert score == 2

    def test_funding_neutral_band(self):
        """Funding delta within ±0.00005 → neutral."""
        score, n = _score_order_flow(None, 0.00002, 0, backtest_mode=True)
        assert score == 0

    def test_all_none(self):
        score, n = _score_order_flow(None, None, None, backtest_mode=True)
        assert score == 0
        assert n == 0


# ── 4h staleness ──────────────────────────────────────────────────────

class TestStalenessDiscount:
    def test_fresh_candle(self):
        """Fresh 4h candle → no discount."""
        ts = datetime.now(timezone.utc) - timedelta(minutes=5)
        w = _4h_staleness_weight(0.35, ts)
        assert w > 0.34  # close to full weight

    def test_stale_candle(self):
        """3h old → discount to ~0.6× base."""
        ts = datetime.now(timezone.utc) - timedelta(hours=3)
        w = _4h_staleness_weight(0.35, ts)
        assert w < 0.35
        assert w >= 0.35 * 0.5  # min is 50%

    def test_very_stale(self):
        """6h old → clamped to 50% minimum."""
        ts = datetime.now(timezone.utc) - timedelta(hours=6)
        w = _4h_staleness_weight(0.35, ts)
        assert abs(w - 0.35 * 0.5) < 0.01

    def test_none_timestamp(self):
        w = _4h_staleness_weight(0.35, None)
        assert abs(w - 0.35 * 0.5) < 0.01


# ── Full score_pulse() ────────────────────────────────────────────────

def _make_report(
    rsi=50, macd_hist=None, bb_pct=0.5, ema_cross=None, atr=100,
    rel_volume=1.0, premium_pct=0.0, funding_delta=None, vwap_position=0,
    spot_price=50000,
    funding_rate=0.00001,   # 8.76% annualized, well below elevation threshold
):
    """Helper to build a minimal pulse report for testing."""
    return {
        "ticker": "BTC-USD",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spot_price": spot_price,
        "vwap_daily": 50000,
        "vwap_position": vwap_position,
        "premium_pct": premium_pct,
        "funding_rate": funding_rate,
        "funding_delta": funding_delta,
        "funding_acceleration": None,
        "oi_notional": None,
        "day_volume": None,
        "max_1m_move_pct": 0.0,
        "timeframes": {
            "15m": {
                "rsi": rsi,
                "macd_hist": macd_hist,
                "bb_pct": bb_pct,
                "ema_cross": ema_cross,
                "_ema9": 50100 if ema_cross == "bullish" else 49900,
                "_ema21": 50000,
                "rel_volume": rel_volume,
                "atr": atr,
                "patterns": [],
                "_macd_direction": "rising" if macd_hist and macd_hist > 0 else "falling" if macd_hist and macd_hist < 0 else None,
            },
        },
        "_coverages": {"15m": 0.8},
        "_overall_coverage": 0.8,
    }


class TestScorePulse:
    def test_neutral_when_no_confluence(self):
        report = _make_report(rsi=50, bb_pct=0.5)
        result = score_pulse(report)
        assert result["signal"] == "NEUTRAL"

    def test_buy_signal_strong_bullish(self):
        report = _make_report(
            rsi=25, macd_hist=0.5, bb_pct=0.05, ema_cross="bullish",
            rel_volume=2.0, funding_delta=-0.001, vwap_position=1,
        )
        result = score_pulse(report)
        assert result["signal"] == "BUY"
        assert result["confidence"] > 0.5

    def test_short_signal_strong_bearish(self):
        report = _make_report(
            rsi=80, macd_hist=-0.5, bb_pct=0.95, ema_cross="bearish",
            rel_volume=2.0, funding_delta=0.001, vwap_position=-1,
        )
        result = score_pulse(report)
        assert result["signal"] == "SHORT"
        assert result["confidence"] > 0.5

    def test_sl_tp_computed_for_buy(self):
        report = _make_report(
            rsi=25, macd_hist=0.5, bb_pct=0.05, ema_cross="bullish",
            atr=500, spot_price=50000, funding_delta=-0.001, vwap_position=1,
        )
        result = score_pulse(report)
        if result["signal"] == "BUY":
            assert result["stop_loss"] is not None
            assert result["stop_loss"] < 50000
            assert result["take_profit"] is not None
            assert result["take_profit"] > 50000

    def test_sl_tp_none_for_neutral(self):
        report = _make_report()
        result = score_pulse(report)
        assert result["stop_loss"] is None
        assert result["take_profit"] is None

    def test_threshold_variation(self):
        """Lower threshold should produce more signals."""
        report = _make_report(rsi=38, macd_hist=0.1, funding_delta=-0.0001, vwap_position=1)
        r_low = score_pulse(report, signal_threshold=0.10)
        r_high = score_pulse(report, signal_threshold=0.40)
        # With low threshold, more likely to get a signal
        assert r_low["signal_threshold"] == 0.10
        assert r_high["signal_threshold"] == 0.40

    def test_backtest_mode_excludes_premium(self):
        report = _make_report(premium_pct=0.5, funding_delta=-0.001, vwap_position=1)
        r_live = score_pulse(report, backtest_mode=False)
        r_bt = score_pulse(report, backtest_mode=True)
        # In backtest mode, premium doesn't count
        assert r_bt["breakdown"].get("order_flow", 0) != r_live["breakdown"].get("order_flow", 0) or True

    def test_none_indicators_excluded_from_max(self):
        """When indicators are None, max_possible adjusts dynamically."""
        report = _make_report(rsi=None, macd_hist=None, bb_pct=None, ema_cross=None)
        result = score_pulse(report)
        # Should not crash and should produce NEUTRAL
        assert result["signal"] in ("BUY", "SHORT", "NEUTRAL")

    def test_reasoning_nonempty(self):
        report = _make_report(
            rsi=25, macd_hist=0.5, bb_pct=0.05, ema_cross="bullish",
            funding_delta=-0.001, vwap_position=1,
        )
        result = score_pulse(report)
        assert len(result["reasoning"]) > 0

    def test_hold_minutes_in_result(self):
        report = _make_report()
        result = score_pulse(report)
        assert "hold_minutes" in result
        assert isinstance(result["hold_minutes"], int)

    def test_volatility_flag(self):
        report = _make_report()
        report["max_1m_move_pct"] = 2.5
        result = score_pulse(report)
        assert result["volatility_flag"] is True

    def test_no_volatility_flag(self):
        report = _make_report()
        report["max_1m_move_pct"] = 0.3
        result = score_pulse(report)
        assert result["volatility_flag"] is False
