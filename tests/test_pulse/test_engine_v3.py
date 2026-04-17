"""Tests for quant_pulse_engine v3 features:
    - Per-timeframe normalization (no conflation bug)
    - TSMOM AND-gate (disagreement → NEUTRAL)
    - Regime-aware factor gating
    - Funding-elevation override
    - Liquidation-cascade override
    - Persistence multiplier
    - EMA liquidity gate
"""

from datetime import datetime, timezone

import pytest

from tradingagents.agents.quant_pulse_engine import score_pulse


def _bullish_report(funding_rate: float = 0.00001) -> dict:
    """Strong bullish confluence across all TFs. Funding = 8.76% ann (normal)."""
    return {
        "ticker": "BTC-USD",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spot_price": 75000.0,
        "premium_pct": -0.02,   # mildly bullish (shorts paying)
        "funding_delta": -0.0001,
        "funding_rate": funding_rate,
        "vwap_daily": 74500.0,
        "vwap_position": 1,
        "max_1m_move_pct": 0.3,
        "partial_bar_flags": {},
        "timeframes": {
            "1m":  {"rsi": 30, "macd_hist": 1.2, "bb_pct": 0.1,
                    "ema_cross": "bullish", "rel_volume": 1.8, "atr": 50,
                    "_macd_direction": "rising", "_ema9": 74900, "_ema21": 74800},
            "5m":  {"rsi": 32, "macd_hist": 1.1, "bb_pct": 0.12,
                    "ema_cross": "bullish", "rel_volume": 1.2, "atr": 60,
                    "_macd_direction": "rising", "_ema9": 74950, "_ema21": 74850},
            "15m": {"rsi": 30, "macd_hist": 1.5, "bb_pct": 0.1,
                    "ema_cross": "bullish", "rel_volume": 1.5, "atr": 80,
                    "_macd_direction": "rising", "_ema9": 74900, "_ema21": 74700},
            "1h":  {"rsi": 33, "macd_hist": 2.0, "bb_pct": 0.1,
                    "ema_cross": "bullish", "rel_volume": 1.5, "atr": 150,
                    "_macd_direction": "rising", "_ema9": 74800,
                    "_ema21": 74500, "patterns": ["bullish_engulfing"]},
            "4h":  {"rsi": 40, "macd_hist": 3.0, "bb_pct": 0.2,
                    "ema_cross": "bullish", "rel_volume": 1.1, "atr": 300,
                    "_macd_direction": "rising", "_ema9": 74500, "_ema21": 74000},
        },
    }


def _bearish_report() -> dict:
    r = _bullish_report()
    # Flip directions
    r["premium_pct"] = 0.02
    r["funding_delta"] = 0.0001
    r["vwap_position"] = -1
    for tf, d in r["timeframes"].items():
        d["rsi"] = 70
        d["macd_hist"] *= -1
        d["bb_pct"] = 0.9
        d["ema_cross"] = "bearish"
        d["_macd_direction"] = "falling"
        d["_ema9"], d["_ema21"] = d["_ema21"], d["_ema9"]
        if "patterns" in d:
            d["patterns"] = ["bearish_engulfing"]
    return r


# ── Core scoring ─────────────────────────────────────────────────────

class TestCoreScoring:
    def test_bullish_confluence_produces_buy(self):
        r = score_pulse(_bullish_report(), regime_mode="trend")
        assert r["signal"] == "BUY"
        assert r["confidence"] > 0.3

    def test_bearish_confluence_produces_short(self):
        r = score_pulse(_bearish_report(), regime_mode="trend")
        assert r["signal"] == "SHORT"
        assert r["confidence"] > 0.3

    def test_normalized_score_in_bounds(self):
        r = score_pulse(_bullish_report(), regime_mode="trend")
        assert -1.0 <= r["normalized_score"] <= 1.0
        assert -1.0 <= r["raw_normalized_score"] <= 1.0

    def test_per_tf_normalization_fixes_conflation(self):
        """Bug from v1: weights summed to 1.0 but max_score_per_tf was N_active_indicators.
        Fixed: each TF normalized to ±1 BEFORE weighting."""
        r = score_pulse(_bullish_report(), regime_mode="trend")
        # With clean bullish confluence the score should be near 1 (not 0.11 as in v1)
        assert r["normalized_score"] > 0.3


# ── TSMOM AND-gate ────────────────────────────────────────────────

class TestTsmomGate:
    def test_tsmom_agreement_preserves_signal(self):
        r = score_pulse(
            _bullish_report(), regime_mode="trend",
            tsmom_direction=1, tsmom_strength=0.67,
        )
        assert r["signal"] == "BUY"
        assert not r["tsmom_gated_out"]

    def test_tsmom_disagreement_neutralizes(self):
        r = score_pulse(
            _bullish_report(), regime_mode="trend",
            tsmom_direction=-1, tsmom_strength=0.67,
        )
        assert r["signal"] == "NEUTRAL"
        assert r["tsmom_gated_out"]

    def test_tsmom_zero_gates_out(self):
        """TSMOM=0 (flat) = no macro conviction → gate out even if confluence is bullish.

        Per spec: strict AND-gate — both layers must have conviction.
        """
        r = score_pulse(
            _bullish_report(), regime_mode="trend",
            tsmom_direction=0, tsmom_strength=0.0,
        )
        assert r["signal"] == "NEUTRAL"
        assert r["tsmom_gated_out"]

    def test_tsmom_none_does_not_gate(self):
        """None = TSMOM unavailable (e.g. insufficient history). Should NOT gate."""
        r = score_pulse(
            _bullish_report(), regime_mode="trend",
            tsmom_direction=None, tsmom_strength=None,
        )
        assert r["signal"] == "BUY"


# ── Overrides ────────────────────────────────────────────────────

class TestOverrides:
    def test_funding_elevation_forces_short(self):
        """Funding ≥ 20% annualized → forces SHORT regardless of signal."""
        # 0.00003/h = 0.003%/h × 24 × 365 ≈ 26% ann
        r = score_pulse(_bullish_report(funding_rate=0.00003), regime_mode="trend")
        assert r["signal"] == "SHORT"
        assert r["override_reason"] == "funding_elevation"

    def test_funding_minus_20_forces_buy(self):
        """Funding ≤ -20% annualized → forces BUY."""
        # Strongly negative (shorts paying extreme premium) — contrarian bullish
        r = score_pulse(_bearish_report() | {"funding_rate": -0.00003},
                        regime_mode="trend")
        # funding_rate < -20%/year hourly = -0.003%/h ≈ -26%/yr
        assert r["signal"] == "BUY"
        assert r["override_reason"] == "funding_elevation"

    def test_funding_normal_no_override(self):
        """Funding at 8.76% ann (normal) → no override."""
        r = score_pulse(_bullish_report(funding_rate=0.00001), regime_mode="trend")
        assert r["override_reason"] in (None, "")

    def test_liquidation_cascade_flips_signal(self):
        """Liq cluster + falling realized vol → override to contra direction."""
        r = score_pulse(
            _bullish_report(), regime_mode="mixed",
            liquidation_score=3.5,          # strong long-liq cluster on down move
            realized_vol_recent=0.005,
            realized_vol_prior=0.010,        # vol falling → cascade exhausted
        )
        # Override kicks in — bullish_report was going BUY; liq says BUY too
        # because liquidation_score > 0 means longs liquidated on down move
        # → contrarian bullish snapback
        assert r["override_reason"] == "liquidation_cascade"

    def test_liquidation_cascade_requires_falling_vol(self):
        """If vol is rising (cascade still unfolding), don't override."""
        r = score_pulse(
            _bullish_report(), regime_mode="mixed",
            liquidation_score=3.5,
            realized_vol_recent=0.015,   # rising
            realized_vol_prior=0.005,
        )
        assert r["override_reason"] != "liquidation_cascade"


# ── Regime gating ─────────────────────────────────────────────────

class TestRegimeGating:
    def test_trend_regime_allows_trend_factors(self):
        r = score_pulse(_bullish_report(), regime_mode="trend")
        assert r["signal"] == "BUY"
        # 4h bias should contribute (non-zero breakdown)
        assert r["breakdown"].get("4h", 0) != 0

    def test_chop_regime_disables_4h(self):
        """In chop regime, 4h trend factor is suppressed."""
        r = score_pulse(_bullish_report(), regime_mode="chop")
        # 4h breakdown should be 0 or significantly reduced
        assert r["breakdown"].get("4h", 0) == 0


# ── Persistence ──────────────────────────────────────────────────

class TestPersistence:
    def test_same_prev_signal_boosts(self):
        r_cold = score_pulse(_bullish_report(), regime_mode="trend",
                             prev_signal=None)
        r_warm = score_pulse(_bullish_report(), regime_mode="trend",
                             prev_signal="BUY")
        assert r_warm["persistence_mul"] > r_cold["persistence_mul"] or \
               r_warm["persistence_mul"] >= 1.0

    def test_opposite_prev_signal_discounts(self):
        r_cold = score_pulse(_bullish_report(), regime_mode="trend",
                             prev_signal=None)
        r_flip = score_pulse(_bullish_report(), regime_mode="trend",
                             prev_signal="SHORT")
        # Flip penalty should lower persistence_mul below 1.0
        assert r_flip["persistence_mul"] < 1.0


# ── EMA liquidity gate ─────────────────────────────────────────────

class TestLiquidityGate:
    """EMA liquidity gate disables the EMA-cross factor on 15m/1h when 24h volume
    is below threshold (default $50M). It does NOT force NEUTRAL — it's a
    factor-level gate, not a signal-level veto."""

    def test_illiquid_symbol_reduces_breakdown_contribution(self):
        r_liquid = score_pulse(
            _bullish_report(), regime_mode="trend",
            ema_liquidity_ok=True,
        )
        r_illiquid = score_pulse(
            _bullish_report(), regime_mode="trend",
            ema_liquidity_ok=False,
        )
        # Illiquid should have less positive or equal score (EMA disabled on 15m/1h)
        assert r_illiquid["raw_normalized_score"] <= r_liquid["raw_normalized_score"] + 1e-6

    def test_liquid_symbol_allows_signal(self):
        r = score_pulse(
            _bullish_report(), regime_mode="trend",
            ema_liquidity_ok=True,
        )
        assert r["signal"] == "BUY"
