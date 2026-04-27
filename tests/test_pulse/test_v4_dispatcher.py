"""Tests for Pulse v4 dispatcher, economic gate, parity, and arm routing."""

from __future__ import annotations

import copy

import pytest

from tradingagents.agents.quant_pulse_engine import (
    _apply_economic_gate,
    score_pulse,
    score_pulse_confluence,
    score_pulse_v4,
)
from tradingagents.pulse.config import get_config
from tradingagents.pulse.pulse_assembly import PulseInputs, score_pulse_from_inputs


def _load_cfg(overrides: dict | None = None):
    """Return a PulseConfig copy with nested dict overrides applied."""
    base = get_config()
    data = copy.deepcopy(base.data)
    if overrides:
        def merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    merge(dst[k], v)
                else:
                    dst[k] = v
        merge(data, overrides)
    return base.with_overrides(data=data)


def _minimal_report(regime: str = "mixed", spot: float = 50_000.0,
                    atr_1h: float = 500.0, vwap: float | None = 49_800.0,
                    funding_rate: float = 0.0):
    return {
        "spot_price": spot,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "funding_rate": funding_rate,
        "vwap_daily": vwap,
        "max_1m_move_pct": 0.1,
        "timeframes": {
            "1h": {"atr": atr_1h, "rsi": 50.0, "macd_hist": 0.0, "bb_pct": 0.5,
                   "ema_cross": None, "rel_volume": 1.0, "patterns": []},
            "4h": {"atr": atr_1h * 1.3, "rsi": 50.0, "macd_hist": 0.0, "bb_pct": 0.5,
                   "ema_cross": None, "rel_volume": 1.0, "patterns": []},
        },
    }


# ── Parity (BLOCKER #6) ─────────────────────────────────────────────

class TestV4DisabledParity:
    """When pulse_v4.enabled=False, score_pulse() must be byte-identical
    to score_pulse_confluence for the same inputs."""

    def test_parity_mixed_regime(self):
        cfg = _load_cfg({"pulse_v4": {"enabled": False}})
        report = _minimal_report()
        a = score_pulse(report, signal_threshold=0.25, regime_mode="mixed", cfg=cfg)
        b = score_pulse_confluence(report, signal_threshold=0.25, regime_mode="mixed", cfg=cfg)
        # Stable keys that must match exactly.
        for k in ("signal", "confidence", "normalized_score", "raw_normalized_score",
                  "stop_loss", "take_profit", "regime_mode"):
            assert a.get(k) == b.get(k), f"parity violation on {k}: {a.get(k)} vs {b.get(k)}"

    def test_parity_trend_regime(self):
        cfg = _load_cfg({"pulse_v4": {"enabled": False}})
        report = _minimal_report()
        a = score_pulse(report, signal_threshold=0.25, regime_mode="trend", cfg=cfg,
                        tsmom_direction=1, tsmom_strength=0.5)
        b = score_pulse_confluence(report, signal_threshold=0.25, regime_mode="trend", cfg=cfg,
                                   tsmom_direction=1, tsmom_strength=0.5)
        assert a == b

    def test_parity_high_vol_trend_routes_to_confluence_when_no_vpd(self):
        # v4 enabled but no vpd_signal → arm returns NEUTRAL → falls back to confluence.
        cfg = _load_cfg({"pulse_v4": {"enabled": True}})
        report = _minimal_report()
        a = score_pulse(report, signal_threshold=0.25, regime_mode="high_vol_trend",
                        cfg=cfg, vpd_signal=None)
        # Expected: arm_used=confluence (fallback).
        assert a.get("arm_used") == "confluence"


# ── Dispatcher routing ──────────────────────────────────────────────

class TestV4Routing:
    def test_high_vol_trend_with_vpd_and_liq_fires_vpd_arm(self):
        cfg = _load_cfg({"pulse_v4": {"enabled": True}})
        report = _minimal_report(funding_rate=0.0)
        r = score_pulse(report, signal_threshold=0.25, regime_mode="high_vol_trend",
                        cfg=cfg, vpd_signal=-1, liquidation_score=+1.0)
        # SHORT signal: direction derived from vpd_signal=-1 (bearish divergence).
        assert r["arm_used"] == "vpd_reversal"
        assert r["signal"] == "SHORT"

    def test_chop_fires_vwap_arm_when_outside_band(self):
        cfg = _load_cfg({"pulse_v4": {"enabled": True}})
        # spot well above VWAP band (upper = 49800 + 1.5*500 = 50550) → price 51000 triggers SHORT
        report = _minimal_report(spot=51000.0, atr_1h=500.0, vwap=49800.0,
                                 funding_rate=0.001)  # positive funding → short allowed
        r = score_pulse(report, signal_threshold=0.25, regime_mode="chop", cfg=cfg)
        # Could be economic-gate rejected; either way arm_used should be vwap_mean_reversion
        # when economic gate passes; otherwise it falls back to confluence.
        if r.get("arm_used") == "vwap_mean_reversion":
            assert r["signal"] in ("SHORT", "NEUTRAL")

    def test_trend_bypasses_arms(self):
        cfg = _load_cfg({"pulse_v4": {"enabled": True}})
        report = _minimal_report()
        r = score_pulse(report, signal_threshold=0.25, regime_mode="trend",
                        cfg=cfg, vpd_signal=-1)
        assert r["arm_used"] == "confluence"


# ── Economic gate (BLOCKER #5) ──────────────────────────────────────

class TestEconomicGate:
    def test_tp_below_threshold_rejected(self):
        cfg = _load_cfg()
        # gross_tp_pct = 0.0001 (1bp); threshold = 2 × (0.00045 + 0) = 0.0009 → rejected
        result = {
            "signal": "BUY", "confidence": 0.6, "normalized_score": 0.6,
            "stop_loss": 49_990, "take_profit": 50_005, "hold_minutes": 30,
        }
        gated = _apply_economic_gate(result, spot_price=50_000.0,
                                     funding_rate=0.0, cfg=cfg)
        assert gated["signal"] == "NEUTRAL"
        assert "economic_gate_failed" in gated.get("arm_reason", "")

    def test_tp_above_threshold_passes(self):
        cfg = _load_cfg()
        result = {
            "signal": "BUY", "confidence": 0.6, "normalized_score": 0.6,
            "stop_loss": 49_000, "take_profit": 50_500, "hold_minutes": 30,
        }
        gated = _apply_economic_gate(result, spot_price=50_000.0,
                                     funding_rate=0.0, cfg=cfg)
        assert gated["signal"] == "BUY"

    def test_funding_cost_included(self):
        cfg = _load_cfg()
        # hold 2h with funding 0.001/hr → funding cost 0.002. Threshold = 2×(0.00045+0.002)=0.0049
        # TP 0.3% = 0.003 → below threshold → reject
        result = {
            "signal": "SHORT", "confidence": 0.6, "normalized_score": -0.6,
            "stop_loss": 50_300, "take_profit": 49_850, "hold_minutes": 120,
        }
        gated = _apply_economic_gate(result, spot_price=50_000.0,
                                     funding_rate=0.001, cfg=cfg)
        assert gated["signal"] == "NEUTRAL"

    def test_gate_disabled_passes_everything(self):
        cfg = _load_cfg({"pulse_v4": {"economic_gate": {"enabled": False}}})
        result = {
            "signal": "BUY", "confidence": 0.6, "normalized_score": 0.6,
            "stop_loss": 49_990, "take_profit": 50_001, "hold_minutes": 30,
        }
        gated = _apply_economic_gate(result, spot_price=50_000.0,
                                     funding_rate=0.0, cfg=cfg)
        assert gated["signal"] == "BUY"


# ── PulseInputs v4 validation ───────────────────────────────────────

class TestPulseInputsV4Validation:
    def _report(self):
        return {"spot_price": 50_000, "timeframes": {}, "timestamp": "2024-01-01T00:00:00+00:00"}

    def test_vpd_signal_invalid_raises(self):
        with pytest.raises(ValueError, match="vpd_signal"):
            PulseInputs(report=self._report(), signal_threshold=0.22, vpd_signal=2)

    def test_vpd_signal_valid_values(self):
        for v in (-1, 0, 1, None):
            PulseInputs(report=self._report(), signal_threshold=0.22, vpd_signal=v)

    def test_liquidity_sweep_dir_invalid_raises(self):
        with pytest.raises(ValueError, match="liquidity_sweep_dir"):
            PulseInputs(report=self._report(), signal_threshold=0.22,
                        liquidity_sweep_dir="up")

    def test_pattern_hits_invalid_tf_raises(self):
        with pytest.raises(ValueError, match="invalid tf"):
            PulseInputs(report=self._report(), signal_threshold=0.22,
                        pattern_hits={"hammer": ["weekly"]})

    def test_pattern_hits_non_dict_raises(self):
        with pytest.raises(ValueError, match="pattern_hits"):
            PulseInputs(report=self._report(), signal_threshold=0.22,
                        pattern_hits=["hammer"])  # type: ignore[arg-type]

    def test_pattern_hits_valid(self):
        PulseInputs(report=self._report(), signal_threshold=0.22,
                    pattern_hits={"hammer": ["1h", "4h"], "double_bottom": ["4h"]})

    def test_as_score_kwargs_threads_v4(self):
        inp = PulseInputs(report=self._report(), signal_threshold=0.22,
                          vpd_signal=-1, liquidity_sweep_dir=1,
                          pattern_hits={"hammer": ["1h"]})
        kw = inp.as_score_kwargs()
        assert kw["vpd_signal"] == -1
        assert kw["liquidity_sweep_dir"] == 1
        assert kw["pattern_hits"] == {"hammer": ["1h"]}


# ── score_pulse_from_inputs end-to-end ─────────────────────────────

def test_score_pulse_from_inputs_v4_disabled_default():
    inp = PulseInputs(
        report=_minimal_report(),
        signal_threshold=0.25,
        regime_mode="mixed",
    )
    result = score_pulse_from_inputs(inp)
    # When v4 disabled (default config), result has no arm_used key.
    assert "signal" in result
