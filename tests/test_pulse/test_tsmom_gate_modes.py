"""Tests for the 3 TSMOM gate modes: strict / confidence_weighted / disabled,
plus the parabolic soft-gate.

Uses a mutable PulseConfig stub to toggle modes between assertions.
"""

from tradingagents.agents.quant_pulse_engine import _apply_tsmom_gate


class _StubCfg:
    """Minimal stub with `.get(*keys, default=...)` honoring a dict tree."""

    def __init__(self, tree):
        self._tree = tree

    def get(self, *keys, default=None):
        cur = self._tree
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def _cfg(
    mode="confidence_weighted",
    counter_mul=1.2,
    weak_thr=0.4,
    z_soft=3.0,
):
    return _StubCfg({
        "confluence": {
            "tsmom_gate": {
                "mode": mode,
                "counter_trend_confluence_mul": counter_mul,
                "weak_trend_strength_threshold": weak_thr,
                "parabolic_soft_gate_z": z_soft,
            }
        }
    })


class TestStrictMode:
    def test_disagreement_zeros_out(self):
        # confluence says -0.5 (SHORT), TSMOM says +1 → blocked
        out, gated, reason = _apply_tsmom_gate(
            normalized=-0.5, signal_threshold=0.22,
            tsmom_direction=+1, tsmom_strength=0.6,
            z_4h_return=0.0, cfg=_cfg(mode="strict"),
        )
        assert out == 0.0
        assert gated is True
        assert reason == "strict_disagree"

    def test_agreement_passes(self):
        out, gated, _ = _apply_tsmom_gate(
            normalized=0.5, signal_threshold=0.22,
            tsmom_direction=+1, tsmom_strength=0.6,
            z_4h_return=0.0, cfg=_cfg(mode="strict"),
        )
        assert out == 0.5
        assert gated is False


class TestConfidenceWeighted:
    def test_strong_counter_trend_in_weak_trend_passes(self):
        # Counter-trend: confluence SHORT (-0.5), TSMOM UP (+1, strength 0.2 weak)
        # |-0.5| = 0.5 ≥ 1.2×0.22 = 0.264 ✓   |strength| 0.2 < 0.4 ✓
        out, gated, _ = _apply_tsmom_gate(
            normalized=-0.5, signal_threshold=0.22,
            tsmom_direction=+1, tsmom_strength=0.2,
            z_4h_return=0.0, cfg=_cfg(),
        )
        assert gated is False
        assert out == -0.5

    def test_weak_counter_trend_blocked(self):
        # |normalized| = 0.23 < 1.2×0.22 = 0.264 → fail strong_counter
        out, gated, reason = _apply_tsmom_gate(
            normalized=-0.23, signal_threshold=0.22,
            tsmom_direction=+1, tsmom_strength=0.2,
            z_4h_return=0.0, cfg=_cfg(),
        )
        assert out == 0.0
        assert gated is True
        assert reason == "counter_trend_insufficient"

    def test_strong_trend_blocks_counter(self):
        # Counter-trend strong enough, but TSMOM strength 0.7 > 0.4 → blocked
        out, gated, _ = _apply_tsmom_gate(
            normalized=-0.5, signal_threshold=0.22,
            tsmom_direction=+1, tsmom_strength=0.7,
            z_4h_return=0.0, cfg=_cfg(),
        )
        assert out == 0.0
        assert gated is True

    def test_parabolic_soft_gate_attenuates(self):
        # z_4h = 1.5, z_soft = 3.0 → multiplier = 0.5
        out, gated, _ = _apply_tsmom_gate(
            normalized=-0.5, signal_threshold=0.22,
            tsmom_direction=+1, tsmom_strength=0.2,
            z_4h_return=1.5, cfg=_cfg(z_soft=3.0),
        )
        assert gated is False
        assert abs(out - (-0.5 * 0.5)) < 1e-9

    def test_parabolic_blowoff_blocks(self):
        # z_4h >= z_soft → multiplier clamps to 0
        out, gated, reason = _apply_tsmom_gate(
            normalized=-0.5, signal_threshold=0.22,
            tsmom_direction=+1, tsmom_strength=0.2,
            z_4h_return=3.5, cfg=_cfg(z_soft=3.0),
        )
        assert out == 0.0
        assert gated is True
        assert reason == "parabolic_soft_gate"

    def test_tsmom_flat_blocks_both_directions(self):
        out, gated, reason = _apply_tsmom_gate(
            normalized=0.5, signal_threshold=0.22,
            tsmom_direction=0, tsmom_strength=0.0,
            z_4h_return=0.0, cfg=_cfg(),
        )
        assert out == 0.0
        assert gated is True
        assert reason == "tsmom_flat"


class TestDisabled:
    def test_never_gates(self):
        out, gated, _ = _apply_tsmom_gate(
            normalized=-0.5, signal_threshold=0.22,
            tsmom_direction=+1, tsmom_strength=0.9,
            z_4h_return=5.0, cfg=_cfg(mode="disabled"),
        )
        assert out == -0.5
        assert gated is False


class TestTsmomNone:
    def test_none_direction_does_not_gate(self):
        # When TSMOM unavailable (cache miss etc.), pass through.
        out, gated, _ = _apply_tsmom_gate(
            normalized=0.5, signal_threshold=0.22,
            tsmom_direction=None, tsmom_strength=None,
            z_4h_return=0.0, cfg=_cfg(mode="confidence_weighted"),
        )
        assert out == 0.5
        assert gated is False
