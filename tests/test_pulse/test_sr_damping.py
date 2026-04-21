"""R.2 — S/R counter-level damping + breakout-waiver semantics.

Covers the three relevant overlays:
  * baseline           — zero damping (legacy behaviour; bullish into
    resistance gets 0.0 penalty).
  * sr_symmetric       — -0.15 damping for bullish into resistance; no
    waiver so even post-breakout bullish gets penalised.
  * sr_breakout_gate   — damping + waiver, so spot > resistance returns
    to zero penalty (a confirmed breakout is not a trap).

The damping logic is in ``_score_sr_proximity`` which is a pure function
reading from a ``PulseConfig`` — easy to unit-test without the full
pulse-report machinery.
"""

from __future__ import annotations

from tradingagents.agents.quant_pulse_engine import _score_sr_proximity
from tradingagents.pulse.config import get_variant_config


# User's reported trade: spot 75990.5 sitting at resistance 75990.5,
# 1h ATR ~300 → window 0.3×300 = 90. Bullish confluence (positive).
SPOT = 75990.5
RESISTANCE = 75990.5
SUPPORT = 73740.0
ATR_1H = 300.0
BULLISH_CONFLUENCE = +0.5
BEARISH_CONFLUENCE = -0.5


def test_baseline_bullish_into_resistance_gets_no_penalty():
    cfg = get_variant_config("baseline")
    factor, side = _score_sr_proximity(
        SPOT, SUPPORT, RESISTANCE, ATR_1H, BULLISH_CONFLUENCE, cfg,
    )
    assert factor == 0.0, "legacy behaviour must keep zero penalty"
    assert side is None


def test_sr_symmetric_bullish_into_resistance_penalised():
    cfg = get_variant_config("sr_symmetric")
    factor, side = _score_sr_proximity(
        SPOT, SUPPORT, RESISTANCE, ATR_1H, BULLISH_CONFLUENCE, cfg,
    )
    assert factor == -0.15
    assert side == "resistance"


def test_sr_symmetric_bearish_into_support_is_mirrored():
    cfg = get_variant_config("sr_symmetric")
    # Spot sitting right at support with bearish confluence.
    factor, side = _score_sr_proximity(
        SUPPORT, SUPPORT, RESISTANCE, ATR_1H, BEARISH_CONFLUENCE, cfg,
    )
    assert factor == +0.15
    assert side == "support"


def test_sr_breakout_gate_applies_damping_when_spot_below_resistance():
    """Pre-breakout bullish into resistance still penalised."""
    cfg = get_variant_config("sr_breakout_gate")
    factor, side = _score_sr_proximity(
        SPOT, SUPPORT, RESISTANCE, ATR_1H, BULLISH_CONFLUENCE, cfg,
    )
    assert factor == -0.15
    assert side == "resistance"


def test_sr_breakout_gate_waives_damping_when_spot_above_resistance():
    """Confirmed breakout: spot past resistance → penalty waived."""
    cfg = get_variant_config("sr_breakout_gate")
    # One ATR past resistance (still within window) but spot > resistance.
    spot_above = RESISTANCE + 50.0
    factor, side = _score_sr_proximity(
        spot_above, SUPPORT, RESISTANCE, ATR_1H, BULLISH_CONFLUENCE, cfg,
    )
    assert factor == 0.0
    assert side == "resistance"  # near_side still reported for diagnostics


def test_sr_breakout_gate_waives_support_break_for_shorts():
    cfg = get_variant_config("sr_breakout_gate")
    spot_below = SUPPORT - 20.0
    factor, side = _score_sr_proximity(
        spot_below, SUPPORT, RESISTANCE, ATR_1H, BEARISH_CONFLUENCE, cfg,
    )
    assert factor == 0.0
    assert side == "support"


def test_agreement_amplifier_unchanged_under_all_variants():
    """The existing bullish-near-support and bearish-near-resistance
    amplifications must not regress under any variant."""
    for name in ("baseline", "sr_symmetric", "sr_breakout_gate"):
        cfg = get_variant_config(name)
        # Bullish near support
        f, s = _score_sr_proximity(
            SUPPORT, SUPPORT, RESISTANCE, ATR_1H, BULLISH_CONFLUENCE, cfg,
        )
        assert f > 0 and s == "support", f"{name}: lost bullish-into-support amp"
        # Bearish near resistance
        f, s = _score_sr_proximity(
            RESISTANCE, SUPPORT, RESISTANCE, ATR_1H, BEARISH_CONFLUENCE, cfg,
        )
        assert f < 0 and s == "resistance", f"{name}: lost bearish-into-resistance amp"
