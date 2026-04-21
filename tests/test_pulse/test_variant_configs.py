"""R.1 — Pulse variant overlay tests.

Covers:
  * list_variant_names discovers all five canonical variants.
  * baseline is byte-identical behaviour (no config delta).
  * sr_symmetric sets counter_level_damping = 0.15.
  * sr_breakout_gate enables breakout waiver on top of symmetric damping.
  * chart_patterns enables the pattern feedback path with bounded factor.
  * strict raises signal_threshold and switches TSMOM mode.
  * A missing variant falls back to baseline without crashing.
  * An invalid overlay falls back to baseline with a logged error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tradingagents.pulse.config import (
    VARIANTS_DIR,
    get_config,
    get_variant_config,
    list_variant_names,
    load_variant_overlay,
)

CANONICAL = {"baseline", "sr_symmetric", "sr_breakout_gate",
             "chart_patterns", "strict"}


def test_list_variant_names_discovers_canonical_five():
    names = set(list_variant_names())
    missing = CANONICAL - names
    assert not missing, f"Missing canonical variants: {missing}"


def test_baseline_is_passthrough():
    base = get_config()
    v = get_variant_config("baseline")
    # Baseline must preserve the base threshold exactly.
    assert v.get("confluence", "signal_threshold") == \
        base.get("confluence", "signal_threshold")
    # And leave the new keys absent (= default-off) so engine code
    # sees "no damping" at baseline.
    assert v.get("confluence", "sr_proximity", "counter_level_damping",
                 default=None) is None
    assert v.get("confluence", "chart_patterns", "enabled",
                 default=None) is None


def test_sr_symmetric_sets_damping():
    v = get_variant_config("sr_symmetric")
    assert v.get("confluence", "sr_proximity", "counter_level_damping") == 0.15
    # Breakout waiver is off in the symmetric variant — any non-True
    # value is treated as disabled by the engine.
    assert not v.get("confluence", "sr_proximity",
                     "breakout_waiver_enabled", default=False)


def test_sr_breakout_gate_enables_waiver_on_top_of_damping():
    v = get_variant_config("sr_breakout_gate")
    assert v.get("confluence", "sr_proximity", "counter_level_damping") == 0.15
    assert v.get("confluence", "sr_proximity",
                 "breakout_waiver_enabled") is True


def test_chart_patterns_variant_enables_feedback_path():
    v = get_variant_config("chart_patterns")
    cp = v.get("confluence", "chart_patterns")
    assert isinstance(cp, dict)
    assert cp.get("enabled") is True
    # Plan mandates bounded magnitude so a single pattern can't flip direction.
    assert 0 < cp.get("max_factor") <= 0.15


def test_strict_raises_threshold_and_hardens_tsmom():
    base = get_config()
    v = get_variant_config("strict")
    base_threshold = base.get("confluence", "signal_threshold")
    assert v.get("confluence", "signal_threshold") > base_threshold
    assert v.get("confluence", "tsmom_gate", "mode") == "strict"


def test_missing_variant_falls_back_to_baseline(caplog):
    v = get_variant_config("does_not_exist")
    # Should return a PulseConfig whose behaviour matches baseline.
    base = get_variant_config("baseline")
    assert v.get("confluence", "signal_threshold") == \
        base.get("confluence", "signal_threshold")


def test_invalid_overlay_falls_back(tmp_path, caplog):
    # Write a variant with an out-of-range signal_threshold.
    (tmp_path / "bogus.yaml").write_text(yaml.safe_dump({
        "confluence": {"signal_threshold": 42.0},  # out of (0, 1)
    }))
    v = get_variant_config("bogus", variants_dir=tmp_path)
    base = get_variant_config("baseline")
    assert v.get("confluence", "signal_threshold") == \
        base.get("confluence", "signal_threshold")


def test_variant_config_hash_differs_from_baseline():
    """Semantic hash must shift when the overlay actually changes behaviour."""
    base = get_variant_config("baseline")
    strict = get_variant_config("strict")
    assert base.content_hash != strict.content_hash


def test_overlay_wrapper_schema_is_supported(tmp_path):
    """Variant files may use either a flat layout or an ``overlay:``-wrapped
    layout; both must deep-merge identically."""
    flat = tmp_path / "flat.yaml"
    flat.write_text(yaml.safe_dump({
        "description": "flat",
        "confluence": {"signal_threshold": 0.30},
    }))
    wrapped = tmp_path / "wrapped.yaml"
    wrapped.write_text(yaml.safe_dump({
        "description": "wrapped",
        "overlay": {"confluence": {"signal_threshold": 0.30}},
    }))
    assert load_variant_overlay("flat", variants_dir=tmp_path) == \
           load_variant_overlay("wrapped", variants_dir=tmp_path)
