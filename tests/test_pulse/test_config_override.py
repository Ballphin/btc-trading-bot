"""Tests for ContextVar override + semantic config hash + regime overlay.

Phase-A foundation (auto-tune v2). Verifies:
  * :func:`use_config_override` installs + restores cleanly.
  * Override propagates across ``contextvars.copy_context().run(...)`` into
    threadpool executor tasks (the subtle bug class called out in the plan).
  * :func:`compute_config_hash` is YAML-order / formatting insensitive.
  * :func:`compute_config_hash` is volatile-metadata insensitive.
  * :func:`compute_config_hash` ignores an unused ``regime_profiles:`` block
    when ``active_regime == "base"``.
  * :func:`compute_config_hash` DIFFERS across venue / data_source / regime.
  * :func:`deep_merge` / :func:`get_effective_config` overlay correctly.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
from pathlib import Path

import pytest
import yaml

from tradingagents.pulse.config import (
    PulseConfig,
    compute_config_hash,
    deep_merge,
    get_active_override,
    get_config,
    get_effective_config,
    use_config_override,
)


# ── Fixtures ─────────────────────────────────────────────────────────

_MIN_VALID_DATA = {
    "engine_version": 3,
    "scheduler": {"pulse_interval_minutes": 5},
    "confluence": {
        "signal_threshold": 0.22,
        "tf_weights": {"1m": 0.05, "5m": 0.10, "15m": 0.20, "1h": 0.30, "4h": 0.35},
        "persistence": {"same_direction_mul": 1.2, "flip_mul": 0.8, "neutral_prev_mul": 1.0},
        "funding_elevation": {"annualized_threshold": 0.20},
    },
    "forward_return": {"atr_multiplier": 0.5},
    "edge_gate": {"required_deflated_is_sharpe": 0.3},
}


def _make_cfg(data=None, **overrides) -> PulseConfig:
    """Build a PulseConfig without touching the filesystem."""
    d = data or dict(_MIN_VALID_DATA)
    return PulseConfig(
        data=d,
        source_path=Path("/tmp/pulse_scoring.yaml"),
        mtime=0.0,
        content_hash=compute_config_hash(d),
        active_regime=overrides.get("active_regime", "base"),
        venue=overrides.get("venue", "hyperliquid"),
        data_source=overrides.get("data_source", "hyperliquid"),
    )


# ── Semantic hash ────────────────────────────────────────────────────

class TestComputeConfigHash:
    def test_key_order_insensitive(self):
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        assert compute_config_hash(a) == compute_config_hash(b)

    def test_volatile_metadata_ignored(self):
        """Calibration-stamp rewrites must NOT invalidate the hash."""
        base = dict(_MIN_VALID_DATA)
        stamped = dict(base, calibrated_at="2026-01-01T00:00:00Z",
                       deflated_sharpe=0.42, pbo=0.3)
        assert compute_config_hash(base) == compute_config_hash(stamped)

    def test_unused_regime_profiles_ignored_for_base(self):
        """Adding a regime_profiles: block should not churn the base hash."""
        base = dict(_MIN_VALID_DATA)
        with_profiles = dict(base, regime_profiles={
            "bull": {"confluence": {"signal_threshold": 0.20}}
        })
        assert compute_config_hash(base, active_regime="base") == \
               compute_config_hash(with_profiles, active_regime="base")

    def test_nonbase_regime_hash_differs(self):
        data = dict(_MIN_VALID_DATA, regime_profiles={
            "bull": {"confluence": {"signal_threshold": 0.20}}
        })
        h_base = compute_config_hash(data, active_regime="base")
        h_bull = compute_config_hash(data, active_regime="bull")
        assert h_base != h_bull

    def test_venue_distinguishes_hash(self):
        h_hl = compute_config_hash(_MIN_VALID_DATA, venue="hyperliquid")
        h_bn = compute_config_hash(_MIN_VALID_DATA, venue="binance_futures")
        assert h_hl != h_bn

    def test_data_source_distinguishes_hash(self):
        h_hl = compute_config_hash(_MIN_VALID_DATA, data_source="hyperliquid")
        h_st = compute_config_hash(_MIN_VALID_DATA, data_source="binance+hl_stitched")
        assert h_hl != h_st

    def test_value_change_changes_hash(self):
        a = dict(_MIN_VALID_DATA)
        b = dict(_MIN_VALID_DATA, confluence=dict(
            _MIN_VALID_DATA["confluence"], signal_threshold=0.30))
        assert compute_config_hash(a) != compute_config_hash(b)

    def test_hash_is_hex_sha256(self):
        h = compute_config_hash(_MIN_VALID_DATA)
        assert len(h) == 64
        int(h, 16)  # raises if not hex


# ── Deep merge ───────────────────────────────────────────────────────

class TestDeepMerge:
    def test_nested_dict_recursed(self):
        base = {"a": {"x": 1, "y": 2}}
        patch = {"a": {"y": 99}}
        out = deep_merge(base, patch)
        assert out == {"a": {"x": 1, "y": 99}}

    def test_non_dict_replaces(self):
        base = {"a": [1, 2, 3]}
        patch = {"a": [9]}
        assert deep_merge(base, patch) == {"a": [9]}

    def test_inputs_not_mutated(self):
        base = {"a": {"x": 1}}
        patch = {"a": {"x": 2}}
        deep_merge(base, patch)
        assert base == {"a": {"x": 1}}
        assert patch == {"a": {"x": 2}}

    def test_empty_patch(self):
        assert deep_merge({"a": 1}, {}) == {"a": 1}

    def test_new_keys_added(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


# ── ContextVar override ──────────────────────────────────────────────

class TestContextVarOverride:
    def test_override_returned_by_get_config(self, tmp_path):
        # Write a real YAML on disk so get_config has a target if not overridden.
        cfg_path = tmp_path / "pulse_scoring.yaml"
        cfg_path.write_text(yaml.safe_dump(_MIN_VALID_DATA, sort_keys=False))

        override = _make_cfg()
        assert get_active_override() is None
        with use_config_override(override):
            assert get_active_override() is override
            assert get_config() is override
        assert get_active_override() is None

    def test_override_restored_on_exception(self):
        override = _make_cfg()
        with pytest.raises(RuntimeError):
            with use_config_override(override):
                assert get_active_override() is override
                raise RuntimeError("boom")
        assert get_active_override() is None

    def test_override_nesting(self):
        outer = _make_cfg()
        inner = _make_cfg()
        assert outer is not inner
        with use_config_override(outer):
            assert get_active_override() is outer
            with use_config_override(inner):
                assert get_active_override() is inner
            assert get_active_override() is outer
        assert get_active_override() is None

    def test_override_propagates_to_threadpool_via_copy_context(self):
        """The subtle bug: ContextVar must be propagated via copy_context.run."""
        override = _make_cfg(active_regime="bull", venue="binance_futures")

        def worker() -> str:
            cfg = get_config()
            return f"{cfg.active_regime}:{cfg.venue}"

        with use_config_override(override):
            ctx = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(ctx.run, worker)
                result = fut.result(timeout=5)
        assert result == "bull:binance_futures"

    def test_override_does_not_leak_to_naked_thread(self):
        """Without copy_context, the thread should NOT see the override.

        Documents expected behaviour so contributors don't rely on leaking.
        """
        override = _make_cfg(active_regime="bull")

        seen: list[str] = []

        def worker():
            # NOTE: no ctx.run wrapper — override should NOT leak here.
            seen.append(get_active_override() is override)

        with use_config_override(override):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(worker).result(timeout=5)
        # Either behaviour is defensible per interpreter, but if it leaks it
        # means our plan's "propagate explicitly" rule is a lie. Assert
        # the safe default: it does NOT leak.
        assert seen == [False]


# ── get_effective_config (regime overlay) ────────────────────────────

class TestGetEffectiveConfig:
    def _cfg_with_profiles(self) -> PulseConfig:
        data = dict(_MIN_VALID_DATA, regime_profiles={
            "bull": {"confluence": {"signal_threshold": 0.20}},
            "bear": {"confluence": {"signal_threshold": 0.26}},
            "sideways": {"confluence": {"signal_threshold": 0.30}},
        })
        return _make_cfg(data=data)

    def test_base_returns_unchanged_config(self):
        base = self._cfg_with_profiles()
        eff = get_effective_config("base", base_config=base)
        assert eff.get("confluence", "signal_threshold") == 0.22
        assert eff.active_regime == "base"

    def test_bull_overlay_applied(self):
        base = self._cfg_with_profiles()
        eff = get_effective_config("bull", base_config=base)
        assert eff.get("confluence", "signal_threshold") == 0.20
        assert eff.active_regime == "bull"

    def test_unknown_regime_falls_back_to_base(self):
        base = self._cfg_with_profiles()
        eff = get_effective_config("nonexistent", base_config=base)
        assert eff.get("confluence", "signal_threshold") == 0.22
        assert eff.active_regime == "base"

    def test_invalid_overlay_falls_back_to_base(self):
        # Construct base with overlay that would violate validator.
        data = dict(_MIN_VALID_DATA, regime_profiles={
            "bad": {"confluence": {"signal_threshold": 5.0}},  # out of range
        })
        base = _make_cfg(data=data)
        eff = get_effective_config("bad", base_config=base)
        assert eff.active_regime == "base"  # fallback
        assert eff.get("confluence", "signal_threshold") == 0.22

    def test_venue_and_source_stamped(self):
        base = self._cfg_with_profiles()
        eff = get_effective_config(
            "bull", base_config=base,
            venue="binance_futures", data_source="binance_futures",
        )
        assert eff.venue == "binance_futures"
        assert eff.data_source == "binance_futures"
        # Hash differs from a hyperliquid-bull config
        eff_hl = get_effective_config("bull", base_config=base)
        assert eff.content_hash != eff_hl.content_hash

    def test_hash_same_regardless_of_profile_addition(self):
        """Adding regime_profiles to YAML should not invalidate base scorecard hash."""
        base_plain = _make_cfg(data=dict(_MIN_VALID_DATA))
        base_with_profiles = self._cfg_with_profiles()
        eff_plain = get_effective_config("base", base_config=base_plain)
        eff_profiled = get_effective_config("base", base_config=base_with_profiles)
        assert eff_plain.content_hash == eff_profiled.content_hash
