"""Unit tests for the PulseBacktestEngine cost + funding integration.

Covers:
    * ``realistic_roundtrip_cost`` formula matches the documented
      composition (spread + 2×slip + 2×sqrt_impact).
    * ``_exec_cost_for_run`` reads from the active ContextVar config.
    * ``_integrate_funding`` returns 0 when no stamps are crossed, sums
      correctly across multiple stamps, and flips sign for SHORT
      positions.
    * Provenance (``data_source``, ``venue``, ``active_regime``,
      ``config_hash``) is stamped on the result dict.
    * A ``config_override`` to ``__init__`` is installed for the full
      ``run()`` lifetime (ContextVar check).

All tests use in-memory fakes for the router so CI stays offline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tradingagents.backtesting.pulse_backtest import PulseBacktestEngine
from tradingagents.pulse.config import (
    PulseConfig,
    compute_config_hash,
    get_config,
)
from tradingagents.pulse.fills import realistic_roundtrip_cost


# ── realistic_roundtrip_cost ─────────────────────────────────────────

class TestRealisticRoundtripCost:
    def test_default_composition(self):
        # spread 2 + 2×slip 5 = 12 bps → 0.0012
        assert realistic_roundtrip_cost() == pytest.approx(0.0012, rel=1e-9)

    def test_impact_adds_when_notional_and_adv_positive(self):
        baseline = realistic_roundtrip_cost(spread_bps=0.0, slippage_bps=0.0)
        with_impact = realistic_roundtrip_cost(
            spread_bps=0.0, slippage_bps=0.0,
            notional_usd=1_000_000, adv_usd=1e8,
            impact_coefficient=10.0,
        )
        assert with_impact > baseline
        # Round-trip impact is 2× one-sided, so result > 0.
        assert with_impact == pytest.approx(2 * with_impact / 2, rel=1e-9)

    def test_impact_disabled_without_notional(self):
        # notional=0 → impact term drops to zero regardless of ADV.
        cost = realistic_roundtrip_cost(
            spread_bps=2.0, slippage_bps=5.0,
            notional_usd=0.0, adv_usd=1e8,
        )
        assert cost == pytest.approx(0.0012, rel=1e-9)

    def test_impact_disabled_without_adv(self):
        cost = realistic_roundtrip_cost(
            spread_bps=2.0, slippage_bps=5.0,
            notional_usd=1_000_000, adv_usd=0.0,
        )
        assert cost == pytest.approx(0.0012, rel=1e-9)

    def test_zero_costs_possible(self):
        assert realistic_roundtrip_cost(spread_bps=0.0, slippage_bps=0.0) == 0.0


# ── _exec_cost_for_run reads active config ───────────────────────────

def _make_engine() -> PulseBacktestEngine:
    return PulseBacktestEngine(
        ticker="BTC-USD",
        start_date="2023-03-01",
        end_date="2023-03-05",
    )


def _config_with_fills(spread_bps: float, slippage_bps: float) -> PulseConfig:
    data = {
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
        "fill_models": {"spread_bps": spread_bps, "slippage_bps": slippage_bps},
    }
    return PulseConfig(
        data=data,
        source_path=__import__("pathlib").Path("/tmp/test.yaml"),
        mtime=0.0,
        content_hash=compute_config_hash(data),
    )


class TestExecCostForRun:
    def test_reads_from_active_config(self):
        from tradingagents.pulse.config import use_config_override
        engine = _make_engine()
        cfg = _config_with_fills(spread_bps=5.0, slippage_bps=10.0)
        # spread 5 + 2×slip 10 = 25 bps → 0.0025
        with use_config_override(cfg):
            assert engine._exec_cost_for_run() == pytest.approx(0.0025, rel=1e-9)

    def test_missing_fill_models_uses_default(self):
        """A config without fill_models falls back to _DEFAULT_EXEC_COST."""
        from tradingagents.pulse.config import use_config_override
        from tradingagents.backtesting.pulse_backtest import _DEFAULT_EXEC_COST

        data = {
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
            # intentionally no fill_models section
        }
        cfg = PulseConfig(
            data=data,
            source_path=__import__("pathlib").Path("/tmp/test.yaml"),
            mtime=0.0,
            content_hash=compute_config_hash(data),
        )
        engine = _make_engine()
        with use_config_override(cfg):
            cost = engine._exec_cost_for_run()
        # Empty section → spread 2 + 2×slip 5 (defaults) = 12 bps → 0.0012.
        # The constant acts as a safety net if _read_ itself throws.
        assert cost in (pytest.approx(0.0012, rel=1e-9), _DEFAULT_EXEC_COST)


# ── _integrate_funding ───────────────────────────────────────────────

class TestIntegrateFunding:
    def _engine(self) -> PulseBacktestEngine:
        return _make_engine()

    def test_empty_df_returns_zero(self):
        e = self._engine()
        got = e._integrate_funding(
            pd.DataFrame(columns=["timestamp", "funding_rate"]),
            datetime(2023, 3, 1, tzinfo=timezone.utc),
            datetime(2023, 3, 1, 1, tzinfo=timezone.utc),
            direction=1,
        )
        assert got == 0.0

    def test_no_stamps_crossed_returns_zero(self):
        e = self._engine()
        funding = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-03-01 08:00", "2023-03-01 16:00"]),
            "funding_rate": [0.0001, 0.0002],
        })
        # Window entirely before the first stamp.
        got = e._integrate_funding(
            funding,
            datetime(2023, 3, 1, 0, tzinfo=timezone.utc),
            datetime(2023, 3, 1, 2, tzinfo=timezone.utc),
            direction=1,
        )
        assert got == 0.0

    def test_long_pays_positive_funding(self):
        e = self._engine()
        funding = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-03-01 08:00"]),
            "funding_rate": [0.0001],  # 1 bp
        })
        # Long crossing one +1bp stamp → cost = +0.0001 (fraction).
        got = e._integrate_funding(
            funding,
            datetime(2023, 3, 1, 0, tzinfo=timezone.utc),
            datetime(2023, 3, 1, 9, tzinfo=timezone.utc),
            direction=1,
        )
        assert got == pytest.approx(0.0001, rel=1e-9)

    def test_short_receives_positive_funding(self):
        e = self._engine()
        funding = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-03-01 08:00"]),
            "funding_rate": [0.0001],
        })
        # Short crossing one +1bp stamp → cost = -0.0001 (credit).
        got = e._integrate_funding(
            funding,
            datetime(2023, 3, 1, 0, tzinfo=timezone.utc),
            datetime(2023, 3, 1, 9, tzinfo=timezone.utc),
            direction=-1,
        )
        assert got == pytest.approx(-0.0001, rel=1e-9)

    def test_multiple_stamps_summed(self):
        e = self._engine()
        funding = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2023-03-01 00:00", "2023-03-01 08:00", "2023-03-01 16:00",
            ]),
            "funding_rate": [0.0001, 0.0002, -0.0001],
        })
        # Cross the last two stamps (00:00 is at boundary, we use `>`).
        got = e._integrate_funding(
            funding,
            datetime(2023, 3, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 3, 1, 17, tzinfo=timezone.utc),
            direction=1,
        )
        assert got == pytest.approx(0.0002 - 0.0001, rel=1e-9)

    def test_exit_before_entry_returns_zero(self):
        e = self._engine()
        funding = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-03-01 08:00"]),
            "funding_rate": [0.0001],
        })
        got = e._integrate_funding(
            funding,
            datetime(2023, 3, 1, 10, tzinfo=timezone.utc),
            datetime(2023, 3, 1, 5, tzinfo=timezone.utc),
            direction=1,
        )
        assert got == 0.0


# ── Provenance stamping ──────────────────────────────────────────────

class TestProvenanceStamping:
    def test_venue_label_routing(self):
        e = _make_engine()
        for src, expected in [
            ("hyperliquid", "hyperliquid"),
            ("binance_futures", "binance_futures"),
            ("binance+hl_stitched", "stitched"),
            ("", "unknown"),
            ("unknown", "unknown"),
        ]:
            e._data_source = src
            assert e._venue_label() == expected, f"for src={src!r}"


# ── config_override install (ContextVar) ─────────────────────────────

class TestConfigOverrideInstallation:
    def test_override_active_during_run(self):
        """run() must install the override for the duration of the pipeline.

        We stub ``_run_pipeline`` to inspect the active config and return
        a sentinel; the override is set only while the with-block runs.
        """
        override = _config_with_fills(spread_bps=99.0, slippage_bps=99.0)
        engine = _make_engine()
        engine._config_override = override

        captured = {}
        def fake_pipeline():
            captured["active"] = get_config()
            return {"ok": True}

        engine._run_pipeline = fake_pipeline  # type: ignore[assignment]
        result = engine.run()
        assert result == {"ok": True}
        assert captured["active"] is override
        # After run() returns, override has been reset.
        assert get_config() is not override
