"""R.2 — Ensemble scoring tests.

Verifies:
  * Five variants × one tick produce five results sharing one tick_id.
  * UUID suffix is present (collision-proof).
  * A failing variant does not abort the ensemble.
  * Deepcopy isolation: mutating one result's report does not leak to
    other variant results on the same tick.
  * Baseline is scored first (result-ordering invariant consumed by
    the champion fallback path).
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from tradingagents.pulse.config import get_config, get_variant_config
from tradingagents.pulse.ensemble import (
    generate_ensemble_tick_id,
    score_ensemble,
    score_variant,
)
from tradingagents.pulse.pulse_assembly import PulseInputs


def _minimal_report() -> dict:
    """A report skinny enough to be scorable without live data."""
    return {
        "timestamp": "2026-04-20T09:30:00Z",
        "spot_price": 75990.0,
        "max_1m_move_pct": 0.1,
        "timeframes": {
            "1h": {"indicators": {}, "patterns": [], "atr": 300.0},
            "4h": {"indicators": {}, "patterns": [], "atr": 500.0},
            "15m": {"indicators": {}, "patterns": [], "atr": 80.0},
            "5m": {"indicators": {}, "patterns": [], "atr": 40.0},
            "1m": {"indicators": {}, "patterns": [], "atr": 10.0},
        },
    }


def _base_inputs() -> PulseInputs:
    cfg = get_config()
    return PulseInputs(
        report=_minimal_report(),
        signal_threshold=float(cfg.get("confluence", "signal_threshold", default=0.22)),
        cfg=cfg,
    )


def test_generate_tick_id_has_uuid_suffix():
    tid = generate_ensemble_tick_id()
    # ISO timestamp + '-' + 8 hex chars
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z-[0-9a-f]{8}$", tid), tid


def test_generate_tick_id_is_unique_under_same_second():
    """Two calls within the same wall-clock second must still differ."""
    ids = {generate_ensemble_tick_id() for _ in range(50)}
    assert len(ids) == 50


def test_score_ensemble_runs_all_canonical_variants():
    inputs = _base_inputs()
    results = score_ensemble(inputs)
    assert set(results.keys()) >= {
        "baseline", "sr_symmetric", "sr_breakout_gate",
        "chart_patterns", "strict",
    }


def test_all_results_share_one_tick_id():
    inputs = _base_inputs()
    results = score_ensemble(inputs)
    tids = {r["ensemble_tick_id"] for r in results.values()}
    assert len(tids) == 1


def test_tick_id_passthrough_when_supplied():
    inputs = _base_inputs()
    given = "2026-04-20T09:30:00Z-deadbeef"
    results = score_ensemble(inputs, ensemble_tick_id=given)
    for r in results.values():
        assert r["ensemble_tick_id"] == given


def test_every_result_carries_its_config_name():
    inputs = _base_inputs()
    results = score_ensemble(inputs)
    for name, r in results.items():
        assert r["config_name"] == name


def test_failing_variant_does_not_abort_ensemble():
    inputs = _base_inputs()
    from tradingagents.pulse import ensemble as ensemble_mod

    real_score = ensemble_mod.score_variant

    def boom_on_strict(base, name, **kw):
        if name == "strict":
            raise RuntimeError("synthetic")
        return real_score(base, name, **kw)

    with patch.object(ensemble_mod, "score_variant", side_effect=boom_on_strict):
        results = score_ensemble(inputs)
    assert "baseline" in results
    assert "strict" not in results  # failed — omitted, not raised


def test_deepcopy_isolation_between_variants():
    """Mutating one variant's report must not affect other variants."""
    inputs = _base_inputs()
    results = score_ensemble(inputs)
    # Mutate baseline result's breakdown dict
    results["baseline"].setdefault("breakdown", {})["injected"] = 999
    # Other variants must be untouched.
    for name, r in results.items():
        if name == "baseline":
            continue
        assert "injected" not in (r.get("breakdown") or {})


def test_baseline_scored_first_in_iteration_order():
    """The dict may be unordered, but the implementation iterates
    baseline first so the legacy fallback still works if later variants
    blow up mid-tick."""
    import tradingagents.pulse.ensemble as ensemble_mod
    seen: list[str] = []

    real = ensemble_mod.score_variant

    def spy(base, name, **kw):
        seen.append(name)
        return real(base, name, **kw)

    with patch.object(ensemble_mod, "score_variant", side_effect=spy):
        score_ensemble(_base_inputs())
    assert seen[0] == "baseline"


def test_strict_variant_produces_higher_threshold_in_result():
    """Sanity check: the strict overlay actually changes the scored
    threshold used for the NEUTRAL/BUY boundary."""
    inputs = _base_inputs()
    base_res = score_variant(inputs, "baseline")
    strict_res = score_variant(inputs, "strict")
    assert strict_res["signal_threshold"] > base_res["signal_threshold"]
