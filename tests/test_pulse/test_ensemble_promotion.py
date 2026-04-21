"""R.7 — per-config promotion state machine + champion selector."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.pulse import ensemble_promotion as ep


def _seed_metrics(base: Path, ticker: str, config: str, metrics: dict):
    d = base / ticker / "configs" / config
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics))


def _seed_outcomes(base: Path, ticker: str, config: str, outcomes: list[dict]):
    d = base / ticker / "configs" / config
    d.mkdir(parents=True, exist_ok=True)
    with (d / "outcomes.jsonl").open("w") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")


def _seed_state(base: Path, ticker: str, config: str, state: str):
    d = base / ticker / "configs" / config
    d.mkdir(parents=True, exist_ok=True)
    (d / "promotion.json").write_text(json.dumps({"state": state}))


def _metrics(
    n_overall: int = 60, oos_dsr: float = 0.8, weekend_n: int = 12,
    sharpe: float = 1.5,
) -> dict:
    return {
        "overall": {"n_signals": n_overall, "sharpe_point": sharpe,
                    "deflated_sharpe": oos_dsr},
        "weekend": {"n_signals": weekend_n,
                    "champion_eligible": weekend_n >= ep.WEEKEND_MIN_N},
        "oos_validation": {"n_signals": 20, "sharpe_point": sharpe,
                           "deflated_sharpe": oos_dsr},
    }


def _outcomes_with_weekend() -> list[dict]:
    """A mix that satisfies the weekend-span check."""
    out = [{"is_weekend": False} for _ in range(48)]
    out += [{"is_weekend": True} for _ in range(12)]
    return out


def test_shadow_promotes_on_all_gates_passed(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline", _metrics())
    _seed_outcomes(tmp_path, "BTC-USD", "baseline", _outcomes_with_weekend())
    r = ep.classify("BTC-USD", "baseline", pulse_dir=tmp_path,
                    drift_dir=tmp_path / "drift")
    assert r.new_state == "candidate"


def test_shadow_blocked_by_low_n_signals(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline",
                  _metrics(n_overall=20))
    _seed_outcomes(tmp_path, "BTC-USD", "baseline", _outcomes_with_weekend())
    r = ep.classify("BTC-USD", "baseline", pulse_dir=tmp_path,
                    drift_dir=tmp_path / "drift")
    assert r.new_state == "shadow"
    assert "n_signals" in r.reason


def test_shadow_blocked_by_no_weekend_span(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline", _metrics())
    _seed_outcomes(tmp_path, "BTC-USD", "baseline",
                   [{"is_weekend": False}] * 60)
    r = ep.classify("BTC-USD", "baseline", pulse_dir=tmp_path,
                    drift_dir=tmp_path / "drift")
    assert r.new_state == "shadow"
    assert "weekend" in r.reason


def test_shadow_blocked_by_weak_weekend_sample(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline",
                  _metrics(weekend_n=5))
    _seed_outcomes(tmp_path, "BTC-USD", "baseline", _outcomes_with_weekend())
    r = ep.classify("BTC-USD", "baseline", pulse_dir=tmp_path,
                    drift_dir=tmp_path / "drift")
    assert r.new_state == "shadow"


def test_shadow_blocked_by_negative_oos_dsr(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline", _metrics(oos_dsr=-0.1))
    _seed_outcomes(tmp_path, "BTC-USD", "baseline", _outcomes_with_weekend())
    r = ep.classify("BTC-USD", "baseline", pulse_dir=tmp_path,
                    drift_dir=tmp_path / "drift")
    assert r.new_state == "shadow"


def test_candidate_promotes_to_live_at_n_150(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline", _metrics(n_overall=155))
    _seed_state(tmp_path, "BTC-USD", "baseline", "candidate")
    r = ep.classify("BTC-USD", "baseline", pulse_dir=tmp_path,
                    drift_dir=tmp_path / "drift")
    assert r.new_state == "live"


def test_candidate_blocked_under_n_150(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline", _metrics(n_overall=100))
    _seed_state(tmp_path, "BTC-USD", "baseline", "candidate")
    r = ep.classify("BTC-USD", "baseline", pulse_dir=tmp_path,
                    drift_dir=tmp_path / "drift")
    assert r.new_state == "candidate"


def test_live_to_retired_on_drift_alerts(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline", _metrics(n_overall=200))
    _seed_state(tmp_path, "BTC-USD", "baseline", "live")
    drift = tmp_path / "drift"
    drift.mkdir()
    for i in range(5):
        (drift / f"BTC-USD_bull_{i}.json").write_text(json.dumps({
            "status": "below_ci" if i < 4 else "in_ci",
        }))
    r = ep.classify("BTC-USD", "baseline", pulse_dir=tmp_path, drift_dir=drift)
    assert r.new_state == "retired"


def test_apply_persists_state_change(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline", _metrics())
    _seed_outcomes(tmp_path, "BTC-USD", "baseline", _outcomes_with_weekend())
    r = ep.classify("BTC-USD", "baseline", pulse_dir=tmp_path,
                    drift_dir=tmp_path / "drift")
    ep.apply(r, pulse_dir=tmp_path)
    assert ep._load_state("BTC-USD", "baseline", pulse_dir=tmp_path) == "candidate"


# ── Champion selector tests ─────────────────────────────────────────

def test_select_champion_requires_margin(tmp_path):
    """Two eligible configs with DSR 0.80 and 0.78 → margin 0.02, too
    tight → no promotion."""
    _seed_metrics(tmp_path, "BTC-USD", "baseline",
                  _metrics(oos_dsr=0.80, sharpe=1.0))
    _seed_state(tmp_path, "BTC-USD", "baseline", "candidate")
    _seed_metrics(tmp_path, "BTC-USD", "sr_symmetric",
                  _metrics(oos_dsr=0.78, sharpe=1.0))
    _seed_state(tmp_path, "BTC-USD", "sr_symmetric", "candidate")
    out = ep.select_champion("BTC-USD", pulse_dir=tmp_path)
    assert out is None, "near-ties must not promote a champion"


def test_select_champion_promotes_on_wide_margin(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline",
                  _metrics(oos_dsr=0.95, sharpe=1.0))
    _seed_state(tmp_path, "BTC-USD", "baseline", "candidate")
    _seed_metrics(tmp_path, "BTC-USD", "sr_symmetric",
                  _metrics(oos_dsr=0.30, sharpe=1.0))
    _seed_state(tmp_path, "BTC-USD", "sr_symmetric", "candidate")
    out = ep.select_champion("BTC-USD", pulse_dir=tmp_path)
    assert out is not None
    assert out[0] == "baseline"


def test_select_champion_ignores_shadow_configs(tmp_path):
    _seed_metrics(tmp_path, "BTC-USD", "baseline",
                  _metrics(oos_dsr=0.95, sharpe=1.0))
    # baseline is still shadow — not eligible.
    _seed_state(tmp_path, "BTC-USD", "baseline", "shadow")
    out = ep.select_champion("BTC-USD", pulse_dir=tmp_path)
    assert out is None
