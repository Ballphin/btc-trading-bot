"""Stage 2 Commit N — per-regime scorecard split."""

from __future__ import annotations

import json
from pathlib import Path

from tradingagents.backtesting.scorecard import get_scorecard


def _write(dir_: Path, ticker: str, decisions):
    d = dir_ / "shadow" / ticker
    d.mkdir(parents=True, exist_ok=True)
    with (d / "decisions_scored.jsonl").open("w") as f:
        for rec in decisions:
            f.write(json.dumps(rec) + "\n")
    with (d / "decisions.jsonl").open("w") as f:
        for rec in decisions:
            f.write(json.dumps(rec) + "\n")


def _dec(regime_dir, regime_stat, was_correct, i=0):
    return {
        "ticker": "BTC-USD",
        "date": f"2026-03-{i + 1:02d}",
        "signal": "BUY",
        "price": 100.0,
        "confidence": 0.7,
        "directional_regime": regime_dir,
        "statistical_regime": regime_stat,
        "scored": True,
        "was_correct_primary": bool(was_correct),
        "actual_return_primary": 0.01 if was_correct else -0.01,
        "net_return_primary": 0.008 if was_correct else -0.012,
    }


def test_per_directional_and_statistical_regime(tmp_path):
    decs = [
        _dec("bull", "trend", True, i) for i in range(5)
    ] + [
        _dec("bear", "chop", False, i + 5) for i in range(3)
    ] + [
        _dec("range_bound", "mixed", True, i + 8) for i in range(2)
    ]
    _write(tmp_path, "BTC-USD", decs)
    sc = get_scorecard("BTC-USD", results_dir=str(tmp_path))

    assert "per_directional_regime" in sc
    assert "per_statistical_regime" in sc
    pdr = sc["per_directional_regime"]
    assert pdr["bull"]["win_rate"] == 1.0
    assert pdr["bear"]["win_rate"] == 0.0
    assert pdr["range_bound"]["sample_size"] == 2
    assert pdr["bull"]["thin_sample"] is True  # <30

    psr = sc["per_statistical_regime"]
    assert "trend" in psr and "chop" in psr and "mixed" in psr


def test_missing_regime_fields_fall_back_to_legacy_regime(tmp_path):
    decs = [
        {"ticker": "BTC-USD", "date": "2026-03-01", "signal": "BUY",
         "price": 100.0, "confidence": 0.7,
         "regime": "trend", "scored": True,
         "was_correct_primary": True,
         "actual_return_primary": 0.01,
         "net_return_primary": 0.008},
    ]
    _write(tmp_path, "BTC-USD", decs)
    sc = get_scorecard("BTC-USD", results_dir=str(tmp_path))
    # With no directional_regime field, directional bucket is "unknown".
    assert "unknown" in sc["per_directional_regime"]
    # Statistical regime falls back to the 'regime' field.
    assert "trend" in sc["per_statistical_regime"]
