"""R.4 — ensemble metrics aggregator tests.

Covers:
  * OOS validation window is the last K=20 outcomes exclusively.
  * Deflated Sharpe is called with n_strategies=5.
  * Weekend segmentation produces a correct count + eligibility flag.
  * Per-directional-regime split with thin_sample flag.
  * Empty outcomes → zero-sample well-formed JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.pulse.ensemble_metrics import (
    N_STRATEGIES_FOR_DSR,
    OOS_VALIDATION_WINDOW,
    WEEKEND_ELIGIBILITY_MIN,
    _exit_type_breakdown,
    aggregate,
    write_metrics,
)


def _outcome(i: int, *, ret: float = 0.01, weekend: bool = False,
             regime: str = "bull", exit_type: str = "tp_hit") -> dict:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    return {
        "ensemble_tick_id": f"t-{i:04d}",
        "exit_ts": (base + timedelta(minutes=i)).isoformat(),
        "net_return_pct": ret,
        "is_weekend": weekend,
        "directional_regime": regime,
        "regime_at_entry": "mixed",
        "exit_type": exit_type,
    }


def test_empty_outcomes_produce_zero_sample_blocks():
    m = aggregate([])
    for block in ("overall", "weekend", "oos_validation"):
        assert m[block]["n_signals"] == 0
        assert m[block]["sharpe_point"] == 0.0
        assert m[block]["deflated_sharpe"] == 0.0


def test_oos_validation_is_last_k_outcomes_only():
    # 60 outcomes → OOS block is LAST 20 only, overall has all 60.
    outcomes = [_outcome(i, ret=0.01) for i in range(60)]
    m = aggregate(outcomes)
    assert m["overall"]["n_signals"] == 60
    assert m["oos_validation"]["n_signals"] == OOS_VALIDATION_WINDOW
    assert OOS_VALIDATION_WINDOW == 20  # plan invariant


def test_oos_block_empty_when_leq_window():
    """If we don't have MORE than K outcomes, the OOS block is empty —
    otherwise configs would graduate on their earliest handful of
    outcomes, defeating the holdout."""
    outcomes = [_outcome(i) for i in range(OOS_VALIDATION_WINDOW)]
    m = aggregate(outcomes)
    assert m["oos_validation"]["n_signals"] == 0


def test_deflated_sharpe_uses_n_strategies_5():
    """compute_deflated_sharpe MUST be invoked with n_strategies=5 — this
    is the FWER-correction that justifies picking the best-of-5 config."""
    outcomes = [_outcome(i, ret=0.02) for i in range(60)]
    with patch(
        "tradingagents.backtesting.walk_forward.compute_deflated_sharpe",
        return_value=0.42,
    ) as mock_dsr:
        aggregate(outcomes)
    for call in mock_dsr.call_args_list:
        assert call.kwargs.get("n_strategies") == N_STRATEGIES_FOR_DSR


def test_weekend_segmentation_and_eligibility_flag():
    outcomes = (
        [_outcome(i, weekend=False) for i in range(40)]
        + [_outcome(100 + i, weekend=True) for i in range(12)]
    )
    m = aggregate(outcomes)
    assert m["weekend"]["n_signals"] == 12
    assert m["weekend"]["champion_eligible"] is True
    # Below the floor → ineligible.
    fewer = outcomes[:45]  # 40 weekday + 5 weekend
    m2 = aggregate(fewer)
    assert m2["weekend"]["n_signals"] == 5
    assert m2["weekend"]["champion_eligible"] is False
    assert WEEKEND_ELIGIBILITY_MIN == 10  # plan invariant


def test_per_directional_regime_split_with_thin_sample_flag():
    outcomes = (
        [_outcome(i, regime="bull") for i in range(35)]
        + [_outcome(100 + i, regime="bear") for i in range(10)]
    )
    m = aggregate(outcomes)
    pdr = m["overall"]["per_directional_regime"]
    assert pdr["bull"]["n"] == 35 and pdr["bull"]["thin_sample"] is False
    assert pdr["bear"]["n"] == 10 and pdr["bear"]["thin_sample"] is True


def test_exit_type_breakdown_counts():
    outcomes = (
        [_outcome(i, exit_type="tp_hit") for i in range(10)]
        + [_outcome(100 + i, exit_type="sl_hit") for i in range(7)]
        + [_outcome(200 + i, exit_type="time_expiry") for i in range(3)]
    )
    m = aggregate(outcomes)
    br = m["overall"]["exit_type_breakdown"]
    assert br == {"tp_hit": 10, "sl_hit": 7, "time_expiry": 3}


def test_write_metrics_round_trip(tmp_path):
    ticker = "BTC-USD"
    cfg_dir = tmp_path / ticker / "configs" / "baseline"
    cfg_dir.mkdir(parents=True)
    outcomes = [_outcome(i, ret=0.005) for i in range(30)]
    with (cfg_dir / "outcomes.jsonl").open("w") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")

    m = write_metrics(ticker, "baseline", pulse_dir=tmp_path)
    assert m["overall"]["n_signals"] == 30
    # Readable from disk.
    disk = json.loads((cfg_dir / "metrics.json").read_text())
    assert disk["overall"]["n_signals"] == 30
