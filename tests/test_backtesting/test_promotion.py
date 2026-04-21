"""Stage 2 Commit P — shadow→live promotion pipeline tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts import pulse_promotion as pp


def _dec(seq, ret, state="shadow", entry_date="2026-03-02", **extra):
    """Mon Mar 2 2026 is a Monday; use Sat=2026-03-07 to span weekend."""
    d = {
        "ticker": "BTC-USD",
        "decision_sequence_number": seq,
        "net_return_primary": float(ret),
        "promotion_state": state,
        "entry_date": entry_date,
    }
    d.update(extra)
    return d


def _write(dir_: Path, ticker: str, decisions):
    d = dir_ / ticker
    d.mkdir(parents=True, exist_ok=True)
    with (d / "decisions_scored.jsonl").open("w") as f:
        for rec in decisions:
            f.write(json.dumps(rec) + "\n")


class TestShadowToCandidate:
    def test_promotes_with_weekend_span_and_positive_sharpe(self, tmp_path):
        rng = np.random.default_rng(1)
        rets = rng.normal(0.01, 0.005, 60)
        decisions = []
        for i, r in enumerate(rets):
            # Put one decision on a Saturday to span weekend.
            d = _dec(i, r, entry_date="2026-03-07" if i == 10 else "2026-03-02")
            decisions.append(d)
        _write(tmp_path, "BTC-USD", decisions)
        out = pp.classify_state(
            "BTC-USD", shadow_dir=tmp_path, drift_dir=tmp_path / "drift",
            n=50, m=10,
        )
        assert out.new_state == "candidate"

    def test_blocked_without_weekend_exposure(self, tmp_path):
        # All weekday entries → gate rejects despite good Sharpe
        rng = np.random.default_rng(2)
        rets = rng.normal(0.01, 0.005, 60)
        decisions = [_dec(i, r, entry_date="2026-03-02") for i, r in enumerate(rets)]
        _write(tmp_path, "BTC-USD", decisions)
        out = pp.classify_state(
            "BTC-USD", shadow_dir=tmp_path, drift_dir=tmp_path / "drift",
            n=50,
        )
        assert out.new_state == "shadow"
        assert "weekend" in out.reason

    def test_blocked_on_negative_sharpe(self, tmp_path):
        rng = np.random.default_rng(3)
        rets = rng.normal(-0.005, 0.02, 60)
        decisions = [_dec(i, r, entry_date="2026-03-07" if i == 5 else "2026-03-02")
                     for i, r in enumerate(rets)]
        _write(tmp_path, "BTC-USD", decisions)
        out = pp.classify_state("BTC-USD", shadow_dir=tmp_path,
                                drift_dir=tmp_path / "drift", n=50)
        assert out.new_state == "shadow"


class TestCandidateToLive:
    def test_promotes_after_m_consecutive_good(self, tmp_path):
        rng = np.random.default_rng(4)
        decisions = [
            _dec(i, r, state="candidate",
                 entry_date="2026-03-07" if i == 3 else "2026-03-02")
            for i, r in enumerate(rng.normal(0.02, 0.003, 12))
        ]
        _write(tmp_path, "BTC-USD", decisions)
        out = pp.classify_state("BTC-USD", shadow_dir=tmp_path,
                                drift_dir=tmp_path / "drift", m=10)
        assert out.new_state == "live"


class TestSequenceNumberKeying:
    """Backfill-replay regression — even if wall-clock is non-monotonic,
    the state machine operates on decision_sequence_number."""

    def test_backfill_decision_doesnt_promote_prematurely(self, tmp_path):
        decisions = [
            _dec(10, 0.02, state="shadow"),
            # A backfilled decision with EARLIER wall-clock but later seq
            _dec(11, -0.05, state="shadow", entry_date="2026-02-01"),
        ]
        _write(tmp_path, "BTC-USD", decisions)
        out = pp.classify_state("BTC-USD", shadow_dir=tmp_path,
                                drift_dir=tmp_path / "drift", n=50)
        # Too few decisions — must stay shadow.
        assert out.new_state == "shadow"
        # Sequence number equals the *last* by seq number (not by time).
        assert out.sequence_number == 11


class TestRetirement:
    def test_live_to_retired_on_three_drift_alerts(self, tmp_path):
        decisions = [_dec(i, 0.01, state="live") for i in range(10)]
        _write(tmp_path, "BTC-USD", decisions)
        drift = tmp_path / "drift"
        drift.mkdir()
        for w in range(5):
            (drift / f"BTC-USD_bull_{w}.json").write_text(json.dumps({
                "status": "below_ci" if w < 4 else "in_ci",
            }))
        out = pp.classify_state("BTC-USD", shadow_dir=tmp_path, drift_dir=drift,
                                retirement_alerts=3)
        assert out.new_state == "retired"


class TestPersistence:
    def test_apply_promotion_stamps_last_line(self, tmp_path):
        decisions = [_dec(0, 0.01, state="shadow"), _dec(1, 0.01, state="shadow")]
        _write(tmp_path, "BTC-USD", decisions)
        decision = pp.PromotionDecision(
            "BTC-USD", "shadow", "candidate", 1, "ok",
        )
        pp.apply_promotion(decision, shadow_dir=tmp_path)
        lines = (tmp_path / "BTC-USD" / "decisions_scored.jsonl").read_text().splitlines()
        last = json.loads(lines[-1])
        assert last["promotion_state"] == "candidate"
        assert last["promotion_reason"] == "ok"
        # First line untouched.
        assert json.loads(lines[0])["promotion_state"] == "shadow"
