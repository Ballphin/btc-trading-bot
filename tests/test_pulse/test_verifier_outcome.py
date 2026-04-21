"""R.3 — verifier outcome computation.

Covers:
  * Funding deducted correctly for longs and shorts (BLOCKER #2).
  * No-lookahead: only bars > entry_ts contribute.
  * Time-expiry exit uses last-bar close.
  * Post-expiry diagnostics only populated for time-expiry outcomes.
  * Day-bucketing produces one fetch per day regardless of config count.
  * Pending pulses deduplicate against existing outcomes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts.pulse_verifier import (
    bucket_by_day,
    compute_outcome,
    load_pending_pulses,
)


def _make_1m_frame(entry_ts: datetime, n_bars: int, *,
                   hi: float = 101.0, lo: float = 99.0,
                   close: float = 100.0) -> pd.DataFrame:
    ts = [entry_ts + timedelta(minutes=i) for i in range(1, n_bars + 1)]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts, utc=True),
        "open": [100.0] * n_bars,
        "high": [hi] * n_bars,
        "low": [lo] * n_bars,
        "close": [close] * n_bars,
        "volume": [1.0] * n_bars,
    })


def _base_pulse(ts: datetime, **kw) -> dict:
    p = {
        "ts": ts.isoformat(),
        "price": 100.0,
        "signal": "BUY",
        "stop_loss": 98.0,
        "take_profit": 104.0,
        "hold_minutes": 60,
        "ensemble_tick_id": "2026-04-20T09:30:00Z-aaaaaaaa",
        "config_name": "baseline",
        "confidence": 0.7,
        "regime_mode": "mixed",
    }
    p.update(kw)
    return p


def test_funding_deducted_for_long(monkeypatch):
    """Long paying +0.0001 funding for 1 settlement → net = gross - 0.0001."""
    entry_ts = datetime(2026, 4, 20, 7, 59, 0, tzinfo=timezone.utc)  # 1 min before 08:00 settle
    pulse = _base_pulse(entry_ts, hold_minutes=120)
    # Make TP hit cleanly on bar 2 so gross_return is deterministic.
    frame = _make_1m_frame(entry_ts, 10, hi=104.0)  # hits TP @ 104
    frame.loc[0, "high"] = 101.0  # bar 1 does NOT hit
    frame.loc[1, "high"] = 104.0  # bar 2 hits TP

    def fake_funding(ticker, ts_str):
        # Return +0.0001 only for the 08:00 settle; else None.
        return 0.0001 if "08:00:00" in ts_str else None

    out = compute_outcome(
        pulse, frame, ticker="BTC-USD",
        atr_5m=5.0, funding_lookup=fake_funding,
    )
    assert out is not None
    assert out["exit_type"] == "tp_hit"
    assert out["funding_cost_pct"] == -0.0001  # longs pay positive rate
    assert out["net_return_pct"] == pytest.approx(
        out["gross_return_pct"] + out["fees_pct"] + out["funding_cost_pct"],
        abs=1e-9,
    )


def test_funding_mirrored_for_short(monkeypatch):
    entry_ts = datetime(2026, 4, 20, 7, 59, 0, tzinfo=timezone.utc)
    pulse = _base_pulse(entry_ts, signal="SHORT", stop_loss=102.0,
                        take_profit=96.0, hold_minutes=120)
    frame = _make_1m_frame(entry_ts, 10, lo=96.0, hi=100.5)
    frame.loc[0, "low"] = 99.0
    frame.loc[1, "low"] = 96.0  # hit TP short

    def fake_funding(ticker, ts_str):
        return 0.0001 if "08:00:00" in ts_str else None

    out = compute_outcome(
        pulse, frame, ticker="BTC-USD",
        atr_5m=5.0, funding_lookup=fake_funding,
    )
    assert out is not None
    # Short RECEIVES funding when rate is positive → funding_cost_pct > 0.
    assert out["funding_cost_pct"] == +0.0001


def test_no_funding_regression(monkeypatch):
    entry_ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
    pulse = _base_pulse(entry_ts, hold_minutes=60)
    frame = _make_1m_frame(entry_ts, 60, hi=104.0)
    frame.loc[0, "high"] = 101.0
    frame.loc[1, "high"] = 104.0

    out = compute_outcome(
        pulse, frame, ticker="BTC-USD",
        atr_5m=5.0, funding_lookup=lambda *a, **k: None,
    )
    assert out["funding_cost_pct"] == 0.0


def test_no_lookahead_bar_at_entry_excluded():
    """A bar stamped at exactly entry_ts must NOT contribute — otherwise
    the verifier would peek at the entry bar's outcome."""
    entry_ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
    # Bar at entry_ts shows TP already hit; subsequent bars flat.
    rows = [{
        "timestamp": pd.Timestamp(entry_ts),
        "open": 100.0, "high": 104.0, "low": 100.0, "close": 104.0, "volume": 1.0,
    }]
    rows += [{
        "timestamp": pd.Timestamp(entry_ts + timedelta(minutes=i)),
        "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0,
    } for i in range(1, 61)]
    frame = pd.DataFrame(rows)

    pulse = _base_pulse(entry_ts, hold_minutes=60)
    now = entry_ts + timedelta(minutes=90)  # ensure time-expiry path
    out = compute_outcome(
        pulse, frame, ticker="BTC-USD", atr_5m=5.0,
        funding_lookup=lambda *a, **k: None, now=now,
    )
    assert out["exit_type"] == "time_expiry", (
        "Entry-bar TP wick must not register — expected time-expiry"
    )


def test_time_expiry_records_post_expiry_diagnostics():
    entry_ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
    pulse = _base_pulse(entry_ts, hold_minutes=30)
    # 30 flat bars → time_expiry at bar 30 close @ 100.0.
    in_window = [{
        "timestamp": pd.Timestamp(entry_ts + timedelta(minutes=i)),
        "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 1.0,
    } for i in range(1, 31)]
    # Post-expiry: price climbs to 101 at +10min, 102 at +30min.
    post = [
        {"timestamp": pd.Timestamp(entry_ts + timedelta(minutes=40)),
         "open": 101.0, "high": 101.2, "low": 100.8, "close": 101.0, "volume": 1.0},
        {"timestamp": pd.Timestamp(entry_ts + timedelta(minutes=60)),
         "open": 102.0, "high": 102.2, "low": 101.8, "close": 102.0, "volume": 1.0},
    ]
    frame = pd.DataFrame(in_window + post)
    out = compute_outcome(
        pulse, frame, ticker="BTC-USD", atr_5m=5.0,
        funding_lookup=lambda *a, **k: None,
        post_expiry_ohlc=frame,
        now=entry_ts + timedelta(minutes=120),
    )
    assert out["exit_type"] == "time_expiry"
    # Exit at 100, +10min post-expiry close 101 → +1%.
    assert out["post_expiry_10min_return"] == pytest.approx(0.01, abs=1e-6)
    assert out["post_expiry_30min_return"] == pytest.approx(0.02, abs=1e-6)


def test_sl_tp_exits_leave_post_expiry_fields_null():
    entry_ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
    pulse = _base_pulse(entry_ts, hold_minutes=60)
    frame = _make_1m_frame(entry_ts, 60, hi=104.0)
    frame.loc[0, "high"] = 104.0  # TP hit at bar 1

    out = compute_outcome(
        pulse, frame, ticker="BTC-USD", atr_5m=5.0,
        funding_lookup=lambda *a, **k: None,
        post_expiry_ohlc=frame,
    )
    assert out["exit_type"] == "tp_hit"
    assert out["post_expiry_10min_return"] is None
    assert out["post_expiry_30min_return"] is None


def test_pending_pulse_not_resolved_returns_none():
    entry_ts = datetime.now(timezone.utc) - timedelta(minutes=5)
    pulse = _base_pulse(entry_ts, hold_minutes=60)
    # No bars yet.
    frame = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = compute_outcome(
        pulse, frame, ticker="BTC-USD", atr_5m=5.0,
        funding_lookup=lambda *a, **k: None,
        now=entry_ts + timedelta(minutes=10),
    )
    assert out is None


def test_is_weekend_flag_populated():
    # Saturday 2026-04-18
    entry_ts = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
    pulse = _base_pulse(entry_ts, hold_minutes=60)
    frame = _make_1m_frame(entry_ts, 60, hi=104.0)
    frame.loc[0, "high"] = 104.0

    out = compute_outcome(pulse, frame, ticker="BTC-USD", atr_5m=5.0,
                          funding_lookup=lambda *a, **k: None)
    assert out["is_weekend"] is True


def test_bucket_by_day_groups_correctly():
    pulses = [
        {"ts": "2026-04-20T09:30:00Z"},
        {"ts": "2026-04-20T23:45:00Z"},
        {"ts": "2026-04-21T00:15:00Z"},
        {"ts": "2026-04-22T12:00:00Z"},
    ]
    out = bucket_by_day(pulses)
    assert set(out.keys()) == {"2026-04-20", "2026-04-21", "2026-04-22"}
    assert len(out["2026-04-20"]) == 2


def test_load_pending_dedupes_against_outcomes(tmp_path):
    ticker = "BTC-USD"
    cfg_dir = tmp_path / ticker / "configs" / "baseline"
    cfg_dir.mkdir(parents=True)
    # Two pulses, one already resolved.
    entry_ts = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
    p1 = _base_pulse(entry_ts, ensemble_tick_id="tick-1")
    p2 = _base_pulse(entry_ts + timedelta(minutes=5), ensemble_tick_id="tick-2")
    with (cfg_dir / "pulse.jsonl").open("w") as f:
        f.write(json.dumps(p1) + "\n")
        f.write(json.dumps(p2) + "\n")
    with (cfg_dir / "outcomes.jsonl").open("w") as f:
        f.write(json.dumps({"ensemble_tick_id": "tick-1"}) + "\n")

    pending = load_pending_pulses(
        ticker, pulse_dir=tmp_path,
        now=entry_ts + timedelta(hours=2),
    )
    ids = {p["ensemble_tick_id"] for p in pending}
    assert ids == {"tick-2"}


def test_load_pending_skips_neutral():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        cfg_dir = base / "BTC-USD" / "configs" / "baseline"
        cfg_dir.mkdir(parents=True)
        entry_ts = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        with (cfg_dir / "pulse.jsonl").open("w") as f:
            f.write(json.dumps(_base_pulse(entry_ts, signal="NEUTRAL",
                                           ensemble_tick_id="n1")) + "\n")
        pending = load_pending_pulses("BTC-USD", pulse_dir=base,
                                      now=entry_ts + timedelta(hours=1))
        assert pending == []
