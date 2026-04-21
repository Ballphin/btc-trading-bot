"""Stage 2 Commit O — decision attribution tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradingagents.backtesting.attribution import (
    per_decision_attribution, weekly_feature_ranking,
)


def test_per_decision_picks_top_positive_and_negative():
    pulse = {
        "breakdown": {
            "1h": 0.35, "4h": 0.20, "1d": 0.05,
            "order_flow": -0.12, "sr_proximity": 0.15,
            "book_imbalance": -0.02, "regime": -0.08,
        },
        "persistence_mul": 1.2,
    }
    out = per_decision_attribution(pulse, top_n=3)
    pos_features = [e["feature"] for e in out["top_positive"]]
    neg_features = [e["feature"] for e in out["top_negative"]]
    assert pos_features == ["1h", "4h", "sr_proximity"]
    assert neg_features == ["order_flow", "regime", "book_imbalance"]
    assert out["persistence_mul"] == 1.2
    assert out["total_abs_contribution"] > 0.9


def test_empty_breakdown_returns_empty_lists():
    out = per_decision_attribution({"breakdown": {}})
    assert out["top_positive"] == []
    assert out["top_negative"] == []


def test_weekly_ranking_aggregates_and_respects_lookback(tmp_path):
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    ticker = "BTC-USD"
    d = tmp_path / ticker
    d.mkdir(parents=True)
    pulses = [
        # Inside the 7d window — should aggregate
        {"ts": (now - timedelta(days=1)).isoformat(),
         "breakdown": {"1h": 0.3, "4h": -0.2}},
        {"ts": (now - timedelta(days=3)).isoformat(),
         "breakdown": {"1h": -0.1, "order_flow": 0.4}},
        # Outside the window — should be ignored
        {"ts": (now - timedelta(days=30)).isoformat(),
         "breakdown": {"1h": 999.0}},
    ]
    with (d / "pulse.jsonl").open("w") as f:
        for p in pulses:
            f.write(json.dumps(p) + "\n")

    ranked = weekly_feature_ranking(ticker, now=now, pulse_dir=tmp_path, top_n=5)
    feats = {r["feature"]: r["cumulative_abs"] for r in ranked}
    # 1h: |0.3| + |-0.1| = 0.4 ; 4h: 0.2 ; order_flow: 0.4
    assert abs(feats["1h"] - 0.4) < 1e-9
    assert abs(feats["4h"] - 0.2) < 1e-9
    assert abs(feats["order_flow"] - 0.4) < 1e-9
    # Outside-window pulse didn't leak.
    assert feats["1h"] < 1.0
