"""R.5 — champion indirection + /api/pulse/ensemble/* endpoints.

Verifies the HIGH #7 invariant: every live-path reader of
``pulse.jsonl`` routes through ``_champion_pulse_path`` so promoting a
variant swaps what the UI and risk consumers see.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect PULSE_DIR into a temp tree for each test."""
    monkeypatch.setattr(server, "PULSE_DIR", tmp_path)
    return TestClient(server.app)


def _seed_variant(tmp: Path, ticker: str, variant: str, entries: list[dict]):
    d = tmp / ticker / "configs" / variant
    d.mkdir(parents=True, exist_ok=True)
    with (d / "pulse.jsonl").open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _seed_legacy(tmp: Path, ticker: str, entries: list[dict]):
    d = tmp / ticker
    d.mkdir(parents=True, exist_ok=True)
    with (d / "pulse.jsonl").open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_default_champion_is_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PULSE_DIR", tmp_path)
    assert server._read_champion_name("BTC-USD") == "baseline"


def test_champion_path_falls_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PULSE_DIR", tmp_path)
    # No variant tree exists → path resolves to legacy pulse.jsonl.
    p = server._champion_pulse_path("BTC-USD")
    assert p.name == "pulse.jsonl"
    assert "configs" not in str(p)


def test_champion_path_prefers_variant_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PULSE_DIR", tmp_path)
    _seed_variant(tmp_path, "BTC-USD", "baseline", [{"signal": "BUY"}])
    p = server._champion_pulse_path("BTC-USD")
    assert "configs/baseline" in str(p)


def test_set_champion_swaps_read_path(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PULSE_DIR", tmp_path)
    _seed_variant(tmp_path, "BTC-USD", "baseline",
                  [{"signal": "BUY", "ts": "2026-04-20T09:00:00Z",
                    "price": 100.0}])
    _seed_variant(tmp_path, "BTC-USD", "sr_symmetric",
                  [{"signal": "NEUTRAL", "ts": "2026-04-20T09:00:00Z",
                    "price": 100.0}])

    # Default (baseline) → live reader sees BUY.
    got = server._read_last_pulse_entry("BTC-USD")
    assert got["signal"] == "BUY"

    # Promote sr_symmetric → live reader flips to NEUTRAL atomically.
    server._write_champion_name("BTC-USD", "sr_symmetric")
    got = server._read_last_pulse_entry("BTC-USD")
    assert got["signal"] == "NEUTRAL"


def test_set_champion_endpoint_rejects_unknown_variant(client):
    resp = client.post("/api/pulse/ensemble/BTC-USD/champion",
                       json={"config": "does_not_exist"})
    assert resp.status_code == 400
    assert "Unknown variant" in resp.json()["detail"]


def test_set_champion_endpoint_accepts_known_variant(client, tmp_path):
    resp = client.post("/api/pulse/ensemble/BTC-USD/champion",
                       json={"config": "sr_symmetric"})
    assert resp.status_code == 200
    assert resp.json()["config"] == "sr_symmetric"
    # Round-trip.
    resp2 = client.get("/api/pulse/ensemble/BTC-USD/champion")
    assert resp2.json()["config"] == "sr_symmetric"


def test_ensemble_latest_endpoint_shape(client, tmp_path):
    _seed_variant(tmp_path, "BTC-USD", "baseline",
                  [{"signal": "BUY", "confidence": 0.8,
                    "ensemble_tick_id": "tick-1",
                    "ts": "2026-04-20T09:00:00Z"}])
    _seed_variant(tmp_path, "BTC-USD", "sr_symmetric",
                  [{"signal": "NEUTRAL", "confidence": 0.3,
                    "ensemble_tick_id": "tick-1",
                    "ts": "2026-04-20T09:00:00Z"}])
    resp = client.get("/api/pulse/ensemble/BTC-USD")
    data = resp.json()
    assert data["champion"] == "baseline"
    assert data["champion_signal"] == "BUY"
    assert data["n_variants"] == 2
    # 1 of 2 agreed with champion → 0.5.
    assert data["agreement_score"] == 0.5


def test_ensemble_disagreements_endpoint(client, tmp_path):
    _seed_variant(tmp_path, "BTC-USD", "baseline", [
        {"signal": "BUY", "ensemble_tick_id": "t1", "ts": "2026-04-20T09:00:00Z"},
        {"signal": "BUY", "ensemble_tick_id": "t2", "ts": "2026-04-20T09:05:00Z"},
    ])
    _seed_variant(tmp_path, "BTC-USD", "strict", [
        {"signal": "BUY", "ensemble_tick_id": "t1", "ts": "2026-04-20T09:00:00Z"},
        # Disagrees at t2.
        {"signal": "NEUTRAL", "ensemble_tick_id": "t2", "ts": "2026-04-20T09:05:00Z"},
    ])
    resp = client.get("/api/pulse/ensemble/BTC-USD/disagreements")
    data = resp.json()
    tids = {d["ensemble_tick_id"] for d in data["disagreements"]}
    assert tids == {"t2"}, "only t2 has ≥2 unique signals"


def test_ensemble_metrics_endpoint(client, tmp_path):
    d = tmp_path / "BTC-USD" / "configs" / "baseline"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"overall": {"n_signals": 42}}))
    resp = client.get("/api/pulse/ensemble/BTC-USD/metrics")
    data = resp.json()
    assert data["metrics"]["baseline"]["overall"]["n_signals"] == 42


def test_scorecard_endpoint_uses_champion_path(client, tmp_path, monkeypatch):
    """Regression for the HIGH #7 audit — scorecard must NOT bypass
    the champion indirection."""
    # Seed only the variant path (no legacy file) with a BUY.
    _seed_variant(tmp_path, "BTC-USD", "baseline",
                  [{"signal": "BUY", "scored": False,
                    "ensemble_tick_id": "t1",
                    "ts": "2026-04-20T09:00:00Z"}])
    resp = client.get("/api/pulse/scorecard/BTC-USD")
    assert resp.status_code == 200
    # If champion indirection works, scorecard sees the baseline pulse;
    # otherwise it reports 0 total because the legacy path is empty.
    assert resp.json()["total"] >= 1
