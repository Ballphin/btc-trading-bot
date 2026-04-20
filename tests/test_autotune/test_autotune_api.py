"""HTTP-level tests for the /api/pulse/autotune/* endpoints.

These tests use FastAPI's ``TestClient`` and mock ``AutoTuner.run`` so
they don't actually fetch market data. The goal is to verify:

    * POST validation (dates, regime name, n_folds / n_configs bounds)
    * Apply flow — 404 on missing, 409 on PROVISIONAL, 409 on hash
      mismatch, 400 on invalid proposed config, 200 on happy path.
    * Artifact listing + fetch endpoints.
    * Job poll endpoint returns 404 on unknown job.

The SSE streaming path is NOT tested here (TestClient doesn't handle
SSE well) — core orchestration coverage lives in
``test_autotune_core.py``; SSE is validated manually once the server is
running.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

import server as server_module
from tradingagents.backtesting.autotune import TuneReport, TuneSpec


client = TestClient(server_module.app)


# ── POST /api/pulse/autotune/{ticker} — validation ───────────────────

class TestStartAutoTuneValidation:
    def test_rejects_non_crypto_ticker(self):
        r = client.post(
            "/api/pulse/autotune/AAPL",
            json={"start_date": "2024-01-01", "end_date": "2024-03-01"},
        )
        assert r.status_code == 400
        assert "crypto-only" in r.json()["detail"]

    def test_rejects_bad_date_format(self):
        r = client.post(
            "/api/pulse/autotune/BTC-USD",
            json={"start_date": "01-01-2024", "end_date": "2024-03-01"},
        )
        assert r.status_code == 400
        assert "date format" in r.json()["detail"]

    def test_rejects_end_before_start(self):
        r = client.post(
            "/api/pulse/autotune/BTC-USD",
            json={"start_date": "2024-03-01", "end_date": "2024-01-01"},
        )
        assert r.status_code == 400

    def test_rejects_window_too_short(self):
        r = client.post(
            "/api/pulse/autotune/BTC-USD",
            json={"start_date": "2024-01-01", "end_date": "2024-01-05"},
        )
        assert r.status_code == 400
        assert "too short" in r.json()["detail"].lower()

    def test_rejects_window_too_large(self):
        r = client.post(
            "/api/pulse/autotune/BTC-USD",
            json={"start_date": "2020-01-01", "end_date": "2024-01-01"},
        )
        assert r.status_code == 400
        assert "too large" in r.json()["detail"].lower()

    def test_rejects_invalid_regime(self):
        r = client.post(
            "/api/pulse/autotune/BTC-USD",
            json={
                "start_date": "2024-01-01", "end_date": "2024-03-01",
                "active_regime": "unknown_regime",
            },
        )
        assert r.status_code == 400

    def test_rejects_out_of_range_n_folds(self):
        r = client.post(
            "/api/pulse/autotune/BTC-USD",
            json={"start_date": "2024-01-01", "end_date": "2024-03-01", "n_folds": 1},
        )
        assert r.status_code == 400

    def test_rejects_out_of_range_n_configs(self):
        r = client.post(
            "/api/pulse/autotune/BTC-USD",
            json={"start_date": "2024-01-01", "end_date": "2024-03-01", "n_configs": 999},
        )
        assert r.status_code == 400


# ── GET /api/pulse/autotune/jobs/{job_id} ────────────────────────────

class TestJobPollEndpoint:
    def test_unknown_job_returns_404(self):
        r = client.get("/api/pulse/autotune/jobs/doesnotexist")
        assert r.status_code == 404


# ── GET /api/pulse/autotune/artifacts ────────────────────────────────

class TestArtifactListing:
    def test_returns_empty_when_no_dir(self, tmp_path, monkeypatch):
        # Redirect cwd so Path("results/autotune") doesn't exist.
        monkeypatch.chdir(tmp_path)
        r = client.get("/api/pulse/autotune/artifacts")
        assert r.status_code == 200
        assert r.json() == {"artifacts": []}

    def test_lists_artifact_summary(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        art_dir = tmp_path / "results" / "autotune"
        art_dir.mkdir(parents=True)
        sample = {
            "spec": {"ticker": "BTC-USD", "active_regime": "bull",
                     "start_date": "2024-01-01", "end_date": "2024-03-01",
                     "n_folds": 3, "n_configs": 30,
                     "pulse_interval_minutes": 15, "seed": 42,
                     "max_pbo": 0.5, "min_oos_over_is_ratio": 0.5,
                     "min_deflated_sharpe": 0.3, "regime_floor_ratio": 0.8,
                     "checkpoint_dir": "cp"},
            "verdict": "PROPOSE",
            "reasons": ["all gates passed"],
            "current_config_hash": "aaa111",
            "proposed_config_hash": "bbb222",
            "proposed_config": {"confluence.signal_threshold": 0.25},
            "diff": [{"path": "confluence.signal_threshold", "old": 0.22, "new": 0.25}],
            "metrics": {},
            "per_fold": [],
            "ran_at": "2026-04-20T10:00:00+00:00",
        }
        (art_dir / "job_abc.json").write_text(json.dumps(sample))
        r = client.get("/api/pulse/autotune/artifacts")
        assert r.status_code == 200
        arts = r.json()["artifacts"]
        assert len(arts) == 1
        assert arts[0]["artifact"] == "job_abc.json"
        assert arts[0]["verdict"] == "PROPOSE"
        assert arts[0]["active_regime"] == "bull"
        assert arts[0]["n_changes"] == 1

    def test_corrupt_artifact_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        art_dir = tmp_path / "results" / "autotune"
        art_dir.mkdir(parents=True)
        (art_dir / "bad.json").write_text("{not valid json")
        r = client.get("/api/pulse/autotune/artifacts")
        assert r.status_code == 200
        # Corrupt file is silently skipped (listing shouldn't fail the whole call)
        assert r.json()["artifacts"] == []


# ── GET /api/pulse/autotune/artifacts/{name} ─────────────────────────

class TestArtifactFetchEndpoint:
    def test_path_traversal_rejected(self):
        r = client.get("/api/pulse/autotune/artifacts/..%2Fetc%2Fpasswd")
        assert r.status_code in (400, 404)

    def test_non_json_rejected(self):
        r = client.get("/api/pulse/autotune/artifacts/foo.exe")
        assert r.status_code == 400

    def test_missing_file_404(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = client.get("/api/pulse/autotune/artifacts/ghost.json")
        assert r.status_code == 404

    def test_reads_existing_artifact(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        art_dir = tmp_path / "results" / "autotune"
        art_dir.mkdir(parents=True)
        payload = {"verdict": "PROPOSE", "proposed_config": {}}
        (art_dir / "x.json").write_text(json.dumps(payload))
        r = client.get("/api/pulse/autotune/artifacts/x.json")
        assert r.status_code == 200
        assert r.json() == payload


# ── POST /api/pulse/autotune/apply ───────────────────────────────────

def _write_artifact(
    tmp_path: Path, *,
    verdict: str = "PROPOSE",
    current_hash: str = "base_hash_aaa",
    proposed: dict = None,
    active_regime: str = "base",
) -> Path:
    art_dir = tmp_path / "results" / "autotune"
    art_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": {
            "ticker": "BTC-USD",
            "active_regime": active_regime,
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
        },
        "verdict": verdict,
        "reasons": [],
        "current_config_hash": current_hash,
        "proposed_config_hash": "new_hash_bbb",
        "proposed_config": proposed if proposed is not None else
            {"confluence.signal_threshold": 0.27},
        "diff": [],
        "metrics": {
            "oos_sharpe_point": 0.8, "pbo": 0.3,
            "deflated_oos_sharpe": 0.5,
            "oos_n_trades_total": 500, "n_folds_used": 3, "n_eff": 400,
        },
        "per_fold": [],
        "ran_at": "2026-04-20T10:00:00+00:00",
    }
    path = art_dir / "apply_test.json"
    path.write_text(json.dumps(payload))
    return path


def _write_live_config(tmp_path: Path) -> Path:
    """Create a config/pulse_scoring.yaml matching the validator's schema."""
    cfg_path = tmp_path / "config" / "pulse_scoring.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump({
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
    }, sort_keys=False))
    return cfg_path


class TestApplyEndpoint:
    def test_missing_artifact_404(self):
        r = client.post(
            "/api/pulse/autotune/apply",
            json={"artifact_path": "does/not/exist.json"},
        )
        assert r.status_code == 404

    def test_rejects_provisional(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        art = _write_artifact(tmp_path, verdict="PROVISIONAL")
        r = client.post(
            "/api/pulse/autotune/apply",
            json={"artifact_path": str(art)},
        )
        assert r.status_code == 409
        assert "PROVISIONAL" in r.json()["detail"]

    def test_rejects_reject_verdict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        art = _write_artifact(tmp_path, verdict="REJECT")
        r = client.post(
            "/api/pulse/autotune/apply",
            json={"artifact_path": str(art)},
        )
        assert r.status_code == 409

    def test_rejects_hash_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        art = _write_artifact(tmp_path, current_hash="artifact_hash")
        r = client.post(
            "/api/pulse/autotune/apply",
            json={
                "artifact_path": str(art),
                "expected_current_config_hash": "different_hash",
            },
        )
        assert r.status_code == 409
        assert "drifted" in r.json()["detail"]

    def test_applies_to_base_profile(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = _write_live_config(tmp_path)
        art = _write_artifact(tmp_path)
        # Point DEFAULT_CONFIG_PATH to our fixture so the endpoint reads
        # + writes our test YAML (not the real repo one).
        monkeypatch.setattr(
            "tradingagents.pulse.config.DEFAULT_CONFIG_PATH", cfg_path,
        )
        r = client.post(
            "/api/pulse/autotune/apply",
            json={"artifact_path": str(art)},
        )
        assert r.status_code == 200, r.json()
        body = r.json()
        assert body["applied"] is True
        assert body["active_regime"] == "base"
        # Written YAML reflects the proposed value at the dotted path.
        written = yaml.safe_load(cfg_path.read_text())
        assert written["confluence"]["signal_threshold"] == 0.27
        # Calibration metadata is stamped.
        assert written["calibrated_at"] is not None
        assert written["deflated_sharpe"] == 0.5
        assert written["calibration_window"]["active_regime"] == "base"

    def test_applies_to_regime_profile(self, tmp_path, monkeypatch):
        """Bull regime apply writes into regime_profiles.bull, base untouched."""
        monkeypatch.chdir(tmp_path)
        cfg_path = _write_live_config(tmp_path)
        art = _write_artifact(
            tmp_path, active_regime="bull",
            proposed={"confluence.signal_threshold": 0.19},
        )
        monkeypatch.setattr(
            "tradingagents.pulse.config.DEFAULT_CONFIG_PATH", cfg_path,
        )
        r = client.post(
            "/api/pulse/autotune/apply",
            json={"artifact_path": str(art)},
        )
        assert r.status_code == 200, r.json()
        written = yaml.safe_load(cfg_path.read_text())
        # Base threshold unchanged
        assert written["confluence"]["signal_threshold"] == 0.22
        # Bull override present
        assert written["regime_profiles"]["bull"]["confluence"]["signal_threshold"] == 0.19

    def test_rejects_invalid_proposed_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = _write_live_config(tmp_path)
        # Proposal with signal_threshold = 5.0 → out of [0, 1) range.
        art = _write_artifact(
            tmp_path, proposed={"confluence.signal_threshold": 5.0},
        )
        monkeypatch.setattr(
            "tradingagents.pulse.config.DEFAULT_CONFIG_PATH", cfg_path,
        )
        r = client.post(
            "/api/pulse/autotune/apply",
            json={"artifact_path": str(art)},
        )
        assert r.status_code == 400
        assert "invalid" in r.json()["detail"].lower()

    def test_path_traversal_rejected(self):
        r = client.post(
            "/api/pulse/autotune/apply",
            json={"artifact_path": "../../etc/passwd"},
        )
        assert r.status_code == 400
