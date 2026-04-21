"""Tests for the pulse drift monitor (Stage 2 Commit H)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts import pulse_drift_monitor as dm


def _write_scored(
    dir_: Path, ticker: str, regime: str, returns, ts_base: datetime
):
    d = dir_ / ticker
    d.mkdir(parents=True, exist_ok=True)
    with (d / "decisions_scored.jsonl").open("w") as f:
        for i, r in enumerate(returns):
            f.write(json.dumps({
                "ticker": ticker,
                "active_regime": regime,
                # Spread across last 25d so all fit in the 30d window.
                "scored_at": (ts_base - timedelta(hours=i * 3)).isoformat(),
                "net_return_primary": float(r),
            }) + "\n")


def _write_artifact(dir_: Path, ticker: str, regime: str, ci_lower: float):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{ticker}_{regime}.json").write_text(json.dumps({
        "spec": {"ticker": ticker, "active_regime": regime},
        "metrics": {"oos_sharpe_ci_lower": ci_lower},
    }))


@pytest.fixture
def now():
    return datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)


class TestDriftStatuses:
    def test_in_ci_when_realised_matches_artifact(self, tmp_path, now):
        # 60 synthetic returns with positive mean — Sharpe well above 0
        rng = np.random.default_rng(1)
        rets = rng.normal(0.005, 0.01, 60).tolist()
        _write_scored(tmp_path / "shadow", "BTC-USD", "bull", rets, now)
        _write_artifact(tmp_path / "autotune", "BTC-USD", "bull", ci_lower=1.0)

        r = dm.run_drift_check(
            "BTC-USD", "bull", now=now,
            shadow_dir=tmp_path / "shadow",
            autotune_dir=tmp_path / "autotune",
        )
        assert r.status == "in_ci"

    def test_below_ci_when_realised_is_low(self, tmp_path, now):
        rng = np.random.default_rng(2)
        # Very small positive mean, tight noise → Sharpe ≈ 0; artifact
        # promised a high CI-lower, so z-gap must be << -2σ.
        rets = rng.normal(0.0001, 0.005, 200).tolist()
        _write_scored(tmp_path / "shadow", "BTC-USD", "bull", rets, now)
        _write_artifact(tmp_path / "autotune", "BTC-USD", "bull", ci_lower=10.0)

        r = dm.run_drift_check(
            "BTC-USD", "bull", now=now,
            shadow_dir=tmp_path / "shadow",
            autotune_dir=tmp_path / "autotune",
        )
        assert r.status == "below_ci"
        assert r.z_gap < -dm.Z_GAP_THRESHOLD

    def test_insufficient_n_returns_that_status(self, tmp_path, now):
        _write_scored(tmp_path / "shadow", "BTC-USD", "bull",
                      [0.001] * 5, now)
        _write_artifact(tmp_path / "autotune", "BTC-USD", "bull", ci_lower=1.0)
        r = dm.run_drift_check(
            "BTC-USD", "bull", now=now,
            shadow_dir=tmp_path / "shadow",
            autotune_dir=tmp_path / "autotune",
        )
        assert r.status == "insufficient_n"

    def test_insufficient_n_when_no_artifact(self, tmp_path, now):
        _write_scored(tmp_path / "shadow", "BTC-USD", "bull",
                      [0.001] * 60, now)
        r = dm.run_drift_check(
            "BTC-USD", "bull", now=now,
            shadow_dir=tmp_path / "shadow",
            autotune_dir=tmp_path / "autotune",
        )
        assert r.status == "insufficient_n"
        assert "no prior auto-tune artifact" in r.reason


class TestFlashCrashSuppression:
    def test_flash_crash_suppresses_alert(self, tmp_path, now):
        # Set up a scenario that would otherwise be 'below_ci'
        rets = [0.0001] * 60
        _write_scored(tmp_path / "shadow", "BTC-USD", "bull", rets, now)
        _write_artifact(tmp_path / "autotune", "BTC-USD", "bull", ci_lower=5.0)

        # Inject a vol_z_clipped spike
        vol_z = [0.5] * 100 + [3.8] + [0.7] * 50

        r = dm.run_drift_check(
            "BTC-USD", "bull", now=now,
            shadow_dir=tmp_path / "shadow",
            autotune_dir=tmp_path / "autotune",
            vol_z_series=vol_z,
        )
        assert r.status == "suppressed"
        assert "flash-crash" in r.reason


class TestBonferroniFreeThreshold:
    def test_threshold_is_two_sigma(self):
        """WCT argument: we use flat 2σ, not Bonferroni. Assert the module
        constant isn't silently tightened (which would kill alert rate)."""
        assert dm.Z_GAP_THRESHOLD == 2.0


class TestPersistence:
    def test_write_drift_result(self, tmp_path, now):
        res = dm.DriftResult(
            "BTC-USD", "bull", "in_ci",
            realised_sharpe=1.5, ci_lower=1.0, z_gap=0.8,
            bootstrap_se=0.5, n_decisions=60,
            iso_week="2026W16", ran_at=now.isoformat(),
            reason="test",
        )
        path = dm.write_drift_result(res, drift_dir=tmp_path)
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["status"] == "in_ci"
        assert payload["iso_week"] == "2026W16"
