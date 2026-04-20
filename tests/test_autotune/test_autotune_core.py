"""Unit tests for :mod:`tradingagents.backtesting.autotune`.

Covers:
    * LHS sample bounds + determinism + marginal coverage.
    * Walk-forward fold layout (purge gap, non-overlap, sufficient size).
    * ``candidate_to_config`` writes values at the right dotted paths and
      the resulting ``content_hash`` shifts accordingly.
    * ``compute_verdict`` transitions at each gate boundary.
    * ``compute_diff`` reports changed paths only, with delta.
    * Bootstrap UCB-lower selection prefers the config with highest
      *lower bound*, not highest point estimate (validates the
      max-of-N bias correction).
    * End-to-end :class:`AutoTuner.run` with a deterministic fake
      ``backtest_fn`` produces a plausible ``PROPOSE`` / ``REJECT``.
    * Checkpoint resume: a crash mid-sweep → restart uses the log.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tradingagents.backtesting.autotune import (
    AutoTuner,
    CheckpointLog,
    MIN_TRADES_OOS_BY_REGIME,
    MIN_TRADES_PER_FOLD,
    SEARCH_SPACE,
    TuneSpec,
    _piecewise_regime_gate,
    candidate_to_config,
    compute_diff,
    compute_verdict,
    make_folds,
    sample_candidates,
)
from tradingagents.pulse.config import (
    PulseConfig,
    compute_config_hash,
)
from tradingagents.pulse.stats import bootstrap_sharpe_ci, bonferroni_z_threshold


# ── Fixtures ─────────────────────────────────────────────────────────

_MIN_VALID_DATA = {
    "engine_version": 3,
    "scheduler": {"pulse_interval_minutes": 5},
    "confluence": {
        "signal_threshold": 0.22,
        "tf_weights": {"1m": 0.05, "5m": 0.10, "15m": 0.20, "1h": 0.30, "4h": 0.35},
        "persistence": {"same_direction_mul": 1.2, "flip_mul": 0.8, "neutral_prev_mul": 1.0},
        "funding_elevation": {"annualized_threshold": 0.20},
        "tsmom_gate": {"counter_trend_confluence_mul": 1.2},
        "exits": {
            "buy_sl_atr_mul": 2.0, "buy_tp_atr_mul": 3.0,
            "short_crash_sl_atr_mul": 1.5, "short_tp_atr_mul": 3.0,
        },
    },
    "forward_return": {"atr_multiplier": 0.5},
    "edge_gate": {"required_deflated_is_sharpe": 0.3},
}


def _make_base_cfg() -> PulseConfig:
    return PulseConfig(
        data=_MIN_VALID_DATA,
        source_path=Path("/tmp/test.yaml"),
        mtime=0.0,
        content_hash=compute_config_hash(_MIN_VALID_DATA),
    )


# ── LHS sampling ─────────────────────────────────────────────────────

class TestSampleCandidates:
    def test_count_matches_n(self):
        assert len(sample_candidates(SEARCH_SPACE, 10, seed=1)) == 10

    def test_empty_for_zero(self):
        assert sample_candidates(SEARCH_SPACE, 0) == []

    def test_values_within_bounds(self):
        cands = sample_candidates(SEARCH_SPACE, 50, seed=1)
        for c in cands:
            for path, v in c.items():
                lo, hi = SEARCH_SPACE[path]
                assert lo <= v <= hi, f"{path}={v} outside [{lo}, {hi}]"

    def test_seed_reproducible(self):
        a = sample_candidates(SEARCH_SPACE, 20, seed=42)
        b = sample_candidates(SEARCH_SPACE, 20, seed=42)
        assert a == b

    def test_different_seeds_produce_different_samples(self):
        a = sample_candidates(SEARCH_SPACE, 20, seed=1)
        b = sample_candidates(SEARCH_SPACE, 20, seed=2)
        assert a != b

    def test_lhs_marginal_coverage(self):
        """Each dim should hit every n-bin stratum exactly once."""
        n = 30
        cands = sample_candidates(SEARCH_SPACE, n, seed=1)
        for path, (lo, hi) in SEARCH_SPACE.items():
            vals = np.array([c[path] for c in cands])
            unit = (vals - lo) / (hi - lo)
            bins = (unit * n).astype(int).clip(0, n - 1)
            # Each bin should appear exactly once → LHS property.
            counts = np.bincount(bins, minlength=n)
            assert (counts == 1).all(), \
                f"dim {path!r} bin counts {counts} (not LHS)"


# ── Walk-forward folds ───────────────────────────────────────────────

class TestMakeFolds:
    def test_basic_layout(self):
        folds = make_folds("2024-01-01", "2024-04-01", n_folds=3)
        assert len(folds) == 3
        # Train-then-test with purge
        for f in folds:
            assert f.train_start < f.train_end
            assert f.train_end < f.test_start  # purge gap
            assert f.test_start < f.test_end

    def test_folds_non_overlapping(self):
        folds = make_folds("2024-01-01", "2024-04-01", n_folds=3)
        for i in range(1, len(folds)):
            assert folds[i].train_start >= folds[i - 1].test_end

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            make_folds("2024-01-01", "2024-01-05", n_folds=5)


# ── candidate_to_config ──────────────────────────────────────────────

class TestCandidateToConfig:
    def test_writes_to_dotted_path(self):
        base = _make_base_cfg()
        cand = {"confluence.signal_threshold": 0.30}
        new = candidate_to_config(base, cand)
        assert new.get("confluence", "signal_threshold") == 0.30

    def test_hash_differs_from_base(self):
        base = _make_base_cfg()
        new = candidate_to_config(base, {"confluence.signal_threshold": 0.30})
        assert new.content_hash != base.content_hash

    def test_all_six_paths_applied(self):
        base = _make_base_cfg()
        cand = {p: (lo + hi) / 2 for p, (lo, hi) in SEARCH_SPACE.items()}
        new = candidate_to_config(base, cand)
        # Each candidate value materializes at its path.
        for path, v in cand.items():
            keys = path.split(".")
            node = new.data
            for k in keys:
                node = node[k]
            assert node == pytest.approx(v, rel=1e-9)

    def test_active_regime_stamped(self):
        base = _make_base_cfg()
        new = candidate_to_config(
            base, {"confluence.signal_threshold": 0.25},
            active_regime="bull", venue="binance_futures",
        )
        assert new.active_regime == "bull"
        assert new.venue == "binance_futures"


# ── Verdict gate ─────────────────────────────────────────────────────

class TestComputeVerdict:
    def _spec(self, **over) -> TuneSpec:
        kw = dict(
            ticker="BTC-USD",
            start_date="2024-01-01", end_date="2024-04-01",
            n_folds=3, n_configs=10,
        )
        kw.update(over)
        return TuneSpec(**kw)

    def test_propose_on_happy_path(self):
        verdict, reasons = compute_verdict(
            is_sharpes=[1.2, 1.3, 1.4],
            oos_sharpes=[0.9, 1.0, 1.1],
            oos_n_trades_total=500,
            active_regime="base",
            pbo=0.25,
            deflated_oos_sharpe=0.5,
            n_folds_used=3,
            spec=self._spec(),
        )
        assert verdict == "PROPOSE"
        assert "all gates passed" in reasons[0]

    def test_provisional_on_insufficient_trades(self):
        verdict, _ = compute_verdict(
            is_sharpes=[1.2], oos_sharpes=[0.9],
            oos_n_trades_total=50,  # below base floor 400
            active_regime="base",
            pbo=0.2, deflated_oos_sharpe=0.5,
            n_folds_used=2,
            spec=self._spec(),
        )
        assert verdict == "PROVISIONAL"

    def test_reject_on_high_pbo(self):
        verdict, reasons = compute_verdict(
            is_sharpes=[1.5], oos_sharpes=[1.0],
            oos_n_trades_total=500,
            active_regime="base",
            pbo=0.8, deflated_oos_sharpe=0.5,
            n_folds_used=2,
            spec=self._spec(),
        )
        assert verdict == "REJECT"
        assert "PBO" in reasons[0]

    def test_reject_on_poor_generalization(self):
        verdict, _ = compute_verdict(
            is_sharpes=[2.0, 2.5, 2.2],
            oos_sharpes=[0.3, 0.4, 0.5],  # OOS/IS ≈ 0.17
            oos_n_trades_total=500,
            active_regime="base",
            pbo=0.2, deflated_oos_sharpe=0.5,
            n_folds_used=3,
            spec=self._spec(),
        )
        assert verdict == "REJECT"

    def test_reject_on_low_deflated_sharpe(self):
        verdict, _ = compute_verdict(
            is_sharpes=[1.0], oos_sharpes=[0.9],
            oos_n_trades_total=500,
            active_regime="base",
            pbo=0.2, deflated_oos_sharpe=0.1,  # below 0.3 floor
            n_folds_used=2,
            spec=self._spec(),
        )
        assert verdict == "REJECT"

    def test_reject_on_insufficient_folds(self):
        verdict, _ = compute_verdict(
            is_sharpes=[1.0], oos_sharpes=[0.9],
            oos_n_trades_total=500,
            active_regime="base",
            pbo=0.2, deflated_oos_sharpe=0.5,
            n_folds_used=1,
            spec=self._spec(),
        )
        assert verdict == "REJECT"

    def test_bear_has_lower_trade_minimum(self):
        # 350 trades — below base(400) but above bear(300)
        verdict, _ = compute_verdict(
            is_sharpes=[1.0, 1.1], oos_sharpes=[0.8, 0.9],
            oos_n_trades_total=350,
            active_regime="bear",
            pbo=0.2, deflated_oos_sharpe=0.5,
            n_folds_used=2,
            spec=self._spec(),
        )
        assert verdict == "PROPOSE"

    def test_sideways_has_lowest_trade_minimum(self):
        # 250 trades — below bear(300), passes sideways(200)
        verdict, _ = compute_verdict(
            is_sharpes=[1.0, 1.1], oos_sharpes=[0.8, 0.9],
            oos_n_trades_total=250,
            active_regime="sideways",
            pbo=0.2, deflated_oos_sharpe=0.5,
            n_folds_used=2,
            spec=self._spec(),
        )
        assert verdict == "PROPOSE"

    def test_regime_degradation_blocks_apply(self):
        verdict, reasons = compute_verdict(
            is_sharpes=[1.0, 1.1], oos_sharpes=[0.8, 0.9],
            oos_n_trades_total=500,
            active_regime="base",
            pbo=0.2, deflated_oos_sharpe=0.5,
            n_folds_used=2,
            spec=self._spec(),
            regime_sharpes_current={"trend": 1.2, "chop": 0.6},
            regime_sharpes_proposed={"trend": 1.1, "chop": 0.1},  # chop collapses
        )
        assert verdict == "REJECT"
        assert "chop" in reasons[0]


# ── Piecewise regime gate ────────────────────────────────────────────

class TestPiecewiseRegimeGate:
    def test_ratio_applied_when_current_high(self):
        assert _piecewise_regime_gate(0.85, 1.0, ratio_floor=0.8)  # 0.85 >= 0.8
        assert not _piecewise_regime_gate(0.70, 1.0, ratio_floor=0.8)

    def test_absolute_band_when_current_mid(self):
        # current=0.4, proposed must be >= 0.3
        assert _piecewise_regime_gate(0.35, 0.4)
        assert not _piecewise_regime_gate(0.20, 0.4)

    def test_absolute_floor_when_current_negative(self):
        # current=-0.5, ratio-naive would allow -0.4; piecewise requires > 0.
        assert not _piecewise_regime_gate(-0.4, -0.5)  # would be "0.8×" naively
        assert _piecewise_regime_gate(0.1, -0.5)


# ── compute_diff ─────────────────────────────────────────────────────

class TestComputeDiff:
    def test_unchanged_path_omitted(self):
        current = {"confluence": {"signal_threshold": 0.22}}
        proposed = {"confluence": {"signal_threshold": 0.22}}
        d = compute_diff(current, proposed, ["confluence.signal_threshold"])
        assert d == []

    def test_changed_path_reports_delta(self):
        current = {"confluence": {"signal_threshold": 0.22}}
        proposed = {"confluence": {"signal_threshold": 0.30}}
        d = compute_diff(current, proposed, ["confluence.signal_threshold"])
        assert len(d) == 1
        assert d[0]["path"] == "confluence.signal_threshold"
        assert d[0]["old"] == 0.22
        assert d[0]["new"] == 0.30
        assert d[0]["delta"] == pytest.approx(0.08, rel=1e-9)

    def test_missing_current_becomes_none(self):
        current = {"confluence": {}}
        proposed = {"confluence": {"signal_threshold": 0.30}}
        d = compute_diff(current, proposed, ["confluence.signal_threshold"])
        assert d[0]["old"] is None

    def test_float_tolerance_skip(self):
        """0.220000001 vs 0.22 should NOT show as a diff."""
        current = {"x": 0.22}
        proposed = {"x": 0.220000001}
        d = compute_diff(current, proposed, ["x"])
        assert d == []


# ── Bootstrap UCB-lower prefers the robust config ───────────────────

class TestBootstrapUCBLowerSelection:
    def test_narrow_beats_wide_at_same_mean(self):
        """Two candidates with the same mean — the one with smaller
        variance must have a higher CI-lower bound."""
        narrow = [0.005] * 100                    # near-zero variance
        wide = [0.12, -0.11, 0.09, -0.10] * 25    # same mean ~0.005, huge var
        lo_narrow, *_ = bootstrap_sharpe_ci(
            narrow, n_bootstrap=500, periods_per_year=8760.0, seed=7,
        )
        lo_wide, *_ = bootstrap_sharpe_ci(
            wide, n_bootstrap=500, periods_per_year=8760.0, seed=7,
        )
        assert lo_narrow > lo_wide, f"narrow={lo_narrow} wide={lo_wide}"


# ── Checkpoint log ───────────────────────────────────────────────────

class TestCheckpointLog:
    def test_append_and_load(self, tmp_path):
        log = CheckpointLog(tmp_path / "cp.jsonl")
        log.append({"fold": 0, "config_idx": 0, "is_sharpe": 1.0, "oos_sharpe": 0.8,
                    "is_n_trades": 50, "oos_n_trades": 40, "oos_returns": [0.01, -0.005]})
        log.append({"fold": 0, "config_idx": 1, "is_sharpe": 0.5, "oos_sharpe": 0.4,
                    "is_n_trades": 55, "oos_n_trades": 45, "oos_returns": [0.02]})
        got = log.load_completed()
        assert (0, 0) in got and (0, 1) in got
        assert got[(0, 0)]["oos_returns"] == [0.01, -0.005]

    def test_corrupt_line_tolerated(self, tmp_path):
        log = CheckpointLog(tmp_path / "cp.jsonl")
        log.append({"fold": 0, "config_idx": 0, "is_sharpe": 1.0, "oos_sharpe": 0.8,
                    "is_n_trades": 50, "oos_n_trades": 40})
        # Corrupt the file
        log.path.open("a").write("{invalid json\n")
        got = log.load_completed()
        assert (0, 0) in got


# ── AutoTuner end-to-end (fake backtest_fn) ──────────────────────────

class TestAutoTunerE2E:
    def _fake_backtest_fn(self, seed_factor: float = 1.0):
        """Returns a closure that produces deterministic 'trades'.

        Each candidate's return series is seeded by
        ``hash(str(cfg)) × seed_factor`` so different candidates get
        different means but the same fake fn is deterministic per call.
        """
        def fn(ticker, start, end, cfg, regime):
            rng = np.random.default_rng(
                abs(hash((cfg.content_hash, start, end))) % (2**31)
            )
            n = 80  # >= MIN_TRADES_PER_FOLD
            # Mean determined by signal_threshold — lower threshold = more trades,
            # slightly higher mean (simulates a true positive direction in space).
            st = cfg.get("confluence", "signal_threshold")
            mean = 0.002 + (0.30 - st) * 0.01 * seed_factor
            returns = rng.normal(mean, 0.005, n).tolist()
            return {"trade_returns": returns}
        return fn

    def test_runs_end_to_end_and_produces_verdict(self, tmp_path, monkeypatch):
        # Point checkpoints + artifacts under tmp_path to keep CI clean.
        monkeypatch.chdir(tmp_path)
        # Stub get_effective_config → base cfg
        from tradingagents.backtesting import autotune as at_mod
        monkeypatch.setattr(at_mod, "get_effective_config",
                            lambda regime: _make_base_cfg())
        spec = TuneSpec(
            ticker="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-03-01",
            n_folds=3, n_configs=6, seed=7,
            checkpoint_dir=str(tmp_path / "cp"),
        )
        tuner = AutoTuner(
            spec=spec,
            backtest_fn=self._fake_backtest_fn(seed_factor=1.0),
            job_id="test_job",
        )
        report = tuner.run()
        # We expect REJECT or PROVISIONAL — our fake returns are weak.
        # The point: no exceptions, artifact written, verdict computed.
        assert report.verdict in {"PROPOSE", "PROVISIONAL", "REJECT"}
        assert Path(report.artifact_path).exists()
        assert report.current_config_hash
        # Either path reports fold layout
        if report.verdict != "REJECT" or report.per_fold:
            assert isinstance(report.diff, list)

    def test_checkpoint_resume(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from tradingagents.backtesting import autotune as at_mod
        monkeypatch.setattr(at_mod, "get_effective_config",
                            lambda regime: _make_base_cfg())

        spec = TuneSpec(
            ticker="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-03-01",
            n_folds=2, n_configs=3, seed=1,
            checkpoint_dir=str(tmp_path / "cp"),
        )
        call_count = {"n": 0}

        def counting_fn(ticker, start, end, cfg, regime):
            call_count["n"] += 1
            rng = np.random.default_rng(1)
            return {"trade_returns": rng.normal(0.001, 0.005, 50).tolist()}

        tuner = AutoTuner(
            spec=spec, backtest_fn=counting_fn, job_id="resume_test",
        )
        tuner.run()
        first_calls = call_count["n"]

        # Second run should short-circuit using the checkpoint.
        call_count["n"] = 0
        tuner2 = AutoTuner(
            spec=spec, backtest_fn=counting_fn, job_id="resume_test",
        )
        tuner2.run()
        assert call_count["n"] == 0, \
            f"expected 0 calls on resume, got {call_count['n']}"


# ── Bonferroni z threshold ───────────────────────────────────────────

class TestBonferroniZ:
    def test_single_test_near_196(self):
        assert 1.95 < bonferroni_z_threshold(1) < 1.97

    def test_eighteen_tests_near_3(self):
        """Plan says z≥3.0 for 3 regimes × 6 params = 18 tests."""
        z = bonferroni_z_threshold(18)
        assert 2.9 < z < 3.1

    def test_monotone_increasing(self):
        zs = [bonferroni_z_threshold(n) for n in [1, 5, 10, 20]]
        assert zs == sorted(zs)
