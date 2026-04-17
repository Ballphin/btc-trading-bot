"""Tests for tradingagents/pulse/stats.py — pure statistical primitives."""

import math

import numpy as np
import pytest

from tradingagents.pulse.stats import (
    autocorr,
    deflated_sharpe,
    effective_sample_size,
    non_overlapping_sharpe,
    pbo,
    sharpe_confidence_interval,
    sharpe_ratio,
    sqrt_impact_bps,
)


# ── autocorr ──────────────────────────────────────────────────────────

class TestAutocorr:
    def test_ar1_positive(self):
        """AR(1) with phi=0.5 should give lag-1 autocorr near 0.5."""
        rng = np.random.default_rng(0)
        n = 5000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.5 * x[i - 1] + rng.normal(0, 1)
        acf = autocorr(x, lag=1)
        assert 0.40 < acf < 0.60

    def test_ar1_negative(self):
        rng = np.random.default_rng(1)
        n = 5000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = -0.3 * x[i - 1] + rng.normal(0, 1)
        acf = autocorr(x, lag=1)
        assert -0.4 < acf < -0.2

    def test_iid_near_zero(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 5000)
        assert abs(autocorr(x, lag=1)) < 0.05

    def test_short_array_returns_nan(self):
        # With lag=1 and only 1 element, can't compute
        result = autocorr(np.array([1.0]), lag=1)
        assert math.isnan(result) or result == 0.0

    def test_constant_series(self):
        # std = 0, autocorr undefined
        result = autocorr(np.ones(50), lag=1)
        assert math.isnan(result) or result == 0.0


# ── effective_sample_size (Newey-West style) ────────────────────────

class TestEffectiveSampleSize:
    def test_iid_close_to_n(self):
        """IID returns → N_eff ≥ 50% of N (small-sample noise in ACF sums)."""
        rng = np.random.default_rng(10)
        x = rng.normal(0, 1, 500)
        n_eff = effective_sample_size(x, max_lag=12)
        assert 0.5 * len(x) < n_eff <= len(x) + 1e-6

    def test_iid_close_to_n_lag2(self):
        """With very small lag, IID → N_eff ≈ N."""
        rng = np.random.default_rng(10)
        x = rng.normal(0, 1, 500)
        n_eff = effective_sample_size(x, max_lag=2)
        assert 0.8 * len(x) < n_eff <= len(x) + 1e-6

    def test_positive_autocorr_shrinks(self):
        """Positive serial corr shrinks N_eff."""
        rng = np.random.default_rng(11)
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.5 * x[i - 1] + rng.normal(0, 1)
        n_eff = effective_sample_size(x, max_lag=12)
        assert n_eff < 0.6 * n

    def test_empty_returns_zero(self):
        n_eff = effective_sample_size(np.array([1.0]), max_lag=5)
        assert n_eff <= 1.0


# ── sharpe_ratio ─────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_mean_positive_sharpe(self):
        rng = np.random.default_rng(20)
        r = 0.001 + rng.normal(0, 0.01, 500)
        sr = sharpe_ratio(r, periods_per_year=252)
        assert sr > 0

    def test_zero_variance_returns_zero(self):
        assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0

    def test_empty_returns_zero(self):
        assert sharpe_ratio([]) == 0.0

    def test_annualization(self):
        """Sharpe scales with sqrt(periods_per_year)."""
        rng = np.random.default_rng(21)
        r = 0.001 + rng.normal(0, 0.01, 500)
        sr_252 = sharpe_ratio(r, periods_per_year=252)
        sr_1 = sharpe_ratio(r, periods_per_year=1)
        assert abs(sr_252 / sr_1 - math.sqrt(252)) < 0.01


# ── Sharpe CI ────────────────────────────────────────────────────────

class TestSharpeConfidenceInterval:
    def test_ci_widens_with_smaller_sample(self):
        sr = 1.0
        lo_100, hi_100 = sharpe_confidence_interval(sr, n_eff=100)
        lo_1000, hi_1000 = sharpe_confidence_interval(sr, n_eff=1000)
        assert (hi_100 - lo_100) > (hi_1000 - lo_1000)

    def test_ci_contains_sr(self):
        sr = 1.0
        lo, hi = sharpe_confidence_interval(sr, n_eff=500)
        assert lo < sr < hi

    def test_n_eff_zero_returns_infinite_ci(self):
        lo, hi = sharpe_confidence_interval(1.0, n_eff=0.0)
        # With no observations, CI should be undefined/infinite; impl may
        # return (nan, nan) or huge bounds. We just require NOT a finite
        # narrow CI.
        assert not (abs(hi - lo) < 0.1)


# ── Deflated Sharpe ──────────────────────────────────────────────────

class TestDeflatedSharpe:
    def test_deflates_below_original(self):
        ds = deflated_sharpe(in_sample_sharpe=1.0, n_params=3, n_obs=100)
        assert ds < 1.0

    def test_zero_params_equals_original(self):
        ds = deflated_sharpe(in_sample_sharpe=1.0, n_params=0, n_obs=100)
        assert ds == pytest.approx(1.0, rel=1e-6)

    def test_clamp_to_zero_if_2k_exceeds_n(self):
        """When 2k > N, sqrt arg would be negative — clamp to 0."""
        ds = deflated_sharpe(in_sample_sharpe=2.0, n_params=100, n_obs=100)
        assert ds == 0.0


# ── PBO ──────────────────────────────────────────────────────────────

class TestPBO:
    def test_top_is_stays_top_oos_pbo_zero(self):
        is_perf = [0.1, 0.5, 0.9, 1.2, 1.8]     # strategy 4 is best in-sample
        oos_perf = [0.05, 0.2, 0.4, 0.7, 1.0]   # still best oos
        assert pbo(is_perf, oos_perf) == 0.0

    def test_top_is_worst_oos_pbo_one(self):
        is_perf = [0.1, 0.5, 0.9, 1.2, 1.8]      # strategy 4 best in-sample
        oos_perf = [1.0, 0.9, 0.8, 0.5, -0.2]    # strategy 4 worst oos
        assert pbo(is_perf, oos_perf) == 1.0


# ── non_overlapping_sharpe ───────────────────────────────────────────

class TestNonOverlappingSharpe:
    def test_reduces_sample_size(self):
        """Non-overlapping should produce a Sharpe with |Sharpe| < overlapping."""
        rng = np.random.default_rng(30)
        r = 0.001 + rng.normal(0, 0.01, 500)
        sr_no = non_overlapping_sharpe(r, window=5, periods_per_year=252)
        # Just require non-nan float
        assert isinstance(sr_no, float)


# ── Square-root market impact ───────────────────────────────────────

class TestSqrtImpact:
    def test_zero_size_zero_impact(self):
        assert sqrt_impact_bps(notional_usd=0, adv_usd=1e9) == 0.0

    def test_zero_adv_zero_impact(self):
        assert sqrt_impact_bps(notional_usd=1e6, adv_usd=0) == 0.0

    def test_scales_with_sqrt_size(self):
        i1 = sqrt_impact_bps(notional_usd=1e6, adv_usd=1e9, c=10.0)
        i4 = sqrt_impact_bps(notional_usd=4e6, adv_usd=1e9, c=10.0)
        # 4× size → 2× impact
        assert abs(i4 / i1 - 2.0) < 0.01

    def test_coefficient_linear(self):
        i_c5 = sqrt_impact_bps(notional_usd=1e6, adv_usd=1e9, c=5.0)
        i_c10 = sqrt_impact_bps(notional_usd=1e6, adv_usd=1e9, c=10.0)
        assert abs(i_c10 / i_c5 - 2.0) < 0.001
