"""Tests for sharpe_standard_error helper implementing Lo (2002) formula.

Validates the unified SE calculation with skewness and kurtosis corrections.
"""

import math

import pytest

from tradingagents.backtesting.stats import (
    sharpe_standard_error,
    compute_skewness_kurtosis,
)


class TestSharpeStandardError:
    """Test sharpe_standard_error helper correctness."""

    def test_reduces_to_normal_form_under_gaussian(self):
        """With skew=0 and kurtosis=3, should match simplified Gaussian SE."""
        sharpe = 1.2
        n = 100

        # Full Lo(2002) formula with Gaussian parameters
        full_se = sharpe_standard_error(sharpe, n, skew=0.0, kurtosis=3.0)

        # Simplified Gaussian formula
        simple_se = math.sqrt((1 + 0.5 * sharpe ** 2) / n)

        assert full_se == pytest.approx(simple_se, rel=1e-10)
        # Numerical check: SE ≈ sqrt((1 + 0.5*1.44)/100) = sqrt(1.72/100) ≈ 0.131
        assert full_se == pytest.approx(0.1311, rel=0.01)

    def test_negative_skew_inflates_se(self):
        """Negative skewness (left tail) should increase SE vs Gaussian."""
        sharpe = 1.2
        n = 100

        gaussian_se = sharpe_standard_error(sharpe, n, skew=0.0, kurtosis=3.0)
        skewed_se = sharpe_standard_error(sharpe, n, skew=-1.5, kurtosis=3.0)

        # Negative skew increases uncertainty
        assert skewed_se > gaussian_se
        # Analytical: -skew*SR = -(-1.5)*1.2 = +1.8 adds to numerator
        # So skewed SE should be ~sqrt((1.72 + 1.8)/100) ≈ 0.185 vs 0.131
        assert skewed_se == pytest.approx(0.185, rel=0.05)

    def test_positive_skew_deflates_se(self):
        """Positive skewness (right tail) should decrease SE vs Gaussian."""
        sharpe = 0.8  # Lower SR to keep numerator positive with moderate positive skew
        n = 100

        gaussian_se = sharpe_standard_error(sharpe, n, skew=0.0, kurtosis=3.0)
        skewed_se = sharpe_standard_error(sharpe, n, skew=0.8, kurtosis=3.0)

        # Moderate positive skew reduces uncertainty
        assert skewed_se < gaussian_se

    def test_fat_tails_inflate_se(self):
        """High kurtosis (fat tails) should increase SE vs Gaussian."""
        sharpe = 1.2
        n = 100

        normal_se = sharpe_standard_error(sharpe, n, skew=0.0, kurtosis=3.0)
        fat_tail_se = sharpe_standard_error(sharpe, n, skew=0.0, kurtosis=10.0)

        # Fat tails increase uncertainty
        assert fat_tail_se > normal_se
        # Analytical: excess=7, term adds (7/4)*1.44 = 2.52 to numerator
        # So fat tail SE ≈ sqrt((1.72 + 2.52)/100) ≈ 0.206 vs 0.131
        assert fat_tail_se == pytest.approx(0.206, rel=0.05)

    def test_tiny_sample_returns_inf(self):
        """n <= 1 should return infinity (undefined SE)."""
        assert sharpe_standard_error(1.0, 0) == float("inf")
        assert sharpe_standard_error(1.0, 1) == float("inf")

    def test_small_sample_finite(self):
        """n = 2 should return finite (though large) SE."""
        se = sharpe_standard_error(1.0, 2)
        assert math.isfinite(se)
        assert se > 0

    def test_crypto_fat_tail_parameters(self):
        """Typical crypto parameters: skew=-1.5, kurt=8 should materially inflate SE."""
        sharpe = 1.2
        n = 100

        gaussian_se = sharpe_standard_error(sharpe, n, skew=0.0, kurtosis=3.0)
        crypto_se = sharpe_standard_error(sharpe, n, skew=-1.5, kurtosis=8.0)

        # Combined effect should be substantial (~1.8x inflation)
        ratio = crypto_se / gaussian_se
        assert ratio > 1.5  # At least 50% inflation
        assert ratio < 2.5  # But not absurd

        # Numerical sanity check: numerator = 1 + 0.72 - (-1.8) + (5/4)*1.44
        # = 1 + 0.72 + 1.8 + 1.8 = 5.32
        # SE = sqrt(5.32/100) = 0.231 vs Gaussian 0.131
        assert crypto_se == pytest.approx(0.231, rel=0.05)


class TestComputeSkewnessKurtosis:
    """Test compute_skewness_kurtosis helper."""

    def test_symmetric_distribution(self):
        """Symmetric distribution should have skew ≈ 0."""
        returns = [-0.02, -0.01, 0.0, 0.01, 0.02]  # symmetric around 0
        skew, kurt = compute_skewness_kurtosis(returns)

        assert abs(skew) < 0.1  # Approximately symmetric
        assert kurt > 1.5  # Some kurtosis

    def test_left_skewed_distribution(self):
        """Left-skewed (more negative outliers) should have negative skew."""
        returns = [0.01, 0.02, 0.01, 0.02, -0.1]  # one large negative
        skew, kurt = compute_skewness_kurtosis(returns)

        assert skew < -0.5  # Clearly left-skewed

    def test_insufficient_data_returns_defaults(self):
        """Less than 4 returns should return (0.0, 3.0)."""
        returns = [0.01, 0.02]
        skew, kurt = compute_skewness_kurtosis(returns)

        assert skew == 0.0
        assert kurt == 3.0

    def test_zero_variance_returns_defaults(self):
        """Zero variance should return (0.0, 3.0)."""
        returns = [0.01, 0.01, 0.01, 0.01]
        skew, kurt = compute_skewness_kurtosis(returns)

        assert skew == 0.0
        assert kurt == 3.0

    def test_normal_like_returns(self):
        """Approximate normal returns should have kurtosis near 3."""
        import random
        random.seed(42)
        # Generate pseudo-normal returns via Box-Muller approximation
        returns = []
        for _ in range(100):
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            returns.append(z * 0.01)  # scale to ~1% vol

        skew, kurt = compute_skewness_kurtosis(returns)

        assert abs(skew) < 0.5  # Roughly symmetric
        assert 2.0 < kurt < 4.0  # Roughly normal kurtosis
