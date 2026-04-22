"""Statistical utilities for backtesting.

Provides standalone statistical helpers that can be reused across
metrics computation, walk-forward validation, and deflated Sharpe calculations.
"""

import math
from typing import Sequence


def sharpe_standard_error(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Standard error of the Sharpe ratio using Lo (2002) / Mertens (2002) formula.

    The formula accounts for skewness and excess kurtosis in return distributions,
    providing a more accurate standard error than the Gaussian approximation.

    Formula:
        SE = sqrt((1 + 0.5*SR^2 - skew*SR + (excess_kurtosis/4)*SR^2) / n)

    When skew=0 and kurtosis=3 (Gaussian), this reduces to:
        SE = sqrt((1 + 0.5*SR^2) / n)

    Args:
        sharpe: Annualized Sharpe ratio
        n: Number of independent periods (sample size)
        skew: Skewness of returns distribution (default 0.0 for symmetric)
        kurtosis: Kurtosis of returns distribution (default 3.0 for normal)

    Returns:
        Standard error of the Sharpe ratio. Returns infinity for n <= 1.

    References:
        - Lo, A.W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal.
        - Mertens, E. (2002). "Variance of the IID Maximally Diversified Portfolio."
    """
    if n <= 1:
        return float("inf")

    excess_kurtosis = kurtosis - 3.0
    numerator = 1.0 + 0.5 * sharpe ** 2 - skew * sharpe + (excess_kurtosis / 4.0) * sharpe ** 2
    # Numerically guard against negative values from extreme parameter combinations
    numerator = max(numerator, 1e-10)
    return math.sqrt(numerator / n)


def compute_skewness_kurtosis(returns: Sequence[float]) -> tuple[float, float]:
    """Compute skewness and kurtosis from a returns series.

    Uses the standard definitions:
    - Skewness: E[(X - mu)^3] / sigma^3
    - Kurtosis: E[(X - mu)^4] / sigma^4

    Args:
        returns: Sequence of return values (e.g., daily returns as decimals)

    Returns:
        Tuple of (skewness, kurtosis). Returns (0.0, 3.0) for insufficient data
        (less than 4 returns or zero standard deviation).
    """
    if len(returns) < 4:
        return 0.0, 3.0

    n = len(returns)
    mean = sum(returns) / n

    # Compute variance
    variance = sum((r - mean) ** 2 for r in returns) / n
    if variance == 0:
        return 0.0, 3.0

    std = math.sqrt(variance)

    # Compute skewness and kurtosis
    skew = sum((r - mean) ** 3 for r in returns) / (n * std ** 3)
    kurtosis = sum((r - mean) ** 4 for r in returns) / (n * std ** 4)

    return skew, kurtosis
