"""Statistical helpers for Pulse scoring & calibration.

All functions are pure (no I/O, no randomness) — easy to test.

Formulas referenced:
  - Newey-West effective sample size (truncated sum of positive autocorrs)
  - Deflated Sharpe (Carver 2015): IS_SR × sqrt(1 − 2·n_params/n_obs)
  - Probability of Backtest Overfitting (López-de-Prado 2014): fraction of
    OOS ranks below median across K combinatorial partitions.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ── Basic statistics ──────────────────────────────────────────────────

def _as_array(returns: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D sequence, got shape {arr.shape}")
    arr = arr[np.isfinite(arr)]
    return arr


def autocorr(returns: Sequence[float] | np.ndarray, lag: int) -> float:
    """Lag-k sample autocorrelation. 0 if insufficient samples."""
    arr = _as_array(returns)
    if lag <= 0 or len(arr) <= lag + 1:
        return 0.0
    x0 = arr[:-lag]
    xk = arr[lag:]
    x0 = x0 - x0.mean()
    xk = xk - xk.mean()
    denom = math.sqrt((x0 ** 2).sum() * (xk ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((x0 * xk).sum() / denom)


def effective_sample_size(
    returns: Sequence[float] | np.ndarray, max_lag: int = 12
) -> float:
    """Newey-West-style effective sample size.

    N_eff = N / (1 + 2 · Σ max(0, ρ_k) for k = 1..max_lag)
    Truncation at 0 (never inflate N) matches Newey-West convention.
    """
    arr = _as_array(returns)
    n = len(arr)
    if n < 3:
        return float(n)
    max_lag = max(1, min(max_lag, n - 2))
    acf_sum = 0.0
    for k in range(1, max_lag + 1):
        rho = autocorr(arr, k)
        if rho > 0:
            acf_sum += rho
    denom = 1.0 + 2.0 * acf_sum
    return float(n / denom) if denom > 0 else float(n)


def sharpe_ratio(
    returns: Sequence[float] | np.ndarray, periods_per_year: float = 252.0
) -> float:
    """Annualized Sharpe. 0 if zero variance or <2 samples."""
    arr = _as_array(returns)
    if len(arr) < 2:
        return 0.0
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float((mu / sigma) * math.sqrt(periods_per_year))


def sharpe_confidence_interval(
    sharpe: float, n_eff: float, confidence: float = 0.95
) -> Tuple[float, float]:
    """CI for annualized Sharpe using Jobson-Korkie / Mertens approximation.

    SE(SR) ≈ sqrt((1 + SR²/2) / N_eff).
    """
    if n_eff < 2 or not math.isfinite(sharpe):
        return (float("-inf"), float("inf"))
    z = 1.959963984540054 if confidence == 0.95 else _z_from_confidence(confidence)
    se = math.sqrt((1.0 + 0.5 * sharpe * sharpe) / max(n_eff, 1.0))
    return (sharpe - z * se, sharpe + z * se)


def _z_from_confidence(confidence: float) -> float:
    # Inverse standard normal CDF at (1+conf)/2; only common values used.
    # Use scipy-free approximation (Abramowitz & Stegun 26.2.23).
    p = (1.0 + confidence) / 2.0
    if not (0 < p < 1):
        raise ValueError("confidence must be in (0, 1)")
    # Beasley-Springer-Moro
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    q = p - 0.5
    if abs(q) <= 0.425:
        r = q * q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0
        return num / den
    r = p if q < 0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    num = ((((2.938163982698783e+00*r + 4.374664141464968e+00)*r
             + -2.549732539343734e+00)*r + -2.400758277161838e+00)*r
             + -3.223964580411365e-01)*r + -7.784894002430293e-03
    den = (((3.754408661907416e+00*r + 2.445134137142996e+00)*r
             + 3.224671290700398e-01)*r + 7.784695709041462e-03)*r + 1.0
    z = num / den
    return z if q >= 0 else -z


# ── Deflated Sharpe (Carver's simple form) ────────────────────────────

def deflated_sharpe(
    in_sample_sharpe: float, n_params: int, n_obs: int
) -> float:
    """Carver (2015): SR_deflated = SR × sqrt(1 − 2·k/N).

    Returns 0 if the deflation factor would be ≤ 0 (overparameterized).
    """
    if n_obs <= 2 * n_params or n_obs <= 0:
        return 0.0
    factor = 1.0 - 2.0 * n_params / n_obs
    if factor <= 0:
        return 0.0
    return float(in_sample_sharpe * math.sqrt(factor))


# ── Probability of Backtest Overfitting (López-de-Prado 2014) ─────────

def pbo(
    is_performance: Sequence[float],
    oos_performance: Sequence[float],
) -> float:
    """Probability of Backtest Overfitting.

    Given N candidate configs with (IS_i, OOS_i) pairs, PBO is the fraction
    of times the best-IS config finishes below the median OOS.

    This is the simple "rank-degradation" variant — not the combinatorial
    CPCV variant (that requires raw OOS return paths per fold). Still
    useful as a calibration-time warning signal.
    """
    is_arr = np.asarray(is_performance, dtype=float)
    oos_arr = np.asarray(oos_performance, dtype=float)
    if len(is_arr) != len(oos_arr):
        raise ValueError("IS and OOS arrays must match in length")
    n = len(is_arr)
    if n < 4:
        return float("nan")
    # The best IS is our selection; measure where its OOS rank lands.
    best_idx = int(np.argmax(is_arr))
    oos_rank = int((oos_arr < oos_arr[best_idx]).sum())  # 0..N-1
    # PBO = P(best-IS ranks below median OOS). With one selection we
    # approximate by: below-median iff rank < N/2.
    return 1.0 if oos_rank < n / 2 else 0.0


def pbo_bootstrap(
    is_matrix: np.ndarray,
    oos_matrix: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> float:
    """Bootstrap PBO estimate. Each column is a candidate config.

    Rows are folds/partitions. For each bootstrap draw, we:
      1) Pick random fold as IS; rest as OOS.
      2) Best IS candidate → check if its OOS rank < median.
      3) Average across draws → PBO estimate in [0, 1].
    """
    if is_matrix.shape != oos_matrix.shape:
        raise ValueError("IS and OOS matrices must have the same shape")
    n_folds, n_candidates = is_matrix.shape
    if n_folds < 2 or n_candidates < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    below = 0
    for _ in range(n_bootstrap):
        fold = int(rng.integers(0, n_folds))
        is_vec = is_matrix[fold]
        oos_vec = oos_matrix[fold]
        best_idx = int(np.argmax(is_vec))
        oos_rank = int((oos_vec < oos_vec[best_idx]).sum())
        if oos_rank < n_candidates / 2:
            below += 1
    return below / n_bootstrap


# ── Sharpe decomposition for overlapping windows ──────────────────────

def non_overlapping_sharpe(
    returns: Sequence[float] | np.ndarray,
    window: int,
    periods_per_year: float = 252.0,
) -> float:
    """Sharpe over every `window`-th sample to avoid overlap-induced bias."""
    arr = _as_array(returns)
    if window <= 1 or len(arr) < 2 * window:
        return sharpe_ratio(arr, periods_per_year)
    sampled = arr[::window]
    if len(sampled) < 2:
        return 0.0
    return sharpe_ratio(sampled, periods_per_year / window)


# ── Market impact (square-root model) ─────────────────────────────────

def sqrt_impact_bps(notional_usd: float, adv_usd: float, c: float = 10.0) -> float:
    """Square-root market-impact model: bps = C × sqrt(size / ADV).

    Args:
        notional_usd: dollar notional of the order.
        adv_usd: 30-day average daily USD volume of the asset.
        c: impact coefficient (≈10 for liquid crypto perps, Cont/Stoikov).

    Returns:
        Impact in basis points. 0 if either input is non-positive.
    """
    if notional_usd <= 0 or adv_usd <= 0:
        return 0.0
    return float(c * math.sqrt(notional_usd / adv_usd))
