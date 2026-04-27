"""Tests for the VPD (volume-price divergence) detector."""

from __future__ import annotations

import numpy as np

from tradingagents.pulse.vpd import compute_vpd


def _integrate(start: float, dlogs: np.ndarray) -> np.ndarray:
    return start * np.exp(np.concatenate([[0.0], np.cumsum(dlogs)]))


def test_bearish_divergence_price_up_volume_down():
    rng = np.random.default_rng(0)
    n = 20
    # Engineer anti-correlated Δlog series: when price tick is positive, volume tick is negative.
    base = rng.normal(0, 0.005, n)
    dlogp = 0.005 + base                       # mostly positive ticks
    dlogv = -0.005 - base                      # opposite sign
    prices = _integrate(100.0, dlogp)
    volumes = _integrate(200.0, dlogv)
    result = compute_vpd(prices, volumes, lookback_bars=20, corr_threshold=-0.30)
    assert result.correlation < -0.9
    assert result.signal == -1
    assert result.kind == "bearish_div"


def test_bullish_divergence_price_down_volume_up():
    rng = np.random.default_rng(1)
    n = 20
    base = rng.normal(0, 0.005, n)
    dlogp = -0.005 + base                      # mostly down
    dlogv = 0.005 - base                       # opposite
    prices = _integrate(110.0, dlogp)
    volumes = _integrate(100.0, dlogv)
    result = compute_vpd(prices, volumes, lookback_bars=20, corr_threshold=-0.30)
    assert result.correlation < -0.9
    assert result.signal == +1
    assert result.kind == "bullish_div"


def test_no_divergence_aligned_trend():
    rng = np.random.default_rng(2)
    n = 20
    base = rng.normal(0, 0.005, n)
    dlogp = 0.005 + base
    dlogv = 0.005 + base                       # SAME direction
    prices = _integrate(100.0, dlogp)
    volumes = _integrate(100.0, dlogv)
    result = compute_vpd(prices, volumes)
    assert result.correlation > 0
    assert result.signal == 0
    assert result.kind is None


def test_no_divergence_when_no_extreme():
    # Strong negative correlation but price stuck in the middle.
    rng = np.random.default_rng(42)
    n = 21
    base = np.linspace(100, 100, n)
    noise = rng.normal(0, 0.5, n)
    prices = base + noise
    volumes = 200 - (prices - 100) * 10 + rng.normal(0, 1, n)
    # Engineer: price[-1] is not a new high/low.
    prices[-1] = prices[:-1].mean()   # middle of range
    result = compute_vpd(prices, volumes)
    # May or may not trigger threshold; but if it does, price isn't at an extreme → signal must be 0.
    if result.correlation <= -0.30:
        assert result.signal == 0


def test_insufficient_data():
    result = compute_vpd([100, 101, 102], [1, 2, 3])
    assert result.signal == 0
    assert result.correlation == 0.0
