"""Volume-Price Divergence (VPD) detector — Pulse v4.

Replaces the cross-sectional rank formula from "101 Formulaic Alphas"
(Kakushadze) which is inappropriate for single-asset time series. We use
a rolling Pearson correlation between log-price-changes and log-volume-
changes; when correlation breaks down at the same time price prints a
new local extreme, that's the divergence signal.

Worked example
--------------
20-bar window. Suppose price marches steadily upward
``Δlog P = [0.01]*20`` while volume contracts
``Δlog V = [-(i / 20)]_{i=0..19}``. Then::

    pearson_corr(ΔP, ΔV) ≈ -1.0   (price up, volume down)
    P[-1] > max(P[-20:-1])        (new high)
    → signal = -1                 (bearish divergence; trend exhaustion)

Conversely, price falling on declining volume signals bullish absorption
(``signal = +1``). Otherwise ``0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

DEFAULT_LOOKBACK_BARS = 20
DEFAULT_CORR_THRESHOLD = -0.30


@dataclass(frozen=True)
class VPDResult:
    signal: int                # -1 bearish, 0 none, +1 bullish
    correlation: float         # rolling Pearson corr(Δlog P, Δlog V)
    kind: Optional[str]        # 'bearish_div' | 'bullish_div' | None


def compute_vpd(
    prices: Sequence[float],
    volumes: Sequence[float],
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> VPDResult:
    """Compute VPD signal on the most recent ``lookback_bars + 1`` bars.

    Args:
        prices: closing prices, ascending in time. Need ≥ ``lookback_bars + 1``.
        volumes: same length and order as ``prices``.
        lookback_bars: rolling window for correlation.
        corr_threshold: divergence triggers when ``corr ≤ corr_threshold``.

    Returns ``VPDResult(signal=0, correlation=0.0, kind=None)`` on
    insufficient data or degenerate volume.
    """
    p = np.asarray(list(prices), dtype=float)
    v = np.asarray(list(volumes), dtype=float)
    if len(p) < lookback_bars + 1 or len(v) != len(p):
        return VPDResult(0, 0.0, None)
    p_window = p[-(lookback_bars + 1):]
    v_window = v[-(lookback_bars + 1):]
    if np.any(p_window <= 0) or np.any(v_window <= 0):
        return VPDResult(0, 0.0, None)
    dlogp = np.diff(np.log(p_window))
    dlogv = np.diff(np.log(v_window))
    if np.std(dlogp, ddof=1) < 1e-12 or np.std(dlogv, ddof=1) < 1e-12:
        return VPDResult(0, 0.0, None)
    corr = float(np.corrcoef(dlogp, dlogv)[0, 1])
    if not np.isfinite(corr):
        return VPDResult(0, 0.0, None)

    if corr > corr_threshold:
        return VPDResult(0, corr, None)

    last = float(p[-1])
    prev_window = p[-(lookback_bars + 1):-1]   # all but the latest bar
    if last >= float(np.max(prev_window)):
        return VPDResult(-1, corr, "bearish_div")
    if last <= float(np.min(prev_window)):
        return VPDResult(+1, corr, "bullish_div")
    return VPDResult(0, corr, None)
