"""Local-extrema detection on smoothed price series (Pulse v4).

Used as the input layer for structural patterns (H&S, Double Bottom,
Channels). Smoothing follows the Lo–Mamaysky–Wang (2000) approach: a
Nadaraya–Watson kernel regression (Gaussian kernel) of closes, then
extrema by sign-change of the smoothed first difference.

Determinism contract: ``find_extrema(closes[:t]) == find_extrema(closes[:t+k])``
for all returned extrema with index ≤ ``t − bandwidth`` (i.e., extrema in
the bandwidth-edge region may shift, but **confirmed** extrema do not).
Callers must use ``confirmation_idx = idx + bandwidth`` to get a stable
extremum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np

ExtremumKind = Literal["min", "max"]


@dataclass(frozen=True)
class Extremum:
    idx: int                       # raw index into the input series
    confirmation_idx: int          # idx + bandwidth (smallest stable index)
    kind: ExtremumKind
    price: float                   # actual close at idx (NOT smoothed)


def _gaussian_smooth(values: np.ndarray, bandwidth: int) -> np.ndarray:
    """Symmetric Gaussian-kernel smoother. Edges shrink the window.

    ``bandwidth`` is the kernel half-width in bars (σ). The kernel
    extends ``±2σ`` so the effective window is ``4σ + 1``.
    """
    if bandwidth <= 0:
        return values.copy()
    half = 2 * bandwidth
    x = np.arange(-half, half + 1, dtype=float)
    w = np.exp(-0.5 * (x / bandwidth) ** 2)
    w /= w.sum()
    n = len(values)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        wlo = lo - (i - half)
        whi = wlo + (hi - lo)
        kernel = w[wlo:whi]
        kernel = kernel / kernel.sum()
        out[i] = float(np.dot(values[lo:hi], kernel))
    return out


def find_extrema(
    closes: Sequence[float],
    bandwidth: int = 8,
    raw_prices: Sequence[float] | None = None,
) -> List[Extremum]:
    """Return local minima/maxima on the smoothed close series.

    Args:
        closes: 1-D sequence of close prices (length ≥ 2 × bandwidth + 3).
        bandwidth: kernel σ in bars. Extrema confirm at ``idx + bandwidth``.
        raw_prices: if provided, the ``Extremum.price`` field uses this
            series at ``idx`` (e.g., raw highs for tops, raw lows for
            bottoms). Defaults to ``closes``.

    Determinism: an extremum at raw index ``i`` is reported only if
    ``i + bandwidth ≤ len(closes) - 1`` — i.e., its confirmation bar has
    been observed. This guarantees prefix-stability.
    """
    arr = np.asarray(list(closes), dtype=float)
    n = len(arr)
    if n < 2 * bandwidth + 3:
        return []
    smoothed = _gaussian_smooth(arr, bandwidth)
    diffs = np.diff(smoothed)            # length n-1
    out: List[Extremum] = []
    raw = np.asarray(list(raw_prices), dtype=float) if raw_prices is not None else arr
    # The Gaussian kernel reaches ±2σ, so smoothed[i] depends on values up
    # to i + 2*bandwidth. Therefore an extremum at idx i is prefix-stable
    # only once we've observed bar i + 2*bandwidth + 1 (need diffs[i] which
    # uses smoothed[i+1] which depends on values up to i + 1 + 2*bandwidth).
    half = 2 * bandwidth
    confirmation_offset = half + 1
    last_safe_idx = n - 1 - confirmation_offset
    for i in range(1, len(diffs)):
        if i > last_safe_idx:
            break
        prev_d = diffs[i - 1]
        curr_d = diffs[i]
        if prev_d > 0 and curr_d < 0:
            out.append(Extremum(idx=i, confirmation_idx=i + confirmation_offset,
                                kind="max", price=float(raw[i])))
        elif prev_d < 0 and curr_d > 0:
            out.append(Extremum(idx=i, confirmation_idx=i + confirmation_offset,
                                kind="min", price=float(raw[i])))
    return out
