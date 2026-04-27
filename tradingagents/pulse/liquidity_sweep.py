"""Liquidity-sweep detector (Pulse v4).

Reads raw 1m OHLCV directly — does **not** route through
``quant_pulse_data.detect_patterns()`` and **bypasses** the legacy
``_wick_filter`` so genuine sweep candles (wick/body ≫ 3) are not
discarded as "liquidation artifacts."

Sweep semantics
---------------
A sweep occurs when price spikes through a recent N-bar extreme and is
then absorbed back into the prior range. Direction of resulting trade:

    * Long-trap (high broken, immediate reclaim down) → SHORT signal (-1)
    * Short-trap (low broken, immediate reclaim up)   → LONG  signal (+1)

Confirmation rules
------------------
1. Within the last bar, price has moved beyond the ``extreme_lookback_bars``
   prior extreme (high for long-trap, low for short-trap).
2. Within ``reclaim_within_bars`` bars after the breach (inclusive of the
   breach bar), price reclaims the extreme on a candle whose volume is
   ≥ ``reclaim_volume_mul`` × the rolling ``extreme_lookback_bars`` mean.
3. Aligned-funding rejection: if ``reject_aligned_funding`` is True and
   funding sign matches sweep direction (longs already paying → bearish
   sweep is just expected; shorts already paying → bullish sweep is just
   expected), reject — a true stop-hunt is contrarian to the carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class SweepResult:
    direction: int                 # -1 short, +1 long, 0 none
    breach_idx: Optional[int]      # row index of the breach bar
    reclaim_idx: Optional[int]     # row index of the reclaim bar
    extreme_price: Optional[float] # the extreme that was swept
    reason: Optional[str] = None   # e.g. 'aligned_funding_reject'


def detect_liquidity_sweep(
    candles_1m: pd.DataFrame,
    *,
    extreme_lookback_bars: int = 60,
    reclaim_within_bars: int = 10,
    reclaim_volume_mul: float = 2.0,
    funding_rate: Optional[float] = None,
    reject_aligned_funding: bool = True,
) -> SweepResult:
    """Detect a sweep that completes within the most recent
    ``reclaim_within_bars`` bars.

    Returns ``SweepResult(direction=0)`` when no sweep is found or data
    is insufficient.
    """
    if candles_1m is None or candles_1m.empty:
        return SweepResult(0, None, None, None)
    if len(candles_1m) < extreme_lookback_bars + 2:
        return SweepResult(0, None, None, None)

    df = candles_1m.reset_index(drop=True)
    n = len(df)
    win_start = max(extreme_lookback_bars, n - reclaim_within_bars - 1)
    if win_start >= n:
        return SweepResult(0, None, None, None)

    rolling_vol = df["volume"].rolling(extreme_lookback_bars, min_periods=10).mean()

    # Look for a breach in the last (reclaim_within_bars + 1) bars and a
    # reclaim in the same window. Iterate from earliest candidate breach
    # to most-recent so the *first* completed sweep wins.
    for breach_idx in range(win_start, n):
        prior_lo = breach_idx - extreme_lookback_bars
        if prior_lo < 0:
            continue
        prior = df.iloc[prior_lo:breach_idx]
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())
        bar = df.iloc[breach_idx]

        # Long-trap: bar broke prior_high, then a reclaim downward
        if float(bar["high"]) > prior_high:
            for j in range(breach_idx, min(n, breach_idx + reclaim_within_bars + 1)):
                rb = df.iloc[j]
                vol_mean = rolling_vol.iloc[j] if j < len(rolling_vol) else None
                if vol_mean is None or pd.isna(vol_mean) or vol_mean <= 0:
                    continue
                if float(rb["close"]) < prior_high and float(rb["volume"]) >= reclaim_volume_mul * float(vol_mean):
                    direction = -1
                    if reject_aligned_funding and funding_rate is not None and funding_rate > 0:
                        return SweepResult(0, breach_idx, j, prior_high, "aligned_funding_reject")
                    return SweepResult(direction, breach_idx, j, prior_high)
            continue

        # Short-trap: bar broke prior_low, then a reclaim upward
        if float(bar["low"]) < prior_low:
            for j in range(breach_idx, min(n, breach_idx + reclaim_within_bars + 1)):
                rb = df.iloc[j]
                vol_mean = rolling_vol.iloc[j] if j < len(rolling_vol) else None
                if vol_mean is None or pd.isna(vol_mean) or vol_mean <= 0:
                    continue
                if float(rb["close"]) > prior_low and float(rb["volume"]) >= reclaim_volume_mul * float(vol_mean):
                    direction = +1
                    if reject_aligned_funding and funding_rate is not None and funding_rate < 0:
                        return SweepResult(0, breach_idx, j, prior_low, "aligned_funding_reject")
                    return SweepResult(direction, breach_idx, j, prior_low)
            continue

    return SweepResult(0, None, None, None)
