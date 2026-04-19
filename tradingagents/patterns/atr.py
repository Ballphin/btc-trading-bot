"""Reference ATR computation for pattern-tolerance thresholds.

Crypto-specific: Hyperliquid funding reset happens on the top of each hour
(:00 UTC). These bars have mechanically elevated wick size that shouldn't
influence pattern tolerance calculations. We drop them from the ATR sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _true_range(df: pd.DataFrame) -> np.ndarray:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return tr


def compute_ref_atr(
    df: pd.DataFrame,
    period: int = 14,
    exclude_funding_bars: bool = True,
) -> float:
    """Return a single reference ATR value (price units) for the window.

    Args:
        df: DataFrame with timestamp + OHLC columns.
        period: ATR lookback.
        exclude_funding_bars: If True and timestamps are available, drop bars
            whose UTC minute == 0 (Hyperliquid funding reset) before averaging.
    """
    if len(df) < 2:
        return 0.0

    work = df
    if exclude_funding_bars and "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], utc=True)
            mask = ts.dt.minute != 0
            # Only drop if we have enough bars remaining
            if mask.sum() >= max(period + 1, len(df) // 2):
                work = df.loc[mask].reset_index(drop=True)
        except Exception:
            pass

    tr = _true_range(work)
    if len(tr) <= period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0
    # Exponential moving average (Wilder's smoothing)
    atr = float(np.mean(tr[:period]))
    alpha = 1.0 / period
    for t in tr[period:]:
        atr = (1 - alpha) * atr + alpha * float(t)
    return atr


def atr_tol(k_atr: float, atr_ref: float, k_pct: float, price: float) -> float:
    """Canonical tolerance: ``max(k_atr * atr_ref, k_pct * price)``.

    ATR floor prevents micro-noise matches in low-vol regimes where a
    pure-ATR threshold would collapse toward zero.
    """
    return max(k_atr * atr_ref, k_pct * price)
