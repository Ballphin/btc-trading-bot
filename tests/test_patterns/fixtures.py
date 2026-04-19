"""Synthetic candle generators with known anchor indices.

Tests assert detector finds anchors within ±1 bar of the known positions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Tuple

import numpy as np
import pandas as pd


def _df_from_closes(closes: np.ndarray, interval_min: int = 60,
                    volumes: np.ndarray | None = None) -> pd.DataFrame:
    """Build an OHLCV DataFrame from a close-price series.

    Each candle has open=prev_close, high/low with small ATR-proportional wicks.
    Volume defaults to 1000 if not provided.
    """
    n = len(closes)
    ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=interval_min * i)
          for i in range(n)]
    opens = np.concatenate([[closes[0]], closes[:-1]])
    # small wicks proportional to scale
    scale = float(np.mean(np.abs(np.diff(closes)))) + 0.1
    rng = np.random.default_rng(42)
    wick = rng.uniform(0.1, 0.3, n) * scale
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick
    vols = volumes if volumes is not None else np.full(n, 1000.0)
    return pd.DataFrame({
        "timestamp": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })


def synthetic_head_and_shoulders(
    n_bars: int = 60,
    shoulder_price: float = 100.0,
    head_price: float = 110.0,
    neckline: float = 95.0,
    noise: float = 0.2,
    volume_declining: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Generate a classic H&S with known anchor indices.

    Returns (df, anchors) where anchors has keys: ls, ln, hd, rn, rs (bar indices).
    """
    n = n_bars
    # Anchor positions: evenly spaced across first 80% of window
    ls, ln, hd, rn, rs = [int(round(n * f)) for f in (0.10, 0.22, 0.35, 0.48, 0.60)]
    break_idx = int(round(n * 0.72))

    closes = np.full(n, neckline * 0.98, dtype=float)
    # Rising into left shoulder
    closes[:ls] = np.linspace(neckline * 0.9, shoulder_price, ls, endpoint=True)
    # Drop to left neck
    closes[ls:ln] = np.linspace(shoulder_price, neckline, ln - ls)
    # Rise to head
    closes[ln:hd] = np.linspace(neckline, head_price, hd - ln)
    # Drop to right neck
    closes[hd:rn] = np.linspace(head_price, neckline, rn - hd)
    # Rise to right shoulder
    closes[rn:rs] = np.linspace(neckline, shoulder_price, rs - rn)
    # Break below neckline
    closes[rs:break_idx] = np.linspace(shoulder_price, neckline * 0.96, break_idx - rs)
    # Continuation
    closes[break_idx:] = np.linspace(neckline * 0.96, neckline * 0.85, n - break_idx)

    rng = np.random.default_rng(123)
    closes = closes + rng.uniform(-noise, noise, n) * (head_price - neckline) * 0.05

    if volume_declining:
        vols = np.full(n, 1500.0)
        vols[ls] = 2000.0
        vols[hd] = 1500.0
        vols[rs] = 900.0
    else:
        vols = np.full(n, 1000.0)

    df = _df_from_closes(closes, interval_min=60, volumes=vols)
    return df, {"ls": ls, "ln": ln, "hd": hd, "rn": rn, "rs": rs, "break": break_idx}


def synthetic_double_top(
    n_bars: int = 30,
    peak_price: float = 100.0,
    trough_price: float = 92.0,
    noise: float = 0.1,
) -> Tuple[pd.DataFrame, dict]:
    p1, tr, p2 = int(n_bars * 0.2), int(n_bars * 0.5), int(n_bars * 0.8)
    closes = np.full(n_bars, trough_price, dtype=float)
    closes[:p1] = np.linspace(trough_price * 0.95, peak_price, p1)
    closes[p1:tr] = np.linspace(peak_price, trough_price, tr - p1)
    closes[tr:p2] = np.linspace(trough_price, peak_price, p2 - tr)
    closes[p2:] = np.linspace(peak_price, trough_price * 0.95, n_bars - p2)
    rng = np.random.default_rng(7)
    closes = closes + rng.uniform(-noise, noise, n_bars)
    return _df_from_closes(closes, interval_min=60), {"p1": p1, "tr": tr, "p2": p2}


def synthetic_ascending_triangle(n_bars: int = 30) -> pd.DataFrame:
    """Resistance at 100, rising support from 90 to 99."""
    closes = []
    rng = np.random.default_rng(11)
    resistance = 100.0
    for i in range(n_bars):
        support = 90.0 + (9.0 * i / n_bars)
        # Zigzag between support and just under resistance
        if i % 4 == 0:
            closes.append(resistance - rng.uniform(0.3, 0.8))
        elif i % 4 == 2:
            closes.append(support + rng.uniform(0.2, 0.6))
        else:
            mid = (support + resistance) / 2
            closes.append(mid + rng.uniform(-1, 1))
    return _df_from_closes(np.array(closes), interval_min=60)
