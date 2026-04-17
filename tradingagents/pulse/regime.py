"""Regime detector — leading-indicator version of GARCH-residual diagnostics.

Rather than fit a full GARCH(1,1) (which adds a heavy dependency), we use
two residual-style statistics on rolling 1h returns:

    returns_acf_lag1          — sign of momentum; +ve → trend-following.
    abs_returns_acf_lag1      — volatility clustering (GARCH-effect proxy).

Combined with a realized-vol z-score, this gives:
    mode ∈ {"trend", "chop", "high_vol_trend", "mixed"}.

Safety backstop: 90-day percentile clamp on vol z-score to prevent regime
flags from being stuck during multi-day volatility dislocations (e.g. the
2022-05 LUNA collapse where EWMA z-score saturated for 96h straight).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tradingagents.pulse.stats import autocorr

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    mode: str                       # trend / chop / high_vol_trend / mixed
    vol_z_score: float
    vol_z_clipped: float
    returns_acf1: float
    abs_returns_acf1: float
    directional_bias: int           # +1 / 0 / -1  (EMA50 vs EMA200 on 4h)
    realized_vol_annualized: float
    sample_size: int
    insufficient_history: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ── Core detector ─────────────────────────────────────────────────────

def detect_regime(
    candles_1h: pd.DataFrame,
    candles_4h: Optional[pd.DataFrame] = None,
    vol_window_bars: int = 500,
    percentile_floor_pct: float = 10.0,
    percentile_ceiling_pct: float = 90.0,
    min_bars_required: int = 240,
) -> RegimeResult:
    """Detect regime from 1h OHLCV (and optional 4h for directional bias).

    Args:
        candles_1h: DataFrame with column "close", sorted ascending.
        candles_4h: optional, used for directional bias via EMA50/EMA200.
        vol_window_bars: rolling window for EWMA-like vol z-score baseline.
        percentile_floor_pct / ceiling_pct: hard clamp on vol z (LUNA backstop).
        min_bars_required: below this, result flagged insufficient_history.
    """
    if candles_1h is None or len(candles_1h) == 0 or "close" not in candles_1h.columns:
        return _empty_regime()

    closes = candles_1h["close"].astype(float).values
    n = len(closes)
    if n < 30:
        return _empty_regime(sample_size=n)

    insufficient = n < min_bars_required

    # Log returns ----------------------------------------------------
    log_rets = np.diff(np.log(np.clip(closes, 1e-12, None)))
    if len(log_rets) < 5:
        return _empty_regime(sample_size=n)

    # Realized vol (last 30 days = 720 bars, annualized)
    vol_window = min(720, len(log_rets))
    recent_rets = log_rets[-vol_window:]
    per_bar_sigma = float(np.std(recent_rets, ddof=1)) if len(recent_rets) > 1 else 0.0
    annualized_vol = per_bar_sigma * math.sqrt(24 * 365)

    # EWMA-style vol z-score: compare recent 30d vol to trailing baseline.
    # Baseline = std of (vol_window_bars) older returns (min 60 bars).
    baseline_window = min(vol_window_bars, len(log_rets) - 5)
    if baseline_window < 60:
        vol_z = 0.0
    else:
        baseline = log_rets[-baseline_window - vol_window : -vol_window] \
            if baseline_window + vol_window < len(log_rets) else log_rets[:-vol_window]
        if len(baseline) > 10:
            baseline_sigma = float(np.std(baseline, ddof=1))
            sigma_distribution = baseline_sigma
            # Use rolling 30-bar std as distribution estimator
            rolling = pd.Series(log_rets).rolling(30, min_periods=10).std().dropna().values
            if len(rolling) > 10 and np.std(rolling) > 1e-10:
                median_sigma = float(np.median(rolling))
                sigma_std = float(np.std(rolling))
                vol_z = (per_bar_sigma - median_sigma) / sigma_std
            elif sigma_distribution > 1e-10:
                vol_z = (per_bar_sigma - sigma_distribution) / sigma_distribution
            else:
                vol_z = 0.0
        else:
            vol_z = 0.0

    # 90-day percentile floor/ceiling (LUNA backstop)
    pct_window = min(90 * 24, len(log_rets))
    if pct_window >= 240:
        rolling_vols = (
            pd.Series(log_rets[-pct_window:]).rolling(30, min_periods=10).std().dropna().values
        )
        if len(rolling_vols) > 30:
            floor = float(np.percentile(rolling_vols, percentile_floor_pct))
            ceil = float(np.percentile(rolling_vols, percentile_ceiling_pct))
            clipped_sigma = max(min(per_bar_sigma, ceil), floor)
            if ceil > floor + 1e-10:
                vol_z_clipped = (clipped_sigma - float(np.median(rolling_vols))) / (
                    (ceil - floor) / 2.0
                )
            else:
                vol_z_clipped = vol_z
        else:
            vol_z_clipped = vol_z
    else:
        vol_z_clipped = vol_z

    # Autocorrs on returns + abs-returns (GARCH-residual proxy).
    # Lag 1 captures intra-hour momentum/reversion; abs-lag-1 captures
    # volatility clustering (GARCH effect).
    returns_acf1 = autocorr(log_rets, lag=1)
    abs_returns_acf1 = autocorr(np.abs(log_rets), lag=1)

    # Directional bias from 4h EMAs
    directional_bias = 0
    if candles_4h is not None and len(candles_4h) >= 220 and "close" in candles_4h.columns:
        closes_4h = candles_4h["close"].astype(float).values
        ema50 = _ema(closes_4h, 50)
        ema200 = _ema(closes_4h, 200)
        if np.isfinite(ema50) and np.isfinite(ema200):
            if ema50 > ema200 * 1.001:
                directional_bias = 1
            elif ema50 < ema200 * 0.999:
                directional_bias = -1

    # Regime classification — thresholds calibrated to crypto 1h ACF
    # (Makarov & Schoar 2020: BTC 1h lag-1 ACF ∈ [-0.05, +0.08] typically).
    if insufficient:
        mode = "mixed"
    elif vol_z_clipped > 1.5 and returns_acf1 > 0.02:
        mode = "high_vol_trend"
    elif returns_acf1 > 0.03 and vol_z_clipped > -0.5:
        mode = "trend"
    elif returns_acf1 < -0.02 and abs(vol_z_clipped) < 0.5:
        mode = "chop"
    else:
        mode = "mixed"

    return RegimeResult(
        mode=mode,
        vol_z_score=round(float(vol_z), 4),
        vol_z_clipped=round(float(vol_z_clipped), 4),
        returns_acf1=round(float(returns_acf1), 4),
        abs_returns_acf1=round(float(abs_returns_acf1), 4),
        directional_bias=int(directional_bias),
        realized_vol_annualized=round(float(annualized_vol), 4),
        sample_size=int(n),
        insufficient_history=bool(insufficient),
    )


def _empty_regime(sample_size: int = 0) -> RegimeResult:
    return RegimeResult(
        mode="mixed", vol_z_score=0.0, vol_z_clipped=0.0,
        returns_acf1=0.0, abs_returns_acf1=0.0, directional_bias=0,
        realized_vol_annualized=0.0, sample_size=sample_size,
        insufficient_history=True,
    )


def _ema(values: np.ndarray, span: int) -> float:
    """Simple EMA last value (pandas-equivalent)."""
    if len(values) < span:
        return float("nan")
    alpha = 2.0 / (span + 1.0)
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return float(ema)
