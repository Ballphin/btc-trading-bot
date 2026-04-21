"""Directional regime classifier — Stage 2 Commit G.

Complements the statistical regime detector in ``regime.py`` (which
classifies into trend / chop / high_vol_trend / mixed) with a
**directional** taxonomy used by the auto-tuner's regime-profile system:

    bull        — sustained uptrend.
    bear        — sustained downtrend.
    range_bound — low realised range, flat drift.
    ambiguous   — insufficient history, conflicting signals, transitioning.

Crypto-specific thresholds (WCT fix from debate round):

    * bull: 90d total return > +15% AND ≥60% of last 30 closes above
      their trailing 30d SMA.
    * bear: 90d total return < -15% AND ≥60% of last 30 closes below
      their trailing 30d SMA.
    * range_bound: 30d (high-low) ÷ 30d mean ATR < 1.2 AND
      |30d return| < 5%.
    * ambiguous: everything else, including insufficient data.

The 90d return captures trend strength and the 30d fraction-above-SMA
captures *consistency* — together they resist the "one big candle"
flapping that a single-lookback momentum rule suffers from.

This module is log-only by design. It **never** switches the active
regime profile automatically; the user always picks ``active_regime``
manually via the Auto-Tune UI.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Latency budget (SSE fix): classifier must never extend the live
# pulse path by more than 50ms. Configurable via env for tests.
PULSE_REGIME_TIMEOUT_MS = int(os.getenv("PULSE_REGIME_TIMEOUT_MS", "50"))

# Thresholds — centralised so the "open question" re-tuning pass in
# week-1 observation can bump them without hunting through the module.
BULL_RETURN_90D = 0.15
BEAR_RETURN_90D = -0.15
CONSISTENCY_FRAC = 0.60
RANGE_RETURN_ABS = 0.05
RANGE_ATR_RATIO = 1.2

LOG_DIR = Path("results/regime")


@dataclass
class DirectionalRegime:
    label: str                 # bull / bear / range_bound / ambiguous
    return_90d: float
    frac_above_sma30: float    # fraction of last 30 closes above their 30d SMA
    return_30d: float
    range_atr_ratio: float     # 30d range ÷ mean ATR
    sample_size: int
    insufficient_history: bool
    reason: str                # short human-readable explanation
    timestamp: str             # UTC ISO, when classified

    def to_dict(self) -> dict:
        return asdict(self)


def _true_range(df: pd.DataFrame) -> pd.Series:
    hi, lo, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(hi - lo), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr


def _classify_sync(df: pd.DataFrame) -> DirectionalRegime:
    """Synchronous classification — wrapped by the timeout driver below.

    ``df`` is expected to be daily OHLC sorted ascending, with columns
    ``open``, ``high``, ``low``, ``close``. Minimum 90 rows for a
    non-ambiguous result.
    """
    now = datetime.now(timezone.utc).isoformat()
    n = len(df)
    if n < 90:
        return DirectionalRegime(
            label="ambiguous",
            return_90d=float("nan"),
            frac_above_sma30=float("nan"),
            return_30d=float("nan"),
            range_atr_ratio=float("nan"),
            sample_size=n,
            insufficient_history=True,
            reason=f"only {n} daily candles; need ≥90",
            timestamp=now,
        )

    closes = df["close"].astype(float).to_numpy()

    return_90d = float(closes[-1] / closes[-90] - 1.0)
    return_30d = float(closes[-1] / closes[-30] - 1.0)

    # 30d SMA consistency: fraction of the last 30 closes that sit
    # above their trailing 30d simple-moving-average.
    sma30 = pd.Series(closes).rolling(30).mean().to_numpy()
    mask = ~np.isnan(sma30[-30:])
    last30 = closes[-30:][mask]
    sma30_tail = sma30[-30:][mask]
    frac_above = float(np.mean(last30 > sma30_tail)) if last30.size else float("nan")

    # 30d range ÷ mean 30d ATR — dimensionless "range tightness".
    last30_df = df.iloc[-30:]
    rng_30 = float(last30_df["high"].max() - last30_df["low"].min())
    atr_mean = float(_true_range(df).rolling(14).mean().iloc[-30:].mean())
    range_atr_ratio = rng_30 / atr_mean if atr_mean > 0 else float("inf")

    # Composite verdict.
    if return_90d > BULL_RETURN_90D and frac_above >= CONSISTENCY_FRAC:
        label, reason = "bull", (
            f"90d return {return_90d:+.1%} > {BULL_RETURN_90D:+.0%} and "
            f"{frac_above:.0%} of 30d closes above SMA30"
        )
    elif return_90d < BEAR_RETURN_90D and (1.0 - frac_above) >= CONSISTENCY_FRAC:
        label, reason = "bear", (
            f"90d return {return_90d:+.1%} < {BEAR_RETURN_90D:+.0%} and "
            f"{1.0 - frac_above:.0%} of 30d closes below SMA30"
        )
    elif range_atr_ratio < RANGE_ATR_RATIO and abs(return_30d) < RANGE_RETURN_ABS:
        label, reason = "range_bound", (
            f"30d range/ATR {range_atr_ratio:.2f} < {RANGE_ATR_RATIO} and "
            f"|30d return| {abs(return_30d):.1%} < {RANGE_RETURN_ABS:.0%}"
        )
    else:
        label, reason = "ambiguous", (
            f"return_90d={return_90d:+.1%} frac_above_sma30={frac_above:.2f} "
            f"range_atr={range_atr_ratio:.2f} return_30d={return_30d:+.1%}"
        )

    return DirectionalRegime(
        label=label,
        return_90d=return_90d,
        frac_above_sma30=frac_above,
        return_30d=return_30d,
        range_atr_ratio=range_atr_ratio,
        sample_size=n,
        insufficient_history=False,
        reason=reason,
        timestamp=now,
    )


def classify_directional(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    *,
    log: bool = True,
    timeout_ms: Optional[int] = None,
) -> DirectionalRegime:
    """Classify directional regime with a hard latency budget.

    Runs ``_classify_sync`` in a small threadpool and cancels with an
    ``ambiguous`` fallback if it exceeds ``timeout_ms`` (default
    ``PULSE_REGIME_TIMEOUT_MS`` from env). Logs the result to
    ``results/regime/{ticker}_regime.jsonl`` when ``log=True`` and a
    ticker is supplied.
    """
    budget_ms = timeout_ms if timeout_ms is not None else PULSE_REGIME_TIMEOUT_MS
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_classify_sync, df)
        try:
            result = fut.result(timeout=budget_ms / 1000.0)
        except FuturesTimeout:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning(
                "classify_directional timed out after %.1fms (budget=%dms) — "
                "returning ambiguous fallback",
                elapsed_ms, budget_ms,
            )
            result = DirectionalRegime(
                label="ambiguous",
                return_90d=float("nan"),
                frac_above_sma30=float("nan"),
                return_30d=float("nan"),
                range_atr_ratio=float("nan"),
                sample_size=len(df),
                insufficient_history=False,
                reason=f"classifier timeout ({budget_ms}ms budget)",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    if log and ticker:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            path = LOG_DIR / f"{ticker.upper()}_regime.jsonl"
            with path.open("a") as f:
                f.write(json.dumps({"ticker": ticker.upper(), **result.to_dict()}) + "\n")
        except Exception as e:  # pragma: no cover — disk-full etc.
            logger.warning("regime log write failed for %s: %s", ticker, e)

    return result
