"""Time-Series Momentum (TSMOM) — primary alpha layer.

Based on Moskowitz, Ooi & Pedersen (2012), "Time-Series Momentum":
    * Multi-lookback: 21d / 63d / 252d on 1h closes (504 / 1512 / 6048 bars).
    * Direction = sign of average of three lookback sign-returns.
    * Strength = |average| ∈ {0, 1/3, 2/3, 1}.
    * Size weight = target_vol / realized_vol_30d (inverse-vol scaling).

Layer role: output `direction ∈ {-1, 0, +1}` AND-gated with confluence engine
— confluence must agree with TSMOM direction to produce a signal.

Cache: result per ticker in `eval_results/tsmom/{ticker}.json`; refreshed on
1h cadence (not per 5-min pulse).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TsmomResult:
    ticker: str
    direction: int              # -1, 0, +1
    strength: float             # 0.0, 0.333, 0.667, 1.0
    size_weight: float          # target_vol / realized_vol_30d
    realized_vol_30d: float     # annualized
    lookback_returns: Dict[str, float]   # e.g. {"21d": 0.12, "63d": 0.05, "252d": -0.03}
    computed_at: float          # unix ts
    n_bars_used: int
    lookbacks_hours: List[int]
    target_annualized_vol: float
    insufficient_history: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ── Pure computation ──────────────────────────────────────────────────

def compute_tsmom(
    candles_1h: pd.DataFrame,
    ticker: str,
    lookbacks_hours: List[int] = (504, 1512, 6048),
    target_annualized_vol: float = 0.20,
    min_bars_required: int = 500,
) -> TsmomResult:
    """Compute TSMOM direction/strength/sizing from 1h close history.

    Args:
        candles_1h: DataFrame with column "close" (and optionally "timestamp"),
                    sorted ascending by time.
        ticker: symbol for logging and result.
        lookbacks_hours: lookback windows in 1h bars. Must be ≥1 element.
        target_annualized_vol: annualized vol target for inverse-vol scaling.
        min_bars_required: below this, flag insufficient_history=True.

    Returns:
        TsmomResult. If history is too short, direction=0 / strength=0.
    """
    if candles_1h is None or len(candles_1h) == 0 or "close" not in candles_1h.columns:
        return TsmomResult(
            ticker=ticker, direction=0, strength=0.0, size_weight=0.0,
            realized_vol_30d=0.0, lookback_returns={},
            computed_at=time.time(), n_bars_used=0,
            lookbacks_hours=list(lookbacks_hours),
            target_annualized_vol=target_annualized_vol,
            insufficient_history=True,
        )

    closes = candles_1h["close"].astype(float).values
    n_bars = len(closes)
    insufficient = n_bars < min_bars_required

    # Per-lookback sign returns (use the max available lookback if < requested)
    returns: Dict[str, float] = {}
    signs: List[float] = []
    for lb in lookbacks_hours:
        effective_lb = min(int(lb), n_bars - 1)
        if effective_lb < 24:          # need at least 1 day
            continue
        ret = (closes[-1] / closes[-effective_lb - 1]) - 1.0
        returns[f"{lb}h"] = float(ret)
        signs.append(1.0 if ret > 0 else -1.0 if ret < 0 else 0.0)

    # Direction = majority sign; strength = magnitude of average.
    if not signs:
        direction = 0
        strength = 0.0
    else:
        avg = sum(signs) / len(signs)
        direction = 1 if avg > 0.33 else -1 if avg < -0.33 else 0
        strength = min(abs(avg), 1.0)

    # Realized vol (30 days = 720 1h bars) — annualized
    vol_window = min(720, n_bars - 1)
    if vol_window >= 24:
        log_rets = np.diff(np.log(np.clip(closes[-vol_window - 1:], 1e-12, None)))
        per_bar_sigma = float(np.std(log_rets, ddof=1)) if len(log_rets) > 1 else 0.0
        annualized_vol = per_bar_sigma * math.sqrt(24 * 365)
    else:
        annualized_vol = 0.0

    size_weight = (
        float(target_annualized_vol / annualized_vol)
        if annualized_vol > 1e-6 else 0.0
    )
    # Clamp extreme sizes (e.g. ultra-low-vol history) at 5× target
    size_weight = min(size_weight, 5.0)

    return TsmomResult(
        ticker=ticker,
        direction=int(direction),
        strength=round(float(strength), 4),
        size_weight=round(float(size_weight), 4),
        realized_vol_30d=round(float(annualized_vol), 4),
        lookback_returns={k: round(v, 6) for k, v in returns.items()},
        computed_at=time.time(),
        n_bars_used=int(n_bars),
        lookbacks_hours=list(lookbacks_hours),
        target_annualized_vol=float(target_annualized_vol),
        insufficient_history=bool(insufficient),
    )


# ── Cache I/O ─────────────────────────────────────────────────────────

def _cache_path(ticker: str, results_dir: str = "./eval_results") -> Path:
    return Path(results_dir) / "tsmom" / f"{ticker}.json"


def load_tsmom(
    ticker: str, results_dir: str = "./eval_results", max_age_sec: float = 3900
) -> Optional[TsmomResult]:
    """Load cached TSMOM result if fresh. Default max_age = 65 min (>1h cadence)."""
    path = _cache_path(ticker, results_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        if time.time() - float(raw.get("computed_at", 0)) > max_age_sec:
            return None
        return TsmomResult(**raw)
    except Exception as e:
        logger.warning(f"[TSMOM] Failed to load cache for {ticker}: {e}")
        return None


def save_tsmom_atomic(result: TsmomResult, results_dir: str = "./eval_results") -> Path:
    """Atomic write of TSMOM cache (tmp + os.replace)."""
    path = _cache_path(result.ticker, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result.to_dict(), indent=2))
    os.replace(tmp, path)
    return path


# ── Refresh helper (fetches 1h history + writes cache) ───────────────

def refresh_tsmom(
    ticker: str,
    hl_client=None,
    lookbacks_hours: List[int] = (504, 1512, 6048),
    target_annualized_vol: float = 0.20,
    min_bars_required: int = 500,
    results_dir: str = "./eval_results",
) -> TsmomResult:
    """Fetch enough 1h history and compute + cache TSMOM for a ticker."""
    from tradingagents.dataflows.hyperliquid_client import HyperliquidClient
    from datetime import datetime, timedelta, timezone

    hl = hl_client or HyperliquidClient()
    base_asset = ticker.replace("-USD", "").replace("USDT", "").upper()
    max_lb = max(lookbacks_hours) if lookbacks_hours else 720
    # Request ~1.1× the max lookback so we have a cushion
    bars_needed = int(max_lb * 1.1)
    start_dt = datetime.now(timezone.utc) - timedelta(hours=bars_needed)

    try:
        df = hl.get_ohlcv(
            base_asset, "1h",
            start=start_dt.strftime("%Y-%m-%d"),
            end=(datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"),
            max_age_override=1800,
        )
    except Exception as e:
        logger.warning(f"[TSMOM] fetch failed for {ticker}: {e}")
        df = pd.DataFrame()

    result = compute_tsmom(
        df, ticker,
        lookbacks_hours=lookbacks_hours,
        target_annualized_vol=target_annualized_vol,
        min_bars_required=min_bars_required,
    )
    save_tsmom_atomic(result, results_dir)
    return result
