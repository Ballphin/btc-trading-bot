"""Four fill models for realistic scorecard PnL estimation.

    best            — idealized mid-price fill at signal time.
    realistic       — mid + spread/2 + slippage (default 5 bps).
    maker_rejected  — if price moves > 100 bps in 10 s after signal, assume
                      the passive (maker) order didn't fill; use taker fill
                      at +15 s.
    maker_adverse   — conditional on maker filling, price 30 s later vs
                      entry = adverse-selection cost.

Plus square-root market impact (Cont/Stoikov): impact_bps = C × √(size/ADV).

Callers pass a tight OHLCV snapshot around the signal timestamp so we can
compute each model purely from historical data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tradingagents.pulse.stats import sqrt_impact_bps


@dataclass
class FillResult:
    model: str
    entry_price: float
    exit_price: float
    gross_return: float      # (exit - entry) / entry × sign
    cost_bps: float
    net_return: float        # gross - cost
    filled: bool             # False only for maker_rejected when fallback price missing
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ──────────────────────────────────────────────────────────

def _price_at_or_after(
    candles_1m: pd.DataFrame, target_ts: pd.Timestamp, max_lag_sec: int = 90
) -> Optional[float]:
    """Return the first close at or after target_ts, within max_lag_sec."""
    if candles_1m is None or candles_1m.empty or "timestamp" not in candles_1m.columns:
        return None
    df = candles_1m
    mask = df["timestamp"] >= target_ts
    sub = df[mask]
    if sub.empty:
        return None
    first_ts = sub.iloc[0]["timestamp"]
    if (first_ts - target_ts).total_seconds() > max_lag_sec:
        return None
    return float(sub.iloc[0]["close"])


def _price_nearest(candles_1m: pd.DataFrame, target_ts: pd.Timestamp) -> Optional[float]:
    if candles_1m is None or candles_1m.empty or "timestamp" not in candles_1m.columns:
        return None
    deltas = (candles_1m["timestamp"] - target_ts).dt.total_seconds().abs()
    idx = deltas.idxmin()
    return float(candles_1m.loc[idx, "close"])


# ── Core fill computation ────────────────────────────────────────────

def compute_four_fills(
    signal_ts: pd.Timestamp,
    horizon_minutes: int,
    direction: int,
    candles_1m: pd.DataFrame,
    entry_price_hint: float,
    spread_bps: float = 2.0,
    slippage_bps: float = 5.0,
    maker_reject_move_bps: float = 100.0,
    maker_reject_window_sec: int = 10,
    maker_adverse_window_sec: int = 30,
    notional_usd: float = 0.0,
    adv_usd: float = 0.0,
    impact_coefficient: float = 10.0,
) -> Dict[str, FillResult]:
    """Compute all four fill models for a single signal.

    Args:
        signal_ts: timestamp of the signal.
        horizon_minutes: exit horizon (5, 15, 60, …).
        direction: +1 BUY, -1 SHORT, 0 returns empty.
        candles_1m: 1m candles DataFrame [timestamp, close, …] sorted ascending.
                    Must span signal_ts to signal_ts + horizon_minutes.
        entry_price_hint: typically the spot/mid at signal time.
        spread_bps: half-spread (bps) added to realistic fill.
        slippage_bps: additional slippage bps (realistic + maker_rejected).
        maker_reject_move_bps: threshold for "price ran away from maker order".
        maker_reject_window_sec: window to check for the move.
        maker_adverse_window_sec: window to measure maker adverse selection.
        notional_usd: order notional for impact calc (0 disables impact).
        adv_usd: 30-day ADV for impact scaling.
        impact_coefficient: C in sqrt impact (10 for liquid crypto perps).

    Returns:
        Dict keyed by model name with FillResult each.
    """
    if direction == 0:
        return {}

    exit_ts = signal_ts + pd.Timedelta(minutes=horizon_minutes)
    exit_px = _price_at_or_after(candles_1m, exit_ts, max_lag_sec=90)
    if exit_px is None:
        exit_px = _price_nearest(candles_1m, exit_ts)
    if exit_px is None or entry_price_hint <= 0:
        return {}

    gross = (exit_px - entry_price_hint) / entry_price_hint * direction

    # Impact (applied on both entry and exit for realistic model)
    impact_one_side = sqrt_impact_bps(notional_usd, adv_usd, impact_coefficient) if notional_usd > 0 else 0.0
    impact_rt = 2.0 * impact_one_side  # round-trip

    # Model 1: BEST — mid-price, zero cost
    best = FillResult(
        model="best", entry_price=entry_price_hint, exit_price=exit_px,
        gross_return=gross, cost_bps=0.0, net_return=gross, filled=True,
    )

    # Model 2: REALISTIC — mid + spread/2 + slippage (round-trip) + impact
    cost_bps_real = spread_bps + slippage_bps + impact_rt
    realistic = FillResult(
        model="realistic", entry_price=entry_price_hint, exit_price=exit_px,
        gross_return=gross, cost_bps=cost_bps_real,
        net_return=gross - cost_bps_real / 10_000.0, filled=True,
    )

    # Model 3: MAKER_REJECTED — if |move in 10s| > 100 bps, maker order missed.
    # Fall back to taker fill at +15 s from signal with full cost.
    reject_check_ts = signal_ts + pd.Timedelta(seconds=maker_reject_window_sec)
    px_at_10s = _price_at_or_after(candles_1m, reject_check_ts, max_lag_sec=60)
    maker_rejected = None
    if px_at_10s is not None and entry_price_hint > 0:
        move_bps = abs(px_at_10s - entry_price_hint) / entry_price_hint * 10_000
        if move_bps > maker_reject_move_bps:
            # Maker didn't fill — use taker at +15s
            fallback_entry = _price_at_or_after(
                candles_1m,
                signal_ts + pd.Timedelta(seconds=15),
                max_lag_sec=60,
            ) or px_at_10s
            gross_mr = (exit_px - fallback_entry) / fallback_entry * direction
            cost_mr = spread_bps * 2 + slippage_bps * 2 + impact_rt   # full taker RT
            maker_rejected = FillResult(
                model="maker_rejected",
                entry_price=fallback_entry, exit_price=exit_px,
                gross_return=gross_mr, cost_bps=cost_mr,
                net_return=gross_mr - cost_mr / 10_000.0, filled=True,
                notes=f"price moved {move_bps:.1f}bps in {maker_reject_window_sec}s",
            )
    if maker_rejected is None:
        # No rejection — treat identically to realistic but labeled
        maker_rejected = FillResult(
            model="maker_rejected", entry_price=entry_price_hint,
            exit_price=exit_px, gross_return=gross, cost_bps=cost_bps_real,
            net_return=gross - cost_bps_real / 10_000.0, filled=True,
            notes="no rejection (price stable)",
        )

    # Model 4: MAKER_ADVERSE — if maker filled, measure additional drift
    # in the 30 s after fill against direction. Negative = adverse.
    adverse_ts = signal_ts + pd.Timedelta(seconds=maker_adverse_window_sec)
    px_30s = _price_at_or_after(candles_1m, adverse_ts, max_lag_sec=60)
    adverse_bps = 0.0
    if px_30s is not None:
        drift = (px_30s - entry_price_hint) / entry_price_hint * direction
        adverse_bps = -drift * 10_000  # negative drift → adverse → positive cost
        adverse_bps = max(0.0, adverse_bps)
    cost_ma = spread_bps + slippage_bps + impact_rt + adverse_bps
    maker_adverse = FillResult(
        model="maker_adverse", entry_price=entry_price_hint,
        exit_price=exit_px, gross_return=gross, cost_bps=cost_ma,
        net_return=gross - cost_ma / 10_000.0, filled=True,
        notes=f"adverse bps: {adverse_bps:.1f}",
    )

    return {
        "best": best,
        "realistic": realistic,
        "maker_rejected": maker_rejected,
        "maker_adverse": maker_adverse,
    }


# ── Simpler API for scorecard path (prices, not DataFrame) ──────────

def simple_fill_returns(
    direction: int,
    entry_price: float,
    exit_price: float,
    *,
    spread_bps: float = 2.0,
    slippage_bps: float = 5.0,
    price_at_10s: Optional[float] = None,
    price_at_30s: Optional[float] = None,
    maker_reject_move_bps: float = 100.0,
    notional_usd: float = 0.0,
    adv_usd: float = 0.0,
    impact_coefficient: float = 10.0,
) -> Dict[str, FillResult]:
    """Back-compat API taking plain scalars — used by server.py scorecard.

    Same models, but called once per scored pulse without DataFrames.
    """
    if direction == 0 or entry_price <= 0 or exit_price <= 0:
        return {}
    gross = (exit_price - entry_price) / entry_price * direction
    impact_one_side = sqrt_impact_bps(notional_usd, adv_usd, impact_coefficient) if notional_usd > 0 else 0.0
    impact_rt = 2.0 * impact_one_side

    best = FillResult("best", entry_price, exit_price, gross, 0.0, gross, True)
    cost_real = spread_bps + slippage_bps + impact_rt
    realistic = FillResult(
        "realistic", entry_price, exit_price, gross, cost_real,
        gross - cost_real / 10_000, True,
    )

    # Maker-rejected: check move at +10 s if provided
    if price_at_10s is not None and entry_price > 0:
        move_bps = abs(price_at_10s - entry_price) / entry_price * 10_000
        if move_bps > maker_reject_move_bps:
            gross_mr = (exit_price - price_at_10s) / price_at_10s * direction
            cost_mr = spread_bps * 2 + slippage_bps * 2 + impact_rt
            maker_rejected = FillResult(
                "maker_rejected", price_at_10s, exit_price, gross_mr, cost_mr,
                gross_mr - cost_mr / 10_000, True,
                notes=f"maker miss: move {move_bps:.0f}bps",
            )
        else:
            maker_rejected = FillResult(
                "maker_rejected", entry_price, exit_price, gross, cost_real,
                gross - cost_real / 10_000, True, notes="no maker miss",
            )
    else:
        maker_rejected = realistic

    # Maker-adverse: drift against direction from 0 → +30s
    adverse_bps = 0.0
    if price_at_30s is not None and entry_price > 0:
        drift = (price_at_30s - entry_price) / entry_price * direction
        adverse_bps = max(0.0, -drift * 10_000)
    cost_ma = spread_bps + slippage_bps + impact_rt + adverse_bps
    maker_adverse = FillResult(
        "maker_adverse", entry_price, exit_price, gross, cost_ma,
        gross - cost_ma / 10_000, True, notes=f"adverse {adverse_bps:.0f}bps",
    )

    return {
        "best": best,
        "realistic": realistic,
        "maker_rejected": maker_rejected,
        "maker_adverse": maker_adverse,
    }
