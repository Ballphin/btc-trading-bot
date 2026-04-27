"""Validate Pulse v4 pattern detectors against real historical OHLCV.

Fetches BTC-USD candles via the era-aware router (Binance / Hyperliquid)
and walks forward in time, printing every pattern hit with timestamp
and price context so a human can eyeball whether the detectors fire on
the "right" structures.

This is a *qualitative* sanity check — not the full validation protocol
(bootstrap Sharpe / SPA / walk-forward). It answers:

    1. Do the detectors fire at all on real data?
    2. When they fire, does the price action "look like" the pattern
       (H&S / double bottom / sweep / VPD divergence)?
    3. Are there obviously wrong fires (e.g., sweep on a calm trend bar)?

Usage:

    python scripts/validate_pulse_v4_patterns.py \
        --ticker BTC-USD --days 14 --step-hours 4

Output: one line per pattern hit, aggregated counts, and fires-per-day
rate. Redirect to a file for log inspection.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict

import pandas as pd

from tradingagents.dataflows.historical_router import (
    fetch_funding_historical,
    fetch_ohlcv_historical,
)
from tradingagents.pulse.v4_inputs import compute_v4_inputs
from tradingagents.pulse.patterns.candles import detect_all as detect_candles
from tradingagents.pulse.patterns.extrema import find_extrema
from tradingagents.pulse.patterns.structural import detect_structural_all
from tradingagents.pulse.vpd import compute_vpd
from tradingagents.pulse.liquidity_sweep import detect_liquidity_sweep

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("validate_v4")
logger.setLevel(logging.INFO)

TIMEFRAMES = ("1m", "15m", "1h", "4h")


def _fetch_all(ticker: str, start: str, end: str) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    candles: Dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        logger.info("fetching %s candles…", tf)
        try:
            res = fetch_ohlcv_historical(ticker, tf, start, end)
            df = res.df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            candles[tf] = df
            logger.info("  %s: %d bars from %s", tf, len(df), res.source)
        except Exception as exc:
            logger.warning("  %s: FAILED (%s)", tf, exc)
            candles[tf] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    try:
        funding_df, src = fetch_funding_historical(ticker, start, end)
        funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"], utc=True, errors="coerce")
        logger.info("funding: %d entries from %s", len(funding_df), src)
    except Exception as exc:
        logger.warning("funding FAILED (%s)", exc)
        funding_df = pd.DataFrame(columns=["timestamp", "funding_rate"])
    return candles, funding_df


def _funding_at(funding_df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if funding_df is None or funding_df.empty:
        return None
    prior = funding_df[funding_df["timestamp"] <= ts]
    if prior.empty:
        return None
    return float(prior.iloc[-1]["funding_rate"])


def _atr(df: pd.DataFrame, n: int = 14) -> float | None:
    if df is None or df.empty or len(df) < n + 1:
        return None
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    prev_c = c[:-1]
    tr = [max(h[i] - l[i], abs(h[i] - prev_c[i - 1]), abs(l[i] - prev_c[i - 1]))
          for i in range(1, len(df))]
    if len(tr) < n:
        return None
    return float(sum(tr[-n:]) / n)


def _run_once(
    candles: Dict[str, pd.DataFrame],
    funding_df: pd.DataFrame,
    ts: pd.Timestamp,
) -> tuple[list[str], dict]:
    """Slice candles up to ``ts`` and run all detectors. Returns (hits, debug)."""
    sliced = {}
    for tf, df in candles.items():
        if df is None or df.empty:
            sliced[tf] = df
            continue
        sliced[tf] = df[df["timestamp"] <= ts].reset_index(drop=True)

    atr_by_tf = {tf: _atr(sliced.get(tf)) for tf in ("1h", "4h")}
    funding_rate = _funding_at(funding_df, ts)

    v4 = compute_v4_inputs(
        candles_by_tf=sliced,
        atr_by_tf=atr_by_tf,
        funding_rate=funding_rate,
    )

    # Flat hit list with tags:
    hit_tags = []
    for name, tfs in v4.pattern_hits.items():
        for tf in tfs:
            hit_tags.append(f"{name}@{tf}")
    if v4.vpd_signal is not None and v4.vpd_signal != 0:
        hit_tags.append(f"vpd={'+1' if v4.vpd_signal > 0 else '-1'}")
    if v4.liquidity_sweep_dir is not None and v4.liquidity_sweep_dir != 0:
        hit_tags.append(f"sweep={'+1' if v4.liquidity_sweep_dir > 0 else '-1'}")

    # Spot for context
    df_1m = sliced.get("1m")
    spot = float(df_1m.iloc[-1]["close"]) if df_1m is not None and not df_1m.empty else None

    return hit_tags, {
        "spot": spot,
        "atr_1h": atr_by_tf.get("1h"),
        "funding_rate": funding_rate,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="BTC-USD")
    p.add_argument("--days", type=int, default=14,
                   help="window size in days (default 14)")
    p.add_argument("--end", default=None,
                   help="end date YYYY-MM-DD (default: today)")
    p.add_argument("--step-hours", type=float, default=4,
                   help="time step between evaluations (default 4h)")
    args = p.parse_args()

    end_dt = (datetime.now(timezone.utc)
              if args.end is None
              else datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    start_dt = end_dt - timedelta(days=args.days + 3)   # +3d warmup
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    print(f"\n=== Pulse v4 pattern validation ===")
    print(f"ticker: {args.ticker}   window: {start_str} → {end_str}   step: {args.step_hours}h\n")

    candles, funding_df = _fetch_all(args.ticker, start_str, end_str)

    # Quick data sanity.
    for tf in TIMEFRAMES:
        df = candles[tf]
        if df.empty:
            print(f"  WARN: {tf} has 0 rows; skipping evaluation steps")
    if candles["1h"].empty:
        print("ERROR: no 1h candles — cannot validate")
        return 1

    # Walk forward in step-hour increments from the start of the evaluable region
    # (skip the first 3 days to let warmup / structural detectors see enough data).
    eval_start = candles["1h"]["timestamp"].iloc[0].to_pydatetime().replace(tzinfo=timezone.utc)
    eval_start += timedelta(days=3)
    if eval_start < start_dt + timedelta(days=3):
        eval_start = start_dt + timedelta(days=3)

    step = timedelta(hours=args.step_hours)
    ts = pd.Timestamp(eval_start).tz_convert("UTC")
    end_ts = pd.Timestamp(end_dt).tz_convert("UTC")

    counter: Counter = Counter()
    n_steps = 0
    n_any_hit = 0

    print(f"{'timestamp':<26}  {'spot':>10}  {'atr1h':>8}  {'fund':>8}  hits")
    print("-" * 100)
    while ts <= end_ts:
        n_steps += 1
        hits, dbg = _run_once(candles, funding_df, ts)
        for h in hits:
            counter[h] += 1
        if hits:
            n_any_hit += 1
            spot_s = f"{dbg['spot']:10,.1f}" if dbg.get("spot") is not None else " " * 10
            atr_s = f"{dbg['atr_1h']:8.1f}" if dbg.get("atr_1h") is not None else " " * 8
            fund = dbg.get("funding_rate")
            fund_s = f"{fund*1e4:+7.2f}b" if fund is not None else " " * 8
            print(f"{ts.isoformat():<26}  {spot_s}  {atr_s}  {fund_s}  {', '.join(hits)}")
        ts = ts + step

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print(f"  evaluated timestamps: {n_steps}")
    print(f"  timestamps with ≥1 hit: {n_any_hit} ({100 * n_any_hit / max(n_steps, 1):.1f}%)")
    days = max(1, (end_ts - pd.Timestamp(eval_start).tz_convert("UTC")).total_seconds() / 86400)
    print(f"  fires-per-day rate: {n_any_hit / days:.2f}\n")
    if counter:
        print("  counts by pattern:")
        for name, n in sorted(counter.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<35} {n:>5}   ({n / days:.2f}/day)")
    else:
        print("  no patterns detected in the window.")

    # Interpretive guidance.
    print("\nSanity thresholds (qualitative):")
    print("  * structural (H&S, double bottom, channel*) — expect 0.1–2 fires/day on 1h+4h")
    print("  * candles (hammer, *_engulfing)             — expect 1–10 fires/day")
    print("  * sweep                                      — expect 0.2–3 fires/day on BTC")
    print("  * vpd                                        — expect 0.5–5 fires/day")
    print("  Far outside these ranges → likely bug (too strict / too loose).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
