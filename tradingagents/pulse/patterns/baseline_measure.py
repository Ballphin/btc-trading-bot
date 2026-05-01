"""Pattern baseline measurement — walk historical candles, detect structural
patterns, measure forward returns, and produce a quality-stratified report.

Usage::

    python -m tradingagents.pulse.patterns.baseline_measure \\
        --ticker BTC-USD --start 2023-06-01 --end 2024-06-01

Output: a summary table and per-hit CSV saved to
``eval_results/pulse/pattern_baseline/``.

This is the empirical foundation for calibrating the regime-pattern
interaction matrix and the structural weight in ``score_pulse_confluence``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tradingagents.dataflows.historical_router import fetch_ohlcv_historical
from tradingagents.pulse.patterns.structural import detect_structural_all
from tradingagents.pulse.regime import detect_regime

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────

_DEFAULT_TICKER = "BTC-USD"
_DEFAULT_INTERVAL = "1h"
_FORWARD_HORIZONS = (1, 4, 12, 24)  # bars ahead to measure returns
_MIN_WINDOW_BARS = 50               # minimum candles before first detection
_SLIDE_STEP = 1                     # slide 1 bar at a time
_ATR_PERIOD = 14                    # for ATR computation


# ── Helpers ──────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = _ATR_PERIOD) -> pd.Series:
    """Compute ATR from OHLCV DataFrame."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


@dataclass
class PatternEvent:
    """A single pattern detection event with forward return metadata."""
    timestamp: str
    pattern: str
    direction: int
    quality: float
    invalidation_price: Optional[float]
    entry_price: float
    regime: str
    # Forward returns at each horizon (NaN if not enough data)
    fwd_returns: Dict[int, float] = field(default_factory=dict)
    # Whether invalidation was hit before each horizon
    invalidated_before: Dict[int, bool] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _measure_forward(
    df: pd.DataFrame,
    entry_idx: int,
    direction: int,
    invalidation_price: Optional[float],
    horizons: tuple = _FORWARD_HORIZONS,
) -> tuple[Dict[int, float], Dict[int, bool]]:
    """Compute forward returns and invalidation checks."""
    n = len(df)
    entry_price = float(df.iloc[entry_idx]["close"])
    fwd = {}
    inval = {}
    for h in horizons:
        target_idx = entry_idx + h
        if target_idx >= n:
            fwd[h] = float("nan")
            inval[h] = False
            continue
        exit_price = float(df.iloc[target_idx]["close"])
        if direction == 1:
            fwd[h] = (exit_price - entry_price) / entry_price
        elif direction == -1:
            fwd[h] = (entry_price - exit_price) / entry_price
        else:
            fwd[h] = 0.0
        # Check invalidation
        if invalidation_price is not None and direction != 0:
            window = df.iloc[entry_idx:target_idx + 1]
            if direction == 1:
                inval[h] = bool((window["low"].astype(float) < invalidation_price).any())
            else:
                inval[h] = bool((window["high"].astype(float) > invalidation_price).any())
        else:
            inval[h] = False
    return fwd, inval


# ── Synthetic Data ──────────────────────────────────────────────────

def _generate_synthetic_data(start: str, end: str, interval: str) -> pd.DataFrame:
    """Generate a synthetic 1-year OHLCV dataset for offline baseline testing."""
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    timestamps = pd.date_range(start_ts, end_ts, freq="1h")
    n = len(timestamps)
    
    np.random.seed(42)
    # Start at 60k, generate random walk
    returns = np.random.normal(0.0001, 0.005, n)
    # Add some sinusoidal regimes (trends, chops)
    regime_wave = np.sin(np.linspace(0, 10 * np.pi, n)) * 0.002
    returns += regime_wave
    
    closes = 60000 * np.exp(np.cumsum(returns))
    highs = closes * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.005, n)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    
    # Fix instances where high/low don't envelop open/close
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    
    volumes = np.random.uniform(100, 1000, n)
    # Boost volume at bottoms/tops
    volumes *= 1 + (np.abs(regime_wave) * 500)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


# ── Main measurement loop ───────────────────────────────────────────

def run_baseline(
    ticker: str = _DEFAULT_TICKER,
    interval: str = _DEFAULT_INTERVAL,
    start: str = "2023-06-01",
    end: str = "2024-06-01",
    bandwidth: int | str = 8,
    output_dir: Optional[str] = None,
    synthetic: bool = False,
) -> pd.DataFrame:
    """Walk historical candles, detect patterns, and measure forward returns.

    Returns a DataFrame with one row per pattern event, including forward
    returns at each horizon and quality scores.
    """
    logger.info(f"[Baseline] Fetching {ticker} {interval} from {start} to {end} (synthetic={synthetic})")
    
    if synthetic:
        df = _generate_synthetic_data(start, end, interval)
        source_label = "synthetic"
    else:
        result = fetch_ohlcv_historical(ticker, interval, start, end)
        df = result.df.copy()
        source_label = result.source

    if df.empty:
        logger.warning("[Baseline] No data returned")
        return pd.DataFrame()

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    logger.info(f"[Baseline] {n} candles fetched (source: {source_label})")

    # Precompute ATR for the full series
    atr_series = _compute_atr(df)

    events: List[PatternEvent] = []
    seen_keys: set = set()  # (pattern_name, fired_at_idx) dedup

    # Slide through the series
    for end_idx in range(_MIN_WINDOW_BARS, n, _SLIDE_STEP):
        start_idx = max(0, end_idx - 500)
        window = df.iloc[start_idx:end_idx + 1]
        atr_val = float(atr_series.iloc[end_idx])

        hits = detect_structural_all(
            window,
            bandwidth=bandwidth,
            atr=atr_val if atr_val > 0 else None,
        )
        for hit in hits:
            # hit indices are relative to window; make them absolute
            abs_fired_idx = hit.fired_at_idx + start_idx
            abs_conf_idx = hit.confirmation_idx + start_idx
            
            key = (hit.name, abs_fired_idx)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Only process if the pattern was detected recently
            # (within the last 5 bars of the current window end)
            if abs_conf_idx > end_idx:
                continue
            if end_idx - abs_conf_idx > 5:
                continue

            # Detect regime at this point
            regime_mode = "mixed"
            if end_idx >= 30:
                try:
                    regime = df.iloc[:end_idx + 1]  # regime detection needs more history, keep using full history
                    regime_mode = detect_regime(regime).mode
                except Exception:
                    pass

            entry_idx = abs_conf_idx
            entry_price = float(df.iloc[entry_idx]["close"])

            fwd, inval = _measure_forward(
                df, entry_idx, hit.direction,
                hit.invalidation_price,
            )

            ts = df.iloc[entry_idx]["timestamp"]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

            events.append(PatternEvent(
                timestamp=ts_str,
                pattern=hit.name,
                direction=hit.direction,
                quality=hit.quality,
                invalidation_price=hit.invalidation_price,
                entry_price=entry_price,
                regime=regime_mode,
                fwd_returns=fwd,
                invalidated_before=inval,
                metadata=dict(hit.metadata) if hit.metadata else {},
            ))

    logger.info(f"[Baseline] Detected {len(events)} pattern events")

    if not events:
        return pd.DataFrame()

    # Build output DataFrame
    rows = []
    for ev in events:
        row = {
            "timestamp": ev.timestamp,
            "pattern": ev.pattern,
            "direction": ev.direction,
            "quality": ev.quality,
            "invalidation_price": ev.invalidation_price,
            "entry_price": ev.entry_price,
            "regime": ev.regime,
        }
        for h in _FORWARD_HORIZONS:
            row[f"fwd_{h}bar"] = ev.fwd_returns.get(h, float("nan"))
            row[f"inval_{h}bar"] = ev.invalidated_before.get(h, False)
        rows.append(row)

    events_df = pd.DataFrame(rows)

    # Save and print summary
    if output_dir is None:
        output_dir = os.path.join("eval_results", "pulse", "pattern_baseline")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{ticker}_{interval}_{start}_{end}.csv")
    events_df.to_csv(csv_path, index=False)
    logger.info(f"[Baseline] Events saved to {csv_path}")

    # Summary report
    summary = _build_summary(events_df)
    summary_path = os.path.join(output_dir, f"{ticker}_{interval}_{start}_{end}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"[Baseline] Summary saved to {summary_path}")

    _print_summary(summary, ticker, interval, start, end)
    return events_df


def _build_summary(df: pd.DataFrame) -> dict:
    """Build a structured summary from the events DataFrame."""
    summary: dict = {"total_events": len(df), "by_pattern": {}, "by_regime": {}}

    for pattern in sorted(df["pattern"].unique()):
        pf = df[df["pattern"] == pattern]
        pat_summary: dict = {"count": len(pf), "avg_quality": round(float(pf["quality"].mean()), 4)}

        for h in _FORWARD_HORIZONS:
            col = f"fwd_{h}bar"
            inval_col = f"inval_{h}bar"
            valid = pf[col].dropna()
            if len(valid) == 0:
                continue
            hit_rate = float((valid > 0).mean())
            avg_ret = float(valid.mean())
            med_ret = float(valid.median())
            inval_rate = float(pf[inval_col].mean()) if inval_col in pf.columns else 0.0
            pat_summary[f"h{h}_hit_rate"] = round(hit_rate, 4)
            pat_summary[f"h{h}_avg_ret"] = round(avg_ret, 6)
            pat_summary[f"h{h}_med_ret"] = round(med_ret, 6)
            pat_summary[f"h{h}_inval_rate"] = round(inval_rate, 4)

        # Quality-stratified breakdown
        q_high = pf[pf["quality"] >= 0.5]
        q_low = pf[pf["quality"] < 0.5]
        for label, subset in [("q_high", q_high), ("q_low", q_low)]:
            if len(subset) == 0:
                continue
            for h in _FORWARD_HORIZONS:
                col = f"fwd_{h}bar"
                valid = subset[col].dropna()
                if len(valid) == 0:
                    continue
                pat_summary[f"{label}_h{h}_hit_rate"] = round(float((valid > 0).mean()), 4)
                pat_summary[f"{label}_h{h}_avg_ret"] = round(float(valid.mean()), 6)

        summary["by_pattern"][pattern] = pat_summary

    # Regime breakdown
    for regime in sorted(df["regime"].unique()):
        rf = df[df["regime"] == regime]
        reg_summary: dict = {"count": len(rf)}
        for h in _FORWARD_HORIZONS:
            col = f"fwd_{h}bar"
            valid = rf[col].dropna()
            if len(valid) == 0:
                continue
            reg_summary[f"h{h}_hit_rate"] = round(float((valid > 0).mean()), 4)
            reg_summary[f"h{h}_avg_ret"] = round(float(valid.mean()), 6)
        summary["by_regime"][regime] = reg_summary

    return summary


def _print_summary(summary: dict, ticker: str, interval: str, start: str, end: str):
    """Print a formatted summary table to stdout."""
    print(f"\n{'='*72}")
    print(f"  Pattern Baseline: {ticker} {interval}  |  {start} → {end}")
    print(f"  Total events: {summary['total_events']}")
    print(f"{'='*72}")

    print(f"\n{'Pattern':<25} {'N':>4} {'Q̄':>5}", end="")
    for h in _FORWARD_HORIZONS:
        print(f" | {'HR':>5} {'AvgR':>8} @{h}b", end="")
    print()
    print("-" * 72)

    for pat, info in sorted(summary["by_pattern"].items()):
        print(f"{pat:<25} {info['count']:>4} {info['avg_quality']:>5.2f}", end="")
        for h in _FORWARD_HORIZONS:
            hr = info.get(f"h{h}_hit_rate", float("nan"))
            ar = info.get(f"h{h}_avg_ret", float("nan"))
            if np.isnan(hr):
                print(f" | {'--':>5} {'--':>8}    ", end="")
            else:
                print(f" | {hr:>5.1%} {ar:>+8.4%}    ", end="")
        print()

    if summary.get("by_regime"):
        print(f"\n{'Regime':<25} {'N':>4}", end="")
        for h in _FORWARD_HORIZONS:
            print(f" | {'HR':>5} {'AvgR':>8} @{h}b", end="")
        print()
        print("-" * 72)
        for reg, info in sorted(summary["by_regime"].items()):
            print(f"{reg:<25} {info['count']:>4}", end="")
            for h in _FORWARD_HORIZONS:
                hr = info.get(f"h{h}_hit_rate", float("nan"))
                ar = info.get(f"h{h}_avg_ret", float("nan"))
                if np.isnan(hr):
                    print(f" | {'--':>5} {'--':>8}    ", end="")
                else:
                    print(f" | {hr:>5.1%} {ar:>+8.4%}    ", end="")
            print()

    print()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure structural pattern baseline hit rates on historical data"
    )
    parser.add_argument("--ticker", default=_DEFAULT_TICKER)
    parser.add_argument("--interval", default=_DEFAULT_INTERVAL)
    parser.add_argument("--start", default="2023-06-01")
    parser.add_argument("--end", default="2024-06-01")
    parser.add_argument("--bandwidth", default="8",
                        help="Kernel bandwidth (int or 'auto' for LOOCV)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic offline data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bw = args.bandwidth
    if bw != "auto":
        bw = int(bw)

    run_baseline(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        end=args.end,
        bandwidth=bw,
        output_dir=args.output_dir,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
