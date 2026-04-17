#!/usr/bin/env python
"""Morning pulse performance report (CLI).

Reads every ``eval_results/pulse/{TICKER}/pulse.jsonl`` and emits a markdown
report with:
    - Total vs. scored pulses per ticker
    - Hit rate + bootstrap 95% CI per horizon per signal
    - 4-fill-model mean net bps + win rates
    - Sharpe (annualized, N_eff-adjusted) at +1h horizon
    - TSMOM direction distribution
    - Regime breakdown
    - Suggested YAML calibration tweaks (thresholds / weights) — dry-run only

Usage:
    python scripts/pulse_morning_report.py
    python scripts/pulse_morning_report.py --lookback-hours 168
    python scripts/pulse_morning_report.py --engine-version v3.0.0
    python scripts/pulse_morning_report.py --output report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add workspace root to sys.path so we can import tradingagents.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tradingagents.pulse.stats import (  # noqa: E402
    effective_sample_size,
    sharpe_confidence_interval,
    sharpe_ratio,
)

PULSE_DIR = ROOT / "eval_results" / "pulse"


# ── Data load ────────────────────────────────────────────────────────

def load_entries(
    ticker_dir: Path,
    lookback_hours: Optional[int],
    engine_version: Optional[str],
) -> List[dict]:
    path = ticker_dir / "pulse.jsonl"
    if not path.exists():
        return []
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        if lookback_hours else None
    )
    entries: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if engine_version and e.get("engine_version") != engine_version:
                continue
            if cutoff is not None:
                ts_str = e.get("ts")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts < cutoff:
                            continue
                    except ValueError:
                        pass
            entries.append(e)
    return entries


# ── Bootstrap CI (re-implemented here to avoid adding to stats.py) ──

def bootstrap_ci(hits: List[int], n_boot: int = 1000, alpha: float = 0.05) -> tuple:
    import random
    if not hits:
        return 0.0, 0.0
    k = len(hits)
    rng = random.Random(0)
    boot_rates = []
    for _ in range(n_boot):
        sample = [hits[rng.randrange(k)] for _ in range(k)]
        boot_rates.append(sum(sample) / k)
    boot_rates.sort()
    lo = boot_rates[int((alpha / 2) * n_boot)]
    hi = boot_rates[int((1 - alpha / 2) * n_boot) - 1]
    return round(lo, 4), round(hi, 4)


# ── Stats aggregation ────────────────────────────────────────────────

@dataclass
class HorizonStats:
    horizon: str
    scored: int
    hits: int
    hit_rate: float
    ci_lo: float
    ci_hi: float
    mean_return: float
    sharpe: float
    sharpe_ci_lo: float
    sharpe_ci_hi: float
    n_eff: float


def per_horizon_stats(entries: List[dict], horizon: str) -> HorizonStats:
    hits = [int(bool(e.get(f"hit_{horizon}"))) for e in entries if e.get("scored") and f"hit_{horizon}" in e]
    rets = [float(e[f"return_{horizon}"]) for e in entries if e.get("scored") and f"return_{horizon}" in e]
    scored = len(hits)
    if scored == 0:
        return HorizonStats(horizon, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    hr = sum(hits) / scored
    lo, hi = bootstrap_ci(hits)
    mean_r = sum(rets) / len(rets) if rets else 0
    # Annualize: horizon in minutes → bars per year
    h_min = {"+5m": 5, "+15m": 15, "+1h": 60}[horizon]
    periods_per_year = 525_600 / h_min
    sr = sharpe_ratio(rets, periods_per_year=periods_per_year) if rets else 0
    n_eff = effective_sample_size(rets, max_lag=5) if len(rets) > 10 else len(rets)
    sr_lo, sr_hi = sharpe_confidence_interval(sr, n_eff=n_eff) if n_eff > 0 else (0, 0)
    return HorizonStats(
        horizon=horizon, scored=scored, hits=sum(hits),
        hit_rate=round(hr, 4),
        ci_lo=lo, ci_hi=hi,
        mean_return=round(mean_r, 6),
        sharpe=round(sr, 3),
        sharpe_ci_lo=round(sr_lo, 3),
        sharpe_ci_hi=round(sr_hi, 3),
        n_eff=round(n_eff, 1),
    )


def fill_model_summary(entries: List[dict], horizon: str) -> Dict[str, dict]:
    by_model: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        if not e.get("scored"):
            continue
        fills = e.get(f"fills_{horizon}") or {}
        if not isinstance(fills, dict):
            continue
        for model, vals in fills.items():
            if not isinstance(vals, dict):
                continue
            nr = vals.get("net_return")
            if isinstance(nr, (int, float)):
                by_model[model].append(float(nr))
    out: Dict[str, dict] = {}
    for model, rets in by_model.items():
        if not rets:
            continue
        out[model] = {
            "count": len(rets),
            "mean_bps": round(sum(rets) / len(rets) * 10_000, 1),
            "win_rate": round(sum(1 for r in rets if r > 0) / len(rets), 4),
            "best": round(max(rets) * 10_000, 1),
            "worst": round(min(rets) * 10_000, 1),
        }
    return out


def tsmom_distribution(entries: List[dict]) -> Dict[str, int]:
    c: Counter = Counter()
    for e in entries:
        d = e.get("tsmom_direction")
        if d is None:
            c["None"] += 1
        elif d == 0:
            c["0 (flat)"] += 1
        elif d > 0:
            c["+1 (up)"] += 1
        else:
            c["-1 (down)"] += 1
    return dict(c)


def regime_distribution(entries: List[dict]) -> Dict[str, int]:
    c: Counter = Counter()
    for e in entries:
        c[e.get("regime_mode") or "unknown"] += 1
    return dict(c)


def signal_distribution(entries: List[dict]) -> Dict[str, int]:
    c: Counter = Counter()
    for e in entries:
        c[e.get("signal") or "unknown"] += 1
    return dict(c)


def override_distribution(entries: List[dict]) -> Dict[str, int]:
    c: Counter = Counter()
    for e in entries:
        r = e.get("override_reason")
        if r:
            c[r] += 1
    return dict(c)


# ── Report rendering ────────────────────────────────────────────────

def render_ticker_report(ticker: str, entries: List[dict]) -> str:
    out = [f"\n## {ticker}\n"]
    out.append(f"- **Total pulses**: {len(entries)}")
    scored = [e for e in entries if e.get("scored")]
    out.append(f"- **Scored**: {len(scored)}")
    if entries:
        first_ts = entries[0].get("ts", "?")
        last_ts = entries[-1].get("ts", "?")
        out.append(f"- **Time range**: {first_ts} → {last_ts}")

    # Signal distribution
    sig_dist = signal_distribution(entries)
    out.append(f"- **Signal mix**: {dict(sig_dist)}")

    # Override distribution
    overrides = override_distribution(entries)
    if overrides:
        out.append(f"- **Overrides fired**: {overrides}")

    # TSMOM distribution
    tm = tsmom_distribution(entries)
    out.append(f"- **TSMOM distribution**: {tm}")

    # Regime distribution
    rg = regime_distribution(entries)
    out.append(f"- **Regime distribution**: {rg}")

    # Per-horizon stats
    if not scored:
        out.append("\n*(no scored pulses yet — scoring runs 1h after pulse)*\n")
        return "\n".join(out)

    out.append("\n### Hit rates (bootstrap 95% CI) + Sharpe (annualized)\n")
    out.append("| Horizon | Scored | Hits | Hit Rate | CI 95% | Mean Ret | Sharpe | SR CI 95% | N_eff |")
    out.append("|---------|--------|------|----------|--------|----------|--------|-----------|-------|")
    for h in ["+5m", "+15m", "+1h"]:
        hs = per_horizon_stats(entries, h)
        out.append(
            f"| {h} | {hs.scored} | {hs.hits} | "
            f"{hs.hit_rate:.2%} | [{hs.ci_lo:.2%}, {hs.ci_hi:.2%}] | "
            f"{hs.mean_return * 10_000:+.1f} bps | {hs.sharpe:.2f} | "
            f"[{hs.sharpe_ci_lo:.2f}, {hs.sharpe_ci_hi:.2f}] | {hs.n_eff:.0f} |"
        )

    # Fill-model summary at +1h
    fs = fill_model_summary(entries, "+1h")
    if fs:
        out.append("\n### Fill-model summary (+1h)\n")
        out.append("| Model | N | Mean (bps) | Win Rate | Best (bps) | Worst (bps) |")
        out.append("|-------|---|------------|----------|------------|-------------|")
        for m in ["best", "realistic", "maker_rejected", "maker_adverse"]:
            if m not in fs:
                continue
            v = fs[m]
            out.append(
                f"| {m} | {v['count']} | {v['mean_bps']:+.1f} | "
                f"{v['win_rate']:.2%} | {v['best']:+.1f} | {v['worst']:+.1f} |"
            )

    return "\n".join(out)


def render_calibration_hints(ticker: str, entries: List[dict]) -> List[str]:
    """Suggest YAML tweaks when per-horizon Sharpe is poor or hits are << 50%."""
    hints = []
    scored = [e for e in entries if e.get("scored")]
    if len(scored) < 30:
        return [f"- **{ticker}**: insufficient sample ({len(scored)} scored) for calibration"]

    hs_1h = per_horizon_stats(entries, "+1h")
    if hs_1h.hit_rate < 0.45 and hs_1h.ci_hi < 0.50:
        hints.append(
            f"- **{ticker}**: +1h hit rate {hs_1h.hit_rate:.2%} "
            f"(CI upper {hs_1h.ci_hi:.2%}); consider raising `signal_threshold` or "
            f"tightening YAML `confluence.weights` on underperforming TFs."
        )
    if hs_1h.sharpe_ci_hi < 0:
        hints.append(
            f"- **{ticker}**: +1h Sharpe CI is entirely negative ({hs_1h.sharpe_ci_lo:.2f}, "
            f"{hs_1h.sharpe_ci_hi:.2f}); **disable ticker from active trading**."
        )

    # If overrides are firing often, flag for review
    overrides = override_distribution(entries)
    if overrides:
        total = sum(overrides.values())
        if total / len(entries) > 0.3:
            hints.append(
                f"- **{ticker}**: {total}/{len(entries)} pulses "
                f"({total/len(entries):.0%}) hit overrides; "
                f"consider widening thresholds ({overrides})."
            )

    return hints


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pulse morning performance report")
    ap.add_argument("--lookback-hours", type=int, default=None,
                    help="Only include pulses within this many hours (default: all)")
    ap.add_argument("--engine-version", type=str, default=None,
                    help="Filter by engine_version (e.g. v3.0.0)")
    ap.add_argument("--tickers", type=str, nargs="*", default=None,
                    help="Ticker(s) to include (default: all)")
    ap.add_argument("--output", type=str, default=None,
                    help="Write report to this markdown file (default: stdout)")
    args = ap.parse_args()

    if not PULSE_DIR.exists():
        print(f"[error] {PULSE_DIR} not found", file=sys.stderr)
        sys.exit(1)

    ticker_dirs = [d for d in PULSE_DIR.iterdir() if d.is_dir()]
    if args.tickers:
        ticker_dirs = [d for d in ticker_dirs if d.name in args.tickers]
    ticker_dirs.sort(key=lambda d: d.name)

    # Header
    report_lines: List[str] = []
    report_lines.append("# Pulse Morning Report")
    report_lines.append("")
    report_lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    if args.lookback_hours:
        report_lines.append(f"- Lookback: {args.lookback_hours}h")
    if args.engine_version:
        report_lines.append(f"- Engine version filter: {args.engine_version}")
    report_lines.append(f"- Tickers scanned: {len(ticker_dirs)}")

    all_hints: List[str] = []
    for td in ticker_dirs:
        entries = load_entries(td, args.lookback_hours, args.engine_version)
        if not entries:
            continue
        report_lines.append(render_ticker_report(td.name, entries))
        all_hints.extend(render_calibration_hints(td.name, entries))

    if all_hints:
        report_lines.append("\n## Calibration hints (dry-run suggestions)\n")
        report_lines.extend(all_hints)

    report = "\n".join(report_lines) + "\n"

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
