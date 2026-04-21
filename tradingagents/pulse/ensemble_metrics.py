"""Pulse ensemble metrics aggregator — R.4.

Reads per-config ``outcomes.jsonl`` and emits a ``metrics.json`` with
three blocks (plan §4):

* ``overall``          — every outcome (diagnostic only, NOT used for
                         champion selection).
* ``weekend``          — outcomes whose ``is_weekend`` flag is True.
                         A config is champion-ineligible until
                         ``weekend_n_signals >= 10`` (HIGH #9).
* ``oos_validation``   — the LAST K=20 outcomes, the exclusive input
                         for champion selection (BLOCKER #5).

Every block carries ``deflated_sharpe`` computed with
``n_strategies=5`` so FWER across the ensemble is corrected (BLOCKER
#4 — LdP 2014 haircut).

The aggregator is a pure function module: input = outcomes list,
output = metrics dict. The scheduler calls ``write_metrics(ticker,
config)`` which reads outcomes from disk, runs the aggregation, and
writes the result. No champion-selection happens here — that's R.7.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

PULSE_DIR = Path("results/pulse")

# Configuration constants — mirrored in the plan and also exported so
# tests / the champion-selector share a single source of truth.
OOS_VALIDATION_WINDOW = 20
N_STRATEGIES_FOR_DSR = 5
WEEKEND_ELIGIBILITY_MIN = 10


def _safe_sharpe(returns: Sequence[float]) -> float:
    """Annualised Sharpe on signal-level returns. No risk-free rate
    (we're comparing configs, not absolute performance)."""
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    # Pulse signals are roughly event-based (one per tick), so we leave
    # the Sharpe in raw per-signal units. The deflated-Sharpe machinery
    # is what matters for promotion gates; absolute scaling is fixed.
    return mu / sd


def _sharpe_block(returns: List[float]) -> Dict[str, Any]:
    """Compute the sub-block fields shared by overall/weekend/oos."""
    n = len(returns)
    if n == 0:
        return {
            "n_signals": 0,
            "mean_net_return": 0.0,
            "sharpe_point": 0.0,
            "deflated_sharpe": 0.0,
        }
    # Lazy import to avoid scipy cost on module load.
    from tradingagents.backtesting.walk_forward import compute_deflated_sharpe
    sharpe = _safe_sharpe(returns)
    dsr = compute_deflated_sharpe(
        sharpe=sharpe,
        n_periods=n,
        n_strategies=N_STRATEGIES_FOR_DSR,
    )
    return {
        "n_signals": n,
        "mean_net_return": round(float(np.mean(returns)), 6),
        "sharpe_point": round(sharpe, 4),
        "deflated_sharpe": round(dsr, 4),
    }


def _per_regime_split(outcomes: List[dict]) -> Dict[str, Dict[str, Any]]:
    """Per-directional-regime split with Stage 2 Commit N ``thin_sample``
    flag. Directional regime is the promoted taxonomy (bull/bear/
    range_bound/ambiguous); falls back to ``regime_at_entry`` when the
    directional field is absent."""
    buckets: Dict[str, List[float]] = {}
    for o in outcomes:
        key = o.get("directional_regime") or o.get("regime_at_entry") or "unknown"
        buckets.setdefault(key, []).append(float(o.get("net_return_pct") or 0.0))
    return {
        k: {
            "n": len(v),
            "mean_net_return": round(float(np.mean(v)), 6),
            "thin_sample": len(v) < 30,
        }
        for k, v in buckets.items()
    }


def aggregate(outcomes: List[dict]) -> Dict[str, Any]:
    """Pure aggregation — outcomes in, metrics-dict out.

    Never raises on empty input; returns zero-sample blocks so the
    downstream JSON is always well-formed.
    """
    # Sort by exit_ts so "last K outcomes" is deterministic. A missing
    # exit_ts sorts to the end (treat as newest).
    ordered = sorted(outcomes, key=lambda o: o.get("exit_ts") or "")

    all_rets = [float(o.get("net_return_pct") or 0.0) for o in ordered]
    weekend = [o for o in ordered if o.get("is_weekend")]
    weekend_rets = [float(o.get("net_return_pct") or 0.0) for o in weekend]

    # OOS window = the most recent K outcomes. If we have ≤K outcomes
    # total, the OOS block is empty (NOT "all of them") — otherwise
    # every config would pass the OOS gate from its very first
    # outcomes, defeating the purpose of the holdout.
    if len(ordered) > OOS_VALIDATION_WINDOW:
        oos = ordered[-OOS_VALIDATION_WINDOW:]
    else:
        oos = []
    oos_rets = [float(o.get("net_return_pct") or 0.0) for o in oos]

    return {
        "overall": {
            **_sharpe_block(all_rets),
            "per_directional_regime": _per_regime_split(ordered),
            "exit_type_breakdown": _exit_type_breakdown(ordered),
        },
        "weekend": {
            **_sharpe_block(weekend_rets),
            "weekend_eligibility_min": WEEKEND_ELIGIBILITY_MIN,
            "champion_eligible": len(weekend_rets) >= WEEKEND_ELIGIBILITY_MIN,
        },
        "oos_validation": {
            **_sharpe_block(oos_rets),
            "window_size": OOS_VALIDATION_WINDOW,
            "exit_type_breakdown": _exit_type_breakdown(oos),
        },
    }


def _exit_type_breakdown(outcomes: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for o in outcomes:
        k = o.get("exit_type") or "unknown"
        out[k] = out.get(k, 0) + 1
    return out


# ── Disk I/O wrappers ────────────────────────────────────────────────

def _load_outcomes(
    ticker: str,
    config: str,
    *,
    pulse_dir: Path = PULSE_DIR,
) -> List[dict]:
    path = pulse_dir / ticker / "configs" / config / "outcomes.jsonl"
    if not path.exists():
        return []
    out: List[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_metrics(
    ticker: str,
    config: str,
    *,
    pulse_dir: Path = PULSE_DIR,
) -> Dict[str, Any]:
    """Aggregate outcomes for ``ticker/config`` and write metrics.json.

    Returns the metrics dict so callers (tests, API endpoints) can
    avoid a second read.
    """
    outcomes = _load_outcomes(ticker, config, pulse_dir=pulse_dir)
    metrics = aggregate(outcomes)
    out_dir = pulse_dir / ticker / "configs" / config
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, indent=2, default=str))
    tmp.replace(out_dir / "metrics.json")
    return metrics


def list_configs(ticker: str, *, pulse_dir: Path = PULSE_DIR) -> List[str]:
    """Enumerate all config subdirectories present for ``ticker``."""
    d = pulse_dir / ticker / "configs"
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def refresh_all(
    ticker: str, *, pulse_dir: Path = PULSE_DIR,
) -> Dict[str, Dict[str, Any]]:
    """Recompute metrics for every config under ``ticker``."""
    out: Dict[str, Dict[str, Any]] = {}
    for cfg in list_configs(ticker, pulse_dir=pulse_dir):
        try:
            out[cfg] = write_metrics(ticker, cfg, pulse_dir=pulse_dir)
        except Exception as e:
            logger.warning(f"[Metrics] {ticker}/{cfg} aggregation failed: {e}")
    return out
