"""Pulse drift monitor — Stage 2 Commit H.

Weekly check that realised 30-day Sharpe for each (ticker, regime) hasn't
degraded below the most recent auto-tune artifact's pessimistic bound.

Spec (debate-revised):
    * Cadence: weekly (cron: sunday 23:00 UTC).
    * Window: 30 trailing days of scored decisions.
    * Threshold: realised point Sharpe must sit within 2 bootstrap-SE of
      the latest artifact's oos_sharpe_ci_lower for that (ticker, regime).
    * Flash-crash suppression: if any 1h candle in the window has
      vol_z_clipped > 3 (existing LUNA backstop from pulse.regime),
      emit "DRIFT SUPPRESSED: OUTLIER" and exit without alerting.
    * Expected alert rate: <=2/month across all regimes — no Bonferroni
      (WCT operator-attention argument: a 40%/month false-positive rate
      from Bonferroni correction defeats the alarm's purpose).

Outputs:
    results/drift/{ticker}_{regime}_{iso_week}.json
        status  : in_ci | below_ci | marginal | insufficient_n | suppressed
        realised_sharpe, ci_lower, z_gap, n_decisions, ran_at

On ``below_ci`` the monitor optionally POSTs to the auto-retune endpoint
(Commit I) — controlled via --trigger flag, off by default for the first
observation week.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from tradingagents.pulse.stats import bootstrap_sharpe_ci, sharpe_ratio

logger = logging.getLogger(__name__)

MIN_DECISIONS = 20          # below this we emit "insufficient_n"
WINDOW_DAYS = 30
FLASH_CRASH_Z = 3.0         # matches regime.py vol_z_clipped backstop
Z_GAP_THRESHOLD = 2.0       # below = below_ci; 0..2 = marginal
DRIFT_DIR = Path("results/drift")
SHADOW_DIR = Path("eval_results/shadow")
AUTOTUNE_DIR = Path("results/autotune")


@dataclass
class DriftResult:
    ticker: str
    regime: str
    status: str
    realised_sharpe: float
    ci_lower: float
    z_gap: float
    bootstrap_se: float
    n_decisions: int
    iso_week: str
    ran_at: str
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Decision & artifact loaders ──────────────────────────────────────

def _load_scored_returns(
    ticker: str,
    regime: str,
    *,
    now: Optional[datetime] = None,
    shadow_dir: Path = SHADOW_DIR,
) -> List[float]:
    """Return per-decision net returns for scored decisions in the last
    ``WINDOW_DAYS`` matching ``regime`` (matches on ``active_regime`` key
    or regime==``any`` which ignores the tag).
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=WINDOW_DAYS)
    path = shadow_dir / ticker / "decisions_scored.jsonl"
    if not path.exists():
        return []
    out: List[float] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Filter by regime tag if present (missing tag → include under "any")
            d_regime = d.get("active_regime") or d.get("regime") or "any"
            if regime != "any" and d_regime != regime:
                continue
            # Filter by scored timestamp
            ts = d.get("scored_at") or d.get("entry_date") or d.get("date")
            try:
                ts_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts_dt < cutoff:
                continue
            r = d.get("net_return_primary")
            if r is None:
                r = d.get("net_return")
            if r is None:
                continue
            try:
                out.append(float(r))
            except (TypeError, ValueError):
                continue
    return out


def _latest_artifact_ci_lower(
    ticker: str,
    regime: str,
    *,
    autotune_dir: Path = AUTOTUNE_DIR,
) -> Optional[float]:
    """Return the oos_sharpe_ci_lower from the newest artifact for the
    (ticker, regime) pair, or None if no artifact exists.
    """
    if not autotune_dir.exists():
        return None
    best_mtime, best_ci = -1.0, None
    for p in autotune_dir.glob("*.json"):
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        spec = payload.get("spec") or {}
        if spec.get("ticker") != ticker:
            continue
        if spec.get("active_regime") != regime:
            continue
        mtime = p.stat().st_mtime
        if mtime <= best_mtime:
            continue
        metrics = payload.get("metrics") or {}
        ci = metrics.get("oos_sharpe_ci_lower")
        if ci is None:
            continue
        best_mtime, best_ci = mtime, float(ci)
    return best_ci


def _flash_crash_in_window(vol_z_series: Iterable[float]) -> bool:
    """True if any 1h vol_z_clipped sample in the window exceeds the
    LUNA-style backstop threshold."""
    return any(abs(float(z)) > FLASH_CRASH_Z for z in vol_z_series)


# ── Core check ──────────────────────────────────────────────────────

def run_drift_check(
    ticker: str,
    regime: str,
    *,
    now: Optional[datetime] = None,
    vol_z_series: Optional[Iterable[float]] = None,
    shadow_dir: Path = SHADOW_DIR,
    autotune_dir: Path = AUTOTUNE_DIR,
) -> DriftResult:
    now = now or datetime.now(timezone.utc)
    iso = now.isocalendar()
    iso_week = f"{iso[0]}W{iso[1]:02d}"
    ran_at = now.isoformat()

    # Flash-crash suppression — honour the existing LUNA backstop.
    if vol_z_series is not None and _flash_crash_in_window(vol_z_series):
        return DriftResult(
            ticker, regime, "suppressed",
            realised_sharpe=float("nan"), ci_lower=float("nan"),
            z_gap=float("nan"), bootstrap_se=float("nan"),
            n_decisions=0, iso_week=iso_week, ran_at=ran_at,
            reason="flash-crash (vol_z_clipped > 3) in 30d window",
        )

    returns = _load_scored_returns(ticker, regime, now=now, shadow_dir=shadow_dir)
    if len(returns) < MIN_DECISIONS:
        return DriftResult(
            ticker, regime, "insufficient_n",
            realised_sharpe=float("nan"), ci_lower=float("nan"),
            z_gap=float("nan"), bootstrap_se=float("nan"),
            n_decisions=len(returns), iso_week=iso_week, ran_at=ran_at,
            reason=f"{len(returns)} scored decisions < {MIN_DECISIONS} min",
        )

    ci_lower_artifact = _latest_artifact_ci_lower(
        ticker, regime, autotune_dir=autotune_dir,
    )
    if ci_lower_artifact is None:
        return DriftResult(
            ticker, regime, "insufficient_n",
            realised_sharpe=float("nan"), ci_lower=float("nan"),
            z_gap=float("nan"), bootstrap_se=float("nan"),
            n_decisions=len(returns), iso_week=iso_week, ran_at=ran_at,
            reason="no prior auto-tune artifact for this (ticker, regime)",
        )

    # Realised Sharpe + bootstrap SE for the z-gap.
    lo, point, hi = bootstrap_sharpe_ci(
        returns, n_bootstrap=1000, ci_low=15.87, ci_high=84.13,  # ±1σ
    )
    # ±1σ percentile range ≈ 2·SE → SE ≈ (hi-lo)/2.
    se = max((hi - lo) / 2.0, 1e-9)
    z_gap = (point - ci_lower_artifact) / se

    if z_gap < -Z_GAP_THRESHOLD:
        status = "below_ci"
        reason = (
            f"realised Sharpe {point:.2f} is {-z_gap:.1f}σ below artifact "
            f"CI-lower {ci_lower_artifact:.2f}"
        )
    elif z_gap < 0:
        status = "marginal"
        reason = f"realised Sharpe {point:.2f} within {-z_gap:.1f}σ of CI-lower"
    else:
        status = "in_ci"
        reason = f"realised Sharpe {point:.2f} ≥ CI-lower {ci_lower_artifact:.2f}"

    return DriftResult(
        ticker, regime, status,
        realised_sharpe=round(point, 4),
        ci_lower=round(ci_lower_artifact, 4),
        z_gap=round(z_gap, 3),
        bootstrap_se=round(se, 4),
        n_decisions=len(returns),
        iso_week=iso_week, ran_at=ran_at, reason=reason,
    )


# ── Persistence & CLI ───────────────────────────────────────────────

def write_drift_result(result: DriftResult, *, drift_dir: Path = DRIFT_DIR) -> Path:
    drift_dir.mkdir(parents=True, exist_ok=True)
    path = drift_dir / f"{result.ticker}_{result.regime}_{result.iso_week}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


def _trigger_autotune(ticker: str, regime: str, base_url: str) -> Tuple[int, str]:
    """POST to /api/pulse/autotune/{ticker} — Commit I's entry point.

    Returns (status_code, body). Returns (0, 'no-trigger') if imports fail.
    """
    try:
        import requests  # local import — only needed when --trigger is set
    except ImportError:
        return 0, "requests not installed"
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=60)).isoformat()
    r = requests.post(
        f"{base_url}/api/pulse/autotune/{ticker}",
        json={
            "start_date": start, "end_date": end,
            "active_regime": regime, "n_folds": 3, "n_configs": 30,
        },
        timeout=10,
    )
    return r.status_code, r.text[:200]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pulse drift monitor (weekly)")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--regime", required=True,
                        choices=["base", "bull", "bear", "range_bound", "ambiguous"])
    parser.add_argument("--trigger", action="store_true",
                        help="POST to auto-tune endpoint on below_ci (Commit I)")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_drift_check(args.ticker.upper(), args.regime)
    path = write_drift_result(result)
    logger.info("wrote %s — status=%s reason=%s", path, result.status, result.reason)

    if args.trigger and result.status == "below_ci":
        code, body = _trigger_autotune(args.ticker.upper(), args.regime, args.base_url)
        logger.info("auto-retune trigger → %s %s", code, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
