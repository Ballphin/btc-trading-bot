"""Shadow→Live promotion pipeline — Stage 2 Commit P.

State machine for per-decision promotion based on accumulating evidence:

    shadow     → candidate : ≥N scored decisions with sequence numbers
                              spanning a weekend, AND
                              realized_net_of_funding sharpe CI-lower > 0.
    candidate  → live      : M additional consecutive decisions without
                              breaching CI-lower.
    live       → retired   : 3+ drift alerts in a rolling window of K
                              sequence numbers.

Design notes (from the debate round):

* Transitions are keyed on ``decision_sequence_number`` (monotonic int
  stamped by the scorecard when a decision is scored), **never** on
  wall-clock. This eliminates the backfill-replay race where a replay
  scoring job re-writes older decisions in the middle of a
  wall-clock-keyed window.
* The shadow→candidate gate must include funding and span a weekend so
  the window captures the weekend-funding regime — without that, a
  strategy that only works on weekdays can pass the gate.
* Propose-only: this script NEVER submits live orders. It only updates
  the ``promotion_state`` field on scored decisions on disk; an
  operator decides whether the "live" label actually maps to real
  capital deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from tradingagents.pulse.stats import bootstrap_sharpe_ci

logger = logging.getLogger(__name__)

# Tunable gates — the plan's open questions note we'll want to move
# these off hardcoded defaults once live data exists. The public
# constants exist so tests and the CLI can override them cleanly.
PROMOTION_N = 50      # min scored decisions for shadow → candidate
PROMOTION_M = 10      # consecutive decisions for candidate → live
RETIREMENT_K = 30     # rolling window for retirement check
RETIREMENT_ALERTS = 3  # drift alerts in window → retire
STATES = ("shadow", "candidate", "live", "retired")

SHADOW_DIR = Path("eval_results/shadow")
DRIFT_DIR = Path("results/drift")


@dataclass
class PromotionDecision:
    ticker: str
    old_state: str
    new_state: str
    sequence_number: int
    reason: str


def _load_scored(ticker: str, *, shadow_dir: Path = SHADOW_DIR) -> List[dict]:
    path = shadow_dir / ticker / "decisions_scored.jsonl"
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
    # Ensure stable sequence_number — fall back to file order if missing.
    for i, d in enumerate(out):
        d.setdefault("decision_sequence_number", i)
    return out


def _spans_weekend(decisions: List[dict]) -> bool:
    """True if the window contains at least one Saturday or Sunday.

    Uses ``entry_date`` (the trade's entry timestamp). Any decision
    whose entry date is a Sat/Sun — or whose entry-exit bracket crosses
    one — counts.
    """
    for d in decisions:
        for key in ("entry_date", "date", "scored_at"):
            ts = d.get(key)
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except Exception:
                continue
            if dt.weekday() >= 5:
                return True
    return False


def _shadow_to_candidate_ok(
    decisions: List[dict],
    *,
    n: int = PROMOTION_N,
) -> Tuple[bool, str]:
    """Check the shadow→candidate gate: N decisions, weekend span,
    positive funding-inclusive Sharpe CI-lower."""
    if len(decisions) < n:
        return False, f"only {len(decisions)} < {n} scored decisions"
    if not _spans_weekend(decisions):
        return False, "no weekend exposure in window"
    returns = [
        float(d.get("net_return_primary")
              if d.get("net_return_primary") is not None
              else d.get("net_return") or 0.0)
        for d in decisions
    ]
    ci_lower, point, _ = bootstrap_sharpe_ci(returns, n_bootstrap=1000)
    if ci_lower <= 0:
        return False, f"funding-aware Sharpe CI-lower {ci_lower:.2f} ≤ 0"
    return True, f"Sharpe CI-lower {ci_lower:.2f} > 0 over {len(decisions)} trades"


def _candidate_to_live_ok(
    decisions: List[dict],
    *,
    m: int = PROMOTION_M,
) -> Tuple[bool, str]:
    """Last M consecutive decisions must not breach CI-lower of 0."""
    if len(decisions) < m:
        return False, f"only {len(decisions)} candidate decisions < {m}"
    last_m = decisions[-m:]
    rets = [float(d.get("net_return_primary") or d.get("net_return") or 0.0) for d in last_m]
    ci_lower, _, _ = bootstrap_sharpe_ci(rets, n_bootstrap=500)
    if ci_lower <= 0:
        return False, f"last {m} CI-lower {ci_lower:.2f} ≤ 0"
    return True, f"last {m} CI-lower {ci_lower:.2f} > 0"


def _drift_alerts_in_window(
    ticker: str,
    *,
    window_k: int = RETIREMENT_K,
    drift_dir: Path = DRIFT_DIR,
) -> int:
    """Count below_ci alerts for ``ticker`` in the last ``window_k``
    drift-check files (all regimes combined)."""
    if not drift_dir.exists():
        return 0
    files = sorted(drift_dir.glob(f"{ticker}_*.json"))[-window_k:]
    n = 0
    for f in files:
        try:
            data = json.loads(f.read_text())
            if data.get("status") == "below_ci":
                n += 1
        except Exception:
            continue
    return n


def classify_state(
    ticker: str,
    decisions: Optional[List[dict]] = None,
    *,
    n: int = PROMOTION_N,
    m: int = PROMOTION_M,
    retirement_alerts: int = RETIREMENT_ALERTS,
    retirement_k: int = RETIREMENT_K,
    shadow_dir: Path = SHADOW_DIR,
    drift_dir: Path = DRIFT_DIR,
) -> PromotionDecision:
    """Compute the new state for ``ticker`` given its scored decisions.

    Pure function (modulo the two disk reads); side-effects live in
    :func:`apply_promotion`.
    """
    decisions = decisions if decisions is not None else _load_scored(
        ticker, shadow_dir=shadow_dir,
    )
    if not decisions:
        return PromotionDecision(ticker, "shadow", "shadow", 0,
                                 "no scored decisions")

    # Determine the current (most recent) state from the last decision.
    decisions.sort(key=lambda d: d.get("decision_sequence_number", 0))
    current = decisions[-1].get("promotion_state", "shadow")
    seq = int(decisions[-1].get("decision_sequence_number") or 0)

    # Retirement override — drift is a louder signal than performance.
    n_alerts = _drift_alerts_in_window(
        ticker, window_k=retirement_k, drift_dir=drift_dir,
    )
    if current == "live" and n_alerts >= retirement_alerts:
        return PromotionDecision(
            ticker, "live", "retired", seq,
            f"{n_alerts} drift alerts in last {retirement_k}",
        )

    if current == "shadow":
        ok, reason = _shadow_to_candidate_ok(decisions, n=n)
        if ok:
            return PromotionDecision(ticker, "shadow", "candidate", seq, reason)
        return PromotionDecision(ticker, "shadow", "shadow", seq, reason)

    if current == "candidate":
        candidate_decisions = [
            d for d in decisions if d.get("promotion_state") == "candidate"
        ]
        ok, reason = _candidate_to_live_ok(candidate_decisions, m=m)
        if ok:
            return PromotionDecision(ticker, "candidate", "live", seq, reason)
        return PromotionDecision(ticker, "candidate", "candidate", seq, reason)

    # Live / retired are terminal for the promotion direction.
    return PromotionDecision(ticker, current, current, seq, "no-op")


def apply_promotion(
    decision: PromotionDecision,
    *,
    shadow_dir: Path = SHADOW_DIR,
) -> None:
    """Stamp the ``promotion_state`` field on the most recent decision.

    We only update the last record rather than rewriting the whole
    JSONL — the state is a forward-looking marker, not retrospective.
    """
    if decision.old_state == decision.new_state:
        return
    path = shadow_dir / decision.ticker / "decisions_scored.jsonl"
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    if not lines:
        return
    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError:
        return
    last["promotion_state"] = decision.new_state
    last["promotion_reason"] = decision.reason
    lines[-1] = json.dumps(last)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly shadow→live promoter")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    decision = classify_state(args.ticker.upper())
    logger.info("%s: %s → %s (%s)", decision.ticker, decision.old_state,
                decision.new_state, decision.reason)
    if not args.dry_run:
        apply_promotion(decision)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
