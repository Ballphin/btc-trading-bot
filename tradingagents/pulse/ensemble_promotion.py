"""Per-config promotion for the pulse ensemble — R.7.

Builds on Stage 2 Commit P's shadow→candidate→live→retired state
machine, but keyed on ``(ticker, config)`` and with v2 gates from the
post-debate plan:

* ``shadow → candidate``
  - ``n_signals ≥ 50``
  - ``oos_validation.deflated_sharpe > 0`` (Deflated Sharpe with
    ``n_strategies=5``, from R.4 metrics aggregator).
  - The outcomes window spans at least one weekend.
  - ``weekend.n_signals ≥ 10`` (HIGH #9).

* ``candidate → live``
  - ``n_signals ≥ 150``
  - ``oos_validation.deflated_sharpe > 0``
  - (Champion-selector layer additionally enforces a margin-of-victory
    rule when picking the single live champion across configs; see
    :func:`select_champion`.)

* ``live → retired``
  - 3+ drift alerts in a rolling window of 30 (same as Commit P, reused
    unchanged because drift is config-agnostic).

Promotion state is persisted per config at
``results/pulse/<ticker>/configs/<config>/promotion.json`` so
champion-selection is pure and can re-run any time without state
thrash. The operator's champion.json file (R.5) is a *separate*
artifact: promotion tells us which configs are ELIGIBLE; champion
tells us which ELIGIBLE config is live.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Gate constants — exported so tests + champion selector share one source.
CANDIDATE_MIN_N = 50
LIVE_MIN_N = 150
WEEKEND_MIN_N = 10
RETIREMENT_K = 30
RETIREMENT_ALERTS = 3

STATES = ("shadow", "candidate", "live", "retired")

PULSE_DIR = Path("results/pulse")
DRIFT_DIR = Path("results/drift")


@dataclass
class PromotionResult:
    ticker: str
    config: str
    old_state: str
    new_state: str
    reason: str


def _load_metrics(
    ticker: str, config: str, *, pulse_dir: Path = PULSE_DIR,
) -> Optional[Dict[str, Any]]:
    path = pulse_dir / ticker / "configs" / config / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_state(
    ticker: str, config: str, *, pulse_dir: Path = PULSE_DIR,
) -> str:
    path = pulse_dir / ticker / "configs" / config / "promotion.json"
    if not path.exists():
        return "shadow"
    try:
        doc = json.loads(path.read_text())
        s = str(doc.get("state") or "shadow")
        return s if s in STATES else "shadow"
    except Exception:
        return "shadow"


def _write_state(
    ticker: str, config: str, state: str, reason: str,
    *, pulse_dir: Path = PULSE_DIR,
) -> None:
    d = pulse_dir / ticker / "configs" / config
    d.mkdir(parents=True, exist_ok=True)
    doc = {
        "state": state,
        "reason": reason,
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = d / "promotion.json.tmp"
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(d / "promotion.json")


def _window_spans_weekend(outcomes_iter: List[dict]) -> bool:
    """True if any outcome's ``is_weekend`` flag was set. The metric
    aggregator has already segmented them; we just need a presence
    check at the config level."""
    return any(bool(o.get("is_weekend")) for o in outcomes_iter)


def _load_outcomes(
    ticker: str, config: str, *, pulse_dir: Path = PULSE_DIR,
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


def _drift_alerts(
    ticker: str,
    *,
    window_k: int = RETIREMENT_K,
    drift_dir: Path = DRIFT_DIR,
) -> int:
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


def _shadow_to_candidate_ok(metrics: Dict[str, Any], outcomes: List[dict]) -> Tuple[bool, str]:
    overall = metrics.get("overall") or {}
    oos = metrics.get("oos_validation") or {}
    weekend = metrics.get("weekend") or {}

    if (overall.get("n_signals") or 0) < CANDIDATE_MIN_N:
        return False, f"n_signals {overall.get('n_signals')} < {CANDIDATE_MIN_N}"
    if (weekend.get("n_signals") or 0) < WEEKEND_MIN_N:
        return False, f"weekend_n_signals {weekend.get('n_signals')} < {WEEKEND_MIN_N}"
    if not _window_spans_weekend(outcomes):
        return False, "window does not span a weekend"
    dsr = oos.get("deflated_sharpe")
    if dsr is None or dsr <= 0:
        return False, f"oos deflated_sharpe {dsr} ≤ 0"
    return True, f"oos DSR {dsr:.3f}, n={overall['n_signals']}, weekend_n={weekend['n_signals']}"


def _candidate_to_live_ok(metrics: Dict[str, Any]) -> Tuple[bool, str]:
    overall = metrics.get("overall") or {}
    oos = metrics.get("oos_validation") or {}
    if (overall.get("n_signals") or 0) < LIVE_MIN_N:
        return False, f"n_signals {overall.get('n_signals')} < {LIVE_MIN_N}"
    dsr = oos.get("deflated_sharpe")
    if dsr is None or dsr <= 0:
        return False, f"oos deflated_sharpe {dsr} ≤ 0"
    return True, f"oos DSR {dsr:.3f}, n={overall['n_signals']}"


def classify(
    ticker: str,
    config: str,
    *,
    pulse_dir: Path = PULSE_DIR,
    drift_dir: Path = DRIFT_DIR,
) -> PromotionResult:
    """Compute the new state for ``(ticker, config)``.

    Pure-ish: reads metrics.json + outcomes.jsonl + drift logs but does
    not mutate anything; :func:`apply` handles the persistence.
    """
    metrics = _load_metrics(ticker, config, pulse_dir=pulse_dir)
    outcomes = _load_outcomes(ticker, config, pulse_dir=pulse_dir)
    current = _load_state(ticker, config, pulse_dir=pulse_dir)

    if metrics is None:
        return PromotionResult(ticker, config, current, current,
                               "no metrics.json — leave unchanged")

    # Retirement override — drift is a louder signal than performance.
    if current == "live" and _drift_alerts(ticker, drift_dir=drift_dir) >= RETIREMENT_ALERTS:
        return PromotionResult(ticker, config, "live", "retired",
                               "≥3 drift alerts in last 30 windows")

    if current == "shadow":
        ok, reason = _shadow_to_candidate_ok(metrics, outcomes)
        new = "candidate" if ok else "shadow"
        return PromotionResult(ticker, config, current, new, reason)

    if current == "candidate":
        ok, reason = _candidate_to_live_ok(metrics)
        new = "live" if ok else "candidate"
        return PromotionResult(ticker, config, current, new, reason)

    # live / retired are terminal wrt this machine.
    return PromotionResult(ticker, config, current, current, "no-op")


def apply(result: PromotionResult, *, pulse_dir: Path = PULSE_DIR) -> None:
    if result.old_state == result.new_state:
        return
    _write_state(
        result.ticker, result.config, result.new_state, result.reason,
        pulse_dir=pulse_dir,
    )
    logger.info(
        "[Promotion] %s/%s: %s → %s (%s)",
        result.ticker, result.config, result.old_state,
        result.new_state, result.reason,
    )


# ── Champion selector ────────────────────────────────────────────────

def select_champion(
    ticker: str,
    *,
    pulse_dir: Path = PULSE_DIR,
) -> Optional[Tuple[str, str]]:
    """Return ``(config, reason)`` of the config that should be
    champion, or ``None`` if no config is both eligible and clearly
    superior.

    Eligibility: state ∈ {candidate, live}.
    Winning rule: highest ``oos_validation.deflated_sharpe`` wins, but
    only if it beats the runner-up by more than half the CI width. The
    margin guard prevents argmax-biased flapping when two configs are
    near-tied (SQR Round 1 #5).
    """
    from tradingagents.pulse.ensemble_metrics import list_configs

    candidates: List[Tuple[str, float, float, float]] = []
    # (config, dsr, sharpe_point, range_proxy)
    for cfg in list_configs(ticker, pulse_dir=pulse_dir):
        state = _load_state(ticker, cfg, pulse_dir=pulse_dir)
        if state not in ("candidate", "live"):
            continue
        m = _load_metrics(ticker, cfg, pulse_dir=pulse_dir)
        if not m:
            continue
        oos = m.get("oos_validation") or {}
        dsr = oos.get("deflated_sharpe")
        if dsr is None:
            continue
        sharpe = float(oos.get("sharpe_point") or 0.0)
        # Range proxy: use |sharpe| as a crude CI-width surrogate since
        # the metrics block doesn't yet carry CI bounds. The margin
        # test then asks "does winner beat runner-up by ≥ half of its
        # own |sharpe|?" — strict enough to reject near-ties.
        ci_width = abs(sharpe) * 0.5
        candidates.append((cfg, float(dsr), sharpe, ci_width))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[1], reverse=True)
    if len(candidates) == 1:
        winner = candidates[0]
        return winner[0], f"sole eligible config; DSR {winner[1]:.3f}"

    top, runner = candidates[0], candidates[1]
    margin = top[1] - runner[1]
    required = max(0.05, (top[3] + runner[3]) / 2.0)
    if margin < required:
        return None  # too close → no promotion, keep current champion
    return top[0], (
        f"DSR {top[1]:.3f} beats {runner[0]} {runner[1]:.3f} "
        f"by {margin:.3f} ≥ {required:.3f}"
    )
