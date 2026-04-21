"""Decision attribution — Stage 2 Commit O.

Per-pulse feature contribution analysis. The Pulse scoring engine
already emits a ``breakdown`` dict per signal (e.g., ``{"1h": 0.35,
"4h": 0.20, "order_flow": -0.1, "sr_proximity": 0.15}``) — this module
surfaces the top-3 positive and top-3 negative contributors per
decision and an aggregate weekly ranking of features by cumulative
|contribution|.

Per SQR rebuttal during the debate, attribution operates only on pure
Pulse features (the ``breakdown`` dict) — we do **not** bring Kalshi /
FRED / Polymarket context into the attribution math; those are agent
narrative inputs, not Pulse score inputs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PULSE_DIR = Path("results/pulse")


def per_decision_attribution(
    pulse_record: Dict[str, Any],
    *,
    top_n: int = 3,
) -> Dict[str, Any]:
    """Return top-N positive / top-N negative contributors for one pulse.

    Shape of return:
        {
            "top_positive": [{"feature": "1h", "contribution": 0.35}, ...],
            "top_negative": [{"feature": "order_flow", "contribution": -0.12}, ...],
            "persistence_mul": 1.0,   # passthrough from pulse
            "total_abs_contribution": 0.72,
        }
    """
    breakdown: Dict[str, Any] = pulse_record.get("breakdown") or {}
    entries: List[Tuple[str, float]] = [
        (k, float(v)) for k, v in breakdown.items() if v is not None
    ]
    pos = sorted([e for e in entries if e[1] > 0], key=lambda x: x[1], reverse=True)
    neg = sorted([e for e in entries if e[1] < 0], key=lambda x: x[1])
    return {
        "top_positive": [{"feature": k, "contribution": round(v, 4)} for k, v in pos[:top_n]],
        "top_negative": [{"feature": k, "contribution": round(v, 4)} for k, v in neg[:top_n]],
        "persistence_mul": pulse_record.get("persistence_mul"),
        "total_abs_contribution": round(sum(abs(v) for _, v in entries), 4),
    }


def _iter_pulses(ticker: str, *, pulse_dir: Path = PULSE_DIR) -> Iterable[Dict[str, Any]]:
    path = pulse_dir / ticker.upper() / "pulse.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def weekly_feature_ranking(
    ticker: str,
    *,
    lookback_days: int = 7,
    top_n: int = 5,
    now: Optional[datetime] = None,
    pulse_dir: Path = PULSE_DIR,
) -> List[Dict[str, Any]]:
    """Aggregate |contribution| by feature across the last ``lookback_days``.

    Returns a list ``[{"feature": ..., "cumulative_abs": ..., "n_pulses": ...}, ...]``
    sorted by cumulative |contribution| desc, truncated to ``top_n``.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    cumulative: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for p in _iter_pulses(ticker, pulse_dir=pulse_dir):
        ts = p.get("ts") or p.get("timestamp")
        try:
            ts_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts_dt < cutoff:
            continue
        for k, v in (p.get("breakdown") or {}).items():
            try:
                cumulative[k] += abs(float(v))
                counts[k] += 1
            except (TypeError, ValueError):
                continue
    ranked = sorted(cumulative.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        {
            "feature": k,
            "cumulative_abs": round(v, 4),
            "n_pulses": counts[k],
        }
        for k, v in ranked
    ]
