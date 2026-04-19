"""Unified scoring for pattern matches.

Single source of truth (SSE+SQR consolidated blocker): every detector must
return scores computed by these functions. API field renamed from
`confidence` → `fit_score` to avoid users reading it as "historical win rate".
"""

from __future__ import annotations

from typing import Iterable, Sequence


def fit_score_from_violations(
    violations: Iterable[float],
    max_violations: Iterable[float],
) -> float:
    """Compute geometric fit score from per-dimension tolerance violations.

    Each ``violation`` is the absolute deviation from the ideal geometry
    (e.g. |left_shoulder - right_shoulder| for H&S symmetry). Each
    ``max_violation`` is the allowed tolerance for that dimension. Values
    above max_violation yield 0; at 0 yield 1; scales linearly in between.

    Returns sqrt(mean(fit_per_dim^2)) style score in [0, 1] where higher is
    better fit.
    """
    pairs = list(zip(violations, max_violations))
    if not pairs:
        return 0.0
    sq_fits = []
    for v, m in pairs:
        if m <= 0:
            # Division by zero guard — treat as perfect if violation is 0.
            sq_fits.append(1.0 if v == 0 else 0.0)
            continue
        ratio = min(max(abs(v) / m, 0.0), 1.0)
        fit = (1.0 - ratio) ** 2
        sq_fits.append(fit)
    return float(sum(sq_fits) / len(sq_fits)) ** 0.5


def duration_score(bars_in_pattern: int, target_bars: int = 40) -> float:
    """Longer patterns are rarer and more reliable. Scales to 1.0 at target_bars."""
    if bars_in_pattern <= 0:
        return 0.0
    return min(bars_in_pattern / float(target_bars), 1.0)


def volume_score_monotonic(
    volumes: Sequence[float],
    direction: str = "declining",
) -> float:
    """Score how monotonically a volume sequence trends.

    For H&S / Top patterns we want DECLINING volume through left-shoulder →
    head → right-shoulder. For Bottom patterns we want INCLINING on the way up.

    Returns 1.0 for perfect monotone, 0.0 for strictly opposite, ~0.5 otherwise.
    """
    if len(volumes) < 2:
        return 0.5
    diffs = [volumes[i + 1] - volumes[i] for i in range(len(volumes) - 1)]
    if direction == "declining":
        correct = sum(1 for d in diffs if d < 0)
    elif direction == "rising":
        correct = sum(1 for d in diffs if d > 0)
    else:
        return 0.5
    return correct / len(diffs)


def combined_score(
    fit: float,
    duration: float,
    volume: float,
    w_fit: float = 0.6,
    w_dur: float = 0.2,
    w_vol: float = 0.2,
) -> float:
    """Final aggregate score for ranking patterns.

    Weights default per Round-2 consolidation: 0.6/0.2/0.2.
    """
    return max(0.0, min(1.0, w_fit * fit + w_dur * duration + w_vol * volume))
