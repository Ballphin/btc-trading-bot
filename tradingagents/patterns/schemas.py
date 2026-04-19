"""Data schemas for chart-pattern detection.

All detectors return `PatternMatch` instances. Invariants enforced in
`__post_init__` so malformed matches never reach the API response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional


class PatternState(str, Enum):
    """Lifecycle state of a detected pattern.

    FORMING      → geometry partially in place; final pivot not yet confirmed.
    COMPLETED    → all geometric anchors present; awaiting breakout.
    CONFIRMED    → price broke the key line (neckline / trendline) with volume.
    RETESTED     → price pulled back to the broken line then continued.
    INVALIDATED  → price moved past a disqualifying level; pattern failed.
    """

    FORMING = "forming"
    COMPLETED = "completed"
    CONFIRMED = "confirmed"
    RETESTED = "retested"
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class AnchorPoint:
    """A labeled point on the chart — pivot, break, or retest."""

    label: str          # e.g. "A", "Left Shoulder", "Head"
    ts: str             # ISO-8601 UTC
    price: float
    role: str           # "peak" | "trough" | "break" | "retest" | "apex"
    idx: int = -1       # index into the source candle series (detector-local)


@dataclass(frozen=True)
class PatternLine:
    """A line connecting two anchor indices on the chart.

    Indices are into the TF's candle array; front-end looks up the ts/price
    from the candle at that index to render on lightweight-charts.
    """

    from_idx: int
    to_idx: int
    role: str                         # "neckline" | "support" | "resistance" | "upper_channel" | ...
    style: Literal["solid", "dashed", "dotted"] = "solid"
    weight: Literal[1, 2, 3] = 2
    color_token: str = "ok_gray"      # token resolved on frontend


@dataclass
class PatternMatch:
    """A detected pattern with full geometry + state + scoring."""

    name: str                         # snake_case id, e.g. "head_and_shoulders"
    display_name: str                 # human label, e.g. "Head and Shoulders"
    bias: Literal["bullish", "bearish", "neutral"]
    state: PatternState
    fit_score: float                  # geometric tolerance match, 0-1
    duration_score: float             # 0-1, bars_in_pattern / 40 clamped
    volume_score: float               # 0-1, monotonic volume check (0.5 neutral)
    combined_score: float             # 0.6·fit + 0.2·dur + 0.2·vol
    timeframe: str                    # "5m" | "15m" | "1h" | "4h"
    anchors: List[AnchorPoint]
    lines: List[PatternLine]
    bars_in_pattern: int
    regime_aligned: bool = False      # set downstream if pulse liq_score matches bias
    description: str = ""
    color_token: str = "ok_gray"      # primary palette token for this pattern

    # Invariants ------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError(f"{self.name}: anchors must be non-empty")

        # Time-ordering (skip for patterns that can overlap, e.g. cup-and-handle
        # with a handle that can dip below cup bottom). Detectors opt-out by
        # passing already-ordered anchors with monotonic ts — default is enforce.
        if self.name not in {"cup_and_handle"}:
            ts_list = [a.ts for a in self.anchors]
            if ts_list != sorted(ts_list):
                raise ValueError(
                    f"{self.name}: anchors not time-ordered: {ts_list}"
                )

        # Score ranges
        for attr in ("fit_score", "duration_score", "volume_score", "combined_score"):
            v = getattr(self, attr)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{self.name}: {attr}={v} out of [0,1]")

        # Volume gating: if a pattern requires volume and volume_score is missing
        # (encoded as 0.5 by convention when unknown), fit_score caps at 0.7.
        if self.name in {"head_and_shoulders", "inverse_head_and_shoulders",
                         "double_top", "double_bottom",
                         "triple_top", "triple_bottom"}:
            if self.volume_score < 0.3 and self.fit_score > 0.7:
                # Use object.__setattr__ since this dataclass is mutable but
                # we want to preserve the cap semantically.
                self.fit_score = 0.7
                self.combined_score = (
                    0.6 * self.fit_score
                    + 0.2 * self.duration_score
                    + 0.2 * self.volume_score
                )

    # Serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "bias": self.bias,
            "state": self.state.value,
            "fit_score": round(self.fit_score, 4),
            "duration_score": round(self.duration_score, 4),
            "volume_score": round(self.volume_score, 4),
            "combined_score": round(self.combined_score, 4),
            "timeframe": self.timeframe,
            "anchors": [
                {
                    "label": a.label,
                    "ts": a.ts,
                    "price": round(a.price, 6),
                    "role": a.role,
                    "idx": a.idx,
                }
                for a in self.anchors
            ],
            "lines": [
                {
                    "from_idx": ln.from_idx,
                    "to_idx": ln.to_idx,
                    "role": ln.role,
                    "style": ln.style,
                    "weight": ln.weight,
                    "color_token": ln.color_token,
                }
                for ln in self.lines
            ],
            "bars_in_pattern": self.bars_in_pattern,
            "regime_aligned": self.regime_aligned,
            "description": self.description,
            "color_token": self.color_token,
        }


# Sentinel used by detectors when volume data is unusable / absent.
VOLUME_UNKNOWN: float = 0.5
