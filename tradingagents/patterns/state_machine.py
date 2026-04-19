"""Pattern lifecycle state classification.

After a detector identifies geometric anchors, we classify where the pattern
is in its lifecycle based on price action AFTER the final anchor:

    FORMING      → final anchor is near the right edge of the window; the
                   confirming move (neckline break for H&S, bounce for
                   triangles) hasn't happened yet.
    COMPLETED    → geometry is fully in place AND the window extends at
                   least N bars past the final anchor, but the key line
                   hasn't been broken.
    CONFIRMED    → price closed through the key line (e.g. neckline for
                   H&S, trendline for triangles) in the pattern's bias
                   direction.
    RETESTED     → after confirmation, price pulled back to the broken
                   line and bounced, before continuing in the bias dir.
    INVALIDATED  → price moved to a disqualifying level (e.g. closed
                   above the H&S head, invalidating the pattern).

Each detector supplies `classify_params`; the classifier does NOT hard-code
detector-specific geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .schemas import PatternState


@dataclass
class StateParams:
    """What the state classifier needs to know about a pattern.

    key_line_price:  horizontal price level whose break = CONFIRMATION
                     (e.g. H&S neckline). Pass None to skip state FSM
                     (pattern will be marked COMPLETED).
    bias:            "bullish" → break DOWNWARD invalidates, break UPWARD confirms
                     "bearish" → mirror
                     "neutral" → neither, stays COMPLETED
    invalidation_price: price that, if CLOSED past, marks pattern INVALIDATED
                        (e.g. above the H&S head for a bearish H&S).
    final_anchor_idx: index of the last pivot anchor in the candle array.
    min_post_bars:   minimum bars past final_anchor_idx before we call it
                     anything other than FORMING.
    """

    key_line_price: Optional[float]
    bias: Literal["bullish", "bearish", "neutral"]
    invalidation_price: Optional[float]
    final_anchor_idx: int
    min_post_bars: int = 2


def classify_state(df: pd.DataFrame, params: StateParams) -> PatternState:
    """Run the FSM against the post-pattern price action."""
    n = len(df)
    if n == 0 or params.final_anchor_idx >= n - 1:
        return PatternState.FORMING

    post = df.iloc[params.final_anchor_idx + 1 :]
    post_closes = post["close"].to_numpy(dtype=float)
    if len(post_closes) < params.min_post_bars:
        return PatternState.FORMING

    # Invalidation check FIRST — highest priority.
    if params.invalidation_price is not None:
        inv = params.invalidation_price
        if params.bias == "bearish":
            if (post_closes > inv).any():
                return PatternState.INVALIDATED
        elif params.bias == "bullish":
            if (post_closes < inv).any():
                return PatternState.INVALIDATED

    # Neutral / no key line: just return COMPLETED after min bars.
    if params.key_line_price is None or params.bias == "neutral":
        return PatternState.COMPLETED

    kl = params.key_line_price
    # Confirmation: bias-direction close past the key line.
    break_mask = (
        (post_closes < kl) if params.bias == "bearish" else (post_closes > kl)
    )
    if not break_mask.any():
        return PatternState.COMPLETED

    first_break = int(np.argmax(break_mask))  # first True index
    post_break = post_closes[first_break + 1 :]
    if len(post_break) == 0:
        return PatternState.CONFIRMED

    # Retest: price returned to within 0.25% of key line, then continued.
    retest_band = abs(kl) * 0.0025
    touched = np.abs(post_break - kl) <= retest_band
    if touched.any():
        touch_idx = int(np.argmax(touched))
        after_touch = post_break[touch_idx + 1 :]
        if len(after_touch) > 0:
            continuation = (
                (after_touch < kl) if params.bias == "bearish" else (after_touch > kl)
            )
            if continuation.any():
                return PatternState.RETESTED

    return PatternState.CONFIRMED
