"""Orchestrator: run all detectors with isolated error handling.

SSE-blocker: a single detector exception must NOT 500 the endpoint. Each
detector runs in its own try/except; failures collected and surfaced.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Tuple

import pandas as pd

from .atr import compute_ref_atr
from .channels import detect_auto_trendlines, detect_channels
from .config import global_cfg
from .continuation import (
    detect_flags_pennants,
    detect_triangles,
    detect_wedges,
)
from .reversal import (
    detect_cup_and_handle,
    detect_double_bottom,
    detect_double_top,
    detect_head_and_shoulders,
    detect_inverse_head_and_shoulders,
    detect_triple_bottom,
    detect_triple_top,
)
from .schemas import PatternMatch

logger = logging.getLogger(__name__)


DetectorFn = Callable[[pd.DataFrame, str, float], List[PatternMatch]]


# All registered detectors. Order defines execution order (first = highest priority
# for label-collision resolution downstream).
DETECTORS: List[Tuple[str, DetectorFn]] = [
    ("head_and_shoulders", detect_head_and_shoulders),
    ("inverse_head_and_shoulders", detect_inverse_head_and_shoulders),
    ("double_top", detect_double_top),
    ("double_bottom", detect_double_bottom),
    ("triple_top", detect_triple_top),
    ("triple_bottom", detect_triple_bottom),
    ("cup_and_handle", detect_cup_and_handle),
    ("triangles", detect_triangles),
    ("wedges", detect_wedges),
    ("flags_pennants", detect_flags_pennants),
    ("channels", detect_channels),
    ("auto_trendlines", detect_auto_trendlines),
]

# Which TFs to run pattern detection on by default. 5m/15m are noisy; skip.
DEFAULT_TFS = ("1h", "4h")


def detect_all(
    candles_by_tf: Dict[str, pd.DataFrame],
    tfs: Tuple[str, ...] = DEFAULT_TFS,
) -> Tuple[List[PatternMatch], List[dict]]:
    """Run every detector on each requested TF.

    Returns:
        (matches, errors) — matches sorted by combined_score desc; errors is a
        list of ``{"detector": str, "tf": str, "error": str}`` dicts.
    """
    matches: List[PatternMatch] = []
    errors: List[dict] = []
    gcfg = global_cfg()
    atr_period = gcfg.get("atr_period", 14)
    exclude_fund = gcfg.get("exclude_funding_bars", True)

    for tf in tfs:
        df = candles_by_tf.get(tf)
        if df is None or len(df) < 15:
            continue

        try:
            atr_ref = compute_ref_atr(df, period=atr_period, exclude_funding_bars=exclude_fund)
        except Exception as e:
            logger.exception("ATR computation failed for %s", tf)
            errors.append({"detector": "atr", "tf": tf, "error": str(e)})
            continue

        for name, fn in DETECTORS:
            try:
                found = fn(df, tf, atr_ref) or []
                matches.extend(found)
            except Exception as e:
                logger.exception("Detector %s failed on %s", name, tf)
                errors.append({"detector": name, "tf": tf, "error": str(e)[:200]})

    matches.sort(key=lambda m: m.combined_score, reverse=True)
    return matches, errors
