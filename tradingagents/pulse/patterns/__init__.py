"""Pulse v4 pattern detection — strict candlestick + structural patterns.

Single source of truth for candlestick detection; replaces the legacy
``_PATTERN_DETECTORS`` list in ``tradingagents/agents/quant_pulse_data.py``.
"""

from tradingagents.pulse.patterns.candles import detect_all
from tradingagents.pulse.patterns.extrema import Extremum, find_extrema
from tradingagents.pulse.patterns.structural import (
    PatternHit,
    detect_ascending_triangle,
    detect_channel_down,
    detect_channel_up,
    detect_double_bottom,
    detect_double_top,
    detect_head_shoulders,
    detect_inverse_head_shoulders,
    detect_rectangle,
    detect_structural_all,
)

__all__ = [
    "detect_all",
    "Extremum",
    "find_extrema",
    "PatternHit",
    "detect_ascending_triangle",
    "detect_channel_down",
    "detect_channel_up",
    "detect_double_bottom",
    "detect_double_top",
    "detect_head_shoulders",
    "detect_inverse_head_shoulders",
    "detect_rectangle",
    "detect_structural_all",
]
