"""Chart-pattern detection package.

Display-only detectors for Pulse Explain Chart. NOT wired into signal scoring.
Entry point: `registry.detect_all(candles_by_tf)`.
"""

from .schemas import (
    AnchorPoint,
    PatternLine,
    PatternMatch,
    PatternState,
)
from .registry import detect_all

__all__ = [
    "AnchorPoint",
    "PatternLine",
    "PatternMatch",
    "PatternState",
    "detect_all",
]
