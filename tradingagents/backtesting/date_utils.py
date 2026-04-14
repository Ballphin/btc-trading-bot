"""Shared date-parsing utilities for the backtesting module.

Centralises the multi-format date parser so that scorecard, walk-forward,
and any future module all handle the system's various date formats
consistently.
"""

import re
from datetime import datetime

# Date formats produced by the system
DATE_FORMATS = [
    "%Y-%m-%d",            # daily: 2026-04-08
    "%Y-%m-%dT%H",         # 4H scheduler: 2026-04-08T16
    "%Y-%m-%dT%H:%M",      # intraday: 2026-04-08T16:00
    "%Y-%m-%d-%I-%M-%p",   # manual runs: 2026-04-13-12-45-AM
]


def parse_any_date(date_str: str) -> datetime:
    """Parse a date string in any of the system's known formats.

    Returns datetime (date portion only for scoring purposes).
    Raises ValueError if no format matches.
    """
    clean = date_str.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    # Last resort: try just the date portion
    # Handles edge cases like "2026-04-13-12-45-AM" where split("T") doesn't help
    date_part = re.match(r'(\d{4}-\d{2}-\d{2})', clean)
    if date_part:
        return datetime.strptime(date_part.group(1), "%Y-%m-%d")
    raise ValueError(f"Cannot parse date: {date_str!r}")
