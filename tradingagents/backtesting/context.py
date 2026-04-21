"""Thread-safe context variables for execution mode.

Stage 2 Commit M.2 — three-mode refactor.

The original single ``BACKTEST_MODE`` flag conflated two distinct
concerns: "am I allowed to fetch live market data?" and "am I allowed
to route orders?". That made paper-trading (live data, no fills)
impossible to express cleanly.

The three flags are now:

    BACKTEST_MODE      — am I simulating historical data? If True,
                         vendor clients must refuse live-only endpoints
                         (Kalshi, realtime snapshots) to prevent
                         look-ahead leakage.
    LIVE_DATA_OK       — am I allowed to hit real market-data vendors?
                         True in live AND paper. False only in backtest.
    EXECUTE_TRADES     — am I allowed to submit orders? True in live
                         only; False in paper and backtest.

The three supported modes expressed as tuples ``(BT, LIVE, EXEC)``:

    live      → (False, True,  True)      — default
    paper     → (False, True,  False)     — live data, no order routing
    backtest  → (True,  False, False)     — historical replay

``BACKTEST_MODE`` is preserved as the authoritative source of truth for
backward compatibility; ``LIVE_DATA_OK`` / ``EXECUTE_TRADES`` are
additive flags callers set explicitly when they need the finer
distinction. Helper :func:`set_mode` makes the common case a one-liner.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Literal

BACKTEST_MODE = contextvars.ContextVar('backtest_mode', default=False)
LIVE_DATA_OK = contextvars.ContextVar('live_data_ok', default=True)
EXECUTE_TRADES = contextvars.ContextVar('execute_trades', default=True)

Mode = Literal["live", "paper", "backtest"]


def set_mode(mode: Mode) -> None:
    """Set all three ContextVars consistently for ``mode``.

    Intended for use in entry points (CLI, server startup, backtest
    engine). For scoped overrides inside a request handler or test,
    prefer :func:`mode_override` which restores prior state on exit.
    """
    if mode == "live":
        BACKTEST_MODE.set(False); LIVE_DATA_OK.set(True); EXECUTE_TRADES.set(True)
    elif mode == "paper":
        BACKTEST_MODE.set(False); LIVE_DATA_OK.set(True); EXECUTE_TRADES.set(False)
    elif mode == "backtest":
        BACKTEST_MODE.set(True); LIVE_DATA_OK.set(False); EXECUTE_TRADES.set(False)
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected live/paper/backtest)")


@contextmanager
def mode_override(mode: Mode):
    """Scoped version of :func:`set_mode` — restores prior tokens on exit."""
    bt_tok = BACKTEST_MODE.set(BACKTEST_MODE.get())
    ld_tok = LIVE_DATA_OK.set(LIVE_DATA_OK.get())
    ex_tok = EXECUTE_TRADES.set(EXECUTE_TRADES.get())
    try:
        set_mode(mode)
        yield
    finally:
        BACKTEST_MODE.reset(bt_tok)
        LIVE_DATA_OK.reset(ld_tok)
        EXECUTE_TRADES.reset(ex_tok)
