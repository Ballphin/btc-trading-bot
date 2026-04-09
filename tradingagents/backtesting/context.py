"""Thread-safe context variables for backtesting.

Uses contextvars to allow backtest mode detection without global state.
This is necessary because prediction_tools.py needs to know if it's running
in backtest mode, but the engine's local config isn't accessible via the
global get_config() singleton.
"""

import contextvars

BACKTEST_MODE = contextvars.ContextVar('backtest_mode', default=False)
