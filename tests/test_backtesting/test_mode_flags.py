"""Stage 2 Commit M.2 — three-mode flag refactor tests."""

from __future__ import annotations

import pytest

from tradingagents.backtesting.context import (
    BACKTEST_MODE, EXECUTE_TRADES, LIVE_DATA_OK,
    mode_override, set_mode,
)


@pytest.mark.parametrize("mode,expected", [
    ("live",     (False, True,  True)),
    ("paper",    (False, True,  False)),
    ("backtest", (True,  False, False)),
])
def test_set_mode_configures_all_three_flags(mode, expected):
    with mode_override(mode):
        assert (BACKTEST_MODE.get(), LIVE_DATA_OK.get(), EXECUTE_TRADES.get()) == expected


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        set_mode("turbo")  # type: ignore[arg-type]


def test_mode_override_restores_prior_state():
    set_mode("live")
    before = (BACKTEST_MODE.get(), LIVE_DATA_OK.get(), EXECUTE_TRADES.get())
    with mode_override("backtest"):
        assert BACKTEST_MODE.get() is True
        assert LIVE_DATA_OK.get() is False
    assert (BACKTEST_MODE.get(), LIVE_DATA_OK.get(), EXECUTE_TRADES.get()) == before


def test_paper_mode_allows_live_data_but_no_execution():
    """The core distinction M.2 was written to express."""
    with mode_override("paper"):
        assert LIVE_DATA_OK.get() is True
        assert EXECUTE_TRADES.get() is False


def test_backtest_mode_disables_live_data():
    with mode_override("backtest"):
        assert BACKTEST_MODE.get() is True
        assert LIVE_DATA_OK.get() is False
