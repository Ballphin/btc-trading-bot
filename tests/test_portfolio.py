"""Tests for portfolio and position tracking."""

import pytest
from tradingagents.backtesting.portfolio import Portfolio, Position, PositionSide


class TestPosition:
    """Test Position dataclass."""

    def test_long_position_pnl(self):
        pos = Position(
            side=PositionSide.LONG,
            entry_price=100.0,
            entry_date="2024-01-01",
            size=10.0,
        )
        assert pos.is_open is True
        pos.close(110.0, "2024-01-15")
        assert pos.is_open is False
        assert pos.pnl == pytest.approx(100.0)  # (110-100)*10

    def test_short_position_pnl(self):
        pos = Position(
            side=PositionSide.SHORT,
            entry_price=100.0,
            entry_date="2024-01-01",
            size=10.0,
        )
        pos.close(90.0, "2024-01-15")
        assert pos.pnl == pytest.approx(100.0)  # (100-90)*10

    def test_short_position_loss(self):
        pos = Position(
            side=PositionSide.SHORT,
            entry_price=100.0,
            entry_date="2024-01-01",
            size=10.0,
        )
        pos.close(120.0, "2024-01-15")
        assert pos.pnl == pytest.approx(-200.0)  # (100-120)*10


class TestPortfolio:
    """Test Portfolio class."""

    def test_initial_state(self):
        p = Portfolio(initial_capital=100_000)
        assert p.cash == 100_000
        assert p.position_side == PositionSide.FLAT
        assert p.portfolio_value(50000) == 100_000

    def test_buy_signal(self):
        p = Portfolio(initial_capital=100_000, position_size_pct=0.5)
        action = p.process_signal("BUY", 50_000.0, "2024-01-01")
        assert "ENTERED LONG" in action
        assert p.position_side == PositionSide.LONG

    def test_sell_closes_long(self):
        p = Portfolio(initial_capital=100_000, position_size_pct=0.5)
        p.process_signal("BUY", 50_000.0, "2024-01-01")
        p.process_signal("SELL", 55_000.0, "2024-01-15")
        assert p.position_side == PositionSide.FLAT
        assert len(p.closed_positions) == 1
        assert p.closed_positions[0].pnl > 0  # price went up

    def test_short_signal(self):
        p = Portfolio(initial_capital=100_000, position_size_pct=0.5)
        action = p.process_signal("SHORT", 50_000.0, "2024-01-01")
        assert "ENTERED SHORT" in action
        assert p.position_side == PositionSide.SHORT

    def test_cover_closes_short(self):
        p = Portfolio(initial_capital=100_000, position_size_pct=0.5)
        p.process_signal("SHORT", 50_000.0, "2024-01-01")
        p.process_signal("COVER", 45_000.0, "2024-01-15")
        assert p.position_side == PositionSide.FLAT
        assert len(p.closed_positions) == 1
        assert p.closed_positions[0].pnl > 0  # price went down, short wins

    def test_hold_no_action(self):
        p = Portfolio(initial_capital=100_000)
        action = p.process_signal("HOLD", 50_000.0, "2024-01-01")
        assert "HOLD" in action
        assert p.position_side == PositionSide.FLAT
        assert len(p.trade_history) == 1

    def test_sell_when_flat(self):
        p = Portfolio(initial_capital=100_000)
        action = p.process_signal("SELL", 50_000.0, "2024-01-01")
        assert "no position" in action.lower()

    def test_cover_when_not_short(self):
        p = Portfolio(initial_capital=100_000)
        action = p.process_signal("COVER", 50_000.0, "2024-01-01")
        assert "no short" in action.lower()

    def test_buy_when_short_flips(self):
        p = Portfolio(initial_capital=100_000, position_size_pct=0.25)
        p.process_signal("SHORT", 50_000.0, "2024-01-01")
        p.process_signal("BUY", 48_000.0, "2024-01-15")
        assert p.position_side == PositionSide.LONG
        assert len(p.closed_positions) == 1  # short was closed

    def test_short_when_long_flips(self):
        p = Portfolio(initial_capital=100_000, position_size_pct=0.25)
        p.process_signal("BUY", 50_000.0, "2024-01-01")
        p.process_signal("SHORT", 52_000.0, "2024-01-15")
        assert p.position_side == PositionSide.SHORT
        assert len(p.closed_positions) == 1  # long was closed

    def test_equity_curve_tracking(self):
        p = Portfolio(initial_capital=100_000, position_size_pct=0.25)
        p.process_signal("BUY", 50_000.0, "2024-01-01")
        p.process_signal("HOLD", 55_000.0, "2024-01-08")
        p.process_signal("SELL", 60_000.0, "2024-01-15")
        assert len(p.equity_curve) == 3
        # Values should be increasing since price went up while long
        assert p.equity_curve[-1]["portfolio_value"] > p.equity_curve[0]["portfolio_value"]

    def test_force_close(self):
        p = Portfolio(initial_capital=100_000, position_size_pct=0.25)
        p.process_signal("BUY", 50_000.0, "2024-01-01")
        p.force_close(55_000.0, "2024-06-30")
        assert p.position_side == PositionSide.FLAT
        assert len(p.closed_positions) == 1
