"""Tests for backtest metrics computation."""

import pytest
from tradingagents.backtesting.metrics import compute_metrics
from tradingagents.backtesting.portfolio import Position, PositionSide


class TestComputeMetrics:
    """Test compute_metrics function."""

    def _make_equity_curve(self, values, start_date="2024-01-01"):
        """Helper to create equity curve from a list of values."""
        from datetime import datetime, timedelta
        start = datetime.strptime(start_date, "%Y-%m-%d")
        return [
            {"date": (start + timedelta(days=i*7)).strftime("%Y-%m-%d"),
             "portfolio_value": v, "cash": v, "position_side": "FLAT"}
            for i, v in enumerate(values)
        ]

    def _make_positions(self, pnls):
        """Helper to create closed positions from a list of P&Ls."""
        positions = []
        for pnl in pnls:
            p = Position(
                side=PositionSide.LONG,
                entry_price=100.0,
                entry_date="2024-01-01",
                size=1.0,
            )
            p.close(100.0 + pnl, "2024-01-15")
            positions.append(p)
        return positions

    def test_empty_equity_curve(self):
        m = compute_metrics([], [], 100_000)
        assert m["total_return_pct"] == 0.0
        assert m["total_trades"] == 0

    def test_single_point(self):
        curve = self._make_equity_curve([100_000])
        m = compute_metrics(curve, [], 100_000)
        assert m["total_return_pct"] == 0.0

    def test_positive_return(self):
        curve = self._make_equity_curve([100_000, 105_000, 110_000])
        positions = self._make_positions([5000, 5000])
        m = compute_metrics(curve, positions, 100_000)
        assert m["total_return_pct"] == pytest.approx(10.0)
        assert m["total_pnl"] == pytest.approx(10_000)
        assert m["total_trades"] == 2
        assert m["win_rate_pct"] == pytest.approx(100.0)

    def test_negative_return(self):
        curve = self._make_equity_curve([100_000, 95_000, 90_000])
        positions = self._make_positions([-5000, -5000])
        m = compute_metrics(curve, positions, 100_000)
        assert m["total_return_pct"] == pytest.approx(-10.0)
        assert m["win_rate_pct"] == pytest.approx(0.0)

    def test_max_drawdown(self):
        curve = self._make_equity_curve([100_000, 110_000, 90_000, 95_000])
        m = compute_metrics(curve, [], 100_000)
        # Peak was 110k, trough was 90k => drawdown = 20/110 = 18.18%
        assert m["max_drawdown_pct"] == pytest.approx(18.1818, rel=0.01)

    def test_win_loss_stats(self):
        positions = self._make_positions([1000, -500, 2000, -300, 500])
        curve = self._make_equity_curve([100_000, 101_000, 100_500, 102_500, 102_200, 102_700])
        m = compute_metrics(curve, positions, 100_000)
        assert m["total_trades"] == 5
        assert m["winning_trades"] == 3
        assert m["losing_trades"] == 2
        assert m["win_rate_pct"] == pytest.approx(60.0)

    def test_profit_factor(self):
        positions = self._make_positions([1000, -500, 2000])
        curve = self._make_equity_curve([100_000, 101_000, 100_500, 102_500])
        m = compute_metrics(curve, positions, 100_000)
        # Gross profit = 3000, gross loss = 500
        assert m["profit_factor"] == pytest.approx(6.0)

    def test_sharpe_ratio_positive(self):
        # Steadily increasing equity => positive Sharpe
        curve = self._make_equity_curve([100_000, 101_000, 102_000, 103_000, 104_000])
        m = compute_metrics(curve, [], 100_000)
        assert m["sharpe_ratio"] > 0
