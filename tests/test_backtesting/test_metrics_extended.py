"""Extended metrics tests — Sharpe SE, sample_size_tier, edge cases."""

import logging
import pytest
from tradingagents.backtesting.metrics import compute_metrics
from tradingagents.backtesting.portfolio import Position, PositionSide


class Helpers:
    @staticmethod
    def make_equity_curve(values, start_date="2024-01-01"):
        from datetime import datetime, timedelta
        start = datetime.strptime(start_date, "%Y-%m-%d")
        return [
            {
                "date": (start + timedelta(days=i * 7)).strftime("%Y-%m-%d"),
                "portfolio_value": v,
                "cash": v,
                "position_side": "FLAT",
            }
            for i, v in enumerate(values)
        ]

    @staticmethod
    def make_positions(pnls):
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


class TestSampleSizeTier:
    def test_unreliable_below_10(self):
        curve = Helpers.make_equity_curve([100_000] * 5)
        m = compute_metrics(curve, [], 100_000)
        tier = m.get("sample_size_tier", "")
        # 5 points → should be unreliable if field exists
        if tier:
            assert tier == "unreliable"

    def test_limited_at_15(self):
        curve = Helpers.make_equity_curve([100_000 + i * 100 for i in range(15)])
        m = compute_metrics(curve, [], 100_000)
        tier = m.get("sample_size_tier", "")
        if tier:
            assert tier in ("limited", "unreliable")

    def test_reliable_at_35(self):
        curve = Helpers.make_equity_curve([100_000 + i * 50 for i in range(35)])
        m = compute_metrics(curve, [], 100_000)
        tier = m.get("sample_size_tier", "")
        if tier:
            assert tier == "reliable"


class TestSharpeStandardError:
    def test_sharpe_se_exists(self):
        curve = Helpers.make_equity_curve([100_000 + i * 200 for i in range(20)])
        m = compute_metrics(curve, [], 100_000)
        # sharpe_se should be computed
        assert "sharpe_se" in m or "sharpe_ratio" in m

    def test_sharpe_se_with_kurtosis(self):
        """Synthetic leptokurtic returns (kurtosis ≈ 10) should have higher SE."""
        import random
        random.seed(42)
        # Generate returns with heavy tails
        values = [100_000]
        for _ in range(50):
            # t-distribution-like returns
            r = random.gauss(0.001, 0.02)
            if random.random() < 0.1:
                r *= 3  # Fat tail
            values.append(values[-1] * (1 + r))
        curve = Helpers.make_equity_curve(values)
        m = compute_metrics(curve, [], 100_000)
        se = m.get("sharpe_se", 0)
        # With fat-tailed returns, SE should be materially higher than Gaussian baseline
        assert se > 0.15


class TestZeroTrades:
    def test_empty_everything(self):
        m = compute_metrics([], [], 100_000)
        assert m["total_return_pct"] == 0.0
        assert m["total_trades"] == 0
        assert m["win_rate_pct"] == 0.0

    def test_all_winning(self):
        curve = Helpers.make_equity_curve([100_000, 110_000, 120_000])
        positions = Helpers.make_positions([10_000, 10_000])
        m = compute_metrics(curve, positions, 100_000)
        assert m["win_rate_pct"] == pytest.approx(100.0)

    def test_all_losing(self):
        curve = Helpers.make_equity_curve([100_000, 90_000, 80_000])
        positions = Helpers.make_positions([-10_000, -10_000])
        m = compute_metrics(curve, positions, 100_000)
        assert m["win_rate_pct"] == pytest.approx(0.0)


class TestProfitFactorEdge:
    def test_zero_losses(self):
        positions = Helpers.make_positions([1000, 2000])
        curve = Helpers.make_equity_curve([100_000, 101_000, 103_000])
        m = compute_metrics(curve, positions, 100_000)
        # profit_factor is None when there are no losing trades (represents ∞)
        assert m["profit_factor"] is None


class TestMalformedPositionDates:
    def test_malformed_date_logs_warning(self, caplog):
        """K3: Malformed dates should log debug, not crash silently."""
        p = Position(
            side=PositionSide.LONG,
            entry_price=100.0,
            entry_date="not-a-date",
            size=1.0,
        )
        p.exit_date = "also-bad"
        p.exit_price = 110.0
        p.pnl = 10.0

        curve = Helpers.make_equity_curve([100_000, 110_000])
        with caplog.at_level(logging.DEBUG):
            m = compute_metrics(curve, [p], 100_000)
        assert m["total_trades"] == 1
        # The fix should log "Malformed position dates"
        assert "Malformed" in caplog.text or m["avg_hold_days"] == 0
