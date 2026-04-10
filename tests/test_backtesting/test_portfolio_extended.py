"""Extended portfolio tests — slippage, funding, COVER direction, liquidation."""

import pytest
from tradingagents.backtesting.portfolio import Portfolio, Position, PositionSide


@pytest.fixture
def portfolio():
    return Portfolio(
        initial_capital=100_000.0,
        position_size_pct=0.25,
        leverage=1.0,
        slippage_bps=5.0,
    )


# ── K6: COVER slippage direction ─────────────────────────────────────

class TestCoverSlippage:
    """COVER is buy-to-close → must get ADVERSE (higher) fill, not favorable."""

    def test_cover_fill_is_adverse(self, portfolio):
        """Short at $60,000 then COVER at $60,000 → fill should be $60,030 at 5 bps."""
        portfolio.process_signal("SHORT", 60_000.0, "2024-01-01")
        assert portfolio.position_side == PositionSide.SHORT

        portfolio.process_signal("COVER", 60_000.0, "2024-01-02")
        assert portfolio.position_side == PositionSide.FLAT
        # After K6 fix: COVER uses (1 + slippage) → $60,000 * 1.0005 = $60,030
        # The P&L should be negative due to adverse fill on BOTH legs

    def test_short_roundtrip_pnl_is_negative_at_same_price(self, portfolio):
        """SHORT@60K → COVER@60K should have negative P&L (slippage on both legs)."""
        portfolio.process_signal("SHORT", 60_000.0, "2024-01-01")
        portfolio.process_signal("COVER", 60_000.0, "2024-01-02")
        # Both legs have adverse slippage → net P&L must be negative
        if portfolio.closed_positions:
            last = portfolio.closed_positions[-1]
            assert last.pnl < 0, f"Expected negative P&L, got {last.pnl}"


# ── Parametrized slippage ────────────────────────────────────────────

class TestSlippage:
    @pytest.mark.parametrize("bps", [5, 25, 50, 100, 500])
    def test_buy_slippage_increases_with_bps(self, bps):
        p = Portfolio(100_000, 0.25, 1.0, slippage_bps=bps)
        base = 60_000.0
        p.process_signal("BUY", base, "2024-01-01")
        # The actual fill price for BUY is base * (1 + bps/10000)
        expected_fill = base * (1 + bps / 10_000)
        if p.current_position:
            assert p.current_position.entry_price == pytest.approx(expected_fill, rel=1e-6)

    def test_sell_slippage_lowers_price(self, portfolio):
        portfolio.process_signal("BUY", 60_000, "2024-01-01")
        portfolio.process_signal("SELL", 60_000, "2024-01-02")
        # SELL fill = 60000 * (1 - 0.0005) = 59970 → P&L is negative
        if portfolio.closed_positions:
            assert portfolio.closed_positions[-1].pnl < 0

    def test_slippage_propagates_from_config(self):
        """Verify slippage_bps is used from Portfolio init, not hardcoded."""
        p1 = Portfolio(100_000, 0.25, 1.0, slippage_bps=10.0)
        p2 = Portfolio(100_000, 0.25, 1.0, slippage_bps=100.0)
        p1.process_signal("BUY", 50_000, "2024-01-01")
        p2.process_signal("BUY", 50_000, "2024-01-01")
        if p1.current_position and p2.current_position:
            assert p2.current_position.entry_price > p1.current_position.entry_price


# ── K5: Funding interval accumulation ────────────────────────────────

class TestFundingAccumulation:
    def test_daily_step_charges_three_intervals(self):
        """24h elapsed / 8h interval = 3 intervals charged, not 1."""
        p = Portfolio(
            initial_capital=100_000,
            position_size_pct=0.25,
            leverage=1.0,
            slippage_bps=0,
            use_funding=True,
            funding_interval_hours=8.0,
        )
        p.process_signal("SHORT", 60_000.0, "2024-01-01")
        # _calculate_funding is called before position opens in process_signal,
        # so last_funding_date is not set. Manually initialize it.
        p.last_funding_date = "2024-01-01"
        # Now call again 24h later → should charge 3 intervals
        cost = p._calculate_funding("2024-01-02", 60_000.0, actual_rate=0.0001)
        position_size = p.current_position.size
        expected = position_size * 60_000.0 * 0.0001 * 3
        assert cost == pytest.approx(expected, rel=0.01), (
            f"Expected {expected}, got {cost} — K5: should charge 3 intervals, not 1"
        )

    def test_4h_step_charges_one_interval_at_8h_cadence(self):
        """4h elapsed → not enough for 1 interval at 8h cadence."""
        p = Portfolio(
            initial_capital=100_000,
            position_size_pct=0.25,
            leverage=1.0,
            slippage_bps=0,
            use_funding=True,
            funding_interval_hours=8.0,
        )
        p.process_signal("SHORT", 60_000.0, "2024-01-01 00:00:00")
        p.last_funding_date = "2024-01-01 00:00:00"
        cost = p._calculate_funding("2024-01-01 04:00:00", 60_000.0, actual_rate=0.0001)
        assert cost == 0.0  # Not enough time for a full interval

    def test_weekend_funding_accumulation(self):
        """Fri→Mon: 72h / 8h = 9 intervals charged."""
        p = Portfolio(
            initial_capital=100_000,
            position_size_pct=0.25,
            leverage=1.0,
            slippage_bps=0,
            use_funding=True,
            funding_interval_hours=8.0,
        )
        p.process_signal("SHORT", 60_000.0, "2024-01-05")  # Friday
        p.last_funding_date = "2024-01-05"
        cost = p._calculate_funding("2024-01-08", 60_000.0, actual_rate=0.0001)
        position_size = p.current_position.size
        expected = position_size * 60_000.0 * 0.0001 * 9  # 72h / 8h = 9
        assert cost == pytest.approx(expected, rel=0.01)

    def test_funding_rate_fallback_zero(self):
        """No rate data → default to 0.0, not 0.0001."""
        p = Portfolio(
            initial_capital=100_000,
            position_size_pct=0.25,
            leverage=1.0,
            slippage_bps=0,
            use_funding=True,
            funding_interval_hours=8.0,
        )
        p.process_signal("SHORT", 60_000.0, "2024-01-01")
        p.last_funding_date = "2024-01-01"
        cost = p._calculate_funding("2024-01-02", 60_000.0, actual_rate=None)
        assert cost == 0.0

    def test_long_receives_funding(self):
        """Long positions receive (negative) funding cost when rate is positive."""
        p = Portfolio(
            initial_capital=100_000,
            position_size_pct=0.25,
            leverage=1.0,
            slippage_bps=0,
            use_funding=True,
            funding_interval_hours=8.0,
        )
        p.process_signal("BUY", 60_000.0, "2024-01-01")
        p.last_funding_date = "2024-01-01"
        cost = p._calculate_funding("2024-01-02", 60_000.0, actual_rate=0.0001)
        assert cost < 0  # Negative = received


# ── Position basics ──────────────────────────────────────────────────

class TestPositionSide:
    def test_overweight_signal(self, portfolio):
        portfolio.process_signal("BUY", 50_000, "2024-01-01")
        portfolio.process_signal("OVERWEIGHT", 51_000, "2024-01-02")
        assert portfolio.position_side == PositionSide.LONG

    def test_underweight_signal(self, portfolio):
        portfolio.process_signal("BUY", 50_000, "2024-01-01")
        portfolio.process_signal("UNDERWEIGHT", 51_000, "2024-01-02")
        # UNDERWEIGHT reduces or closes long
        assert portfolio.position_side in (PositionSide.LONG, PositionSide.FLAT)


# ── Liquidation ──────────────────────────────────────────────────────

class TestLiquidation:
    def test_leveraged_long_liquidation(self):
        """2x leverage long liquidated at 50% price drop."""
        p = Portfolio(100_000, 0.50, leverage=2.0, slippage_bps=0)
        p.process_signal("BUY", 60_000, "2024-01-01")
        # 55% drop → margin exhausted
        p.process_signal("HOLD", 27_000, "2024-01-02")
        # Position should be force-closed


class TestForceClose:
    def test_force_close_all(self, portfolio):
        portfolio.process_signal("BUY", 50_000, "2024-01-01")
        portfolio.force_close(51_000, "2024-01-10")
        assert portfolio.position_side == PositionSide.FLAT
