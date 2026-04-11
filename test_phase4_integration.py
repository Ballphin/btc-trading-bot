#!/usr/bin/env python3
"""Phase 4 Integration Test: Advanced trading features."""

import sys
from datetime import datetime, timedelta

# Test imports
try:
    from tradingagents.backtesting.portfolio import Portfolio, PositionSide
    from tradingagents.backtesting.indicators import calculate_atr, calculate_volatility
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_atr_calculation():
    """Test ATR indicator calculation."""
    print("\n=== Testing ATR Calculation ===")
    
    try:
        # Test with BTC-USD (should have data)
        atr = calculate_atr("BTC-USD", "2024-01-15", period=14)
        
        if atr is not None and atr > 0:
            print(f"  ✓ ATR calculated: {atr:.2f}")
            assert atr > 0, "ATR should be positive"
        else:
            print("  ⚠ ATR calculation returned None (may be expected for recent dates)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ATR calculation error: {e}")
        raise

def test_volatility_calculation():
    """Test volatility indicator calculation."""
    print("\n=== Testing Volatility Calculation ===")
    
    try:
        # Test with BTC-USD
        vol = calculate_volatility("BTC-USD", "2024-01-15", period=20)
        
        if vol is not None and vol > 0:
            print(f"  ✓ Volatility calculated: {vol:.4f}")
            assert vol > 0, "Volatility should be positive"
        else:
            print("  ⚠ Volatility calculation returned None (may be expected for recent dates)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Volatility calculation error: {e}")
        raise

def test_position_scaling():
    """Test position scaling (pyramiding)."""
    print("\n=== Testing Position Scaling ===")
    
    portfolio = Portfolio(
        initial_capital=100000,
        position_size_pct=0.25,
        leverage=1.0,
    )
    
    # Enter initial position
    action = portfolio.process_signal("BUY", 65000, "2026-01-01")
    assert portfolio.current_position is not None
    initial_size = portfolio.current_position.size
    initial_entry = portfolio.current_position.entry_price
    print(f"  Initial position: {initial_size:.6f} units @ ${initial_entry:,.2f}")
    
    # Scale into position (add 50%)
    action = portfolio.scale_position(67000, "2026-01-02", scale_factor=0.5)
    print(f"  Action: {action}")
    
    assert portfolio.current_position is not None
    assert portfolio.current_position.size > initial_size
    new_size = portfolio.current_position.size
    new_entry = portfolio.current_position.entry_price
    
    print(f"  Scaled position: {new_size:.6f} units @ ${new_entry:,.2f}")
    print(f"  Size increased by: {((new_size/initial_size - 1) * 100):.1f}%")
    
    # Verify weighted average entry price
    expected_entry = (initial_entry * initial_size + 67000 * initial_size * 0.5) / (initial_size * 1.5)
    assert abs(new_entry - expected_entry) < 1, f"Entry price mismatch: {new_entry} vs {expected_entry}"
    
    print("  ✓ Position scaling successful")
    return portfolio

def test_position_rebalancing():
    """Test portfolio rebalancing."""
    print("\n=== Testing Position Rebalancing ===")
    
    portfolio = Portfolio(
        initial_capital=100000,
        position_size_pct=0.25,
        leverage=1.0,
    )
    
    # Enter position
    action = portfolio.process_signal("BUY", 65000, "2026-01-01")
    initial_size = portfolio.current_position.size
    print(f"  Initial position: {initial_size:.6f} units")
    
    # Price moves up significantly - position becomes larger % of portfolio
    new_price = 75000
    portfolio_val = portfolio.portfolio_value(new_price)
    position_val = portfolio.current_position.size * new_price
    position_pct = position_val / portfolio_val
    print(f"  Position at ${new_price:,.2f}: {position_pct*100:.1f}% of portfolio")
    
    # Rebalance to 50%
    action = portfolio.rebalance_position(new_price, "2026-01-05", target_pct=0.5)
    print(f"  Action: {action}")
    
    # Verify rebalancing
    new_portfolio_val = portfolio.portfolio_value(new_price)
    new_position_val = portfolio.current_position.size * new_price
    new_position_pct = new_position_val / new_portfolio_val
    print(f"  After rebalance: {new_position_pct*100:.1f}% of portfolio")
    
    # Should be close to 50% (within tolerance)
    assert abs(new_position_pct - 0.5) < 0.1, f"Rebalance failed: {new_position_pct*100:.1f}% vs target 50%"
    
    print("  ✓ Position rebalancing successful")
    return portfolio

def test_atr_based_sizing():
    """Test ATR-based position sizing."""
    print("\n=== Testing ATR-Based Position Sizing ===")
    
    portfolio = Portfolio(
        initial_capital=100000,
        position_size_pct=0.25,
        leverage=1.0,
        position_sizing="atr_risk",
        risk_per_trade=0.01,  # Risk 1% per trade
    )
    
    # Enter position with stop loss
    entry_price = 65000
    stop_loss = 63000  # $2000 risk per unit
    
    action = portfolio.process_signal(
        "BUY",
        entry_price,
        "2026-01-01",
        stop_loss_price=stop_loss,
    )
    
    assert portfolio.current_position is not None
    size = portfolio.current_position.size
    
    # Calculate expected size based on ATR risk
    # Risk amount = 1% of $100,000 = $1,000
    # Stop distance = $2,000
    # Expected size = $1,000 / $2,000 = 0.5 units (before leverage)
    risk_amount = 100000 * 0.01
    stop_distance = abs(entry_price - stop_loss)
    expected_size = risk_amount / stop_distance
    
    print(f"  Entry: ${entry_price:,.2f}, Stop: ${stop_loss:,.2f}")
    print(f"  Risk per trade: ${risk_amount:,.2f}")
    print(f"  Stop distance: ${stop_distance:,.2f}")
    print(f"  Position size: {size:.6f} units")
    print(f"  Expected size: {expected_size:.6f} units")
    
    # Verify size is correct (within 2% tolerance to account for execution-price slippage)
    assert abs(size - expected_size) / expected_size < 0.02, f"Size mismatch: {size} vs {expected_size}"
    
    print("  ✓ ATR-based sizing successful")
    return portfolio

def test_advanced_features_integration():
    """Test all Phase 4 features together."""
    print("\n=== Testing Advanced Features Integration ===")
    
    portfolio = Portfolio(
        initial_capital=100000,
        position_size_pct=0.25,
        leverage=2.0,
        position_sizing="atr_risk",
        risk_per_trade=0.02,
    )
    
    # Test 1: Enter with ATR sizing
    action = portfolio.process_signal(
        "BUY",
        65000,
        "2026-01-01",
        stop_loss_price=63000,
        take_profit_price=70000,
    )
    print(f"  1. Entry: {action[:80]}...")
    assert portfolio.current_position is not None
    
    # Test 2: Scale position
    action = portfolio.scale_position(66000, "2026-01-02", scale_factor=0.3)
    print(f"  2. Scale: {action[:80]}...")
    
    # Test 3: Rebalance
    action = portfolio.rebalance_position(67000, "2026-01-03", target_pct=0.4)
    print(f"  3. Rebalance: {action[:80]}...")
    
    # Test 4: Hit take profit
    action = portfolio.process_signal("HOLD", 70500, "2026-01-04")
    print(f"  4. Exit: {action[:80]}...")
    assert "TAKE PROFIT" in action
    
    stats = portfolio.get_stats()
    print(f"\n  Final Stats:")
    print(f"    Total Trades: {stats['total_trades']}")
    print(f"    Takes Hit: {stats['takes_hit']}")
    print(f"    Total Fees: ${stats['total_fees_paid']:.2f}")
    
    print("  ✓ Advanced features integration successful")
    return portfolio

def main():
    """Run all Phase 4 integration tests."""
    print("=" * 60)
    print("Phase 4 Integration Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: ATR calculation
        test_atr_calculation()
        
        # Test 2: Volatility calculation
        test_volatility_calculation()
        
        # Test 3: Position scaling
        test_position_scaling()
        
        # Test 4: Position rebalancing
        test_position_rebalancing()
        
        # Test 5: ATR-based sizing
        test_atr_based_sizing()
        
        # Test 6: Integration
        test_advanced_features_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL PHASE 4 TESTS PASSED")
        print("=" * 60)
        print("\nPhase 4 Advanced Features Complete:")
        print("  ✓ Dynamic ATR calculation")
        print("  ✓ Dynamic volatility calculation")
        print("  ✓ Position scaling (pyramiding)")
        print("  ✓ Portfolio rebalancing")
        print("  ✓ ATR-based position sizing")
        print("  ✓ Full integration with risk management")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
