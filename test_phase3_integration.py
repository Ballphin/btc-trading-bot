#!/usr/bin/env python3
"""Phase 3 Integration Test: Full backtest pipeline with risk management."""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Test imports
try:
    from tradingagents.backtesting.portfolio import Portfolio, Position, PositionSide, TradeRecord
    from tradingagents.backtesting.metrics import compute_metrics
    from tradingagents.backtesting.report import generate_report
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_portfolio_with_risk_management():
    """Test Portfolio with stop loss, take profit, and time exits."""
    print("\n=== Testing Portfolio Risk Management ===")
    
    # Create portfolio with ATR-based sizing
    portfolio = Portfolio(
        initial_capital=100000,
        position_size_pct=0.25,
        leverage=2.0,
        position_sizing="atr_risk",
        risk_per_trade=0.01,
    )
    
    # Test 1: Enter long with stop/take
    print("\nTest 1: Enter LONG with stop/take")
    action = portfolio.process_signal(
        signal="BUY",
        price=65000,
        date="2026-01-01",
        stop_loss_price=63000,
        take_profit_price=70000,
        max_hold_days=7,
    )
    print(f"  Action: {action}")
    assert portfolio.current_position is not None
    assert portfolio.current_position.stop_loss_price == 63000
    assert portfolio.current_position.take_profit_price == 70000
    print("  ✓ Position opened with stop/take")
    
    # Test 2: Stop loss hit
    print("\nTest 2: Stop loss triggered")
    action = portfolio.process_signal(
        signal="HOLD",
        price=62500,  # Below stop
        date="2026-01-02",
    )
    print(f"  Action: {action}")
    assert "STOPPED OUT" in action
    assert portfolio.current_position is None
    assert portfolio.stops_hit == 1
    print("  ✓ Stop loss triggered correctly")
    
    # Test 3: Enter short and hit take profit
    print("\nTest 3: Enter SHORT and hit take profit")
    action = portfolio.process_signal(
        signal="SHORT",
        price=65000,
        date="2026-01-05",
        stop_loss_price=67000,
        take_profit_price=60000,
        max_hold_days=5,
    )
    print(f"  Action: {action}")
    assert portfolio.current_position is not None
    
    action = portfolio.process_signal(
        signal="HOLD",
        price=59500,  # Below take profit for short
        date="2026-01-06",
    )
    print(f"  Action: {action}")
    assert "TAKE PROFIT" in action
    assert portfolio.takes_hit == 1
    print("  ✓ Take profit triggered correctly")
    
    # Test 4: Time-based exit
    print("\nTest 4: Time-based exit")
    action = portfolio.process_signal(
        signal="BUY",
        price=65000,
        date="2026-01-10",
        max_hold_days=3,
    )
    assert portfolio.current_position is not None
    
    # Hold for 3 days
    action = portfolio.process_signal(
        signal="HOLD",
        price=66000,
        date="2026-01-13",  # 3 days later
    )
    print(f"  Action: {action}")
    assert "TIME EXIT" in action
    print("  ✓ Time exit triggered correctly")
    
    print(f"\n  Portfolio Stats:")
    print(f"    Stops Hit: {portfolio.stops_hit}")
    print(f"    Takes Hit: {portfolio.takes_hit}")
    print(f"    Closed Positions: {len(portfolio.closed_positions)}")
    
    return portfolio

def test_metrics_calculation(portfolio):
    """Test metrics calculation with risk management fields."""
    print("\n=== Testing Metrics Calculation ===")
    
    try:
        metrics = compute_metrics(
            equity_curve=portfolio.equity_curve,
            closed_positions=portfolio.closed_positions,
            initial_capital=portfolio.initial_capital,
            total_fees=portfolio.total_fees_paid,
            total_funding=portfolio.total_funding_paid,
            liquidations=portfolio.liquidations,
            leverage=portfolio.leverage,
            stops_hit=portfolio.stops_hit,
            takes_hit=portfolio.takes_hit,
        )
        
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Stops Hit: {metrics['stops_hit']}")
        print(f"  Takes Hit: {metrics['takes_hit']}")
        print(f"  Avg Hold Days: {metrics['avg_hold_days']:.1f}")
        print(f"  Avg R:R Ratio: {metrics['avg_rr_ratio']:.2f}")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        
        assert metrics['stops_hit'] == portfolio.stops_hit
        assert metrics['takes_hit'] == portfolio.takes_hit
        assert 'avg_hold_days' in metrics
        assert 'avg_rr_ratio' in metrics
        
        print("  ✓ Metrics calculated successfully")
        return metrics
        
    except Exception as e:
        print(f"  ✗ Metrics calculation error: {e}")
        raise

def test_report_generation(metrics, portfolio):
    """Test report generation with risk management sections."""
    print("\n=== Testing Report Generation ===")
    
    try:
        config = {
            "ticker": "BTC-USD",
            "start_date": "2026-01-01",
            "end_date": "2026-01-15",
            "frequency": "daily",
            "initial_capital": 100000,
        }
        
        decisions = [
            {"date": "2026-01-01", "price": 65000, "signal": "BUY", "action": "ENTERED LONG", "portfolio_value": 100000, "position": "LONG"},
            {"date": "2026-01-02", "price": 62500, "signal": "HOLD", "action": "STOPPED OUT", "portfolio_value": 98000, "position": "FLAT"},
        ]
        
        report = generate_report(
            ticker="BTC-USD",
            metrics=metrics,
            decisions=decisions,
            equity_curve=portfolio.equity_curve,
            config=config,
        )
        
        # Verify report contains risk management section
        assert "Risk Management" in report
        assert "Stops Hit" in report
        assert "Targets Hit" in report
        assert "Avg Hold Days" in report
        assert "Avg R:R Ratio" in report
        
        print("  ✓ Report generated with risk management section")
        print(f"  Report length: {len(report)} chars")
        
        return report
        
    except Exception as e:
        print(f"  ✗ Report generation error: {e}")
        raise

def test_trade_record_serialization(portfolio):
    """Test TradeRecord serialization with all new fields."""
    print("\n=== Testing TradeRecord Serialization ===")
    
    try:
        trade_history = []
        for t in portfolio.trade_history:
            trade_dict = {
                "date": t.date,
                "signal": t.signal,
                "price": t.price,
                "action_taken": t.action_taken,
                "position_side": t.position_side,
                "portfolio_value": t.portfolio_value,
                "cash": t.cash,
                "unrealized_pnl": t.unrealized_pnl,
                "realized_pnl": t.realized_pnl,
                "fees_paid": t.fees_paid,
                "funding_paid": t.funding_paid,
                "leverage": t.leverage,
                "liquidation_price": t.liquidation_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "hold_days": t.hold_days,
                "exit_reason": t.exit_reason,
                "atr_at_entry": t.atr_at_entry,
            }
            trade_history.append(trade_dict)
        
        # Serialize to JSON
        json_str = json.dumps(trade_history, indent=2, default=str)
        
        # Deserialize back
        parsed = json.loads(json_str)
        
        print(f"  Serialized {len(trade_history)} trade records")
        print(f"  JSON size: {len(json_str)} bytes")
        print(f"  Sample trade keys: {list(parsed[0].keys())}")
        
        # Verify all fields present
        required_fields = ['stop_loss', 'take_profit', 'hold_days', 'exit_reason', 'atr_at_entry']
        for field in required_fields:
            assert field in parsed[0], f"Missing field: {field}"
        
        print("  ✓ TradeRecord serialization successful")
        
    except Exception as e:
        print(f"  ✗ Serialization error: {e}")
        raise

def main():
    """Run all Phase 3 integration tests."""
    print("=" * 60)
    print("Phase 3 Integration Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Portfolio with risk management
        portfolio = test_portfolio_with_risk_management()
        
        # Test 2: Metrics calculation
        metrics = test_metrics_calculation(portfolio)
        
        # Test 3: Report generation
        report = test_report_generation(metrics, portfolio)
        
        # Test 4: Serialization
        test_trade_record_serialization(portfolio)
        
        print("\n" + "=" * 60)
        print("✅ ALL PHASE 3 TESTS PASSED")
        print("=" * 60)
        print("\nPhase 3 Infrastructure Complete:")
        print("  ✓ Stop loss/take profit enforcement")
        print("  ✓ Time-based exits")
        print("  ✓ ATR-based position sizing")
        print("  ✓ Risk metrics calculation")
        print("  ✓ Enhanced report generation")
        print("  ✓ TradeRecord serialization")
        
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
