#!/usr/bin/env python3
"""Unit tests for SignalProcessor JSON parsing."""

import sys
from tradingagents.graph.signal_processing import SignalProcessor

def test_json_parsing():
    """Test SignalProcessor can parse structured JSON signals."""
    
    # Create processor with None LLM (won't be used for JSON parsing tests)
    processor = SignalProcessor(None)
    
    # Test 1: Valid JSON in markdown code block
    json_signal = '''```json
{
  "signal": "SHORT",
  "stop_loss_price": 68500.00,
  "take_profit_price": 60000.00,
  "confidence": 0.75,
  "max_hold_days": 7,
  "reasoning": "Bearish breakdown below support with extreme long positioning."
}
```'''
    
    result = processor.process_signal(json_signal)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result["signal"] == "SHORT", f"Expected SHORT, got {result['signal']}"
    assert result["stop_loss_price"] == 68500.00
    assert result["take_profit_price"] == 60000.00
    assert result["confidence"] == 0.75
    assert result["max_hold_days"] == 7
    print("✓ TEST 1 PASSED: JSON in markdown code block")
    
    # Test 2: Raw JSON without code block
    raw_json = '''{
  "signal": "BUY",
  "stop_loss_price": 62000.00,
  "take_profit_price": 75000.00,
  "confidence": 0.85,
  "max_hold_days": 10,
  "reasoning": "Strong support bounce with bullish divergence."
}'''
    
    result = processor.process_signal(raw_json)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result["signal"] == "BUY"
    assert result["stop_loss_price"] == 62000.00
    print("✓ TEST 2 PASSED: Raw JSON without code block")
    
    # Test 3: JSON embedded in text
    embedded_json = '''Based on my analysis, here is my decision:

{
  "signal": "HOLD",
  "stop_loss_price": 0,
  "take_profit_price": 0,
  "confidence": 0.5,
  "max_hold_days": 3,
  "reasoning": "Waiting for clearer directional signal."
}

This represents a cautious stance given mixed signals.'''
    
    result = processor.process_signal(embedded_json)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result["signal"] == "HOLD"
    print("✓ TEST 3 PASSED: JSON embedded in text")
    
    # Test 4: Malformed JSON - should fallback to word extraction
    # Note: This will make an actual LLM call, so we'll skip it in automated tests
    # freestyle_text = '''**Rating: BUY**
    # 
    # **Executive Summary**: Strong technical setup with bullish momentum...'''
    # 
    # result = processor.process_signal(freestyle_text)
    # assert isinstance(result, str), f"Expected string fallback, got {type(result)}"
    # assert result == "BUY"
    # print("✓ TEST 4 PASSED: Fallback to word extraction")
    
    print("\n✅ All SignalProcessor tests passed!")

if __name__ == "__main__":
    test_json_parsing()
