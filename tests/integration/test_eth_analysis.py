"""Tests for ETH-USD analysis to ensure it doesn't crash on Bitcoin-specific on-chain metrics."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_get_onchain_data_btc():
    """Verify on-chain data works for BTC-USD."""
    from tradingagents.dataflows.crypto_data import get_onchain_data
    
    # Should not raise for BTC
    with patch('tradingagents.dataflows.crypto_data._get') as mock_get:
        mock_client = MagicMock()
        mock_client.get_onchain_summary.return_value = "# Bitcoin On-Chain Report\nMVRV: 2.5"
        mock_get.return_value = mock_client
        
        result = get_onchain_data("BTC-USD", "2026-04-11")
        assert "Bitcoin On-Chain" in result


def test_get_onchain_data_eth():
    """Verify ETH-USD returns appropriate message instead of crashing."""
    from tradingagents.dataflows.crypto_data import get_onchain_data
    
    # For ETH, should return a message about on-chain data not being available
    result = get_onchain_data("ETH-USD", "2026-04-11")
    
    assert "On-chain metrics" in result
    assert "only available for Bitcoin" in result
    assert "ETH" in result
    assert "do not apply to ETH-USD" in result


def test_get_onchain_data_various_tickers():
    """Verify various non-BTC tickers get appropriate messages."""
    from tradingagents.dataflows.crypto_data import get_onchain_data
    
    non_btc_tickers = ["ETH-USD", "ETH-USDT", "SOL-USD", "ADA-USD", "BTC-USDT"]
    
    for ticker in non_btc_tickers:
        result = get_onchain_data(ticker, "2026-04-11")
        
        base = ticker.replace("-USD", "").replace("USDT", "").upper()
        if base == "BTC":
            # BTC tickers should proceed to fetch data
            continue
        else:
            # Non-BTC should get message
            assert "only available for Bitcoin" in result, f"Failed for {ticker}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
