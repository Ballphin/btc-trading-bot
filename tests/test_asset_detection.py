"""Tests for asset type detection."""

import pytest
from tradingagents.dataflows.asset_detection import detect_asset_type, is_crypto


class TestDetectAssetType:
    """Test detect_asset_type function."""

    def test_crypto_with_usd_suffix(self):
        assert detect_asset_type("BTC-USD") == "crypto"
        assert detect_asset_type("ETH-USD") == "crypto"
        assert detect_asset_type("SOL-USD") == "crypto"

    def test_crypto_with_usdt_suffix(self):
        assert detect_asset_type("BTCUSDT") == "crypto"
        assert detect_asset_type("ETHUSDT") == "crypto"

    def test_crypto_known_symbols(self):
        assert detect_asset_type("BTC") == "crypto"
        assert detect_asset_type("ETH") == "crypto"
        assert detect_asset_type("SOL") == "crypto"
        assert detect_asset_type("DOGE") == "crypto"
        assert detect_asset_type("LINK") == "crypto"
        assert detect_asset_type("AVAX") == "crypto"

    def test_equity_symbols(self):
        assert detect_asset_type("NVDA") == "equity"
        assert detect_asset_type("AAPL") == "equity"
        assert detect_asset_type("MSFT") == "equity"
        assert detect_asset_type("GOOGL") == "equity"
        assert detect_asset_type("TSLA") == "equity"

    def test_equity_with_exchange_suffix(self):
        assert detect_asset_type("RY.TO") == "equity"
        assert detect_asset_type("HSBA.L") == "equity"

    def test_case_insensitive(self):
        assert detect_asset_type("btc-usd") == "crypto"
        assert detect_asset_type("Btc-Usd") == "crypto"
        assert detect_asset_type("nvda") == "equity"

    def test_is_crypto_convenience(self):
        assert is_crypto("BTC-USD") is True
        assert is_crypto("ETH-USD") is True
        assert is_crypto("NVDA") is False
        assert is_crypto("AAPL") is False
