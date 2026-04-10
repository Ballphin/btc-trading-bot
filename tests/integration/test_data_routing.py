"""Integration tests for data routing — vendor dispatch, crypto vs equity paths."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.dataflows.interface import route_to_vendor, VENDOR_METHODS
from tradingagents.dataflows.asset_detection import is_crypto, detect_asset_type


# ── Asset detection ──────────────────────────────────────────────────

class TestAssetDetection:
    def test_btc_usd_is_crypto(self):
        assert is_crypto("BTC-USD") is True

    def test_eth_usd_is_crypto(self):
        assert is_crypto("ETH-USD") is True

    def test_aapl_not_crypto(self):
        assert is_crypto("AAPL") is False

    def test_spy_not_crypto(self):
        assert is_crypto("SPY") is False

    def test_btcusdt_is_crypto(self):
        assert is_crypto("BTCUSDT") is True


class TestDetectAssetType:
    def test_btc_usd(self):
        assert detect_asset_type("BTC-USD") == "crypto"

    def test_aapl(self):
        assert detect_asset_type("AAPL") == "equity"

    def test_sol_usd(self):
        assert detect_asset_type("SOL-USD") == "crypto"


# ── Vendor methods registry ──────────────────────────────────────────

class TestVendorRegistry:
    def test_has_core_methods(self):
        """Core data methods should be registered."""
        expected = ["get_stock_data", "get_fundamentals", "get_news"]
        for m in expected:
            assert m in VENDOR_METHODS, f"Missing method: {m}"

    def test_crypto_methods_present(self):
        """Crypto-specific methods should be registered."""
        crypto_methods = ["get_derivatives_data", "get_intraday_data", "get_realtime_snapshot"]
        for m in crypto_methods:
            assert m in VENDOR_METHODS, f"Missing crypto method: {m}"

    def test_all_vendors_callable(self):
        """All registered vendor methods should be callable dicts."""
        for vendor, methods in VENDOR_METHODS.items():
            assert isinstance(methods, dict), f"{vendor} methods is not a dict"
            for method_name, fn in methods.items():
                assert callable(fn), f"{vendor}.{method_name} is not callable"


# ── route_to_vendor dispatch ─────────────────────────────────────────

class TestRouteToVendor:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            route_to_vendor("nonexistent_method_xyz", "AAPL", "2024-01-15")
