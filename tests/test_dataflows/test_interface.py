"""Tests for interface.py — vendor routing, category resolution, fallback chains."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.dataflows.interface import (
    get_category_for_method,
    get_vendor,
    route_to_vendor,
    VENDOR_METHODS,
    TOOLS_CATEGORIES,
)


class TestGetCategoryForMethod:
    def test_known_method(self):
        cat = get_category_for_method("get_stock_data")
        assert cat == "core_stock_apis"

    def test_derivatives(self):
        cat = get_category_for_method("get_derivatives_data")
        assert cat == "derivatives_data"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_category_for_method("nonexistent_method")


class TestGetVendor:
    @patch("tradingagents.dataflows.interface.get_config")
    def test_category_level(self, mock_config):
        mock_config.return_value = {
            "data_vendors": {"core_stock_apis": "yfinance"},
            "tool_vendors": {},
        }
        vendor = get_vendor("core_stock_apis")
        assert vendor == "yfinance"

    @patch("tradingagents.dataflows.interface.get_config")
    def test_tool_level_override(self, mock_config):
        mock_config.return_value = {
            "data_vendors": {"core_stock_apis": "yfinance"},
            "tool_vendors": {"get_stock_data": "alpha_vantage"},
        }
        vendor = get_vendor("core_stock_apis", method="get_stock_data")
        assert vendor == "alpha_vantage"

    @patch("tradingagents.dataflows.interface.get_config")
    def test_missing_category_returns_default(self, mock_config):
        mock_config.return_value = {"data_vendors": {}, "tool_vendors": {}}
        vendor = get_vendor("some_unknown_category")
        assert vendor == "default"


class TestRouteToVendor:
    @patch("tradingagents.dataflows.interface.get_config")
    def test_routes_to_correct_vendor(self, mock_config):
        mock_config.return_value = {
            "data_vendors": {"derivatives_data": "crypto"},
            "tool_vendors": {},
        }
        mock_impl = MagicMock(return_value="derivatives result")
        with patch.dict(VENDOR_METHODS, {"get_derivatives_data": {"crypto": mock_impl}}):
            result = route_to_vendor("get_derivatives_data", "BTC-USD", "2024-01-01")
        assert result == "derivatives result"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="not found"):
            route_to_vendor("totally_fake_method")

    @patch("tradingagents.dataflows.interface.get_config")
    def test_fallback_chain(self, mock_config):
        mock_config.return_value = {
            "data_vendors": {"core_stock_apis": "alpha_vantage"},
            "tool_vendors": {},
        }
        from tradingagents.dataflows.alpha_vantage_common import AlphaVantageRateLimitError

        mock_av = MagicMock(side_effect=AlphaVantageRateLimitError("rate limited"))
        mock_yf = MagicMock(return_value="yfinance result")

        with patch.dict(
            VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": mock_av, "yfinance": mock_yf}},
        ):
            result = route_to_vendor("get_stock_data", "AAPL", "2024-01-01", "2024-06-01")
        assert result == "yfinance result"
        assert mock_av.called
        assert mock_yf.called


class TestVendorMethodsCompleteness:
    def test_all_tools_have_vendor_methods(self):
        """Every tool listed in TOOLS_CATEGORIES must exist in VENDOR_METHODS."""
        for category, info in TOOLS_CATEGORIES.items():
            for tool in info["tools"]:
                assert tool in VENDOR_METHODS, (
                    f"Tool '{tool}' in category '{category}' missing from VENDOR_METHODS"
                )

    def test_vendor_methods_have_at_least_one_impl(self):
        for method, vendors in VENDOR_METHODS.items():
            assert len(vendors) >= 1, f"Method '{method}' has no vendor implementations"
