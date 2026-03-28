import tradingagents.default_config as default_config
from typing import Dict, Optional

# Use default config but allow it to be overridden
_config: Optional[Dict] = None


def initialize_config():
    """Initialize the configuration with default values."""
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()


def set_config(config: Dict):
    """Update the configuration with custom values."""
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
    _config.update(config)


def get_config() -> Dict:
    """Get the current configuration."""
    if _config is None:
        initialize_config()
    return _config.copy()


def get_vendor_for_asset(category: str, ticker: str) -> str:
    """Get the configured vendor, accounting for asset type.

    For crypto tickers, uses crypto_vendors config. For equities, uses data_vendors.
    """
    from tradingagents.dataflows.asset_detection import detect_asset_type

    config = get_config()
    asset_type = config.get("asset_type", "auto")
    if asset_type == "auto":
        asset_type = detect_asset_type(ticker)

    if asset_type == "crypto":
        crypto_vendors = config.get("crypto_vendors", {})
        if category in crypto_vendors:
            return crypto_vendors[category]

    return config.get("data_vendors", {}).get(category, "yfinance")


# Initialize with default config
initialize_config()
