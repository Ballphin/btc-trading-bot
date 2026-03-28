"""Detect whether a ticker is equity or crypto."""

CRYPTO_SUFFIXES = ("-USD", "-USDT", "USDT", "-BUSD")

KNOWN_CRYPTO = {
    "BTC", "ETH", "SOL", "DOGE", "DOT", "LINK", "AVAX", "XRP", "ADA",
    "MATIC", "NEAR", "UNI", "AAVE", "ARB", "OP", "APT", "SUI",
    "BNB", "LTC", "BCH", "SHIB", "ALGO", "FIL", "ATOM", "HBAR",
    "ICP", "STX", "INJ", "TON", "PEPE", "WIF", "RENDER", "FET",
}


def detect_asset_type(ticker: str) -> str:
    """
    Determine whether a ticker represents a crypto asset or equity.

    Returns:
        'crypto' or 'equity'
    """
    upper = ticker.upper()
    if any(upper.endswith(s) for s in CRYPTO_SUFFIXES):
        return "crypto"
    base = upper.split("-")[0]
    if base in KNOWN_CRYPTO:
        return "crypto"
    return "equity"


def is_crypto(ticker: str) -> bool:
    """Convenience check: returns True if the ticker is a crypto asset."""
    return detect_asset_type(ticker) == "crypto"
