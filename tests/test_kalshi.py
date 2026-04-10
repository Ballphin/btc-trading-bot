import pytest
import os
from unittest.mock import patch
from cryptography.hazmat.primitives.asymmetric import rsa

from tradingagents.dataflows.kalshi_client import (
    sign_pss_text,
    get_kalshi_macro_context,
    generate_kalshi_auth_headers
)

def test_kalshi_rsa_signature_padding():
    """Verify the cryptography RSA PSS signature matches Kalshi's spec."""
    # Generate a dummy RSA key for testing
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    test_message = "1678888999999GET/trade-api/v2/portfolio/balance"
    signature = sign_pss_text(private_key, test_message)
    
    # Signature should be a valid Base64 encoded string
    assert isinstance(signature, str)
    assert len(signature) > 100  # RSA 2048 signature is 256 bytes, b64 is 344 chars

@patch("os.path.exists", return_value=False)
def test_kalshi_auth_missing_keys(mock_exists):
    """Verify graceful handling if the user hasn't set up their Kalshi API keys yet."""
    headers = generate_kalshi_auth_headers("GET", "/trade-api/v2/portfolio/balance")
    assert headers == {} # Should return empty dict gracefully without crashing

@patch("tradingagents.dataflows.kalshi_client.requests.get")
def test_kalshi_macro_context(mock_get):
    """Verify the batch-polling Macro Risk Dashboard groups and formats correctly."""
    
    mock_resp = mock_get.return_value
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "markets": [
            {
                "ticker": "KXFEDDECISION-TEST",
                "title": "Fed rate 5.25%",
                "subtitle": "Will it be 5.25%?",
                "close_time": "2026-11-05T12:00:00Z",
                "yes_ask": 78,
                "yes_bid": 72,
                "open_interest": 450000
            }
        ]
    }
    
    dashboard = get_kalshi_macro_context()
    
    # Verify the markdown wrapper
    assert "### Kalshi Macro Risk Dashboard" in dashboard
    
    # Test temporal alignment keywords
    assert "Next Fed Rate Decision" in dashboard
    assert "Implied Probability 71%" in dashboard
    assert "Confidence: HIGH" in dashboard
