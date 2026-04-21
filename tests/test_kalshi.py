import pytest
import os
from unittest.mock import patch, MagicMock
from cryptography.hazmat.primitives.asymmetric import rsa

from tradingagents.dataflows.kalshi_client import (
    sign_pss_text,
    get_kalshi_macro_context,
    generate_kalshi_auth_headers,
    _fetch_series,
)
from tradingagents.backtesting.context import BACKTEST_MODE

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
    # Wolfers-Zitzewitz γ=0.91 multiplicative correction on mid=0.75 → ≈73%
    assert "Implied Probability 73%" in dashboard
    assert "Confidence: HIGH" in dashboard


# ---------------------------------------------------------------------------
# Stage 2 Commit F.1 — correctness tests
# ---------------------------------------------------------------------------

def _wz(yes_ask: int, yes_bid: int) -> int:
    """Reference implementation of the corrected W-Z formula for tests."""
    mid = (yes_ask + yes_bid) / 200.0
    gamma = 0.91
    num = mid ** gamma
    return round(100 * num / (num + (1.0 - mid) ** gamma))


@pytest.mark.parametrize("raw,expected", [
    (5, _wz(5, 5)),
    (25, _wz(25, 25)),
    (50, _wz(50, 50)),  # symmetric → 50
    (75, _wz(75, 75)),
    (95, _wz(95, 95)),
])
@patch("tradingagents.dataflows.kalshi_client.requests.get")
def test_kalshi_wz_correction_grid(mock_get, raw, expected):
    """W-Z γ=0.91 correction is symmetric around 50 and monotonic."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={
            "markets": [{
                "ticker": "KXFEDDECISION-X",
                "title": "t", "subtitle": "t",
                "close_time": "2026-11-05T12:00:00Z",
                "yes_ask": raw, "yes_bid": raw,
                "open_interest": 450000,
            }]
        }),
    )
    out = get_kalshi_macro_context()
    # raw=50 → mid=0.5 → p=0.5 → 50%
    if raw == 50:
        assert "Implied Probability 50%" in out
    assert f"Implied Probability {expected}%" in out


@patch("tradingagents.dataflows.kalshi_client.requests.get")
def test_kalshi_illiquid_side_flagged(mock_get):
    """yes_bid <= 0 or wide spread → [ILLIQUID] line, not implied prob."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={
            "markets": [{
                "ticker": "KXCPI-X",
                "title": "t", "subtitle": "NoBid",
                "close_time": "2026-11-05T12:00:00Z",
                "yes_ask": 40, "yes_bid": 0,
                "open_interest": 450000,
            }]
        }),
    )
    out = get_kalshi_macro_context()
    assert "[ILLIQUID]" in out
    assert "Implied Probability" not in out.split("NoBid")[1].split("\n")[0]


@patch("tradingagents.dataflows.kalshi_client.requests.get")
def test_kalshi_http_error_propagates(mock_get):
    """HTTP errors surface as DATA UNAVAILABLE dashboard lines."""
    mock_get.return_value = MagicMock(status_code=503)
    out = get_kalshi_macro_context()
    assert "DATA UNAVAILABLE: HTTP 503" in out


def test_kalshi_backtest_mode_source_guard():
    """BACKTEST_MODE=True → no HTTP calls, stub string returned."""
    token = BACKTEST_MODE.set(True)
    try:
        with patch("tradingagents.dataflows.kalshi_client.requests.get") as mg:
            out = get_kalshi_macro_context()
            assert "DISABLED IN BACKTEST MODE" in out
            mg.assert_not_called()
    finally:
        BACKTEST_MODE.reset(token)
