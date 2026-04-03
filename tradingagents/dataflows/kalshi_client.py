import os
import time
import base64
import requests
from datetime import datetime
from itertools import groupby
from concurrent.futures import ThreadPoolExecutor

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

KALSHI_API_BASE = "https://trading-api.kalshi.com/trade-api/v2"

# Master list of curated macro intelligence markers
KALSHI_SERIES_TICKERS = ["KXFEDDECISION", "KXCPI", "KXRECESS"]


def load_private_key_from_file(file_path: str):
    with open(file_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )
    return private_key


def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    """Implement RSA PKCS1 PSS signature per Kalshi's v2 spec."""
    message = text.encode('utf-8')
    try:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    except InvalidSignature as e:
        raise ValueError("RSA sign PSS failed") from e


def generate_kalshi_auth_headers(method: str, path: str) -> dict:
    """Generate headers to securely sign requests.
    Required for executing trades or fetching portfolio balance."""
    api_key = os.getenv("KALSHI_API_KEY")
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not api_key or not key_path or not os.path.exists(key_path):
        return {}

    current_time_ms = str(int(time.time() * 1000))
    msg_string = current_time_ms + method.upper() + path
    priv_key = load_private_key_from_file(key_path)
    sig = sign_pss_text(priv_key, msg_string)

    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": current_time_ms,
        "Content-Type": "application/json"
    }


def _fetch_series(series_ticker: str) -> tuple:
    url = f"{KALSHI_API_BASE}/markets?series_ticker={series_ticker}&status=open"
    try:
        # Public data endpoint doesn't require rigid authentication
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return series_ticker, r.json().get("markets", [])
    except Exception as e:
        pass
    return series_ticker, []


def get_kalshi_macro_context() -> str:
    """
    Proactively batch-polls macro series from Kalshi.
    Implements Data Cleanup, Temporal Selection, and outputs a united Macro Risk Dashboard.
    """
    # Fetch all endpoints concurrently
    with ThreadPoolExecutor(max_workers=len(KALSHI_SERIES_TICKERS)) as executor:
        results = list(executor.map(_fetch_series, KALSHI_SERIES_TICKERS))

    dashboard_lines = ["### Kalshi Macro Risk Dashboard\n"]

    for idx, (series, raw_markets) in enumerate(results, 1):
        # 1. Parse and Filter raw responses
        active_markets = [m for m in raw_markets if m.get('close_time')]
        if not active_markets:
            continue

        # Mathematical sort to ensure we extract the immediate upcoming event exactly
        active_markets.sort(key=lambda x: x['close_time'])

        groups = []
        for k, g in groupby(active_markets, key=lambda x: x['close_time']):
            groups.append((k, list(g)))

        # 2. Temporal Selection: Choose ONLY the closest deadline group
        immediate_close_time, target_markets = groups[0]

        # Aggregate liquidity to assess reliability
        # Open interest is denominated in cents, convert to dollars implicitly or rely on direct value
        total_open_interest = sum(m.get('open_interest', 0) for m in target_markets)

        series_title = series
        if series == "KXFEDDECISION":
            series_title = "Next Fed Rate Decision"
        elif series == "KXCPI":
            series_title = "Upcoming CPI Output"
        elif series == "KXRECESS":
            series_title = "US Recession"

        dt_format = immediate_close_time.replace('Z', '+00:00')
        try:
            close_dt = datetime.fromisoformat(dt_format)
            date_str = close_dt.strftime('%b %Y')
        except ValueError:
            date_str = immediate_close_time[:7]

        dashboard_lines.append(f"**{idx}. {series_title}: {date_str}**")

        # Liquidity Check Threshold
        if total_open_interest < 10000:
            dashboard_lines.append("- [DATA UNAVAILABLE: AGGREGATE LIQUIDITY < $10k]\n")
            continue

        # 3. Market Extrapolation for bracket distributions
        for m in target_markets:
            title = m.get('title', '')
            subtitle = m.get('subtitle', title)
            yes_ask = m.get('yes_ask', 0)
            yes_bid = m.get('yes_bid', 0)
            
            # Kalshi asks/bids are in cents. (e.g., 65 cents = 65% prob)
            # They max out at 100, wait, depending on API version maybe 1.0 or 100.
            # v2 market endpoints generally return yes_ask as an integer of cents (e.g., 65)
            implied_prob = yes_ask

            spread = yes_ask - yes_bid
            if spread > 15:  # Spread larger than 15 cents (15%)
                dashboard_lines.append(f"- *{subtitle}*: [WARNING: WIDE SPREAD > 15%]")
            else:
                dashboard_lines.append(f"- *{subtitle}*: Implied Probability {implied_prob:.0f}%")

        confidence = "HIGH" if total_open_interest > 250000 else ("MEDIUM" if total_open_interest > 50000 else "LOW")
        dashboard_lines.append(f"- Aggregate Event Open Interest: ${total_open_interest:,.0f} (Confidence: {confidence})\n")

    if len(dashboard_lines) == 1:
        return "Kalshi Macro Risk Dashboard unavailable: Connection or liquidity failure."

    return "\n".join(dashboard_lines)
