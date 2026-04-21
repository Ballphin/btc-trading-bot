import os
import time
import base64
import logging
import functools
import requests
from datetime import datetime
from itertools import groupby
from concurrent.futures import ThreadPoolExecutor

from tradingagents.backtesting.context import BACKTEST_MODE

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

KALSHI_API_BASE = "https://trading-api.kalshi.com/trade-api/v2"

# Stage 2 Commit K — curated macro series. Crypto markets deliberately
# excluded (typical OI $5-30k vs Polymarket's $8M); crypto prediction
# markets routed through polymarket_client instead.
KALSHI_SERIES_TICKERS = [
    "KXFEDDECISION",  # FOMC rate decision
    "KXCPI",          # Monthly CPI
    "KXPCE",          # PCE inflation
    "KXJOBS",         # Non-farm payrolls
    "KXGDP",          # Quarterly GDP
    "KXUNEMP",        # Unemployment rate
    "KXRECESS",       # Recession probability
    "KXFEDCHAIR",     # Fed Chair confirmation / transitions
    "KXDEBT",         # Debt ceiling
    "KXGEOPOL",       # Geopolitical (generic basket)
]

# Pretty titles for the macro dashboard — keyed on series ticker.
KALSHI_SERIES_TITLES = {
    "KXFEDDECISION": "Next Fed Rate Decision",
    "KXCPI": "Upcoming CPI Output",
    "KXPCE": "Upcoming PCE Inflation",
    "KXJOBS": "Non-farm Payrolls",
    "KXGDP": "Quarterly GDP",
    "KXUNEMP": "Unemployment Rate",
    "KXRECESS": "US Recession",
    "KXFEDCHAIR": "Fed Chair Transition",
    "KXDEBT": "Debt Ceiling",
    "KXGEOPOL": "Geopolitical Risk",
}

# Daily snapshot directory for agent-output reproducibility.
KALSHI_SNAPSHOT_DIR = "results/kalshi_snapshots"


@functools.lru_cache(maxsize=1)
def load_private_key_from_file(file_path: str):
    """Load and cache RSA private key from PEM file (singleton)."""
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
    """Fetch a single Kalshi series with proper authentication.

    Returns (series_ticker, markets, error_str). error_str is None on success
    so callers can distinguish outage from legitimate absence of markets.
    """
    path = f"/trade-api/v2/markets?series_ticker={series_ticker}&status=open"
    url = KALSHI_API_BASE + f"/markets?series_ticker={series_ticker}&status=open"
    try:
        headers = generate_kalshi_auth_headers("GET", path)
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return series_ticker, r.json().get("markets", []), None
        else:
            logger.warning(f"Kalshi API returned {r.status_code} for {series_ticker}")
            return series_ticker, [], f"HTTP {r.status_code}"
    except Exception as e:
        logger.warning(f"Kalshi fetch failed for {series_ticker}: {e}")
        return series_ticker, [], f"{type(e).__name__}"


def get_kalshi_macro_context() -> str:
    """
    Proactively batch-polls macro series from Kalshi.
    Implements Data Cleanup, Temporal Selection, and outputs a united Macro Risk Dashboard.
    """
    # Source-level backtest guard — prevents any caller (tool wrapper or direct)
    # from leaking live prediction-market data into a historical simulation.
    if BACKTEST_MODE.get():
        return "[KALSHI: DISABLED IN BACKTEST MODE]"

    # Fetch all endpoints concurrently
    with ThreadPoolExecutor(max_workers=len(KALSHI_SERIES_TICKERS)) as executor:
        results = list(executor.map(_fetch_series, KALSHI_SERIES_TICKERS))

    dashboard_lines = ["### Kalshi Macro Risk Dashboard\n"]

    for idx, (series, raw_markets, error_str) in enumerate(results, 1):
        if error_str is not None:
            dashboard_lines.append(
                f"**{idx}. {series}**: [DATA UNAVAILABLE: {error_str}]\n"
            )
            continue
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

        series_title = KALSHI_SERIES_TITLES.get(series, series)

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
        # Stage 2 Commit K — explicit low-confidence flag for the
        # $10-50k OI range (SQR HIGH #14). Don't silently drop — agents
        # should know the signal is weak and reason accordingly.
        if total_open_interest < 50000:
            dashboard_lines.append(
                f"- [LOW CONFIDENCE: OI ${total_open_interest:,.0f} < $50k — "
                f"implied probabilities noisy]"
            )

        # 3. Market Extrapolation for bracket distributions
        for m in target_markets:
            title = m.get('title', '')
            subtitle = m.get('subtitle', title)
            yes_ask = m.get('yes_ask', 0)
            yes_bid = m.get('yes_bid', 0)

            # Unit-check: Kalshi values are cents in [0, 100]. Guards against
            # future API format shifts (e.g., dollar-denominated quotes).
            assert 0 <= yes_ask <= 100, f"yes_ask out of cents range: {yes_ask}"
            assert 0 <= yes_bid <= 100, f"yes_bid out of cents range: {yes_bid}"

            spread = yes_ask - yes_bid
            if yes_bid <= 0 or spread > 15:
                dashboard_lines.append(
                    f"- *{subtitle}*: raw ask {yes_ask}c [ILLIQUID]"
                )
                continue

            # Wolfers-Zitzewitz 2004 favourite-longshot correction (γ=0.91,
            # multiplicative form). Converts bid/ask midpoint (cents → prob)
            # into a bias-corrected implied probability.
            mid = (yes_ask + yes_bid) / 200.0  # cents → prob in [0, 1]
            gamma = 0.91
            num = mid ** gamma
            p = num / (num + (1.0 - mid) ** gamma)
            implied_prob = round(p * 100)
            dashboard_lines.append(
                f"- *{subtitle}*: Implied Probability {implied_prob:.0f}%"
            )

        confidence = "HIGH" if total_open_interest > 250000 else ("MEDIUM" if total_open_interest > 50000 else "LOW")
        dashboard_lines.append(f"- Aggregate Event Open Interest: ${total_open_interest:,.0f} (Confidence: {confidence})\n")

    if len(dashboard_lines) == 1:
        return "Kalshi Macro Risk Dashboard unavailable: Connection or liquidity failure."

    dashboard = "\n".join(dashboard_lines)

    # Stage 2 Commit K — daily snapshot cache for agent-output
    # reproducibility. Failing to write must never break the live call.
    try:
        import os as _os
        from pathlib import Path as _Path
        snap_dir = _Path(KALSHI_SNAPSHOT_DIR)
        snap_dir.mkdir(parents=True, exist_ok=True)
        from datetime import timezone as _tz
        today = datetime.now(_tz.utc).strftime("%Y-%m-%d")
        (snap_dir / f"{today}.md").write_text(dashboard)
    except Exception as _e:  # pragma: no cover
        logger.debug("kalshi snapshot write failed: %s", _e)

    return dashboard
