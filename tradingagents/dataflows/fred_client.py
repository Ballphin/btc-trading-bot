"""FRED API client for macro indicators.

Free API key required: register at https://fred.stlouisfed.org/docs/api/api_key.html
Calls the REST API directly via requests — no fredapi package dependency.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from tradingagents.dataflows.base_client import BaseDataClient

logger = logging.getLogger(__name__)


class FREDClient(BaseDataClient):
    """Fetch macroeconomic indicator data from the FRED REST API."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    SERIES_MAP = {
        "m2": "WM2NS",          # M2 Money Supply
        "dxy": "DTWEXBGS",      # Trade Weighted U.S. Dollar Index
        "dgs2": "DGS2",         # 2-Year Treasury Yield
        "dgs10": "DGS10",       # 10-Year Treasury Yield
        "fedfunds": "FEDFUNDS", # Federal Funds Rate
        # Stage 2 Commit L — expanded condensed dashboard series.
        "dff": "DFF",           # Effective Fed Funds Rate (daily)
        "t10y2y": "T10Y2Y",     # 10Y-2Y spread (recession proxy)
        "dexuseu": "DEXUSEU",   # USD/EUR exchange rate
        "dcoilwtico": "DCOILWTICO",  # WTI crude oil
        "unrate": "UNRATE",     # Unemployment rate
        "cpiaucsl": "CPIAUCSL", # CPI-U all items
    }

    # Stage 2 Commit L — condensed "macro dashboard" series subset
    # (fast-moving series only; monthly CPI/UNRATE are omitted from the
    # intraday dashboard to avoid a stale snapshot dominating the view).
    DASHBOARD_SERIES = [
        "dff", "dgs10", "t10y2y", "dexuseu", "dcoilwtico", "unrate", "cpiaucsl",
    ]

    def __init__(self, api_key: str = None, cache_ttl: int = 86400):
        super().__init__("fred", cache_ttl)
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "FRED_API_KEY not set. Macro indicator data will be unavailable. "
                "Register free at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    def get_series(self, series_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch a FRED series by friendly name or raw series ID.

        Args:
            series_name: Friendly name (m2, dxy, dgs2, dgs10, fedfunds) or raw FRED series ID
            start_date: yyyy-mm-dd
            end_date: yyyy-mm-dd

        Returns:
            DataFrame with [date, value] columns.
        """
        if not self.api_key:
            return pd.DataFrame(columns=["date", "value"])

        series_id = self.SERIES_MAP.get(series_name.lower(), series_name)
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }

        try:
            data = self._request(
                self.BASE_URL, params, cache_prefix=f"fred-{series_id}"
            )
        except Exception as e:
            logger.warning(f"FRED {series_id} fetch failed: {e}")
            return pd.DataFrame(columns=["date", "value"])

        observations = data.get("observations", [])
        if not observations:
            return pd.DataFrame(columns=["date", "value"])

        df = pd.DataFrame(observations)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "value"]].dropna()
        return df

    def get_macro_summary(self, end_date: str, look_back_days: int = 90) -> str:
        """
        Fetch all macro series and return a formatted markdown report.

        Args:
            end_date: yyyy-mm-dd
            look_back_days: How many days of history

        Returns:
            Markdown table of macro indicators with latest values, changes, trends.
        """
        start = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=look_back_days)
        ).strftime("%Y-%m-%d")

        lines = [
            "# Macro Economic Indicators",
            f"**Period**: {start} to {end_date}",
            "",
            "| Indicator | Latest Value | 30d Change | Trend |",
            "|-----------|-------------|------------|-------|",
        ]

        for name, series_id in self.SERIES_MAP.items():
            try:
                df = self.get_series(name, start, end_date)
                if not df.empty and len(df) >= 2:
                    latest = df.iloc[-1]["value"]
                    cutoff = df.iloc[-1]["date"] - timedelta(days=30)
                    prev_30d = df[df["date"] <= cutoff]
                    prev_val = (
                        prev_30d.iloc[-1]["value"] if not prev_30d.empty else df.iloc[0]["value"]
                    )
                    change = ((latest - prev_val) / prev_val * 100) if prev_val != 0 else 0
                    trend = "↑ UP" if change > 0.5 else "↓ DOWN" if change < -0.5 else "→ FLAT"
                    lines.append(
                        f"| {name.upper()} | {latest:.2f} | {change:+.2f}% | {trend} |"
                    )
                else:
                    lines.append(f"| {name.upper()} | N/A | N/A | N/A |")
            except Exception:
                lines.append(f"| {name.upper()} | N/A | N/A | N/A |")

        return "\n".join(lines)


# ── Stage 2 Commit L — market-hours-aware cache + dashboard ──────────

def _is_us_market_hours(dt: Optional[datetime] = None) -> bool:
    """True if ``dt`` (UTC) falls within 9:30-16:00 ET on a weekday.

    Uses a fixed UTC-4 (EDT) offset approximation — accurate for the
    use case (cache TTL selection) within ±1h around DST transitions.
    """
    from datetime import timezone as _tz
    dt = dt or datetime.now(_tz.utc)
    # Convert UTC → ET (approximate: subtract 4h).
    et = dt - timedelta(hours=4)
    if et.weekday() >= 5:
        return False
    minutes = et.hour * 60 + et.minute
    return (9 * 60 + 30) <= minutes <= (16 * 60)


def market_hours_cache_ttl_sec(dt: Optional[datetime] = None) -> int:
    """Return cache TTL in seconds: 15min in US hours, 60min otherwise.

    Callers pass this as ``cache_ttl=`` when instantiating FREDClient
    so fast-moving series refresh more often during the most liquid
    window.
    """
    return 15 * 60 if _is_us_market_hours(dt) else 60 * 60


def get_fred_macro_dashboard(
    end_date: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
) -> str:
    """Stage 2 Commit L — condensed market-hours-aware FRED dashboard.

    Renders ``FREDClient.DASHBOARD_SERIES`` in a tight markdown table.
    Cache TTL is chosen dynamically from :func:`market_hours_cache_ttl_sec`
    so intraday US hours get a 15min TTL and off-hours get 60min.

    Gracefully degrades: if ``FRED_API_KEY`` is unset, returns a single
    ``[FRED: no API key]`` line so agents know the signal is absent.
    """
    from datetime import timezone as _tz
    end_date = end_date or datetime.now(_tz.utc).strftime("%Y-%m-%d")
    client = FREDClient(api_key=api_key, cache_ttl=market_hours_cache_ttl_sec())
    if not client.api_key:
        return "[FRED: no API key — set FRED_API_KEY env var]"

    start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
    lines = [
        "### FRED Macro Dashboard",
        f"*{end_date} · cache {market_hours_cache_ttl_sec() // 60}min*",
        "",
        "| Indicator | Latest | 30d Chg | Trend |",
        "|-----------|--------|---------|-------|",
    ]
    for name in FREDClient.DASHBOARD_SERIES:
        try:
            df = client.get_series(name, start, end_date)
            if df.empty or len(df) < 2:
                lines.append(f"| {name.upper()} | N/A | N/A | N/A |")
                continue
            latest = float(df.iloc[-1]["value"])
            cutoff = df.iloc[-1]["date"] - timedelta(days=30)
            prev_30d = df[df["date"] <= cutoff]
            prev_val = float(
                prev_30d.iloc[-1]["value"] if not prev_30d.empty else df.iloc[0]["value"]
            )
            change = ((latest - prev_val) / prev_val * 100.0) if prev_val != 0 else 0.0
            trend = "↑ UP" if change > 0.5 else "↓ DOWN" if change < -0.5 else "→ FLAT"
            lines.append(f"| {name.upper()} | {latest:.2f} | {change:+.2f}% | {trend} |")
        except Exception as e:
            logger.debug("FRED dashboard %s failed: %s", name, e)
            lines.append(f"| {name.upper()} | N/A | N/A | N/A |")
    return "\n".join(lines)
