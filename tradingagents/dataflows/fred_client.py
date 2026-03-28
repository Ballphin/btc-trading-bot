"""FRED API client for macro indicators.

Free API key required: register at https://fred.stlouisfed.org/docs/api/api_key.html
Calls the REST API directly via requests — no fredapi package dependency.
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd

from tradingagents.dataflows.base_client import BaseDataClient

logger = logging.getLogger(__name__)


class FREDClient(BaseDataClient):
    """Fetch macroeconomic indicator data from the FRED REST API."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    SERIES_MAP = {
        "m2": "WM2NS",        # M2 Money Supply
        "dxy": "DTWEXBGS",    # Trade Weighted U.S. Dollar Index
        "dgs2": "DGS2",       # 2-Year Treasury Yield
        "dgs10": "DGS10",     # 10-Year Treasury Yield
        "fedfunds": "FEDFUNDS",  # Federal Funds Rate
    }

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
