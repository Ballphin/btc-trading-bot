"""BGeometrics free-tier API — Bitcoin on-chain metrics.

Free API key required: register at https://bitcoin-data.com/bguser/pricing
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd

from tradingagents.dataflows.base_client import BaseDataClient

logger = logging.getLogger(__name__)


class BGeometricsClient(BaseDataClient):
    """Fetch Bitcoin on-chain metrics from BGeometrics free-tier API."""

    BASE_URL = "https://bitcoin-data.com/api"

    METRIC_ENDPOINTS = {
        "mvrv": "/v1/mvrv",
        "sopr": "/v1/sopr",
        "exchange_netflow": "/v1/exchange-netflow",
        "exchange_reserve": "/v1/exchange-reserve",
        "lth_supply": "/v1/lth-supply",
        "nupl": "/v1/nupl",
        "puell_multiple": "/v1/puell-multiple",
        "active_addresses": "/v1/active-addresses",
    }

    def __init__(self, api_key: str = None, cache_ttl: int = 7200):
        super().__init__("bgeometrics", cache_ttl)
        self.api_key = api_key or os.getenv("BGEOMETRICS_API_KEY", "")
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def get_metric(
        self,
        metric: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch an on-chain metric by friendly name.

        Args:
            metric: One of the keys in METRIC_ENDPOINTS
            start_date: yyyy-mm-dd
            end_date: yyyy-mm-dd

        Returns:
            DataFrame with [date, value] columns.
        """
        endpoint = self.METRIC_ENDPOINTS.get(metric)
        if not endpoint:
            raise ValueError(
                f"Unknown metric: {metric}. Available: {list(self.METRIC_ENDPOINTS.keys())}"
            )

        params = {}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        try:
            data = self._request(
                f"{self.BASE_URL}{endpoint}",
                params,
                cache_prefix=f"bg-{metric}",
            )
        except Exception as e:
            logger.warning(f"BGeometrics {metric} fetch failed: {e}")
            return pd.DataFrame(columns=["date", "value"])

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(columns=["date", "value"])

        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    def get_onchain_summary(self, end_date: str, look_back_days: int = 60) -> str:
        """
        Fetch key on-chain metrics and return a formatted markdown report.

        Args:
            end_date: yyyy-mm-dd
            look_back_days: How many days of history to include

        Returns:
            Formatted markdown string with on-chain analysis.
        """
        start = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=look_back_days)
        ).strftime("%Y-%m-%d")

        report_lines = [
            "# Bitcoin On-Chain Analysis Report",
            f"**Period**: {start} to {end_date}",
            "",
        ]

        core_metrics = ["mvrv", "sopr", "exchange_netflow", "exchange_reserve", "nupl"]
        failed_metrics: list = []

        for metric_name in core_metrics:
            try:
                df = self.get_metric(metric_name, start, end_date)
                if not df.empty and "value" in df.columns:
                    latest = df.iloc[-1]
                    avg_val = df["value"].mean()
                    label = metric_name.upper().replace("_", " ")

                    report_lines.append(f"## {label}")
                    report_lines.append(f"- **Latest value**: {latest.get('value', 'N/A')}")
                    report_lines.append(f"- **Date**: {latest.get('date', 'N/A')}")
                    report_lines.append(f"- **Period average**: {avg_val:.4f}")

                    if len(df) >= 7:
                        recent_7d = df.tail(7)["value"].mean()
                        report_lines.append(f"- **7-day average**: {recent_7d:.4f}")

                    report_lines.append("")
                else:
                    report_lines.append(f"## {metric_name.upper()}: No data available")
                    report_lines.append("")
            except Exception as e:
                failed_metrics.append(metric_name)
                report_lines.append(f"## {metric_name.upper()}: Data unavailable ({e})")
                report_lines.append("")

        if failed_metrics:
            report_lines.insert(
                1,
                f"*Note: Partial on-chain data — unavailable metrics: {', '.join(failed_metrics)}.*\n",
            )

        return "\n".join(report_lines)
