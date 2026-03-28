"""Alternative.me Fear & Greed Index — completely free, no auth required."""

import logging

import pandas as pd

from tradingagents.dataflows.base_client import BaseDataClient

logger = logging.getLogger(__name__)


class FearGreedClient(BaseDataClient):
    """Fetch the Crypto Fear & Greed Index from Alternative.me."""

    URL = "https://api.alternative.me/fng/"

    def __init__(self, cache_ttl: int = 3600):
        super().__init__("fear_greed", cache_ttl)

    def get_index(self, limit: int = 30) -> pd.DataFrame:
        """
        Fetch Fear & Greed history.

        Args:
            limit: Number of days of history. 0 = all available data.

        Returns:
            DataFrame[timestamp, value, value_classification] sorted ascending.
        """
        params = {"limit": limit, "format": "json"}

        try:
            data = self._request(self.URL, params, cache_prefix="fng")
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return pd.DataFrame(columns=["timestamp", "value", "value_classification"])

        records = data.get("data", [])
        if not records:
            return pd.DataFrame(columns=["timestamp", "value", "value_classification"])

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
            df["value"] = df["value"].astype(int)

        cols = ["timestamp", "value", "value_classification"]
        available_cols = [c for c in cols if c in df.columns]
        return df[available_cols].sort_values("timestamp").reset_index(drop=True)

    def get_sentiment_report(self, look_back_days: int = 30) -> str:
        """
        Formatted markdown report of Fear & Greed data.

        Args:
            look_back_days: How many days of history to include.

        Returns:
            Markdown string with current reading, averages, and contrarian signal.
        """
        df = self.get_index(limit=look_back_days)
        if df.empty:
            return "# Crypto Fear & Greed Index\n\nData unavailable."

        latest = df.iloc[-1]
        current_val = int(latest["value"])
        classification = latest.get("value_classification", "Unknown")

        avg_7d = df.tail(7)["value"].mean() if len(df) >= 7 else current_val
        avg_30d = df["value"].mean()

        if current_val > 75:
            contrarian = "⚠️ Extreme Greed — historically precedes corrections"
        elif current_val < 25:
            contrarian = "⚠️ Extreme Fear — historically precedes rallies"
        elif current_val > 60:
            contrarian = "Elevated greed — watch for overextension"
        elif current_val < 40:
            contrarian = "Elevated fear — potential accumulation zone"
        else:
            contrarian = "Neutral zone — no strong contrarian signal"

        return (
            f"# Crypto Fear & Greed Index\n\n"
            f"- **Current**: {current_val} ({classification})\n"
            f"- **7-day avg**: {avg_7d:.0f}\n"
            f"- **30-day avg**: {avg_30d:.0f}\n"
            f"- **Contrarian signal**: {contrarian}\n"
        )
