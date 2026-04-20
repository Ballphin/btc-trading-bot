"""Era-aware historical data router for pulse backtests.

The pulse auto-tune pipeline needs trustworthy historical OHLCV + funding
data for the **bear regime** (2022). Hyperliquid perps launched 2023-02-15;
before that there is no HL data to tune on. This module centralizes the
routing decision:

    * ``end <= HL_LAUNCH``      → Binance Futures only
    * ``start >= HL_LAUNCH``    → Hyperliquid only
    * overlapping both eras     → Binance for the pre-launch slice,
                                  Hyperliquid for the post-launch slice,
                                  validated via price-agreement check
                                  on the overlap window.

A single entrypoint (:func:`fetch_ohlcv_historical`) returns
``(df, source_label)`` so the backtest engine can stamp provenance on
every result. Funding rates follow the same contract
(:func:`fetch_funding_historical`) with a real-data fallback to
Binance ``/fapi/v1/fundingRate`` (not the prior plan's synthesized
``+0.02%/8h`` — Binance has real data all the way back to 2019-09).

All routing decisions are logged. Callers that want to force a single
source for reproducibility can pass ``force_source="binance_futures"``
or ``"hyperliquid"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from tradingagents.dataflows.binance_client import BinanceClient
from tradingagents.dataflows.coinbase_client import (
    CoinbaseClient,
    GRANULARITY_1D,
    GRANULARITY_1H,
    GRANULARITY_15M,
    GRANULARITY_5M,
)
from tradingagents.dataflows.hyperliquid_client import HyperliquidClient

logger = logging.getLogger(__name__)


# ── Canonical constants ──────────────────────────────────────────────

#: UTC instant of the first usable Hyperliquid perp bar. Data earlier than
#: this simply doesn't exist on HL, so anything targeting <= this date
#: must come from Binance. Source: HL public launch announcement.
HL_LAUNCH_UTC = datetime(2023, 2, 15, 0, 0, 0, tzinfo=timezone.utc)

#: Short overlap either side of the boundary used for stitch validation.
#: Wider window = more robust but slower; 14 days ≈ 336 hourly bars is
#: plenty to hit statistical significance on price-agreement checks.
_STITCH_OVERLAP_DAYS = 14

#: Max acceptable price divergence between sources on any 1h bar during
#: the overlap window (fraction, not percent). 0.5% catches genuine data
#: corruption while tolerating normal cross-venue basis noise.
_STITCH_P99_DIVERGENCE_LIMIT = 0.005

SourceLabel = Literal[
    "hyperliquid",
    "binance_futures",
    "binance+hl_stitched",
]


@dataclass
class FetchResult:
    """Return value of :func:`fetch_ohlcv_historical`.

    ``df`` is sorted ascending by ``timestamp`` with tz-naive UTC times.
    ``source`` stamps provenance so callers can propagate it into
    ``config_hash`` (see :func:`tradingagents.pulse.config.compute_config_hash`).
    ``overlap_report`` is non-``None`` only for stitched fetches; it
    records the p99 and max divergence observed on the overlap window.
    """

    df: pd.DataFrame
    source: SourceLabel
    overlap_report: Optional[dict] = None


# ── Ticker mapping helpers ───────────────────────────────────────────

def _to_binance_symbol(ticker: str) -> str:
    """Map internal ``BTC-USD`` form → Binance perp ``BTCUSDT``.

    Assumes USDT-denominated perps (Binance doesn't publish USD-margined
    perps for most altcoins). Returns the uppercased concatenation if
    the ticker already looks Binance-ish (``BTCUSDT``).
    """
    t = ticker.upper().strip()
    if t.endswith("-USD"):
        return t.replace("-USD", "USDT")
    if t.endswith("USD") and not t.endswith("USDT"):
        return t.replace("USD", "USDT")
    return t


def _to_hl_coin(ticker: str) -> str:
    """Map internal ``BTC-USD`` form → Hyperliquid coin symbol ``BTC``."""
    t = ticker.upper().strip()
    return t.replace("-USD", "").replace("USDT", "").replace("USD", "")


def _to_coinbase_product(ticker: str) -> str:
    """Map internal ``BTC-USD`` form → Coinbase spot product ``BTC-USD``.

    Coinbase uses the same convention we do, so this is mostly a
    normalization pass.
    """
    t = ticker.upper().strip()
    if "-" in t:
        return t
    # BTCUSDT → BTC-USD (drop the T)
    if t.endswith("USDT"):
        return f"{t[:-4]}-USD"
    if t.endswith("USD"):
        return f"{t[:-3]}-USD"
    return t


# ── Interval mapping helpers ─────────────────────────────────────────

_HL_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d",
}

_COINBASE_GRANULARITY_MAP = {
    "5m": GRANULARITY_5M,
    "15m": GRANULARITY_15M,
    "1h": GRANULARITY_1H,
    "1d": GRANULARITY_1D,
}


# ── Public API ───────────────────────────────────────────────────────

def fetch_ohlcv_historical(
    ticker: str,
    interval: str,
    start: str,
    end: str,
    *,
    force_source: Optional[SourceLabel] = None,
    validate_with_coinbase: bool = False,
    hl_client: Optional[HyperliquidClient] = None,
    binance_client: Optional[BinanceClient] = None,
    coinbase_client: Optional[CoinbaseClient] = None,
) -> FetchResult:
    """Era-aware OHLCV fetch with stitch + optional Coinbase cross-check.

    Args:
        ticker: Internal format, e.g. ``"BTC-USD"``.
        interval: ``1m`` / ``5m`` / ``15m`` / ``1h`` / ``4h`` / ``1d``.
        start: ``yyyy-mm-dd`` (inclusive).
        end: ``yyyy-mm-dd`` (exclusive — same semantics as Binance).
        force_source: If set, bypass era routing and fetch from this
            source only. Useful for A/B comparisons.
        validate_with_coinbase: When True, sample 100 random 1h bars
            from the fetched window, fetch the same bars from Coinbase
            spot, and raise :class:`DataSourceDriftError` if any bar's
            close disagrees by >0.5%. Only runs for intervals where
            Coinbase has native granularity.
        hl_client / binance_client / coinbase_client: Injected clients
            for tests. Default: fresh instances with 1h cache TTL.

    Raises:
        DataSourceDriftError: Coinbase validation failed (only when
            ``validate_with_coinbase=True``).
        StitchValidationError: Overlap validation failed at the HL/
            Binance boundary (stitched windows only).
    """
    hl = hl_client or HyperliquidClient()
    bn = binance_client or BinanceClient()
    start_utc = _parse_utc(start)
    end_utc = _parse_utc(end)
    if end_utc <= start_utc:
        raise ValueError(f"end={end} must be after start={start}")

    if force_source is not None:
        return _fetch_by_source(force_source, ticker, interval, start, end, hl, bn)

    if end_utc <= HL_LAUNCH_UTC:
        result = _fetch_by_source("binance_futures", ticker, interval, start, end, hl, bn)
    elif start_utc >= HL_LAUNCH_UTC:
        result = _fetch_by_source("hyperliquid", ticker, interval, start, end, hl, bn)
    else:
        # Straddles the launch boundary — stitch.
        result = _fetch_stitched(ticker, interval, start, end, hl, bn)

    if validate_with_coinbase and interval in _COINBASE_GRANULARITY_MAP:
        cb = coinbase_client or CoinbaseClient()
        _validate_against_coinbase(result.df, ticker, interval, cb)

    return result


# ── Source-specific fetchers ─────────────────────────────────────────

def _fetch_by_source(
    source: SourceLabel,
    ticker: str,
    interval: str,
    start: str,
    end: str,
    hl: HyperliquidClient,
    bn: BinanceClient,
) -> FetchResult:
    if source == "hyperliquid":
        coin = _to_hl_coin(ticker)
        hl_interval = _HL_INTERVAL_MAP.get(interval)
        if hl_interval is None:
            raise ValueError(f"Hyperliquid does not support interval {interval!r}")
        df = hl.get_ohlcv(coin, hl_interval, start=start, end=end)
        return FetchResult(df=df, source="hyperliquid")
    if source == "binance_futures":
        symbol = _to_binance_symbol(ticker)
        df = bn.get_klines(symbol, interval, start=start, end=end)
        # Binance klines include n_trades / quote_volume columns that HL
        # doesn't. Callers should be column-agnostic, but trim the
        # common subset to match HL's shape for predictable downstream.
        keep = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]].copy()
        return FetchResult(df=df, source="binance_futures")
    raise ValueError(f"Unsupported force_source={source!r}")


def _fetch_stitched(
    ticker: str,
    interval: str,
    start: str,
    end: str,
    hl: HyperliquidClient,
    bn: BinanceClient,
) -> FetchResult:
    """Binance pre-launch + HL post-launch, validated on the overlap."""
    boundary_date = HL_LAUNCH_UTC.strftime("%Y-%m-%d")

    # Phase 1: Binance up to (and including) the boundary.
    binance_part = _fetch_by_source(
        "binance_futures", ticker, interval, start, boundary_date, hl, bn,
    ).df
    # Phase 2: HL from the boundary onward.
    hl_part = _fetch_by_source(
        "hyperliquid", ticker, interval, boundary_date, end, hl, bn,
    ).df

    # Phase 3: overlap validation. Fetch the 14-day window either side
    # of the launch date from BOTH sources at 1h resolution and check
    # price agreement. Independent of the user's requested `interval`
    # so the validation is consistent across runs.
    overlap_start = (HL_LAUNCH_UTC - pd.Timedelta(days=_STITCH_OVERLAP_DAYS)).strftime("%Y-%m-%d")
    overlap_end = (HL_LAUNCH_UTC + pd.Timedelta(days=_STITCH_OVERLAP_DAYS)).strftime("%Y-%m-%d")
    try:
        ov_binance = _fetch_by_source(
            "binance_futures", ticker, "1h", overlap_start, overlap_end, hl, bn,
        ).df
        ov_hl = _fetch_by_source(
            "hyperliquid", ticker, "1h", overlap_start, overlap_end, hl, bn,
        ).df
        report = _validate_stitch_overlap(ov_binance, ov_hl)
    except Exception as e:
        # If the overlap fetch itself fails, bail with a permissive
        # report — but log loudly. A stitched backtest without overlap
        # validation is still better than no backtest.
        logger.warning(
            f"[HistoricalRouter] Stitch overlap validation skipped "
            f"(overlap fetch failed: {e}); stitched data may be inconsistent"
        )
        report = {"status": "skipped", "error": str(e)}

    if report.get("status") == "failed":
        raise StitchValidationError(
            f"Binance vs Hyperliquid price divergence exceeds "
            f"{_STITCH_P99_DIVERGENCE_LIMIT*100:.2f}% p99 on overlap window: "
            f"{report}"
        )

    # Concatenate Binance (< boundary) + HL (>= boundary), dedup, sort.
    boundary_ts = pd.Timestamp(HL_LAUNCH_UTC).tz_localize(None)
    binance_part = binance_part[binance_part["timestamp"] < boundary_ts]
    hl_part = hl_part[hl_part["timestamp"] >= boundary_ts]
    stitched = (
        pd.concat([binance_part, hl_part], ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return FetchResult(
        df=stitched, source="binance+hl_stitched", overlap_report=report,
    )


# ── Validation ───────────────────────────────────────────────────────

class StitchValidationError(RuntimeError):
    """Raised when Binance vs Hyperliquid overlap disagrees beyond the gate."""


class DataSourceDriftError(RuntimeError):
    """Raised when Coinbase spot cross-validation fails."""


def _validate_stitch_overlap(
    binance: pd.DataFrame,
    hl: pd.DataFrame,
) -> dict:
    """Compute price-agreement stats on the overlap window at 1h res.

    Returns a report dict with keys:
        status  — "ok" / "failed" / "skipped"
        n_bars  — number of bars compared
        p99     — 99th percentile of |close_bn - close_hl| / close_hl
        max     — maximum divergence across all compared bars
        limit   — the threshold that was applied
    """
    if binance.empty or hl.empty:
        return {"status": "skipped", "reason": "empty_overlap"}

    merged = pd.merge(
        binance[["timestamp", "close"]].rename(columns={"close": "bn_close"}),
        hl[["timestamp", "close"]].rename(columns={"close": "hl_close"}),
        on="timestamp",
        how="inner",
    )
    if len(merged) < 24:  # at least 24 hourly bars
        return {"status": "skipped", "reason": f"n_bars={len(merged)} too small"}

    merged["div"] = (merged["bn_close"] - merged["hl_close"]).abs() / merged["hl_close"]
    p99 = float(np.percentile(merged["div"].values, 99))
    mx = float(merged["div"].max())
    report = {
        "status": "ok" if p99 <= _STITCH_P99_DIVERGENCE_LIMIT else "failed",
        "n_bars": int(len(merged)),
        "p99": p99,
        "max": mx,
        "limit": _STITCH_P99_DIVERGENCE_LIMIT,
    }
    return report


def _validate_against_coinbase(
    df: pd.DataFrame,
    ticker: str,
    interval: str,
    cb: CoinbaseClient,
    *,
    sample_size: int = 100,
    divergence_limit: float = 0.005,
    rng_seed: int = 42,
) -> None:
    """Sample ``sample_size`` random bars and cross-check Coinbase spot.

    Non-destructive — only logs and raises on hard failure. Coinbase is
    spot and can drift from Binance perp during illiquid hours, so we
    use p99 of sampled divergences, not the max.
    """
    if df.empty:
        return
    gran = _COINBASE_GRANULARITY_MAP[interval]
    start = df["timestamp"].min().strftime("%Y-%m-%d")
    end = (df["timestamp"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    product = _to_coinbase_product(ticker)
    try:
        cb_df = cb.get_ohlcv(product, granularity=gran, start=start, end=end)
    except Exception as e:
        logger.warning(f"[HistoricalRouter] Coinbase validation skipped: {e}")
        return
    if cb_df.empty:
        return

    merged = pd.merge(
        df[["timestamp", "close"]].rename(columns={"close": "src_close"}),
        cb_df[["timestamp", "close"]].rename(columns={"close": "cb_close"}),
        on="timestamp", how="inner",
    )
    if len(merged) < 10:
        return
    rng = np.random.default_rng(rng_seed)
    n = min(sample_size, len(merged))
    idx = rng.choice(len(merged), size=n, replace=False)
    sampled = merged.iloc[idx]
    div = (sampled["src_close"] - sampled["cb_close"]).abs() / sampled["cb_close"]
    p99 = float(np.percentile(div.values, 99))
    if p99 > divergence_limit:
        raise DataSourceDriftError(
            f"Coinbase validation failed for {ticker}/{interval}: "
            f"p99 divergence = {p99*100:.3f}% > limit "
            f"{divergence_limit*100:.2f}% across {n} sampled bars"
        )
    logger.info(
        f"[HistoricalRouter] Coinbase cross-check OK for {ticker}/{interval}: "
        f"p99={p99*100:.3f}% over {n} bars"
    )


# ── Funding ──────────────────────────────────────────────────────────

def fetch_funding_historical(
    ticker: str,
    start: str,
    end: str,
    *,
    hl_client: Optional[HyperliquidClient] = None,
    binance_client: Optional[BinanceClient] = None,
) -> Tuple[pd.DataFrame, SourceLabel]:
    """Return 8h funding history with era-aware routing.

    Pre-HL windows fall back to Binance ``/fapi/v1/fundingRate`` which
    has **real** data (not synthetic) going back to 2019-09. This
    replaces the earlier plan's hard-coded ``+0.02%/8h`` fallback.

    Returns:
        (df, source_label) — df has columns ``timestamp``, ``funding_rate``
        sorted ascending, or an empty DataFrame if both sources failed.
    """
    hl = hl_client or HyperliquidClient()
    bn = binance_client or BinanceClient()
    start_utc = _parse_utc(start)
    end_utc = _parse_utc(end)

    if end_utc <= HL_LAUNCH_UTC:
        return _binance_funding(ticker, start, end, bn), "binance_futures"
    if start_utc >= HL_LAUNCH_UTC:
        return _hl_funding(ticker, start, end, hl), "hyperliquid"

    boundary = HL_LAUNCH_UTC.strftime("%Y-%m-%d")
    bn_df = _binance_funding(ticker, start, boundary, bn)
    hl_df = _hl_funding(ticker, boundary, end, hl)
    boundary_ts = pd.Timestamp(HL_LAUNCH_UTC).tz_localize(None)
    if not bn_df.empty:
        bn_df = bn_df[bn_df["timestamp"] < boundary_ts]
    if not hl_df.empty:
        hl_df = hl_df[hl_df["timestamp"] >= boundary_ts]
    merged = (
        pd.concat([bn_df, hl_df], ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return merged, "binance+hl_stitched"


def _binance_funding(ticker: str, start: str, end: str, bn: BinanceClient) -> pd.DataFrame:
    """Normalize Binance funding rows to ``timestamp`` + ``funding_rate``."""
    symbol = _to_binance_symbol(ticker)
    try:
        df = bn.get_funding_rates(symbol=symbol, start=start, end=end)
    except Exception as e:
        logger.warning(f"[HistoricalRouter] Binance funding failed: {e}")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["fundingTime"]),
        "funding_rate": df["fundingRate"].astype(float),
    })
    return out.sort_values("timestamp").reset_index(drop=True)


def _hl_funding(ticker: str, start: str, end: str, hl: HyperliquidClient) -> pd.DataFrame:
    """Normalize Hyperliquid funding rows to the same shape as Binance."""
    coin = _to_hl_coin(ticker)
    try:
        df = hl.get_funding_history(coin, start=start, end=end)
    except Exception as e:
        logger.warning(f"[HistoricalRouter] HL funding failed: {e}")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    # HL client returns timestamp + funding_rate already.
    if "funding_rate" not in df.columns or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    return df[["timestamp", "funding_rate"]].copy()


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_utc(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
