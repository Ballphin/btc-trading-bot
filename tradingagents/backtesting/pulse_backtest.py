"""Pulse Backtest Engine — Historical replay of the deterministic pulse scoring
engine on months of OHLCV candle data with zero LLM calls.

Implements all v3 debate findings:
  - Shared _compute_tf_indicators() (no duplication)
  - premium_pct = 0 in backtest mode (no VWAP proxy)
  - Signal de-duplication (same-direction suppressed for hold_minutes)
  - Stop-and-reverse equity model (single position at a time)
  - Forward return uses candle OPEN (not close)
  - Candle gap detection (>1.1× interval → unscorable)
  - Indicator coverage gating (<60% → excluded from metrics)
  - Regime bucketing via P25/P75 of rolling 24h vol
  - Sharpe annualized by de-duplicated trade count
  - Profitability curve downsampled to ≤500 points for API
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tradingagents.agents.quant_pulse_data import (
    PULSE_TIMEFRAMES,
    _compute_tf_indicators,
    compute_vwap_from_slice,
    compute_volatility_flag,
)
from tradingagents.agents.quant_pulse_engine import score_pulse
from tradingagents.dataflows.hyperliquid_client import (
    HyperliquidClient,
    _INTERVAL_SECONDS,
)
from tradingagents.pulse.config import get_config
from tradingagents.pulse.pulse_assembly import PulseInputs, score_pulse_from_inputs
from tradingagents.pulse.regime import detect_regime
from tradingagents.pulse.support_resistance import compute_support_resistance

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

_EXEC_COST = 0.0005  # 5 bps
_POSITION_SIZE = 0.05  # 5% of equity per trade
_MIN_COVERAGE = 0.60  # exclude pulses with <60% indicator coverage
_MAX_CURVE_POINTS = 500  # downsample profitability curve for API


class PulseBacktestEngine:
    """Replay the pulse scoring engine on historical candle data."""

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        pulse_interval_minutes: int = 15,
        signal_threshold: float = 0.25,
        results_dir: str = "./eval_results",
    ):
        self.ticker = ticker
        self.base_asset = ticker.replace("-USD", "").replace("USDT", "").upper()
        self.start_date = start_date
        self.end_date = end_date
        self.interval_minutes = pulse_interval_minutes
        self.threshold = signal_threshold
        self.results_dir = results_dir

        self.start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def run(self) -> Dict[str, Any]:
        """Execute the full backtest pipeline.

        Returns:
            Results dict matching the plan output shape.
        """
        logger.info(f"[PulseBacktest] Starting {self.ticker} "
                     f"{self.start_date} → {self.end_date}")

        # Phase 1: Pre-fetch
        candles, funding_df = self._prefetch()

        # Phase 2: Replay
        signals = self._replay(candles, funding_df)

        # Phase 3: Score forward returns
        scored = self._score_signals(signals, candles)

        # Phase 4: Compute metrics
        return self._compute_metrics(scored, candles)

    # ── Phase 1: Pre-fetch ────────────────────────────────────────────

    def _prefetch(self) -> tuple:
        """Bulk-fetch all historical data needed for the backtest."""
        hl = HyperliquidClient()

        # Pad 2 days before and 1 day after
        fetch_start = (self.start_dt - timedelta(days=2)).strftime("%Y-%m-%d")
        fetch_end = (self.end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        candles: Dict[str, pd.DataFrame] = {}
        for tf in PULSE_TIMEFRAMES:
            logger.info(f"[PulseBacktest] Fetching {tf} candles...")
            try:
                df = hl.get_ohlcv(
                    self.base_asset, tf,
                    start=fetch_start, end=fetch_end,
                    max_age_override=3600,
                )
                candles[tf] = df
                logger.info(f"[PulseBacktest] {tf}: {len(df)} candles")
            except Exception as e:
                logger.warning(f"[PulseBacktest] Failed to fetch {tf}: {e}")
                candles[tf] = pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

        # Funding history
        logger.info("[PulseBacktest] Fetching funding history...")
        days_range = (self.end_dt - self.start_dt).days + 4
        try:
            funding_df = hl.get_funding_history(
                self.base_asset,
                start=fetch_start,
                end=fetch_end,
                max_age_override=3600,
            )
        except Exception as e:
            logger.warning(f"[PulseBacktest] Funding history failed: {e}")
            funding_df = pd.DataFrame(columns=["timestamp", "funding_rate"])

        logger.info(f"[PulseBacktest] Pre-fetch complete. "
                     f"Funding entries: {len(funding_df)}")
        return candles, funding_df

    # ── Phase 2: Replay ───────────────────────────────────────────────

    def _replay(
        self,
        candles: Dict[str, pd.DataFrame],
        funding_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Step through time at pulse_interval, build reports, score."""
        interval_sec = self.interval_minutes * 60
        current = self.start_dt
        signals = []
        prev_funding_rate = None

        # De-duplication state
        last_signal_direction = None
        suppress_until = None

        # Gap detection on 1m data
        gap_timestamps = set()
        if not candles["1m"].empty:
            ts_1m = candles["1m"]["timestamp"].values
            for i in range(1, len(ts_1m)):
                gap = (pd.Timestamp(ts_1m[i]) - pd.Timestamp(ts_1m[i - 1])).total_seconds()
                if gap > 66:  # >1.1× 60s
                    gap_timestamps.add(pd.Timestamp(ts_1m[i]))

        total_intervals = 0
        n_excluded_warmup = 0

        while current < self.end_dt:
            total_intervals += 1

            # Check if current timestamp is in a gap zone
            in_gap = False
            for gap_ts in gap_timestamps:
                if abs((current - gap_ts.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds()) < 120:
                    in_gap = True
                    break

            if in_gap:
                current += timedelta(seconds=interval_sec)
                continue

            # Build report from historical data
            report = self._build_historical_report(candles, funding_df, current, prev_funding_rate)

            # Check coverage
            overall_cov = report.get("_overall_coverage", 0)
            if overall_cov < _MIN_COVERAGE:
                n_excluded_warmup += 1
                current += timedelta(seconds=interval_sec)
                if report.get("funding_rate") is not None:
                    prev_funding_rate = report["funding_rate"]
                continue

            # Score via unified PulseInputs (v3 parity with live pipeline)
            cfg = get_config()

            # --- Regime ---
            regime_mode = "mixed"
            try:
                df_1h_slice = candles.get("1h")
                if df_1h_slice is not None and not df_1h_slice.empty:
                    mask = df_1h_slice["timestamp"] <= current
                    hist = df_1h_slice[mask]
                    if len(hist) >= 30:
                        regime_mode = detect_regime(hist.tail(500)).mode
            except Exception:
                regime_mode = "mixed"

            # --- Realized vol windows (30m recent vs 30m prior, from 1m) ---
            rv_recent = None
            rv_prior = None
            try:
                df_1m = candles.get("1m")
                if df_1m is not None and not df_1m.empty:
                    mask = df_1m["timestamp"] <= current
                    recent = df_1m[mask].tail(60)
                    if len(recent) >= 60:
                        closes = recent["close"].astype(float).values
                        log_rets = np.diff(np.log(np.clip(closes, 1e-12, None)))
                        if len(log_rets) >= 30:
                            rv_recent = float(np.std(log_rets[-30:], ddof=1))
                            rv_prior = float(np.std(log_rets[:30], ddof=1))
            except Exception:
                pass

            # --- Support/Resistance from historical pivots (pivots only; no L2 in backtest) ---
            support = None
            resistance = None
            sr_source = "none"
            try:
                df_1h_hist = candles.get("1h")
                df_4h_hist = candles.get("4h")
                atr_1h_hist = (report.get("timeframes") or {}).get("1h", {}).get("atr")
                if df_1h_hist is not None and not df_1h_hist.empty:
                    mask1 = df_1h_hist["timestamp"] <= current
                    df_1h_slice = df_1h_hist[mask1].tail(200)
                else:
                    df_1h_slice = None
                if df_4h_hist is not None and not df_4h_hist.empty:
                    mask4 = df_4h_hist["timestamp"] <= current
                    df_4h_slice = df_4h_hist[mask4].tail(120)
                else:
                    df_4h_slice = None
                sr = compute_support_resistance(
                    spot_price=report.get("spot_price"),
                    df_1h=df_1h_slice,
                    df_4h=df_4h_slice,
                    atr_1h=atr_1h_hist,
                    l2_snapshot=None,     # honest: no historical L2 data
                    df_5m=None,
                    now=current,
                )
                support = sr.support
                resistance = sr.resistance
                sr_source = sr.source
            except Exception:
                pass

            # --- 4h return z-score ---
            z_4h_return = None
            try:
                df_1h_hist = candles.get("1h")
                if df_1h_hist is not None and not df_1h_hist.empty:
                    mask = df_1h_hist["timestamp"] <= current
                    closes = df_1h_hist[mask].tail(100)["close"].astype(float).values
                    if len(closes) >= 100:
                        log_rets_1h = np.diff(np.log(np.clip(closes, 1e-12, None)))
                        if len(log_rets_1h) >= 96:
                            ret_4h_series = np.convolve(log_rets_1h, np.ones(4), mode="valid")
                            last = float(ret_4h_series[-1])
                            ref = ret_4h_series[-91:-1]
                            sd = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
                            mu = float(np.mean(ref)) if len(ref) > 0 else 0.0
                            if sd > 1e-12:
                                z_4h_return = (last - mu) / sd
            except Exception:
                pass

            inputs = PulseInputs(
                report=report,
                signal_threshold=self.threshold,
                backtest_mode=True,
                tsmom_direction=None,      # honest: no historical TSMOM reconstruction yet
                tsmom_strength=None,
                regime_mode=regime_mode,
                realized_vol_recent=rv_recent,
                realized_vol_prior=rv_prior,
                liquidation_score=None,    # no historical liq cluster data
                book_imbalance=None,       # no historical L2 data
                prev_signal=last_signal_direction,
                ema_liquidity_ok=True,
                support=support,
                resistance=resistance,
                sr_source=sr_source,
                z_4h_return=z_4h_return,
                cfg=cfg,
            )
            result = score_pulse_from_inputs(inputs)

            if report.get("funding_rate") is not None:
                prev_funding_rate = report["funding_rate"]

            if result["signal"] == "NEUTRAL":
                current += timedelta(seconds=interval_sec)
                continue

            # De-duplication: suppress same-direction for hold_minutes
            if suppress_until is not None and current < suppress_until:
                if result["signal"] == last_signal_direction:
                    current += timedelta(seconds=interval_sec)
                    continue

            # Record signal
            entry = {
                "ts": current.isoformat(),
                "signal": result["signal"],
                "confidence": result["confidence"],
                "normalized_score": result["normalized_score"],
                "price": report.get("spot_price"),
                "stop_loss": result.get("stop_loss"),
                "take_profit": result.get("take_profit"),
                "hold_minutes": result.get("hold_minutes"),
                "timeframe_bias": result.get("timeframe_bias"),
                "breakdown": result.get("breakdown", {}),
            }
            signals.append(entry)

            last_signal_direction = result["signal"]
            suppress_until = current + timedelta(minutes=result.get("hold_minutes", 45))

            current += timedelta(seconds=interval_sec)

        logger.info(f"[PulseBacktest] Replay done: {total_intervals} intervals, "
                     f"{len(signals)} signals, {n_excluded_warmup} warmup-excluded")

        # Store metadata
        self._total_intervals = total_intervals
        self._n_excluded_warmup = n_excluded_warmup
        self._gap_count = len(gap_timestamps)

        return signals

    def _build_historical_report(
        self,
        candles: Dict[str, pd.DataFrame],
        funding_df: pd.DataFrame,
        ts: datetime,
        prev_funding_rate: Optional[float],
    ) -> dict:
        """Build a pulse report from pre-fetched historical data at timestamp ts.

        Uses the SAME _compute_tf_indicators() as the live path.
        """
        timeframes = {}
        coverages = {}

        for tf, cfg in PULSE_TIMEFRAMES.items():
            n_candles = cfg["candles"]
            df_all = candles.get(tf, pd.DataFrame())

            if df_all.empty:
                timeframes[tf] = {
                    "rsi": None, "macd_hist": None, "bb_pct": None,
                    "ema_cross": None, "rel_volume": None, "atr": None,
                    "patterns": [], "_macd_direction": None,
                }
                coverages[tf] = 0.0
                continue

            # Slice candles up to current timestamp
            mask = df_all["timestamp"] <= ts
            sliced = df_all[mask].tail(n_candles)

            indicators, cov = _compute_tf_indicators(
                sliced, tf, detect_patterns_flag=cfg["pattern"]
            )
            timeframes[tf] = indicators
            coverages[tf] = cov

        # Spot price from 1m close
        spot_price = None
        if not candles["1m"].empty:
            mask = candles["1m"]["timestamp"] <= ts
            recent = candles["1m"][mask]
            if not recent.empty:
                spot_price = float(recent.iloc[-1]["close"])

        # VWAP from 1m candles
        vwap_daily = compute_vwap_from_slice(candles["1m"], ts)
        vwap_position = 0
        if vwap_daily is not None and spot_price is not None:
            vwap_position = 1 if spot_price > vwap_daily else -1

        # Premium = 0 in backtest mode (no proxy)
        premium_pct = 0.0

        # Funding from historical data
        funding_rate = None
        funding_delta = None
        funding_acceleration = None
        if not funding_df.empty:
            mask = funding_df["timestamp"] <= ts
            recent_funding = funding_df[mask].tail(3)
            if len(recent_funding) >= 1:
                funding_rate = float(recent_funding.iloc[-1]["funding_rate"])
            if len(recent_funding) >= 2:
                rates = recent_funding["funding_rate"].values
                funding_delta = float(rates[-1] - rates[-2])
                if prev_funding_rate is not None and funding_rate is not None:
                    prev_delta = funding_rate - prev_funding_rate
                    # This is an approximation; acceleration = delta - prev_delta
                if len(rates) >= 3:
                    prev_d = float(rates[-2] - rates[-3])
                    funding_acceleration = float(funding_delta - prev_d)
            elif prev_funding_rate is not None and funding_rate is not None:
                funding_delta = funding_rate - prev_funding_rate

        # Volatility flag
        max_1m_move = 0.0
        if not candles["1m"].empty:
            mask = candles["1m"]["timestamp"] <= ts
            recent_1m = candles["1m"][mask].tail(6)
            if len(recent_1m) >= 2:
                closes = recent_1m["close"].values
                moves = [abs((closes[i] - closes[i-1]) / closes[i-1]) * 100
                         for i in range(1, len(closes)) if closes[i-1] > 0]
                max_1m_move = max(moves) if moves else 0.0

        total_cov = sum(coverages.values()) / len(coverages) if coverages else 0.0

        return {
            "ticker": self.ticker,
            "timestamp": ts.isoformat(),
            "spot_price": spot_price,
            "vwap_daily": vwap_daily,
            "vwap_position": vwap_position,
            "premium_pct": premium_pct,
            "funding_rate": funding_rate,
            "funding_delta": funding_delta,
            "funding_acceleration": funding_acceleration,
            "oi_notional": None,
            "day_volume": None,
            "max_1m_move_pct": max_1m_move,
            "timeframes": timeframes,
            "_coverages": coverages,
            "_overall_coverage": total_cov,
        }

    # ── Phase 3: Score forward returns ────────────────────────────────

    def _score_signals(
        self,
        signals: List[Dict],
        candles: Dict[str, pd.DataFrame],
    ) -> List[Dict]:
        """Score each signal's forward returns using candle OPEN."""
        if not signals or candles["1m"].empty:
            return signals

        candles_1m = candles["1m"]

        thresholds = {"+5m": (5, 0.0005), "+15m": (15, 0.0010), "+1h": (60, 0.0015)}

        for entry in signals:
            price = entry.get("price")
            if not price or price <= 0:
                continue

            try:
                pulse_ts = datetime.fromisoformat(entry["ts"])
                if pulse_ts.tzinfo is None:
                    pulse_ts = pulse_ts.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError):
                continue

            direction = 1 if entry["signal"] == "BUY" else -1

            for horizon, (minutes, min_move) in thresholds.items():
                target_ts = pulse_ts + timedelta(minutes=minutes)
                mask = candles_1m["timestamp"] >= target_ts
                if mask.any():
                    target_candle = candles_1m[mask].iloc[0]
                    fwd_price = float(target_candle["open"])
                    raw_return = (fwd_price - price) / price * direction
                    net_return = raw_return - _EXEC_COST
                    entry[f"hit_{horizon}"] = net_return >= min_move
                    entry[f"return_{horizon}"] = round(net_return, 6)

            # SL/TP hit check within hold period
            hold_min = entry.get("hold_minutes", 45)
            sl = entry.get("stop_loss")
            tp = entry.get("take_profit")
            hold_end = pulse_ts + timedelta(minutes=hold_min)

            mask_hold = (candles_1m["timestamp"] > pulse_ts) & (candles_1m["timestamp"] <= hold_end)
            hold_candles = candles_1m[mask_hold]

            entry["exit_type"] = "timeout"
            entry["exit_return"] = None

            if not hold_candles.empty and sl is not None and tp is not None:
                for _, c in hold_candles.iterrows():
                    if direction == 1:  # BUY
                        if c["low"] <= sl:
                            entry["exit_type"] = "sl_hit"
                            entry["exit_return"] = round((sl - price) / price - _EXEC_COST, 6)
                            break
                        if c["high"] >= tp:
                            entry["exit_type"] = "tp_hit"
                            entry["exit_return"] = round((tp - price) / price - _EXEC_COST, 6)
                            break
                    else:  # SHORT
                        if c["high"] >= sl:
                            entry["exit_type"] = "sl_hit"
                            entry["exit_return"] = round((price - sl) / price - _EXEC_COST, 6)
                            break
                        if c["low"] <= tp:
                            entry["exit_type"] = "tp_hit"
                            entry["exit_return"] = round((price - tp) / price - _EXEC_COST, 6)
                            break

                if entry["exit_type"] == "timeout" and not hold_candles.empty:
                    exit_price = float(hold_candles.iloc[-1]["close"])
                    raw = (exit_price - price) / price * direction
                    entry["exit_return"] = round(raw - _EXEC_COST, 6)

        return signals

    # ── Phase 4: Compute metrics ──────────────────────────────────────

    def _compute_metrics(
        self,
        signals: List[Dict],
        candles: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Aggregate metrics: hit rates, Sharpe, drawdown, regime bucketing."""
        total_intervals = getattr(self, "_total_intervals", 0)
        n_excluded_warmup = getattr(self, "_n_excluded_warmup", 0)
        gap_count = getattr(self, "_gap_count", 0)

        n_signals = len(signals)
        buy_signals = [s for s in signals if s["signal"] == "BUY"]
        short_signals = [s for s in signals if s["signal"] == "SHORT"]

        # Hit rates
        hit_rates = {}
        for horizon in ["+5m", "+15m", "+1h"]:
            key = f"hit_{horizon}"
            scored = [s for s in signals if key in s]
            hits = sum(1 for s in scored if s.get(key))
            overall = round(hits / len(scored), 4) if scored else 0

            buy_scored = [s for s in buy_signals if key in s]
            buy_hits = sum(1 for s in buy_scored if s.get(key))
            short_scored = [s for s in short_signals if key in s]
            short_hits = sum(1 for s in short_scored if s.get(key))

            hit_rates[horizon] = {
                "overall": overall,
                "BUY": round(buy_hits / len(buy_scored), 4) if buy_scored else 0,
                "SHORT": round(short_hits / len(short_scored), 4) if short_scored else 0,
            }

        # Confidence buckets (derived from threshold)
        min_conf = self.threshold / 0.5
        bucket_bounds = [
            ("low", min_conf, min_conf + 0.15),
            ("mid", min_conf + 0.15, min_conf + 0.35),
            ("high", min_conf + 0.35, 1.01),
        ]
        by_confidence = {}
        for label, lo, hi in bucket_bounds:
            bucket = [s for s in signals if lo <= s.get("confidence", 0) < hi]
            scored_bucket = [s for s in bucket if "hit_+1h" in s]
            hits = sum(1 for s in scored_bucket if s.get("hit_+1h"))
            by_confidence[label] = {
                "range": f"[{lo:.2f}, {hi:.2f})",
                "n": len(bucket),
                "hit_1h": round(hits / len(scored_bucket), 4) if scored_bucket else 0,
            }

        # Stop-and-reverse equity curve
        equity = 1.0
        equity_curve = [1.0]
        trade_returns = []

        current_direction = None
        entry_price = None

        for s in signals:
            price = s.get("price")
            if price is None or price <= 0:
                continue

            sig = s["signal"]

            # Close existing position if direction changes (stop-and-reverse)
            if current_direction is not None and sig != current_direction and entry_price is not None:
                dir_sign = 1 if current_direction == "BUY" else -1
                close_pnl = (price - entry_price) / entry_price * dir_sign
                net_pnl = close_pnl - _EXEC_COST
                equity *= (1 + net_pnl * _POSITION_SIZE)
                trade_returns.append(net_pnl)

            # Open new position
            entry_price = price
            current_direction = sig
            equity_curve.append(equity)

        # Close final position at last available price
        if current_direction is not None and entry_price is not None:
            last_price = None
            if not candles["1m"].empty:
                last_price = float(candles["1m"].iloc[-1]["close"])
            if last_price and entry_price > 0:
                dir_sign = 1 if current_direction == "BUY" else -1
                close_pnl = (last_price - entry_price) / entry_price * dir_sign
                net_pnl = close_pnl - _EXEC_COST
                equity *= (1 + net_pnl * _POSITION_SIZE)
                trade_returns.append(net_pnl)
                equity_curve.append(equity)

        # Sharpe (annualized by de-duplicated trade count)
        sharpe_ratio = 0.0
        if len(trade_returns) > 1:
            returns_arr = np.array(trade_returns)
            mean_r = np.mean(returns_arr)
            std_r = np.std(returns_arr, ddof=1)
            if std_r > 1e-10:
                sharpe_per_trade = mean_r / std_r
                days_span = (self.end_dt - self.start_dt).days
                trades_per_year = len(trade_returns) / max(days_span, 1) * 365
                sharpe_ratio = sharpe_per_trade * math.sqrt(trades_per_year)

        # Max drawdown (on full resolution curve)
        max_dd = 0.0
        peak = equity_curve[0]
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Lag-1 autocorrelation of trade returns
        autocorr_lag1 = 0.0
        if len(trade_returns) > 2:
            r = np.array(trade_returns)
            r_mean = np.mean(r)
            var = np.var(r)
            if var > 1e-20:
                autocorr_lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])

        # Regime bucketing (rolling 24h realized vol from 1h returns)
        by_regime = self._compute_regime_buckets(signals, candles)

        # Downtime events
        downtime_events = self._detect_downtime_events(candles)

        # Downsample profitability curve
        curve_ds = _downsample(equity_curve, _MAX_CURVE_POINTS)

        signal_freq = round(n_signals / total_intervals * 100, 1) if total_intervals > 0 else 0

        return {
            "ticker": self.ticker,
            "period": f"{self.start_date} to {self.end_date}",
            "total_intervals": total_intervals,
            "total_signals": n_signals,
            "signal_breakdown": {
                "BUY": len(buy_signals),
                "SHORT": len(short_signals),
                "NEUTRAL": total_intervals - n_signals,
            },
            "signal_frequency_pct": signal_freq,
            "hit_rates": hit_rates,
            "by_confidence_bucket": by_confidence,
            "by_regime": by_regime,
            "sharpe_ratio": round(sharpe_ratio, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "profitability_curve": curve_ds,
            "gap_count": gap_count,
            "downtime_events": downtime_events,
            "premium_note": "Premium set to 0 in backtest (historical data unavailable)",
            "n_excluded_warmup": n_excluded_warmup,
            "return_autocorr_lag1": round(autocorr_lag1, 4),
            "n_trades": len(trade_returns),
            "signals": signals[-200:],  # last 200 for inspection
        }

    # ── Regime bucketing ──────────────────────────────────────────────

    def _compute_regime_buckets(
        self,
        signals: List[Dict],
        candles: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Split metrics by rolling 24h realized vol (P25/P75 percentiles)."""
        if not signals or candles.get("1h") is None or candles["1h"].empty:
            return {}

        candles_1h = candles["1h"].copy()
        if len(candles_1h) < 25:
            return {}

        # Compute rolling 24h realized vol (from 1h close-to-close returns × √24)
        candles_1h = candles_1h.sort_values("timestamp").reset_index(drop=True)
        candles_1h["return_1h"] = candles_1h["close"].pct_change()
        candles_1h["vol_24h"] = candles_1h["return_1h"].rolling(24).std() * math.sqrt(24)
        candles_1h = candles_1h.dropna(subset=["vol_24h"])

        if candles_1h.empty:
            return {}

        # P25/P75 thresholds
        vol_values = candles_1h["vol_24h"].values
        p25 = float(np.percentile(vol_values, 25))
        p75 = float(np.percentile(vol_values, 75))

        # Assign each signal to a regime
        regime_signals = {"low_vol": [], "mid_vol": [], "high_vol": []}

        for s in signals:
            try:
                sig_ts = datetime.fromisoformat(s["ts"])
                if sig_ts.tzinfo is None:
                    sig_ts = sig_ts.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError):
                continue

            # Find nearest 1h vol
            mask = candles_1h["timestamp"] <= sig_ts
            if not mask.any():
                continue
            vol = float(candles_1h[mask].iloc[-1]["vol_24h"])

            if vol < p25:
                regime_signals["low_vol"].append(s)
            elif vol < p75:
                regime_signals["mid_vol"].append(s)
            else:
                regime_signals["high_vol"].append(s)

        result = {}
        for regime, sigs in regime_signals.items():
            scored = [s for s in sigs if "hit_+1h" in s]
            hits = sum(1 for s in scored if s.get("hit_+1h"))
            hit_1h = round(hits / len(scored), 4) if scored else 0

            # Per-regime sharpe from exit returns
            returns = [s.get("exit_return", 0) for s in sigs if s.get("exit_return") is not None]
            regime_sharpe = 0.0
            if len(returns) > 1:
                r = np.array(returns)
                std = np.std(r, ddof=1)
                if std > 1e-10:
                    regime_sharpe = float(np.mean(r) / std * math.sqrt(365 * 24 * 4))

            result[regime] = {
                "n": len(sigs),
                "hit_1h": hit_1h,
                "sharpe": round(regime_sharpe, 4),
            }

        return result

    # ── Downtime detection ────────────────────────────────────────────

    def _detect_downtime_events(
        self, candles: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """Detect contiguous gaps >30 min in 1m data."""
        events = []
        if candles["1m"].empty:
            return events

        ts = candles["1m"]["timestamp"].sort_values().values
        gap_start = None
        for i in range(1, len(ts)):
            gap = (pd.Timestamp(ts[i]) - pd.Timestamp(ts[i - 1])).total_seconds()
            if gap > 1800:  # 30 min
                events.append({
                    "start": str(pd.Timestamp(ts[i - 1])),
                    "end": str(pd.Timestamp(ts[i])),
                    "duration_min": round(gap / 60, 1),
                })

        return events


# ── Utility ───────────────────────────────────────────────────────────

def _downsample(curve: list, max_points: int) -> list:
    """Downsample a list to at most max_points using evenly-spaced indices."""
    if len(curve) <= max_points:
        return [round(v, 6) for v in curve]
    step = len(curve) / max_points
    indices = [int(i * step) for i in range(max_points)]
    indices[-1] = len(curve) - 1  # always include last point
    return [round(curve[i], 6) for i in indices]
