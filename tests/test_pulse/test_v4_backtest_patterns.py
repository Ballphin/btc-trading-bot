"""Tests for v4 pattern detection integrated into pulse_backtest.py replay loop."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from tradingagents.backtesting.pulse_backtest import PulseBacktestEngine


class TestV4BacktestPatternIntegration:
    def _make_engine(self, start="2026-01-01", end="2026-01-02"):
        return PulseBacktestEngine(
            ticker="BTC-USD",
            start_date=start,
            end_date=end,
            pulse_interval_minutes=15,
            signal_threshold=0.25,
        )

    def test_replay_calls_compute_v4_inputs_when_enabled(self):
        """When pulse_v4 is enabled, compute_v4_inputs is called during replay."""
        engine = self._make_engine()
        with patch("tradingagents.backtesting.pulse_backtest.compute_v4_inputs") as mock_v4:
            mock_v4.return_value = MagicMock(
                vpd_signal=None,
                liquidity_sweep_dir=None,
                pattern_hits={},
            )
            # Mock prefetch to return minimal candles
            with patch.object(engine, "_prefetch") as mock_prefetch:
                ts = pd.date_range("2026-01-01", "2026-01-02", freq="1min")
                df = pd.DataFrame({
                    "timestamp": ts,
                    "open": [50000.0] * len(ts),
                    "high": [50100.0] * len(ts),
                    "low": [49900.0] * len(ts),
                    "close": [50000.0 + i * 0.1 for i in range(len(ts))],
                    "volume": [100.0] * len(ts),
                })
                mock_prefetch.return_value = (
                    {"1m": df, "15m": df.iloc[::15].reset_index(drop=True),
                     "1h": df.iloc[::60].reset_index(drop=True),
                     "4h": df.iloc[::240].reset_index(drop=True)},
                    pd.DataFrame(columns=["timestamp", "funding_rate"]),
                )
                with patch.object(engine, "_score_signals", return_value=[]):
                    with patch.object(engine, "_compute_metrics", return_value={}):
                        try:
                            engine.run()
                        except Exception:
                            pass
                        assert mock_v4.called

    def test_pattern_snapshots_list_exists(self):
        engine = self._make_engine()
        assert hasattr(engine, "_pattern_snapshots")
        assert engine._pattern_snapshots == []

    def test_validate_patterns_returns_summary(self):
        engine = self._make_engine()
        engine._pattern_snapshots = [
            {
                "ts": "2026-01-01T00:00:00",
                "name": "engulfing_bullish",
                "timeframe": "1h",
                "direction": 1,
                "signal_ts": "2026-01-01T01:00:00",
            },
            {
                "ts": "2026-01-01T02:00:00",
                "name": "head_shoulders",
                "timeframe": "4h",
                "direction": -1,
                "signal_ts": "2026-01-01T03:00:00",
            },
        ]
        scored = [
            {"ts": "2026-01-01T01:00:00", "hit_+1h": True},
            {"ts": "2026-01-01T03:00:00", "hit_+1h": False},
        ]
        summary = engine._validate_patterns(scored)
        assert summary["total"] == 2
        assert summary["correct"] == 1
        assert summary["incorrect"] == 1
        assert summary["unresolved"] == 0
        assert summary["by_pattern_type"]["engulfing_bullish"]["n"] == 1
        assert summary["by_pattern_type"]["engulfing_bullish"]["correct"] == 1
        assert summary["by_pattern_type"]["head_shoulders"]["n"] == 1
        assert summary["by_pattern_type"]["head_shoulders"]["incorrect"] == 1

    def test_validate_patterns_skips_unassociated(self):
        engine = self._make_engine()
        engine._pattern_snapshots = [
            {
                "ts": "2026-01-01T00:00:00",
                "name": "engulfing_bullish",
                "timeframe": "1h",
                "direction": 1,
                "signal_ts": None,
            },
        ]
        scored = []
        summary = engine._validate_patterns(scored)
        assert summary["total"] == 0

    def test_validate_patterns_marks_unresolved(self):
        engine = self._make_engine()
        engine._pattern_snapshots = [
            {
                "ts": "2026-01-01T00:00:00",
                "name": "test",
                "timeframe": "1h",
                "direction": 1,
                "signal_ts": "2026-01-01T01:00:00",
            },
        ]
        scored = [{"ts": "2026-01-01T01:00:00"}]  # missing hit_+1h
        summary = engine._validate_patterns(scored)
        assert summary["unresolved"] == 1
        assert engine._pattern_snapshots[0]["validation"] == "unresolved"

    def test_signals_receive_arm_used_field(self):
        engine = self._make_engine()
        # Verify the signal dict now contains arm_used and patterns
        with patch.object(engine, "_prefetch") as mock_prefetch:
            ts = pd.date_range("2026-01-01", periods=100, freq="1min")
            df = pd.DataFrame({
                "timestamp": ts,
                "open": [50000.0] * len(ts),
                "high": [50100.0] * len(ts),
                "low": [49900.0] * len(ts),
                "close": [50000.0] * len(ts),
                "volume": [100.0] * len(ts),
            })
            mock_prefetch.return_value = (
                {"1m": df, "15m": df.iloc[::15].reset_index(drop=True)},
                pd.DataFrame(columns=["timestamp", "funding_rate"]),
            )
            with patch("tradingagents.backtesting.pulse_backtest.score_pulse_from_inputs") as mock_score:
                mock_score.return_value = {
                    "signal": "BUY",
                    "confidence": 0.5,
                    "normalized_score": 0.3,
                    "hold_minutes": 45,
                    "arm_used": "confluence",
                }
                with patch("tradingagents.backtesting.pulse_backtest.compute_v4_inputs") as mock_v4:
                    mock_v4.return_value = MagicMock(
                        vpd_signal=1,
                        liquidity_sweep_dir=None,
                        pattern_hits={},
                    )
                    signals = engine._replay(df, pd.DataFrame())
                    if signals:
                        s = signals[0]
                        assert "arm_used" in s
                        assert "patterns" in s
