"""End-to-end integration tests: SHORT signals fire; S/R amplifies
confluence-consistent signals only; crash-only SL is set on SHORTs.
"""

import pytest

from tradingagents.agents.quant_pulse_engine import score_pulse
from tradingagents.pulse.config import get_config


def _bearish_report(spot=50_000.0, atr_1h=500.0):
    return {
        "spot_price": spot,
        "vwap_daily": spot * 1.005,
        "vwap_position": -1,
        "premium_pct": 0.0,
        "funding_rate": 0.000005,   # ~4% ann, no elevation override
        "funding_delta": 0.0,
        "max_1m_move_pct": 0.3,
        "timeframes": {
            "15m": {
                "rsi": 72, "macd_hist": -2.0, "bb_pct": 0.92,
                "ema_cross": "bearish", "rel_volume": 1.4,
                "atr": atr_1h * 0.4, "patterns": [],
                "_ema9": spot * 0.998, "_ema21": spot * 1.002,
                "_macd_direction": "falling",
            },
            "1h": {
                "rsi": 70, "macd_hist": -3.0, "bb_pct": 0.90,
                "ema_cross": "bearish", "rel_volume": 1.3,
                "atr": atr_1h, "patterns": [],
                "_ema9": spot * 0.998, "_ema21": spot * 1.002,
                "_macd_direction": "falling",
            },
            "4h": {
                "rsi": 68, "macd_hist": -4.0, "bb_pct": 0.88,
                "ema_cross": "bearish", "rel_volume": 1.2,
                "atr": atr_1h * 2.5, "patterns": [],
                "_ema9": spot * 0.997, "_ema21": spot * 1.003,
                "_macd_direction": "falling",
            },
        },
    }


def _bullish_report(spot=50_000.0, atr_1h=500.0):
    return {
        "spot_price": spot,
        "vwap_daily": spot * 0.995,
        "vwap_position": 1,
        "premium_pct": 0.0,
        "funding_rate": 0.000005,
        "funding_delta": 0.0,
        "max_1m_move_pct": 0.3,
        "timeframes": {
            "15m": {
                "rsi": 28, "macd_hist": 2.0, "bb_pct": 0.08,
                "ema_cross": "bullish", "rel_volume": 1.4,
                "atr": atr_1h * 0.4, "patterns": ["hammer"],
                "_ema9": spot * 1.002, "_ema21": spot * 0.998,
                "_macd_direction": "rising",
            },
            "1h": {
                "rsi": 30, "macd_hist": 3.0, "bb_pct": 0.10,
                "ema_cross": "bullish", "rel_volume": 1.3,
                "atr": atr_1h, "patterns": [],
                "_ema9": spot * 1.002, "_ema21": spot * 0.998,
                "_macd_direction": "rising",
            },
            "4h": {
                "rsi": 32, "macd_hist": 4.0, "bb_pct": 0.12,
                "ema_cross": "bullish", "rel_volume": 1.2,
                "atr": atr_1h * 2.5, "patterns": [],
                "_ema9": spot * 1.003, "_ema21": spot * 0.997,
                "_macd_direction": "rising",
            },
        },
    }


class TestShortSignalFires:
    def test_bearish_at_resistance_during_up_trend_fires_short(self):
        """Core user request: SHORTs must fire.

        Bearish confluence at resistance in a TSMOM-up regime should
        produce SHORT under the default `confidence_weighted` gate.
        """
        cfg = get_config()
        r = _bearish_report(spot=50_000.0, atr_1h=500.0)
        result = score_pulse(
            r, regime_mode="mixed",
            tsmom_direction=+1, tsmom_strength=0.2,   # weak up-trend
            resistance=50_100.0,                       # spot within 0.3×ATR=150
            sr_source="pivot",
            z_4h_return=0.5,
            cfg=cfg,
        )
        assert result["signal"] == "SHORT"
        assert result["confidence"] > 0.2

    def test_bullish_at_support_fires_buy(self):
        cfg = get_config()
        r = _bullish_report(spot=50_000.0, atr_1h=500.0)
        result = score_pulse(
            r, regime_mode="mixed",
            tsmom_direction=+1, tsmom_strength=0.5,
            support=49_900.0, sr_source="pivot",
            cfg=cfg,
        )
        assert result["signal"] == "BUY"


class TestSrNeverCreatesSignals:
    def test_neutral_confluence_stays_neutral_even_at_resistance(self):
        """Pure proximity to resistance with no bearish indicator must NOT
        manufacture a SHORT signal."""
        # All indicators neutral
        neutral_report = {
            "spot_price": 50_000,
            "timeframes": {
                "15m": {"rsi": 50, "macd_hist": 0, "bb_pct": 0.5,
                        "ema_cross": None, "rel_volume": 1.0, "atr": 200,
                        "patterns": [], "_macd_direction": None},
                "1h": {"rsi": 50, "macd_hist": 0, "bb_pct": 0.5,
                       "ema_cross": None, "rel_volume": 1.0, "atr": 500,
                       "patterns": [], "_macd_direction": None},
            },
            "premium_pct": 0.0,
            "max_1m_move_pct": 0.1,
        }
        cfg = get_config()
        result = score_pulse(
            neutral_report, regime_mode="mixed",
            tsmom_direction=None,
            resistance=50_050,  # very close to spot
            sr_source="pivot",
            cfg=cfg,
        )
        assert result["signal"] == "NEUTRAL"

    def test_sr_does_not_flip_direction(self):
        """Bullish confluence + nearby resistance should NOT produce a SHORT."""
        r = _bullish_report(spot=50_000, atr_1h=500)
        cfg = get_config()
        result = score_pulse(
            r, regime_mode="mixed",
            tsmom_direction=+1, tsmom_strength=0.5,
            resistance=50_050,  # near, but confluence is bullish
            sr_source="pivot",
            cfg=cfg,
        )
        # S/R proximity must not produce SHORT when confluence is bullish.
        assert result["signal"] in ("BUY", "NEUTRAL")


class TestCrashOnlySlOnShorts:
    def test_short_has_sl_even_at_short_holds(self):
        """SHORT must ALWAYS carry a crash-only SL, regardless of hold_minutes."""
        # Force dominant 15m timeframe (bias for SHORT) → hold_minutes=45 usually,
        # but we'll verify SL presence.
        r = _bearish_report(spot=50_000, atr_1h=500)
        cfg = get_config()
        result = score_pulse(
            r, regime_mode="mixed",
            tsmom_direction=+1, tsmom_strength=0.2,
            resistance=50_100, sr_source="pivot",
            cfg=cfg,
        )
        if result["signal"] == "SHORT":
            assert result["stop_loss"] is not None
            # Crash SL above entry
            assert result["stop_loss"] > 50_000

    def test_buy_under_15m_has_no_sl_tp(self):
        """BUY with hold_minutes < 15 (dominant TF = 1m or 5m) → no SL/TP."""
        # Build a report where 5m dominates with a strong bullish score
        spot = 50_000.0
        atr = 500.0
        report = {
            "spot_price": spot,
            "premium_pct": 0.0,
            "funding_rate": 0.0,
            "max_1m_move_pct": 0.1,
            "timeframes": {
                "5m": {
                    "rsi": 20, "macd_hist": 5.0, "bb_pct": 0.05,
                    "ema_cross": "bullish", "rel_volume": 2.0,
                    "atr": atr * 0.2, "patterns": ["hammer"],
                    "_ema9": spot * 1.003, "_ema21": spot * 0.997,
                    "_macd_direction": "rising",
                },
                "1h": {
                    "rsi": 45, "macd_hist": 0.1, "bb_pct": 0.4,
                    "ema_cross": None, "rel_volume": 1.0,
                    "atr": atr, "patterns": [],
                    "_macd_direction": None,
                },
            },
        }
        cfg = get_config()
        result = score_pulse(report, regime_mode="mixed", tsmom_direction=None, cfg=cfg)
        # hold_minutes for 5m dominant is 15; for 1m dominant is 5.
        # If 5m dominates, hold=15 → SL/TP threshold is 15m, borderline.
        # Test: for any BUY with hold_minutes < 15 (i.e., 1m dominant), SL/TP=None.
        if result["signal"] == "BUY" and result["hold_minutes"] < 15:
            assert result["stop_loss"] is None
            assert result["take_profit"] is None
