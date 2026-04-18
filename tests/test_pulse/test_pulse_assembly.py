"""Tests for PulseInputs dataclass validation."""

import pytest

from tradingagents.pulse.pulse_assembly import PulseInputs


def _minimal_report():
    return {"spot_price": 50_000, "timeframes": {}, "timestamp": "2024-01-01T00:00:00+00:00"}


class TestPulseInputsValidation:
    def test_happy_path(self):
        inp = PulseInputs(report=_minimal_report(), signal_threshold=0.22)
        assert inp.regime_mode == "mixed"
        assert inp.sr_source == "none"
        assert inp.tsmom_direction is None

    def test_invalid_regime_mode_raises(self):
        with pytest.raises(ValueError, match="regime_mode"):
            PulseInputs(report=_minimal_report(), signal_threshold=0.22,
                        regime_mode="banana")

    def test_invalid_sr_source_raises(self):
        with pytest.raises(ValueError, match="sr_source"):
            PulseInputs(report=_minimal_report(), signal_threshold=0.22,
                        sr_source="walls")

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="signal_threshold"):
            PulseInputs(report=_minimal_report(), signal_threshold=0.0)
        with pytest.raises(ValueError, match="signal_threshold"):
            PulseInputs(report=_minimal_report(), signal_threshold=1.5)

    def test_invalid_tsmom_direction_raises(self):
        with pytest.raises(ValueError, match="tsmom_direction"):
            PulseInputs(report=_minimal_report(), signal_threshold=0.22,
                        tsmom_direction=2)

    def test_invalid_prev_signal_raises(self):
        with pytest.raises(ValueError, match="prev_signal"):
            PulseInputs(report=_minimal_report(), signal_threshold=0.22,
                        prev_signal="LONG")

    def test_valid_prev_signals(self):
        for s in (None, "BUY", "SHORT", "NEUTRAL"):
            PulseInputs(report=_minimal_report(), signal_threshold=0.22, prev_signal=s)

    def test_as_score_kwargs_contains_all_v3(self):
        inp = PulseInputs(
            report=_minimal_report(), signal_threshold=0.22,
            tsmom_direction=1, tsmom_strength=0.5, regime_mode="trend",
            support=49_000, resistance=51_000, sr_source="pivot",
            z_4h_return=1.2, prev_signal="BUY", book_imbalance=0.1,
        )
        kw = inp.as_score_kwargs()
        # All v3 fields must be present in the kwargs
        required = {
            "signal_threshold", "support", "resistance",
            "tsmom_direction", "tsmom_strength", "regime_mode",
            "liquidation_score", "realized_vol_recent", "realized_vol_prior",
            "book_imbalance", "prev_signal", "ema_liquidity_ok",
            "z_4h_return", "sr_source",
        }
        assert required.issubset(kw.keys())
        assert kw["support"] == 49_000
        assert kw["sr_source"] == "pivot"
        assert kw["z_4h_return"] == 1.2
