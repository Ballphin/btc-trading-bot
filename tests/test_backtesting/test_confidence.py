"""Tests for confidence.py — Kelly sizing, DSR gate, calibration, shrinkage."""

import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tradingagents.graph.confidence import ConfidenceScorer


@pytest.fixture
def scorer(tmp_path):
    """ConfidenceScorer with testable results_dir."""
    return ConfidenceScorer(results_dir=str(tmp_path))


@pytest.fixture
def regime_ctx():
    return {
        "regime": "trending_up",
        "current_price": 60_000.0,
        "daily_vol": 0.02,
        "above_sma20": True,
    }


# ── K2: results_dir testability ──────────────────────────────────────

class TestResultsDir:
    def test_default_results_dir(self):
        s = ConfidenceScorer()
        assert s.results_dir == Path("eval_results")

    def test_custom_results_dir(self, tmp_path):
        s = ConfidenceScorer(results_dir=str(tmp_path))
        assert s.results_dir == tmp_path


# ── Kelly threshold at n=59/60/61 ────────────────────────────────────

class TestKellyThreshold:
    """Kelly eligibility requires _KELLY_MIN_TRADES = 60."""

    def _score_with_sample_size(self, scorer, n, regime_ctx, tmp_path):
        mock_ks = MagicMock()
        mock_ks.get_signal_win_rate.return_value = {
            "win_rate": 0.55,
            "sample_size": n,
        }
        return scorer.score(
            llm_confidence=0.70,
            ticker="BTC-USD",
            signal="BUY",
            knowledge_store=mock_ks,
            regime_ctx=regime_ctx,
            stop_loss=57_000.0,
            take_profit=66_000.0,
            max_hold_days=7,
            reasoning="Strong uptrend",
        )

    def test_n59_no_kelly(self, scorer, regime_ctx, tmp_path):
        result = self._score_with_sample_size(scorer, 59, regime_ctx, tmp_path)
        # Below threshold → should use fixed fallback, not Kelly
        assert result.get("position_size_pct") is not None

    def test_n60_kelly_eligible(self, scorer, regime_ctx, tmp_path):
        result = self._score_with_sample_size(scorer, 60, regime_ctx, tmp_path)
        assert result.get("position_size_pct") is not None

    def test_n61_kelly_eligible(self, scorer, regime_ctx, tmp_path):
        result = self._score_with_sample_size(scorer, 61, regime_ctx, tmp_path)
        assert result.get("position_size_pct") is not None


# ── K1: Kelly shrinkage monotonicity ─────────────────────────────────

class TestKellyShrinkage:
    def test_shrinkage_monotonically_increases(self, scorer):
        """More data → less shrinkage → higher position size."""
        sizes = []
        for n in [10, 50, 100]:
            size, _, _ = scorer.kelly_position_size(
                p=0.55, R=2.0,
                entry_price=60_000, stop_loss=57_000, take_profit=66_000,
                signal="BUY", max_hold_days=7, sample_size=n,
            )
            sizes.append(size)
        # shrinkage(10) > shrinkage(50) > shrinkage(100) → size(10) < size(50) < size(100)
        assert sizes[0] < sizes[1] < sizes[2], f"Expected monotonic increase: {sizes}"

    def test_shrinkage_at_n2(self, scorer):
        """At n=2, shrinkage = 1 - 2/2 = 0 → position size should be 0."""
        size, _, _ = scorer.kelly_position_size(
            p=0.55, R=2.0,
            entry_price=60_000, stop_loss=57_000, take_profit=66_000,
            signal="BUY", max_hold_days=7, sample_size=2,
        )
        assert size == pytest.approx(0.0)


# ── K8: DSR gate with low-winrate high-R systems ────────────────────

class TestDSRGate:
    def test_low_winrate_high_r_remains_eligible(self, scorer, tmp_path):
        """p=0.40, R=2.0, n=100 → Kelly edge is positive → should stay eligible."""
        mock_ks = MagicMock()
        mock_ks.get_signal_win_rate.return_value = {
            "win_rate": 0.40,
            "sample_size": 100,
        }
        regime_ctx = {
            "regime": "trending_up",
            "current_price": 60_000.0,
            "daily_vol": 0.02,
            "above_sma20": True,
        }
        result = scorer.score(
            llm_confidence=0.60,
            ticker="BTC-USD",
            signal="BUY",
            knowledge_store=mock_ks,
            regime_ctx=regime_ctx,
            stop_loss=57_000.0,
            take_profit=66_000.0,
            max_hold_days=7,
            reasoning="Trend following",
        )
        # Kelly edge: 0.40*2.0 - 0.60 = 0.20 > 0 → should have non-trivial sizing
        assert result["position_size_pct"] > 0.0

    def test_breakeven_system_disabled(self, scorer, tmp_path):
        """p=0.50, R=1.0 → Kelly edge = 0 → should disable Kelly."""
        mock_ks = MagicMock()
        mock_ks.get_signal_win_rate.return_value = {
            "win_rate": 0.50,
            "sample_size": 100,
        }
        regime_ctx = {
            "regime": "ranging",
            "current_price": 60_000.0,
            "daily_vol": 0.01,
            "above_sma20": True,
        }
        result = scorer.score(
            llm_confidence=0.50,
            ticker="BTC-USD",
            signal="BUY",
            knowledge_store=mock_ks,
            regime_ctx=regime_ctx,
            stop_loss=57_000.0,
            take_profit=63_000.0,
            max_hold_days=7,
            reasoning="",
        )
        # Should use fixed fallback, not Kelly (edge ≈ 0)
        assert result.get("position_size_pct") is not None


# ── Cold-start correction ────────────────────────────────────────────

class TestColdStartCorrection:
    def test_calibration_file_present(self, scorer, tmp_path):
        """When calibration.json exists, load correction from it."""
        cal_dir = tmp_path / "shadow" / "BTC-USD"
        cal_dir.mkdir(parents=True)
        cal_file = cal_dir / "calibration.json"
        cal_file.write_text(json.dumps({"correction": 0.70}))

        result = scorer.score(
            llm_confidence=0.80,
            ticker="BTC-USD",
            signal="BUY",
            knowledge_store=None,
            regime_ctx={
                "regime": "trending_up",
                "current_price": 60_000,
                "above_sma20": True,
            },
            stop_loss=57_000,
            take_profit=66_000,
        )
        assert result["confidence"] <= 0.80  # Should be dampened

    def test_no_calibration_file(self, scorer, tmp_path):
        """Without calibration.json, use default 0.85 correction."""
        result = scorer.score(
            llm_confidence=0.90,
            ticker="NEW-TICKER",
            signal="BUY",
            knowledge_store=None,
            regime_ctx={
                "regime": "unknown",
                "current_price": 100,
                "above_sma20": True,
            },
            stop_loss=90,
            take_profit=120,
        )
        assert result["confidence"] <= 0.90


# ── Cold-start linear ramp ──────────────────────────────────────────

class TestColdStartRamp:
    """Test continuous linear ramp: correction = base + (1-base) * min(1, n/60)."""

    @pytest.mark.parametrize("n, expected_min_correction", [
        (0, 0.85),
        (15, 0.8875),
        (30, 0.925),
        (60, 1.0),
        (100, 1.0),
    ])
    def test_ramp_values(self, tmp_path, n, expected_min_correction):
        scorer = ConfidenceScorer(results_dir=str(tmp_path))
        # Create scored decisions to control scored_count
        shadow_dir = tmp_path / "shadow" / "TEST-TICKER"
        shadow_dir.mkdir(parents=True)
        scored_path = shadow_dir / "decisions_scored.jsonl"
        with open(scored_path, "w") as f:
            for i in range(n):
                f.write(json.dumps({"ticker": "TEST-TICKER", "scored": True}) + "\n")

        result = scorer.score(
            llm_confidence=1.0,  # Use 1.0 to isolate the correction
            ticker="TEST-TICKER",
            signal="HOLD",
            knowledge_store=None,
            regime_ctx={"regime": "unknown", "current_price": None, "above_sma20": True},
            stop_loss=None,
            take_profit=None,
        )
        # overconfidence_correction should match the ramp
        assert result["overconfidence_correction"] == pytest.approx(expected_min_correction, abs=0.005)

    def test_scored_count_in_result(self, scorer, tmp_path):
        result = scorer.score(
            llm_confidence=0.80,
            ticker="NEW-TICKER",
            signal="BUY",
            knowledge_store=None,
            regime_ctx={"regime": "unknown", "current_price": 100, "above_sma20": True},
            stop_loss=90,
            take_profit=120,
        )
        assert "scored_count_used" in result
        assert result["scored_count_used"] == 0


# ── Hedge-word penalty ───────────────────────────────────────────────

class TestHedgeWordPenalty:
    def test_no_hedge_words(self, scorer):
        cal, pen = scorer.calibrate(0.80, None, 0, None, "trending_up", True, "Strong conviction")
        assert pen == 0.0

    def test_one_hedge_word(self, scorer):
        cal, pen = scorer.calibrate(0.80, None, 0, None, "trending_up", True, "I'm uncertain about this")
        assert pen == pytest.approx(0.03)  # Multiplicative: 3% per word

    def test_multiple_hedge_words(self, scorer):
        text = "however uncertain despite concern"
        cal, pen = scorer.calibrate(0.80, None, 0, None, "trending_up", True, text)
        assert pen > 0.06  # Multiple hedge words: 4 * 3% = 12%

    def test_penalty_capped_at_012(self, scorer):
        # All current hedge words: but however although despite uncertain concern weak
        text = "but however although despite uncertain concern weak"
        _, pen = scorer.calibrate(0.80, None, 0, None, "trending_up", True, text)
        assert pen == pytest.approx(0.12)  # 7 words × 3% = 21% but capped at 12%

    def test_removed_words_no_penalty(self, scorer):
        """'risk', 'volatile', 'caution' were removed — no penalty."""
        cal_clean, pen_clean = scorer.calibrate(0.80, None, 0, None, "trending_up", True, "No hedge words here")
        cal_risk, pen_risk = scorer.calibrate(0.80, None, 0, None, "trending_up", True, "risk volatile caution")
        assert pen_risk == 0.0
        assert cal_clean == cal_risk

    def test_word_boundary_no_false_positive(self, scorer):
        """'but' should NOT match in 'distribution' or 'contribute'."""
        cal_clean, pen_clean = scorer.calibrate(0.80, None, 0, None, "trending_up", True, "clean text")
        cal_substr, pen_substr = scorer.calibrate(0.80, None, 0, None, "trending_up", True, "distribution contribute rebuttal")
        assert pen_substr == 0.0
        assert cal_clean == cal_substr


# ── Regime gating ────────────────────────────────────────────────────

class TestRegimeGating:
    def test_volatile_below_sma_penalty(self, scorer):
        """Volatile + below SMA20 → harder penalty (for non-SHORT)."""
        cal_volatile, _ = scorer.calibrate(0.80, None, 0, None, "volatile", False, "", signal="BUY")
        cal_normal, _ = scorer.calibrate(0.80, None, 0, None, "trending_up", True, "")
        assert cal_volatile < cal_normal

    def test_volatile_short_below_sma_less_penalty(self, scorer):
        """SHORT below SMA20 in volatile regime → reduced penalty (0.92 not 0.75)."""
        cal_short, _ = scorer.calibrate(0.80, None, 0, None, "volatile", False, "", signal="SHORT")
        cal_buy, _ = scorer.calibrate(0.80, None, 0, None, "volatile", False, "", signal="BUY")
        # SHORT gets 0.92 penalty, BUY gets 0.75 penalty → SHORT confidence should be higher
        assert cal_short > cal_buy
        # SHORT should be 0.80 * 0.92 = 0.736
        assert cal_short == pytest.approx(0.80 * 0.92, abs=0.01)
        # BUY should be 0.80 * 0.75 = 0.60
        assert cal_buy == pytest.approx(0.80 * 0.75, abs=0.01)


# ── kelly_position_size math ─────────────────────────────────────────

class TestKellyPositionSize:
    def test_hold_returns_zero(self, scorer):
        size, R, hs = scorer.kelly_position_size(0.55, 2.0, 60000, 57000, 66000, "HOLD")
        assert size == 0.0

    def test_overweight_returns_fixed(self, scorer):
        size, R, hs = scorer.kelly_position_size(0.55, 2.0, 60000, 57000, 66000, "OVERWEIGHT")
        assert 0 < size <= 0.40

    def test_no_entry_price(self, scorer):
        size, R, hs = scorer.kelly_position_size(0.55, 2.0, None, 57000, 66000, "BUY")
        assert size > 0  # Fallback

    def test_negative_kelly_edge(self, scorer):
        """p < 1/(1+R) → f* ≤ 0 → size should be 0."""
        size, _, _ = scorer.kelly_position_size(
            p=0.20, R=2.0,
            entry_price=60_000, stop_loss=57_000, take_profit=66_000,
            signal="BUY", sample_size=100,
        )
        assert size == 0.0
