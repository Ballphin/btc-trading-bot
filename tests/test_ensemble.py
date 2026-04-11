"""Tests for ensemble analysis components."""

import pytest
from tradingagents.graph.consensus_engine import ConsensusEngine, ConsensusResult
from tradingagents.graph.ensemble_orchestrator import should_use_ensemble
from tradingagents.backtesting.ensemble_tracker import EnsembleTracker


class TestConsensusEngine:
    """Test the consensus computation logic."""

    def test_compute_r_ratio_buy(self):
        """R:R computation for BUY signal."""
        engine = ConsensusEngine()
        rr = engine.compute_r_ratio(entry=100, stop=95, take=110, signal="BUY")
        assert rr == 2.0  # (110-100)/(100-95) = 10/5 = 2

    def test_compute_r_ratio_short(self):
        """R:R computation for SHORT signal."""
        engine = ConsensusEngine()
        rr = engine.compute_r_ratio(entry=100, stop=105, take=90, signal="SHORT")
        assert rr == 2.0  # (100-90)/(105-100) = 10/5 = 2

    def test_compute_r_ratio_invalid(self):
        """R:R returns None for invalid inputs."""
        engine = ConsensusEngine()
        assert engine.compute_r_ratio(100, 100, 110, "BUY") is None  # stop == entry
        assert engine.compute_r_ratio(100, 95, 100, "BUY") is None  # take == entry
        assert engine.compute_r_ratio(0, 95, 110, "BUY") is None  # entry == 0

    def test_majority_vote_simple(self):
        """Majority vote with clear winner."""
        engine = ConsensusEngine()
        result = engine.majority_vote(["BUY", "BUY", "SELL"])
        assert result == "BUY"

    def test_compute_consensus_basic(self):
        """Full consensus computation."""
        engine = ConsensusEngine()
        results = [
            {"signal": "BUY", "confidence": 0.75, "stop_loss_price": 95, "take_profit_price": 110, "max_hold_days": 3, "reasoning": "Test 1"},
            {"signal": "BUY", "confidence": 0.70, "stop_loss_price": 94, "take_profit_price": 112, "max_hold_days": 4, "reasoning": "Test 2"},
            {"signal": "BUY", "confidence": 0.80, "stop_loss_price": 96, "take_profit_price": 108, "max_hold_days": 3, "reasoning": "Test 3"},
        ]
        
        consensus = engine.compute_consensus(results, entry_price=100, ticker="TEST")
        
        assert consensus.signal == "BUY"
        assert 0.70 < consensus.confidence < 0.80  # Average of 0.75, 0.70, 0.80
        assert consensus.max_hold_days == 3  # Median
        assert "divergence_metrics" in consensus.ensemble_metadata

    def test_should_rerun_divergence(self):
        """Detect when re-run is needed due to high divergence."""
        engine = ConsensusEngine()
        
        # Low divergence - no rerun
        low_div = [
            {"signal": "BUY", "confidence": 0.72},
            {"signal": "BUY", "confidence": 0.75},
            {"signal": "BUY", "confidence": 0.73},
        ]
        assert not engine.should_rerun(low_div, confidence_range_threshold=0.20)
        
        # High confidence divergence - should rerun
        high_div = [
            {"signal": "BUY", "confidence": 0.90},
            {"signal": "BUY", "confidence": 0.60},
            {"signal": "BUY", "confidence": 0.65},
        ]
        assert engine.should_rerun(high_div, confidence_range_threshold=0.20)

    def test_should_rerun_signal_disagreement(self):
        """Detect when re-run is needed due to signal disagreement."""
        engine = ConsensusEngine()
        
        # 3/3 agreement - should not rerun
        perfect_agreement = [
            {"signal": "BUY", "confidence": 0.70},
            {"signal": "BUY", "confidence": 0.75},
            {"signal": "BUY", "confidence": 0.60},
        ]
        assert not engine.should_rerun(perfect_agreement, agreement_threshold=0.67)
        
        # 2/3 = 0.67 agreement - at threshold boundary (should not rerun with <= threshold)
        ok_agreement = [
            {"signal": "BUY", "confidence": 0.70},
            {"signal": "BUY", "confidence": 0.75},
            {"signal": "SELL", "confidence": 0.60},
        ]
        # Note: 2/3 = 0.666... which is < 0.67, so this WILL trigger rerun at 0.67 threshold
        # This is correct - we want 67% or better agreement
        assert engine.should_rerun(ok_agreement, agreement_threshold=0.67)
        
        # 1/3 agreement - should definitely rerun
        bad_agreement = [
            {"signal": "BUY", "confidence": 0.70},
            {"signal": "SELL", "confidence": 0.75},
            {"signal": "HOLD", "confidence": 0.60},
        ]
        assert engine.should_rerun(bad_agreement, agreement_threshold=0.67)


class TestShouldUseEnsemble:
    """Test provider-specific ensemble rules."""

    def test_openrouter_enabled(self):
        """OpenRouter should use ensemble."""
        config = {
            "ensemble_enabled_providers": ["openrouter"],
            "ensemble_disabled_providers": ["deepseek"],
            "enable_ensemble": True,
        }
        assert should_use_ensemble(config, "openrouter") is True

    def test_deepseek_disabled(self):
        """DeepSeek should not use ensemble."""
        config = {
            "ensemble_enabled_providers": ["openrouter"],
            "ensemble_disabled_providers": ["deepseek"],
            "enable_ensemble": True,
        }
        assert should_use_ensemble(config, "deepseek") is False

    def test_unknown_provider_defaults_to_global(self):
        """Unknown provider uses global setting."""
        config = {
            "ensemble_enabled_providers": ["openrouter"],
            "ensemble_disabled_providers": ["deepseek"],
            "enable_ensemble": True,
        }
        assert should_use_ensemble(config, "unknown") is True

        config["enable_ensemble"] = False
        assert should_use_ensemble(config, "unknown") is False


class TestEnsembleTracker:
    """Test ensemble accuracy tracking."""

    def test_log_result(self, tmp_path):
        """Log an ensemble result."""
        tracker = EnsembleTracker(results_dir=str(tmp_path))
        
        tracker.log_result(
            timestamp="2026-04-11T12:00:00",
            ticker="BTC-USD",
            consensus_signal="BUY",
            consensus_confidence=0.75,
            individual_signals=[
                {"signal": "BUY", "confidence": 0.70},
                {"signal": "BUY", "confidence": 0.75},
                {"signal": "BUY", "confidence": 0.80},
            ],
            market_outcome="win",
        )
        
        stats = tracker.get_accuracy_stats()
        assert stats["total"] == 1
        assert stats["correct"] == 1
        assert stats["accuracy"] == 1.0

    def test_get_stats_by_ticker(self, tmp_path):
        """Filter stats by ticker."""
        tracker = EnsembleTracker(results_dir=str(tmp_path))
        
        tracker.log_result(
            timestamp="2026-04-11T12:00:00",
            ticker="BTC-USD",
            consensus_signal="BUY",
            consensus_confidence=0.75,
            individual_signals=[{"signal": "BUY", "confidence": 0.70}],
        )
        
        tracker.log_result(
            timestamp="2026-04-11T12:00:00",
            ticker="ETH-USD",
            consensus_signal="SELL",
            consensus_confidence=0.65,
            individual_signals=[{"signal": "SELL", "confidence": 0.65}],
        )
        
        btc_stats = tracker.get_accuracy_stats(ticker="BTC-USD")
        assert btc_stats["total"] == 1
        
        all_stats = tracker.get_accuracy_stats()
        assert all_stats["total"] == 2
