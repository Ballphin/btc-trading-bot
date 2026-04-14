"""Comprehensive tests for adaptive scoring: score_pending_decisions, get_scorecard,
compute_brier_decomposition, and run_calibration_study.

All tests write JSONL directly (never via record_decision) to include adaptive fields
like max_hold_days and position_size_pct. OHLC/price fetches are mocked.
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from tradingagents.backtesting.scorecard import (
    score_pending_decisions,
    get_scorecard,
    compute_brier_decomposition,
    run_calibration_study,
    CALIBRATION_HORIZON_DAYS,
    _SPREAD_BPS,
    _FUNDING_RATE_PER_8H,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_decisions(path: Path, decisions: list[dict]):
    """Write a list of decision dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in decisions:
            f.write(json.dumps(d, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    """Read all entries from a JSONL file."""
    entries = []
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def _make_ohlc(entry_dt: datetime, rows: list[dict]) -> pd.DataFrame:
    """Build a mock OHLC DataFrame from row dicts: {day, open, high, low, close}."""
    dates = [entry_dt + timedelta(days=r["day"]) for r in rows]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df = df[["Open", "High", "Low", "Close"]]
    return df


def _base_decision(date: str = "2026-03-01", signal: str = "BUY", price: float = 100.0,
                   confidence: float = 0.7, ticker: str = "BTC-USD", **kwargs) -> dict:
    """Create a base decision dict with all required fields."""
    d = {
        "ticker": ticker,
        "date": date,
        "signal": signal,
        "price": price,
        "confidence": confidence,
        "regime": kwargs.get("regime", "bull_quiet"),
        "stop_loss": kwargs.get("stop_loss"),
        "take_profit": kwargs.get("take_profit"),
        "max_hold_days": kwargs.get("max_hold_days", 3),
        "position_size_pct": kwargs.get("position_size_pct", 0.05),
        "source": "test",
        "recorded_at": datetime.now().isoformat(),
        "scored": False,
    }
    return d


def _scored_decision(date: str = "2026-03-01", signal: str = "BUY", price: float = 100.0,
                     confidence: float = 0.7, was_correct: bool = True,
                     net_return: float = 0.03, exit_type: str = "held_to_expiry",
                     regime: str = "bull_quiet", **kwargs) -> dict:
    """Create a pre-scored decision for get_scorecard / brier / calibration tests."""
    return {
        "ticker": kwargs.get("ticker", "BTC-USD"),
        "date": date,
        "signal": signal,
        "price": price,
        "confidence": confidence,
        "regime": regime,
        "scored": True,
        "was_correct_primary": was_correct,
        "actual_return_primary": net_return + 0.001,
        "net_return_primary": net_return,
        "exit_type": exit_type,
        "exit_price": price * (1 + net_return),
        "exit_day": kwargs.get("exit_day", 3),
        "hold_days_planned": kwargs.get("hold_days_planned", 3),
        "execution_cost": 0.001,
        "brier_score": (confidence - (1.0 if was_correct else 0.0)) ** 2,
        "scored_at": datetime.now().isoformat(),
        **{k: v for k, v in kwargs.items() if k not in ("ticker", "exit_day", "hold_days_planned")},
    }


# ── TestScorePendingDecisions ────────────────────────────────────────────


MOCK_SCORECARD = "tradingagents.backtesting.scorecard"


class TestScorePendingDecisions:
    """Tests for score_pending_decisions with mocked OHLC + datetime."""

    def _setup_dir(self, tmp_path, decisions):
        """Write decisions.jsonl and return results_dir string."""
        shadow = tmp_path / "shadow" / "BTC-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions.jsonl", decisions)
        return str(tmp_path)

    def _ohlc_for_buy(self, entry_dt, close=103.0, sl_hit=False, tp_hit=False):
        """Build OHLC data that either hits SL, TP, or expires."""
        if sl_hit:
            return _make_ohlc(entry_dt, [
                {"day": 1, "open": 100, "high": 101, "low": 93, "close": 95},
            ])
        if tp_hit:
            return _make_ohlc(entry_dt, [
                {"day": 1, "open": 100, "high": 112, "low": 99, "close": 111},
            ])
        return _make_ohlc(entry_dt, [
            {"day": 1, "open": 100, "high": 102, "low": 99, "close": 101},
            {"day": 2, "open": 101, "high": 103, "low": 100, "close": 102},
            {"day": 3, "open": 102, "high": 104, "low": 101, "close": close},
        ])

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_basic_scoring(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """Decisions older than hold period get scored."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        mock_ohlc.return_value = self._ohlc_for_buy(entry_dt)
        decisions = [_base_decision("2026-03-01", max_hold_days=3)]
        results_dir = self._setup_dir(tmp_path, decisions)
        result = score_pending_decisions("BTC-USD", results_dir)
        assert result["scored"] == 1
        assert result["pending"] == 0

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=95.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_sl_hit_exit_type(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """SL hit should produce exit_type=stop_loss_hit."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        mock_ohlc.return_value = self._ohlc_for_buy(entry_dt, sl_hit=True)
        decisions = [_base_decision("2026-03-01", stop_loss=95.0, take_profit=110.0)]
        results_dir = self._setup_dir(tmp_path, decisions)
        score_pending_decisions("BTC-USD", results_dir)
        scored = _read_jsonl(tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl")
        assert scored[0]["exit_type"] == "stop_loss_hit"
        assert scored[0]["was_correct_primary"] is False

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=111.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_tp_hit_exit_type(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """TP hit should produce exit_type=take_profit_hit."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        mock_ohlc.return_value = self._ohlc_for_buy(entry_dt, tp_hit=True)
        decisions = [_base_decision("2026-03-01", stop_loss=90.0, take_profit=110.0)]
        results_dir = self._setup_dir(tmp_path, decisions)
        score_pending_decisions("BTC-USD", results_dir)
        scored = _read_jsonl(tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl")
        assert scored[0]["exit_type"] == "take_profit_hit"
        assert scored[0]["was_correct_primary"] is True

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_no_sl_tp_held_to_expiry(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """Without SL/TP, exits as held_to_expiry."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        mock_ohlc.return_value = self._ohlc_for_buy(entry_dt)
        decisions = [_base_decision("2026-03-01", stop_loss=None, take_profit=None)]
        results_dir = self._setup_dir(tmp_path, decisions)
        score_pending_decisions("BTC-USD", results_dir)
        scored = _read_jsonl(tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl")
        assert scored[0]["exit_type"] == "held_to_expiry"

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_max_hold_none_falls_back_7d(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """max_hold_days=None should use CALIBRATION_HORIZON_DAYS (7)."""
        mock_dt.now.return_value = datetime(2026, 3, 15)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        # Provide 7 days of OHLC
        ohlc_rows = [
            {"day": i, "open": 100 + i, "high": 102 + i, "low": 99 + i, "close": 101 + i}
            for i in range(1, 8)
        ]
        mock_ohlc.return_value = _make_ohlc(entry_dt, ohlc_rows)
        decisions = [_base_decision("2026-03-01", max_hold_days=None)]
        results_dir = self._setup_dir(tmp_path, decisions)
        score_pending_decisions("BTC-USD", results_dir)
        scored = _read_jsonl(tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl")
        assert len(scored) == 1
        assert scored[0]["hold_days_planned"] == CALIBRATION_HORIZON_DAYS

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_max_hold_3_uses_3d(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """max_hold_days=3 should use 3d hold period."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        mock_ohlc.return_value = self._ohlc_for_buy(entry_dt)
        decisions = [_base_decision("2026-03-01", max_hold_days=3)]
        results_dir = self._setup_dir(tmp_path, decisions)
        score_pending_decisions("BTC-USD", results_dir)
        scored = _read_jsonl(tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl")
        assert scored[0]["hold_days_planned"] == 3

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_hold_skips_nondirectional(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """HOLD signals are skipped — not directional."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        decisions = [_base_decision("2026-03-01", signal="HOLD")]
        results_dir = self._setup_dir(tmp_path, decisions)
        result = score_pending_decisions("BTC-USD", results_dir)
        assert result["scored"] == 0

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_invalid_price_skipped(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """Decisions with price=0 or null are skipped."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        decisions = [
            _base_decision("2026-03-01", price=0),
            _base_decision("2026-03-02", price=-1),
        ]
        results_dir = self._setup_dir(tmp_path, decisions)
        result = score_pending_decisions("BTC-USD", results_dir)
        assert result["scored"] == 0

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_unparseable_date_skipped(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """Decisions with unparseable dates are skipped."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        decisions = [_base_decision("not-a-date")]
        results_dir = self._setup_dir(tmp_path, decisions)
        result = score_pending_decisions("BTC-USD", results_dir)
        assert result["scored"] == 0

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_idempotency_file_line_count(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """Scoring twice should not duplicate entries — verify by file line count."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        mock_ohlc.return_value = self._ohlc_for_buy(entry_dt)
        decisions = [_base_decision("2026-03-01")]
        results_dir = self._setup_dir(tmp_path, decisions)

        r1 = score_pending_decisions("BTC-USD", results_dir)
        assert r1["scored"] == 1
        scored_path = tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl"
        lines_after_first = len(scored_path.read_text().strip().split("\n"))

        r2 = score_pending_decisions("BTC-USD", results_dir)
        assert r2["scored"] == 0
        lines_after_second = len(scored_path.read_text().strip().split("\n"))
        assert lines_after_first == lines_after_second

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_execution_cost_deducted_short(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """For crypto SHORT, net_return < actual_return due to spread + funding."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        # SHORT profitable: close < entry
        mock_ohlc.return_value = _make_ohlc(entry_dt, [
            {"day": 1, "open": 100, "high": 101, "low": 96, "close": 97},
            {"day": 2, "open": 97, "high": 98, "low": 95, "close": 96},
            {"day": 3, "open": 96, "high": 97, "low": 94, "close": 95},
        ])
        decisions = [_base_decision("2026-03-01", signal="SHORT", max_hold_days=3)]
        results_dir = self._setup_dir(tmp_path, decisions)
        score_pending_decisions("BTC-USD", results_dir)
        scored = _read_jsonl(tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl")
        s = scored[0]
        assert s["net_return_primary"] < s["actual_return_primary"]
        assert s["execution_cost"] > 0

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_brier_score_from_primary(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """Brier score = (confidence - outcome)^2 using was_correct_primary."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        mock_ohlc.return_value = self._ohlc_for_buy(entry_dt, close=103)
        decisions = [_base_decision("2026-03-01", confidence=0.7)]
        results_dir = self._setup_dir(tmp_path, decisions)
        score_pending_decisions("BTC-USD", results_dir)
        scored = _read_jsonl(tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl")
        s = scored[0]
        expected_brier = (0.7 - 1.0) ** 2  # correct → outcome=1.0
        assert abs(s["brier_score"] - expected_brier) < 1e-5

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_legacy_compat_fields_set(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """Legacy fields was_correct_{N}d and actual_price_{N}d should be set."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        mock_ohlc.return_value = self._ohlc_for_buy(entry_dt)
        decisions = [_base_decision("2026-03-01", max_hold_days=3)]
        results_dir = self._setup_dir(tmp_path, decisions)
        score_pending_decisions("BTC-USD", results_dir)
        scored = _read_jsonl(tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl")
        s = scored[0]
        assert "was_correct_3d" in s
        assert "actual_return_3d" in s
        assert s["scored"] is True

    @patch(f"{MOCK_SCORECARD}.datetime")
    @patch(f"{MOCK_SCORECARD}._get_price_on_date", return_value=103.0)
    @patch(f"{MOCK_SCORECARD}._get_ohlc_range")
    def test_scored_file_created(self, mock_ohlc, mock_price, mock_dt, tmp_path):
        """decisions_scored.jsonl should be created with valid JSON entries."""
        mock_dt.now.return_value = datetime(2026, 3, 10)
        mock_dt.strptime = datetime.strptime
        entry_dt = datetime(2026, 3, 1)
        mock_ohlc.return_value = self._ohlc_for_buy(entry_dt)
        decisions = [_base_decision("2026-03-01")]
        results_dir = self._setup_dir(tmp_path, decisions)
        score_pending_decisions("BTC-USD", results_dir)
        scored_path = tmp_path / "shadow" / "BTC-USD" / "decisions_scored.jsonl"
        assert scored_path.exists()
        entries = _read_jsonl(scored_path)
        assert len(entries) == 1
        assert entries[0]["ticker"] == "BTC-USD"

    def test_no_decisions_file(self, tmp_path):
        """No decisions.jsonl → returns error."""
        shadow = tmp_path / "shadow" / "BTC-USD"
        shadow.mkdir(parents=True)
        result = score_pending_decisions("BTC-USD", str(tmp_path))
        assert "error" in result
        assert result["scored"] == 0


# ── TestGetScorecard ─────────────────────────────────────────────────────


class TestGetScorecard:
    """Tests for get_scorecard() reading from scored JSONL."""

    def _setup(self, tmp_path, scored_decisions, raw_decisions=None):
        shadow = tmp_path / "shadow" / "BTC-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions_scored.jsonl", scored_decisions)
        if raw_decisions:
            _write_decisions(shadow / "decisions.jsonl", raw_decisions)
        else:
            _write_decisions(shadow / "decisions.jsonl", scored_decisions)
        return str(tmp_path)

    def test_exit_type_breakdown_counts(self, tmp_path):
        scored = [
            _scored_decision("2026-03-01", exit_type="take_profit_hit"),
            _scored_decision("2026-03-02", exit_type="take_profit_hit"),
            _scored_decision("2026-03-03", exit_type="stop_loss_hit", was_correct=False),
            _scored_decision("2026-03-04", exit_type="held_to_expiry"),
        ]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        assert card["exit_type_breakdown"]["take_profit_hit"] == 2
        assert card["exit_type_breakdown"]["stop_loss_hit"] == 1
        assert card["exit_type_breakdown"]["held_to_expiry"] == 1

    def test_ev_positive_when_wins_dominate(self, tmp_path):
        # 4 wins at +3%, 1 loss at -2%
        scored = [
            _scored_decision(f"2026-03-0{i}", was_correct=True, net_return=0.03)
            for i in range(1, 5)
        ] + [_scored_decision("2026-03-05", was_correct=False, net_return=-0.02)]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        assert card["ev_per_trade_10k"] > 0

    def test_ev_negative_when_losses_dominate(self, tmp_path):
        # 1 win at +2%, 4 losses at -3%
        scored = [_scored_decision("2026-03-01", was_correct=True, net_return=0.02)] + [
            _scored_decision(f"2026-03-0{i}", was_correct=False, net_return=-0.03)
            for i in range(2, 6)
        ]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        assert card["ev_per_trade_10k"] < 0

    def test_ev_with_none_correctness_exact_arithmetic(self, tmp_path):
        """When _was_correct returns None for some decisions, those are excluded from
        wins and losses but remain in len(scored). Verify exact EV arithmetic:
        10 scored, 4 correct (+3%), 3 incorrect (-2%), 3 None → win_rate_f=0.4, loss_rate_f=0.3
        EV = (0.4 * 0.03 - 0.3 * 0.02) * 10000 = (0.012 - 0.006) * 10000 = 60.0"""
        scored = []
        # 4 correct with was_correct_primary=True
        for i in range(4):
            scored.append(_scored_decision(f"2026-03-0{i+1}", was_correct=True, net_return=0.03))
        # 3 incorrect with was_correct_primary=False
        for i in range(3):
            scored.append(_scored_decision(f"2026-03-0{i+5}", was_correct=False, net_return=-0.02))
        # 3 with NO correctness field (None) — remove was_correct_primary and was_correct_7d
        for i in range(3):
            d = _scored_decision(f"2026-03-{10+i}", net_return=0.01)
            del d["was_correct_primary"]
            scored.append(d)

        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        # win_rate_f = 4/10 = 0.4, loss_rate_f = 3/10 = 0.3
        # avg_win = 0.03, avg_loss = 0.02
        expected_ev = round((0.4 * 0.03 - 0.3 * 0.02) * 10000, 2)
        assert card["ev_per_trade_10k"] == expected_ev

    def test_avg_win_loss_return(self, tmp_path):
        scored = [
            _scored_decision("2026-03-01", was_correct=True, net_return=0.05),
            _scored_decision("2026-03-02", was_correct=True, net_return=0.03),
            _scored_decision("2026-03-03", was_correct=False, net_return=-0.04),
        ]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        assert abs(card["avg_win_return"] - 0.04) < 1e-4  # mean of 0.05 and 0.03
        assert abs(card["avg_loss_return"] - 0.04) < 1e-4  # abs(-0.04) = 0.04

    def test_overall_win_rate_uses_primary(self, tmp_path):
        """Win rate should use was_correct_primary, not was_correct_7d."""
        scored = [
            _scored_decision("2026-03-01", was_correct=True),
            _scored_decision("2026-03-02", was_correct=True),
            _scored_decision("2026-03-03", was_correct=False),
        ]
        # Override was_correct_7d to be opposite — should be ignored
        for d in scored:
            d["was_correct_7d"] = not d["was_correct_primary"]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        # Primary says 2/3 correct
        assert abs(card["overall_win_rate"] - 0.6667) < 0.01

    def test_win_by_signal_populated(self, tmp_path):
        scored = [
            _scored_decision("2026-03-01", signal="BUY", was_correct=True),
            _scored_decision("2026-03-02", signal="SHORT", was_correct=False),
        ]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        assert "BUY" in card["win_by_signal"]
        assert "SHORT" in card["win_by_signal"]
        assert card["win_by_signal"]["BUY"]["win_rate"] == 1.0
        assert card["win_by_signal"]["SHORT"]["win_rate"] == 0.0

    def test_win_by_regime_populated(self, tmp_path):
        scored = [
            _scored_decision("2026-03-01", regime="bull_quiet", was_correct=True),
            _scored_decision("2026-03-02", regime="bear_volatile", was_correct=False),
        ]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        assert "bull_quiet" in card["win_by_regime"]
        assert "bear_volatile" in card["win_by_regime"]

    def test_win_by_combo_excludes_small_samples(self, tmp_path):
        """Combos with < 3 samples should be excluded."""
        scored = [
            _scored_decision(f"2026-03-0{i}", signal="BUY", regime="bull_quiet", was_correct=True)
            for i in range(1, 4)
        ] + [
            _scored_decision("2026-03-04", signal="SHORT", regime="bear", was_correct=False),
        ]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        assert "BUY_bull_quiet" in card["win_by_combo"]
        assert "SHORT_bear" not in card["win_by_combo"]

    def test_recent_decisions_capped_at_20(self, tmp_path):
        scored = [
            _scored_decision(f"2026-03-{i:02d}", was_correct=(i % 2 == 0))
            for i in range(1, 26)  # 25 decisions
        ]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        assert len(card["recent_decisions"]) == 20

    def test_empty_scored_file(self, tmp_path):
        shadow = tmp_path / "shadow" / "BTC-USD"
        shadow.mkdir(parents=True)
        # Only decisions.jsonl, no scored file
        _write_decisions(shadow / "decisions.jsonl", [_base_decision()])
        card = get_scorecard("BTC-USD", str(tmp_path))
        assert card["scored_decisions"] == 0

    def test_brier_decomposition_included(self, tmp_path):
        """≥5 scored decisions → brier_decomposition should be present."""
        scored = [
            _scored_decision(f"2026-03-0{i}", confidence=0.6 + i * 0.02, was_correct=(i % 2 == 0))
            for i in range(1, 7)
        ]
        results_dir = self._setup(tmp_path, scored)
        card = get_scorecard("BTC-USD", results_dir)
        assert card["brier_decomposition"] is not None


# ── TestComputeBrierDecomposition ────────────────────────────────────────


class TestComputeBrierDecomposition:
    """Tests for Brier score decomposition."""

    def _setup(self, tmp_path, scored_decisions):
        shadow = tmp_path / "shadow" / "BTC-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions_scored.jsonl", scored_decisions)
        return str(tmp_path)

    def test_uses_primary_correctness(self, tmp_path):
        scored = [
            _scored_decision(f"2026-03-0{i}", confidence=0.65, was_correct=True)
            for i in range(1, 6)
        ]
        # Override was_correct_7d to False — should use primary
        for d in scored:
            d["was_correct_7d"] = False
        results_dir = self._setup(tmp_path, scored)
        result = compute_brier_decomposition("BTC-USD", results_dir)
        assert "error" not in result
        assert result["base_rate"] == 1.0  # all primary=True

    def test_fallback_to_legacy_7d(self, tmp_path):
        scored = [
            _scored_decision(f"2026-03-0{i}", confidence=0.65, was_correct=True)
            for i in range(1, 6)
        ]
        for d in scored:
            del d["was_correct_primary"]
            d["was_correct_7d"] = False
        results_dir = self._setup(tmp_path, scored)
        result = compute_brier_decomposition("BTC-USD", results_dir)
        assert "error" not in result
        assert result["base_rate"] == 0.0  # all 7d=False

    def test_3_bins_for_small_sample(self, tmp_path):
        scored = [
            _scored_decision(f"2026-03-{i:02d}", confidence=0.50 + (i % 3) * 0.15, was_correct=(i % 2 == 0))
            for i in range(1, 20)
        ]
        results_dir = self._setup(tmp_path, scored)
        result = compute_brier_decomposition("BTC-USD", results_dir)
        assert result["n_bins"] <= 3

    def test_dampen_trigger(self, tmp_path):
        """High reliability (bad calibration) → dampen=True."""
        # All predictions at 0.9 confidence, all wrong → reliability is high
        scored = [
            _scored_decision(f"2026-03-0{i}", confidence=0.9, was_correct=False)
            for i in range(1, 8)
        ]
        results_dir = self._setup(tmp_path, scored)
        result = compute_brier_decomposition("BTC-USD", results_dir)
        assert result["calibration_trigger"]["dampen"] is True

    def test_fewer_than_5_returns_error(self, tmp_path):
        scored = [_scored_decision(f"2026-03-0{i}") for i in range(1, 4)]
        results_dir = self._setup(tmp_path, scored)
        result = compute_brier_decomposition("BTC-USD", results_dir)
        assert "error" in result

    def test_brier_score_decomposition_math(self, tmp_path):
        """Verify brier_score = reliability - resolution + uncertainty."""
        scored = [
            _scored_decision(f"2026-03-{i:02d}", confidence=0.6 + (i % 4) * 0.05,
                             was_correct=(i % 3 != 0))
            for i in range(1, 12)
        ]
        results_dir = self._setup(tmp_path, scored)
        result = compute_brier_decomposition("BTC-USD", results_dir)
        expected = result["reliability"] - result["resolution"] + result["uncertainty"]
        assert abs(result["brier_score"] - expected) < 1e-5

    def test_degenerate_all_same_confidence(self, tmp_path):
        """All confidences = 0.50 → all in one bin, resolution = 0, reliability ≥ 0."""
        scored = [
            _scored_decision(f"2026-03-{i:02d}", confidence=0.50, was_correct=(i % 2 == 0))
            for i in range(1, 10)
        ]
        results_dir = self._setup(tmp_path, scored)
        result = compute_brier_decomposition("BTC-USD", results_dir)
        assert "error" not in result
        # All in one bin → resolution = 0 (single bin mean_outcome == base_rate)
        assert abs(result["resolution"]) < 1e-6
        assert result["reliability"] >= 0


# ── TestRunCalibrationStudy ──────────────────────────────────────────────


class TestRunCalibrationStudy:
    """Tests for calibration study with Bayesian shrinkage and dedup."""

    def _setup(self, tmp_path, scored_decisions):
        shadow = tmp_path / "shadow" / "BTC-USD"
        shadow.mkdir(parents=True)
        _write_decisions(shadow / "decisions_scored.jsonl", scored_decisions)
        return str(tmp_path)

    def test_dedup_keeps_highest_confidence(self, tmp_path):
        """3 decisions on same day → keeps highest confidence for dedup."""
        scored = [
            _scored_decision("2026-03-01", confidence=0.6, was_correct=True),
            _scored_decision("2026-03-01", confidence=0.8, was_correct=False),
            _scored_decision("2026-03-01", confidence=0.7, was_correct=True),
        ] + [
            _scored_decision(f"2026-03-{i:02d}", confidence=0.65, was_correct=True)
            for i in range(2, 12)  # 10 more days for min_decisions
        ]
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        assert "error" not in result
        # 3 decisions on day 1 → 1 after dedup. + 10 unique days = 11 deduped
        assert result["n_decisions_deduped"] == 11
        assert result["n_decisions_total"] == 13

    def test_shrinkage_at_n10(self, tmp_path):
        """At n=10 (deduped): w = max(0.3, 10/30) = 0.333.
        With data_correction = mean_outcome/mean_confidence:
        If 6/10 correct and mean_conf=0.65 → data_corr = 0.6/0.65 ≈ 0.923
        correction = (1-0.333)*0.85 + 0.333*0.923 = 0.567 + 0.307 = 0.874"""
        scored = []
        for i in range(6):
            scored.append(_scored_decision(f"2026-03-{i+1:02d}", confidence=0.65, was_correct=True))
        for i in range(4):
            scored.append(_scored_decision(f"2026-03-{i+7:02d}", confidence=0.65, was_correct=False))
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        assert "error" not in result
        # w = max(0.3, 10/30) = 1/3
        w = 10 / 30.0
        data_corr = 0.6 / 0.65
        expected = (1 - w) * 0.85 + w * data_corr
        expected = max(0.60, min(1.0, expected))
        assert abs(result["correction"] - round(expected, 4)) < 0.001

    def test_shrinkage_at_n20(self, tmp_path):
        """At n=20: w = 20/30 = 0.667."""
        scored = []
        for i in range(12):
            scored.append(_scored_decision(f"2026-03-{i+1:02d}", confidence=0.65, was_correct=True))
        for i in range(8):
            scored.append(_scored_decision(f"2026-04-{i+1:02d}", confidence=0.65, was_correct=False))
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        w = 20 / 30.0
        data_corr = 12 / 20 / 0.65
        expected = max(0.60, min(1.0, (1 - w) * 0.85 + w * data_corr))
        assert abs(result["correction"] - round(expected, 4)) < 0.001

    def test_shrinkage_at_n30(self, tmp_path):
        """At n=30: w = 1.0 → correction = data_correction."""
        scored = []
        for i in range(18):
            scored.append(_scored_decision(f"2026-03-{i+1:02d}", confidence=0.70, was_correct=True))
        for i in range(12):
            scored.append(_scored_decision(f"2026-04-{i+1:02d}", confidence=0.70, was_correct=False))
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        data_corr = 18 / 30 / 0.70
        expected = max(0.60, min(1.0, data_corr))
        assert abs(result["correction"] - round(expected, 4)) < 0.001

    def test_shrinkage_at_n50_same_as_n30(self, tmp_path):
        """At n=50: w = min(1.0, 50/30) = 1.0 → same as n=30."""
        scored = []
        for i in range(30):
            scored.append(_scored_decision(f"2026-03-{i+1:02d}", confidence=0.70, was_correct=True))
        for i in range(20):
            scored.append(_scored_decision(f"2026-04-{i+1:02d}", confidence=0.70, was_correct=False))
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        data_corr = 30 / 50 / 0.70
        expected = max(0.60, min(1.0, data_corr))
        assert abs(result["correction"] - round(expected, 4)) < 0.001

    def test_correction_clamped_low(self, tmp_path):
        """Extreme case: all wrong → data_correction very low → clamped to 0.60."""
        scored = [
            _scored_decision(f"2026-03-{i+1:02d}", confidence=0.9, was_correct=False)
            for i in range(12)
        ]
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        assert result["correction"] == 0.60

    def test_correction_clamped_high(self, tmp_path):
        """All correct with low confidence → data_correction > 1.0 → clamped to 1.0."""
        scored = [
            _scored_decision(f"2026-03-{i+1:02d}", confidence=0.5, was_correct=True)
            for i in range(15)
        ]
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        assert result["correction"] == 1.0

    def test_primary_fallback_to_legacy(self, tmp_path):
        """Without was_correct_primary, should use was_correct_7d."""
        scored = [
            _scored_decision(f"2026-03-{i+1:02d}", confidence=0.65, was_correct=True)
            for i in range(12)
        ]
        for d in scored:
            del d["was_correct_primary"]
            d["was_correct_7d"] = False  # all legacy False
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        # All false → mean_outcome = 0 → data_corr = 0 → clamped to 0.60
        assert result["correction"] == 0.60

    def test_regime_coverage_high(self, tmp_path):
        regimes = ["bull_quiet", "bear_volatile", "range_bound", "breakout", "mean_reverting"]
        scored = []
        for i, r in enumerate(regimes * 3):  # 15 decisions across 5 regimes
            scored.append(_scored_decision(f"2026-03-{i+1:02d}", regime=r, confidence=0.65, was_correct=True))
        results_dir = self._setup(tmp_path, scored[:15])
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        assert result["coverage_quality"] == "high"

    def test_regime_coverage_medium(self, tmp_path):
        regimes = ["bull_quiet", "bear_volatile", "range_bound"]
        scored = []
        for i, r in enumerate(regimes * 4):  # 12 decisions across 3 regimes
            scored.append(_scored_decision(f"2026-03-{i+1:02d}", regime=r, confidence=0.65, was_correct=True))
        results_dir = self._setup(tmp_path, scored[:12])
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        assert result["coverage_quality"] == "medium"

    def test_regime_coverage_low(self, tmp_path):
        scored = [
            _scored_decision(f"2026-03-{i+1:02d}", regime="bull_quiet", confidence=0.65, was_correct=True)
            for i in range(12)
        ]
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        assert result["coverage_quality"] == "low"

    def test_output_fields_complete(self, tmp_path):
        scored = [
            _scored_decision(f"2026-03-{i+1:02d}", confidence=0.65, was_correct=True)
            for i in range(12)
        ]
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        assert "n_decisions_total" in result
        assert "n_decisions_deduped" in result
        assert "correction" in result
        assert "mean_confidence" in result
        assert "mean_outcome" in result
        assert "regimes_covered" in result
        assert "coverage_quality" in result

    def test_calibration_json_written(self, tmp_path):
        scored = [
            _scored_decision(f"2026-03-{i+1:02d}", confidence=0.65, was_correct=True)
            for i in range(12)
        ]
        results_dir = self._setup(tmp_path, scored)
        run_calibration_study("BTC-USD", results_dir=results_dir)
        cal_path = tmp_path / "shadow" / "BTC-USD" / "calibration.json"
        assert cal_path.exists()
        cal = json.loads(cal_path.read_text())
        assert "correction" in cal
        assert isinstance(cal["correction"], float)

    def test_fewer_than_10_returns_error(self, tmp_path):
        scored = [
            _scored_decision(f"2026-03-0{i}", confidence=0.65, was_correct=True)
            for i in range(1, 6)
        ]
        results_dir = self._setup(tmp_path, scored)
        result = run_calibration_study("BTC-USD", results_dir=results_dir)
        assert "error" in result
