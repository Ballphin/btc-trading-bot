"""Tests for the HedgeFund history persistence layer (v2 plan).

Covers:
- Confidence normalization (int [0,100] → float [0,1])
- Per-ticker projection of analyst signals (no leakage between tickers)
- Microsecond-resolution filename + collision suffix fallback
- Decision context capture with yfinance failure
- Persist failure returns struct without raising
- History endpoints surface kind from directory (NOT file body)
- Get-analysis short-circuits hedgefund records past the main scorer
- Scorecard walk-forward reader does NOT pick up hedgefund logs
- Gist sync round-trip carries _kind tag
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from starlette.testclient import TestClient

import server
from server import (
    HedgeFundRequest,
    _persist_hedgefund_run,
    _project_analyst_signals,
    app,
)


@pytest.fixture
def client():
    return TestClient(app)


def _make_req(tickers, **over):
    base = dict(
        tickers=tickers,
        selected_analysts=["warren_buffett_agent"],
        start_date="2026-01-01",
        end_date="2026-03-01",
        model_name="deepseek-v4-pro",
        model_provider="DeepSeek",
        initial_cash=100000.0,
        use_nvidia_deepseek=False,
    )
    base.update(over)
    return HedgeFundRequest(**base)


def _decision(action="short", quantity=62, confidence=82, reasoning="test"):
    """Minimal duck-typed PortfolioDecision-like object."""
    class _D:
        def model_dump(self):
            return {
                "action": action,
                "quantity": quantity,
                "confidence": confidence,
                "reasoning": reasoning,
            }
    return _D()


# ── Helper-level tests ────────────────────────────────────────────────


def test_project_analyst_signals_per_ticker_only():
    all_signals = {
        "warren_buffett_agent": {
            "AAPL": {"signal": "bullish", "confidence": 80, "reasoning": "a"},
            "NVDA": {"signal": "bearish", "confidence": 70, "reasoning": "b"},
        },
        "technical_agent": {
            "AAPL": {"signal": "neutral", "confidence": 50},
        },
    }
    out_aapl = _project_analyst_signals(all_signals, "AAPL")
    out_nvda = _project_analyst_signals(all_signals, "NVDA")

    assert set(out_aapl.keys()) == {"warren_buffett_agent", "technical_agent"}
    assert out_aapl["warren_buffett_agent"]["confidence_0_1"] == pytest.approx(0.80)
    assert out_aapl["warren_buffett_agent"]["raw"]["reasoning"] == "a"

    assert set(out_nvda.keys()) == {"warren_buffett_agent"}
    assert "AAPL" not in str(out_nvda)  # no leakage


def test_project_analyst_signals_handles_missing_and_bad_types():
    assert _project_analyst_signals(None, "AAPL") == {}
    assert _project_analyst_signals({"agent": "not a dict"}, "AAPL") == {}
    assert _project_analyst_signals({"agent": {"AAPL": None}}, "AAPL") == {}
    bad_conf = {"agent": {"AAPL": {"signal": "bullish", "confidence": "oops"}}}
    out = _project_analyst_signals(bad_conf, "AAPL")
    assert out["agent"]["confidence_0_1"] is None


# ── _persist_hedgefund_run ────────────────────────────────────────────


@pytest.fixture
def patched_eval_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "EVAL_RESULTS_DIR", tmp_path)
    # Stub yfinance so tests are hermetic
    df = MagicMock()
    df.empty = False
    df.__getitem__ = lambda self, k: MagicMock(iloc=MagicMock(__getitem__=lambda self, i: 100.0))
    yf_ticker = MagicMock()
    yf_ticker.history.return_value = df
    monkeypatch.setattr(server.yf, "Ticker", lambda t: yf_ticker)
    # pd.notna returns True for the stub
    monkeypatch.setattr(server.pd, "notna", lambda v: True)
    return tmp_path


def test_persist_writes_one_file_per_ticker_with_normalized_confidence(patched_eval_dir):
    req = _make_req(["AAPL", "NVDA"])
    decisions = {"AAPL": _decision(quantity=10, confidence=82), "NVDA": _decision(action="buy", quantity=5, confidence=60)}
    signals = {
        "warren_buffett_agent": {
            "AAPL": {"signal": "bullish", "confidence": 82},
            "NVDA": {"signal": "bullish", "confidence": 60},
        }
    }
    res = _persist_hedgefund_run("job-1", req, decisions, signals)
    assert res["persisted"] is True
    assert res["error"] is None
    assert len(res["files"]) == 2

    for ticker, expected_conf in [("AAPL", 0.82), ("NVDA", 0.60)]:
        hf_dir = patched_eval_dir / ticker / "HedgeFundStrategy_logs"
        files = list(hf_dir.glob("full_states_log_*.json"))
        assert len(files) == 1
        body = json.loads(files[0].read_text())
        ts_key, payload = next(iter(body.items()))
        assert payload["ticker"] == ticker
        assert payload["confidence_0_1"] == pytest.approx(expected_conf)
        assert payload["score_eligible"] is False
        assert payload["kind"] == "hedgefund"
        assert payload["run_id"] == "job-1"
        # Per-ticker projection: NVDA file must not contain AAPL data
        agents = payload["analyst_signals"]
        for a in agents.values():
            raw = a.get("raw") or {}
            assert "AAPL" not in str(raw) if ticker == "NVDA" else True


def test_persist_collision_suffix_kicks_in(patched_eval_dir):
    req = _make_req(["AAPL"])
    decisions = {"AAPL": _decision()}
    res1 = _persist_hedgefund_run("job-1", req, decisions, {})
    assert res1["persisted"] is True

    # Force collision: rename the just-written file to a predictable name,
    # then mock the timestamp so the next write targets the same base name.
    written = Path(res1["files"][0])
    base_name = written.stem  # full_states_log_<ts>
    # Force second write to use the same ts_key by patching datetime.
    fixed_ts_key = base_name.replace("full_states_log_", "")
    with patch.object(server, "datetime") as dt_mock:
        # We need datetime.now(timezone.utc).astimezone(_USER_DISPLAY_TZ).strftime(...)
        # to produce fixed_ts_key.
        instance = MagicMock()
        instance.astimezone.return_value.strftime.return_value = fixed_ts_key
        instance.isoformat.return_value = "2026-01-01T00:00:00+00:00"
        dt_mock.now.return_value = instance
        # Re-export timezone so the function can still resolve it.
        dt_mock.now.return_value.astimezone.return_value.isoformat.return_value = "2026-01-01T00:00:00-05:00"
        res2 = _persist_hedgefund_run("job-2", req, decisions, {})
    assert res2["persisted"] is True
    # The new file must NOT overwrite the first.
    assert Path(res2["files"][0]) != written
    assert Path(res2["files"][0]).name.startswith(base_name)
    assert "-1.json" in Path(res2["files"][0]).name


def test_persist_rejects_unsafe_ticker(patched_eval_dir):
    req = _make_req(["../etc"])
    decisions = {"../etc": _decision()}
    res = _persist_hedgefund_run("job-1", req, decisions, {})
    assert res["persisted"] is False
    assert "invalid ticker" in (res["error"] or "")
    # Must not have created any directory outside the safe area
    assert not (patched_eval_dir.parent / "etc").exists()


def test_persist_yfinance_failure_records_error(patched_eval_dir, monkeypatch):
    bad_ticker = MagicMock()
    bad_ticker.history.side_effect = RuntimeError("no network")
    monkeypatch.setattr(server.yf, "Ticker", lambda t: bad_ticker)

    req = _make_req(["AAPL"])
    decisions = {"AAPL": _decision()}
    res = _persist_hedgefund_run("job-1", req, decisions, {})
    assert res["persisted"] is True
    body = json.loads(Path(res["files"][0]).read_text())
    _, payload = next(iter(body.items()))
    assert payload["price_at_decision_usd"] is None
    assert "no network" in (payload["price_capture_error"] or "")
    assert payload["notional_usd"] is None


def test_persist_write_failure_returns_struct_no_raise(patched_eval_dir, monkeypatch):
    # Force the atomic write step to raise after the cap check
    real_replace = Path.replace
    def boom(self, target):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "replace", boom)

    req = _make_req(["AAPL"])
    decisions = {"AAPL": _decision()}
    res = _persist_hedgefund_run("job-1", req, decisions, {})
    assert res["persisted"] is False
    assert "disk full" in (res["error"] or "")
    # Restore
    monkeypatch.setattr(Path, "replace", real_replace)


def test_persist_missing_decision_for_ticker(patched_eval_dir):
    req = _make_req(["AAPL", "NVDA"])
    decisions = {"AAPL": _decision()}  # NVDA missing
    res = _persist_hedgefund_run("job-1", req, decisions, {})
    # AAPL wrote, NVDA failed → overall not_persisted (error present)
    assert res["persisted"] is False
    assert "NVDA" in (res["error"] or "")
    assert len(res["files"]) == 1


# ── History API ───────────────────────────────────────────────────────


def test_list_tickers_includes_hedgefund_only_ticker(client, tmp_path):
    hf_dir = tmp_path / "AAPL" / "HedgeFundStrategy_logs"
    hf_dir.mkdir(parents=True)
    (hf_dir / "full_states_log_2026-05-16-11-30-45-123456-PM.json").write_text(
        json.dumps({"2026-05-16-11-30-45-123456-PM": {"action": "short", "quantity": 5, "confidence_0_1": 0.7}})
    )
    with patch("server.EVAL_RESULTS_DIR", tmp_path):
        resp = client.get("/api/history")
    assert resp.status_code == 200
    tickers = resp.json()["tickers"]
    assert any(t["ticker"] == "AAPL" and t["analysis_count"] == 1 for t in tickers)


def test_list_analyses_returns_both_kinds(client, tmp_path):
    main = tmp_path / "AAPL" / "TradingAgentsStrategy_logs"
    hf = tmp_path / "AAPL" / "HedgeFundStrategy_logs"
    main.mkdir(parents=True)
    hf.mkdir(parents=True)
    (main / "full_states_log_2026-05-15.json").write_text(
        json.dumps({"2026-05-15": {"final_trade_decision": "BUY"}})
    )
    (hf / "full_states_log_2026-05-16-11-30-45-123456-PM.json").write_text(
        json.dumps({"2026-05-16-11-30-45-123456-PM": {"action": "short", "quantity": 5, "confidence_0_1": 0.7}})
    )
    with patch("server.EVAL_RESULTS_DIR", tmp_path):
        resp = client.get("/api/history/AAPL")
    assert resp.status_code == 200
    rows = resp.json()["analyses"]
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["hedgefund", "main"]
    hf_row = next(r for r in rows if r["kind"] == "hedgefund")
    assert hf_row["action"] == "short"
    assert hf_row["quantity"] == 5
    assert hf_row["candle_time"] == "2026-05-16-11-30-45-123456-PM"


def test_get_analysis_resolves_hedgefund_file(client, tmp_path):
    hf = tmp_path / "AAPL" / "HedgeFundStrategy_logs"
    hf.mkdir(parents=True)
    body = {
        "2026-05-16-11-30-45-123456-PM": {
            "kind": "hedgefund",
            "ticker": "AAPL",
            "action": "short",
            "quantity": 5,
            "confidence_0_1": 0.7,
            "reasoning": "test",
            "analyst_signals": {},
            "analyst_signals_empty": True,
            "score_eligible": False,
            "ts_local": "2026-05-16T23:30:45-04:00",
        }
    }
    (hf / "full_states_log_2026-05-16-11-30-45-123456-PM.json").write_text(json.dumps(body))
    with patch("server.EVAL_RESULTS_DIR", tmp_path):
        resp = client.get("/api/history/AAPL/2026-05-16-11-30-45-123456-PM")
    assert resp.status_code == 200
    env = resp.json()
    assert env["kind"] == "hedgefund"
    assert env["data"]["action"] == "short"
    assert env["data"]["confidence_0_1"] == 0.7


def test_get_analysis_main_precedence_on_ambiguous(client, tmp_path):
    """If both dirs contain the same filename, main wins (defensive)."""
    main = tmp_path / "AAPL" / "TradingAgentsStrategy_logs"
    hf = tmp_path / "AAPL" / "HedgeFundStrategy_logs"
    main.mkdir(parents=True)
    hf.mkdir(parents=True)
    (main / "full_states_log_2026-05-15.json").write_text(
        json.dumps({"2026-05-15": {"final_trade_decision": "BUY MAIN"}})
    )
    (hf / "full_states_log_2026-05-15.json").write_text(
        json.dumps({"2026-05-15": {"action": "short"}})
    )
    with patch("server.EVAL_RESULTS_DIR", tmp_path):
        resp = client.get("/api/history/AAPL/2026-05-15")
    assert resp.status_code == 200
    env = resp.json()
    assert env["kind"] == "main"


# ── Scorecard isolation ───────────────────────────────────────────────


def test_walk_forward_skips_hedgefund_dir(tmp_path):
    """walk_forward globs ONLY TradingAgentsStrategy_logs; hedgefund logs
    must not pollute hit-rate computation."""
    from tradingagents.backtesting.walk_forward import WalkForwardValidator
    hf = tmp_path / "AAPL" / "HedgeFundStrategy_logs"
    hf.mkdir(parents=True)
    (hf / "full_states_log_2026-05-16-11-30-45-123456-PM.json").write_text(
        json.dumps({"2026-05-16-11-30-45-123456-PM": {
            "action": "short", "score_eligible": False
        }})
    )
    v = WalkForwardValidator(ticker="AAPL", results_dir=str(tmp_path))
    decisions = v._load_decisions_from_logs()
    assert decisions == []


# ── Gist sync round-trip ──────────────────────────────────────────────


def test_gist_push_packs_kind_tagged_records(tmp_path, monkeypatch):
    """push_history must include _kind tags so pull can route correctly."""
    from tradingagents.pulse import gist_sync

    main = tmp_path / "AAPL" / "TradingAgentsStrategy_logs"
    hf = tmp_path / "AAPL" / "HedgeFundStrategy_logs"
    main.mkdir(parents=True)
    hf.mkdir(parents=True)
    (main / "full_states_log_2026-05-15.json").write_text(json.dumps({"x": 1}))
    (hf / "full_states_log_2026-05-16-11-30-45-123456-PM.json").write_text(json.dumps({"y": 2}))

    monkeypatch.setattr(gist_sync, "_history_enabled", lambda: True)
    monkeypatch.setenv("HISTORY_GIST_ID", "abc")
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    captured = {}
    fake_resp = MagicMock(status_code=200)
    fake_resp.raise_for_status = lambda: None
    def fake_patch(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return fake_resp
    monkeypatch.setattr("requests.patch", fake_patch)

    assert gist_sync.push_history(tmp_path, "AAPL") is True
    blob = captured["payload"]["files"]["history_AAPL.jsonl"]["content"]
    lines = [l for l in blob.splitlines() if l.strip()]
    assert len(lines) == 2
    kinds = sorted(json.loads(l)["_kind"] for l in lines)
    assert kinds == ["hedgefund", "main"]


def test_gist_pull_routes_by_kind(tmp_path, monkeypatch):
    from tradingagents.pulse import gist_sync

    monkeypatch.setattr(gist_sync, "_history_enabled", lambda: True)
    monkeypatch.setenv("HISTORY_GIST_ID", "abc")
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    # Mock the gist GET to return two records, different _kind
    main_rec = {"filename": "full_states_log_2026-05-15.json", "content": {"x": 1}, "_kind": "main"}
    hf_rec = {"filename": "full_states_log_2026-05-16-11-30-45-123456-PM.json", "content": {"y": 2}, "_kind": "hedgefund"}
    jsonl = json.dumps(main_rec) + "\n" + json.dumps(hf_rec)
    fake_resp = MagicMock()
    fake_resp.raise_for_status = lambda: None
    fake_resp.json = lambda: {"files": {"history_AAPL.jsonl": {"content": jsonl}}}
    monkeypatch.setattr("requests.get", lambda url, headers=None, timeout=None: fake_resp)

    result = gist_sync.pull_history_all(tmp_path)
    assert result["total_files"] == 2
    assert (tmp_path / "AAPL" / "TradingAgentsStrategy_logs" / "full_states_log_2026-05-15.json").exists()
    assert (tmp_path / "AAPL" / "HedgeFundStrategy_logs" / "full_states_log_2026-05-16-11-30-45-123456-PM.json").exists()


# ── Size cap ──────────────────────────────────────────────────────────


def test_persist_size_cap_truncates(patched_eval_dir):
    """A pathologically large analyst_signals payload triggers truncation."""
    req = _make_req(["AAPL"])
    decisions = {"AAPL": _decision(reasoning="r" * 5000)}
    # Build a huge analyst_signals dict so the serialized blob exceeds 256 KB.
    huge_blob = "x" * 30000
    signals = {
        f"agent_{i}": {"AAPL": {"signal": "bullish", "confidence": 50, "blob": huge_blob}}
        for i in range(20)
    }
    res = _persist_hedgefund_run("job-1", req, decisions, signals)
    assert res["persisted"] is True
    body = json.loads(Path(res["files"][0]).read_text())
    _, payload = next(iter(body.items()))
    assert payload["truncated"] is True
    # Raw payloads must have been dropped
    for a in payload["analyst_signals"].values():
        assert "raw" not in a
