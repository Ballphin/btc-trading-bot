"""Lightweight endpoint tests (no live data fetch needed for 404 path)."""

from fastapi.testclient import TestClient


def _client():
    import server
    return TestClient(server.app)


def test_explain_404_on_unknown_ts():
    client = _client()
    # Ticker likely has NO pulse entry at this random ts
    resp = client.get("/api/pulse/explain/XYZ-UNKNOWN/2020-01-01T00:00:00+00:00")
    assert resp.status_code == 404
    body = resp.json()
    assert "No pulse entry" in body["detail"]


def test_read_pulse_at_missing_ticker_returns_none():
    from server import _read_pulse_at
    assert _read_pulse_at("DOES_NOT_EXIST", "2020-01-01T00:00:00+00:00") is None
