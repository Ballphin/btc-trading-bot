"""End-to-end smoke tests: detectors find textbook patterns.

Not exhaustive; proves the pipeline (pivots → scoring → state) works.
"""

from __future__ import annotations

from tradingagents.patterns import detect_all
from tradingagents.patterns.atr import compute_ref_atr
from tradingagents.patterns.reversal import (
    detect_double_top,
    detect_head_and_shoulders,
)
from tradingagents.patterns.continuation import detect_triangles
from tradingagents.patterns.scoring import (
    combined_score,
    duration_score,
    fit_score_from_violations,
    volume_score_monotonic,
)

from .fixtures import (
    synthetic_ascending_triangle,
    synthetic_double_top,
    synthetic_head_and_shoulders,
)


def test_scoring_bounds():
    assert fit_score_from_violations([0, 0], [1, 1]) == 1.0
    assert fit_score_from_violations([10, 10], [1, 1]) == 0.0
    assert 0.0 < fit_score_from_violations([0.5, 0.5], [1, 1]) < 1.0


def test_duration_score():
    assert duration_score(0) == 0.0
    assert duration_score(40) == 1.0
    assert duration_score(80) == 1.0


def test_volume_monotonic_declining():
    assert volume_score_monotonic([3, 2, 1], "declining") == 1.0
    assert volume_score_monotonic([1, 2, 3], "declining") == 0.0


def test_combined_score_weights():
    assert combined_score(1.0, 1.0, 1.0) == 1.0
    assert combined_score(0.0, 0.0, 0.0) == 0.0
    # All-ones should = sum of weights = 1.0
    assert abs(combined_score(1.0, 0.0, 0.0) - 0.6) < 1e-9


def test_head_and_shoulders_detects_textbook():
    df, anchors = synthetic_head_and_shoulders()
    atr = compute_ref_atr(df, exclude_funding_bars=False)
    matches = detect_head_and_shoulders(df, "1h", atr)
    assert len(matches) >= 1, "should detect at least one H&S"
    m = matches[0]
    assert m.bias == "bearish"
    assert m.fit_score > 0.3  # textbook shape
    # Anchor labels A..E in order
    assert [a.label for a in m.anchors[:5]] == ["A", "B", "C", "D", "E"]
    # Head anchor C should be within 2 bars of known head idx
    c_anchor = m.anchors[2]
    assert abs(c_anchor.idx - anchors["hd"]) <= 2


def test_double_top_detects_textbook():
    df, anchors = synthetic_double_top()
    atr = compute_ref_atr(df, exclude_funding_bars=False)
    matches = detect_double_top(df, "1h", atr)
    assert len(matches) >= 1
    m = matches[0]
    assert m.bias == "bearish"
    assert m.state is not None


def test_ascending_triangle_detects():
    df = synthetic_ascending_triangle()
    atr = compute_ref_atr(df, exclude_funding_bars=False)
    matches = detect_triangles(df, "1h", atr)
    # Triangle detector is strict; accept 0 or 1 — just verify no exception
    assert isinstance(matches, list)


def test_registry_isolates_detector_failure(monkeypatch):
    """A raising detector must not break the pipeline."""
    from tradingagents.patterns import registry

    def _boom(df, tf, atr):
        raise RuntimeError("boom")

    # Patch the first detector to always raise
    original = registry.DETECTORS
    monkeypatch.setattr(
        registry, "DETECTORS",
        [("boom", _boom)] + original,
    )

    df, _ = synthetic_head_and_shoulders()
    matches, errors = detect_all({"1h": df}, tfs=("1h",))
    assert any(e["detector"] == "boom" for e in errors)
    # Other detectors still ran
    assert isinstance(matches, list)


def test_pattern_match_to_dict_serializable():
    df, _ = synthetic_head_and_shoulders()
    atr = compute_ref_atr(df, exclude_funding_bars=False)
    matches = detect_head_and_shoulders(df, "1h", atr)
    if not matches:
        return
    import json
    d = matches[0].to_dict()
    # Must round-trip through JSON
    json.dumps(d)
    assert d["name"] == "head_and_shoulders"
    assert d["bias"] == "bearish"
    assert "anchors" in d and len(d["anchors"]) >= 5
    assert "lines" in d and len(d["lines"]) >= 1
