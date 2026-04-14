"""Direct tests for the shared date_utils.parse_any_date function.

Covers all 4 system date formats, edge cases, and regex fallback.
"""

from datetime import datetime

import pytest

from tradingagents.backtesting.date_utils import parse_any_date


class TestParseAnyDate:
    """Tests for parse_any_date covering all system formats and edge cases."""

    def test_daily_format(self):
        result = parse_any_date("2026-04-08")
        assert result == datetime(2026, 4, 8)

    def test_4h_scheduler_format(self):
        result = parse_any_date("2026-04-08T16")
        assert result == datetime(2026, 4, 8, 16, 0)

    def test_intraday_format(self):
        result = parse_any_date("2026-04-08T16:00")
        assert result == datetime(2026, 4, 8, 16, 0)

    def test_manual_run_pm(self):
        result = parse_any_date("2026-04-13-02-30-PM")
        assert result == datetime(2026, 4, 13, 14, 30)

    def test_manual_run_am(self):
        """12:45 AM → hour 0, minute 45."""
        result = parse_any_date("2026-04-13-12-45-AM")
        assert result == datetime(2026, 4, 13, 0, 45)

    def test_midnight_boundary(self):
        """12:00 AM → hour 0, minute 0."""
        result = parse_any_date("2026-04-13-12-00-AM")
        assert result == datetime(2026, 4, 13, 0, 0)

    def test_noon_boundary(self):
        """12:00 PM → hour 12, minute 0."""
        result = parse_any_date("2026-04-13-12-00-PM")
        assert result == datetime(2026, 4, 13, 12, 0)

    def test_whitespace_stripping(self):
        result = parse_any_date("  2026-04-08  ")
        assert result == datetime(2026, 4, 8)

    def test_regex_fallback_extra_suffix(self):
        """Date with unrecognized suffix falls back to date portion via regex."""
        result = parse_any_date("2026-04-13-extra-stuff")
        assert result == datetime(2026, 4, 13)

    def test_garbage_raises_valueerror(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            parse_any_date("not-a-date")

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            parse_any_date("")

    def test_identity_across_imports(self):
        """parse_any_date is the same function whether imported from scorecard or walk_forward."""
        from tradingagents.backtesting.scorecard import parse_any_date as scorecard_parse
        # walk_forward imports via: from tradingagents.backtesting.date_utils import parse_any_date
        from tradingagents.backtesting.date_utils import parse_any_date as direct_parse
        assert scorecard_parse is direct_parse
