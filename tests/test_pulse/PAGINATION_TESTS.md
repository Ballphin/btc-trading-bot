# Pulse Pagination Tests

## Summary

Comprehensive test coverage for the Signal History pagination feature that allows users to load all historical trades via a "Load More" button.

## Test Coverage

### Backend Tests (`TestPulsePagination` class - 11 tests)

#### Pagination Logic Tests
1. **`test_pagination_default_returns_last_50`** - Verifies default behavior returns last 50 pulses in descending order
2. **`test_pagination_with_offset`** - Tests offset parameter skips most recent N pulses
3. **`test_pagination_load_more_sequence`** - Simulates clicking "Load More" multiple times to collect all pulses
4. **`test_pagination_offset_beyond_total`** - Handles edge case when offset exceeds total count
5. **`test_pagination_exact_page_boundary`** - Verifies correct behavior at exact page boundaries (50/100)
6. **`test_pagination_single_pulse`** - Tests pagination with small datasets (< 50 pulses)
7. **`test_pagination_limit_zero`** - Edge case: limit of zero returns empty list with correct metadata

#### Helper Function Tests
8. **`test_count_pulses_helper`** - Tests `_count_pulses()` function directly
9. **`test_read_pulses_with_offset`** - Tests `_read_pulses()` with offset parameter directly
10. **`test_pagination_corrupt_lines_ignored`** - Verifies corrupt JSON lines are ignored in count/pagination

#### Existing Tests Updated
11. **`test_empty_pulses`** - Updated to verify `total`, `has_more`, and `offset` fields
12. **`test_populated_pulses`** - Updated to verify pagination metadata
13. **`test_limit_param`** - Updated to verify `has_more` flag when limit < total

## Test Fixtures

### `seeded_pulse` fixture
Creates 3 pulse entries (BUY, SHORT, NEUTRAL) for basic testing

### `many_pulses` fixture  
Creates 100 pulse entries across multiple days for pagination testing

## API Response Format Verified

```json
{
  "ticker": "BTC-USD",
  "pulses": [...],
  "count": 50,
  "total": 100,
  "has_more": true,
  "offset": 0
}
```

## Key Behaviors Tested

1. **Descending Order**: Pulses are returned newest-first within each page
2. **Offset Calculation**: Correctly skips the most recent N pulses
3. **Page Boundaries**: Works correctly at exact 50-item boundaries
4. **Empty Results**: Returns empty array with correct metadata when offset >= total
5. **Corrupt Line Handling**: Invalid JSON lines are silently ignored
6. **Small Datasets**: Works correctly with < 50 total pulses
7. **Total Count**: `_count_pulses()` accurately counts valid JSON lines only

## Running the Tests

```bash
# Run pagination tests only
cd /Users/daniel/Desktop/TradingAgents
.venv/bin/python -m pytest tests/test_pulse/test_pulse_api.py::TestPulsePagination -v

# Run all pulse API tests
.venv/bin/python -m pytest tests/test_pulse/test_pulse_api.py -v

# Run full pulse test suite
.venv/bin/python -m pytest tests/test_pulse/ -v
```

## Test Results

- **24 tests** in `test_pulse_api.py` - all passing
- **379 tests** in `tests/test_pulse/` - all passing
- No regressions introduced
