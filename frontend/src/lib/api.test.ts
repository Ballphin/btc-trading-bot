import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fetchTickers, fetchAnalyses, fetchPrice } from './api';

describe('api helpers', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('fetchTickers returns tickers array', async () => {
    const payload = { tickers: [{ ticker: 'BTC-USD', analysis_count: 2, latest_date: '2026-01-01' }] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: async () => payload,
    }));

    const result = await fetchTickers();
    expect(result).toHaveLength(1);
    expect(result[0].ticker).toBe('BTC-USD');
  });

  it('fetchAnalyses returns analyses array', async () => {
    const payload = { analyses: [{ date: '2026-01-01', signal: 'BUY', file: 'x.json' }] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: async () => payload,
    }));

    const result = await fetchAnalyses('BTC-USD');
    expect(result).toHaveLength(1);
    expect(result[0].signal).toBe('BUY');
  });

  it('fetchPrice returns normalized records', async () => {
    const payload = {
      data: [{
        date: '2026-01-01',
        open: 100,
        high: 110,
        low: 95,
        close: 105,
        volume: 1234,
        sma50: null,
        sma200: null,
      }],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: async () => payload,
    }));

    const result = await fetchPrice('BTC-USD', 10, '1d');
    expect(result[0].close).toBe(105);
    expect(result[0].volume).toBe(1234);
  });
});
