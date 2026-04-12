import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import History from './History';

const mockFetchTickers = vi.fn();
const mockFetchAnalyses = vi.fn();

vi.mock('../lib/api', () => ({
  fetchTickers: (...args: any[]) => mockFetchTickers(...args),
  fetchAnalyses: (...args: any[]) => mockFetchAnalyses(...args),
}));

describe('History page', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockFetchTickers.mockReset();
    mockFetchAnalyses.mockReset();
  });

  it('renders ticker list and selected ticker analyses', async () => {
    mockFetchTickers.mockResolvedValue([{ ticker: 'BTC-USD', analysis_count: 1, latest_date: '2026-01-01' }]);
    mockFetchAnalyses.mockResolvedValue([{ date: '2026-01-01', signal: 'BUY', file: 'x.json' }]);

    render(
      <MemoryRouter initialEntries={['/history/BTC-USD']}>
        <Routes>
          <Route path="/history/:ticker" element={<History />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('BTC-USD')).toBeInTheDocument();
    });

    expect(screen.getByText(/1 analysis/i)).toBeInTheDocument();
  });

  it('renders new timestamp format with AM/PM correctly', async () => {
    mockFetchTickers.mockResolvedValue([{ ticker: 'BTC-USD', analysis_count: 2, latest_date: '2026-04-11' }]);
    mockFetchAnalyses.mockResolvedValue([
      { date: '2026-04-11', local_date: '2026-04-11', candle_time: '2026-04-11-02-30-PM', time: '2:30 PM', signal: 'BUY', file: 'log1.json' },
      { date: '2026-04-11', local_date: '2026-04-11', candle_time: '2026-04-11-10-15-AM', time: '10:15 AM', signal: 'SHORT', file: 'log2.json' },
    ]);

    render(
      <MemoryRouter initialEntries={['/history/BTC-USD']}>
        <Routes>
          <Route path="/history/:ticker" element={<History />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('2:30 PM')).toBeInTheDocument();
    });

    expect(screen.getByText('10:15 AM')).toBeInTheDocument();
  });

  it('groups multiple analyses by local date with new format', async () => {
    mockFetchTickers.mockResolvedValue([{ ticker: 'BTC-USD', analysis_count: 3, latest_date: '2026-04-11' }]);
    mockFetchAnalyses.mockResolvedValue([
      { date: '2026-04-11', local_date: '2026-04-11', candle_time: '2026-04-11-02-30-PM', time: '2:30 PM', signal: 'BUY', file: 'log1.json' },
      { date: '2026-04-11', local_date: '2026-04-11', candle_time: '2026-04-11-03-45-PM', time: '3:45 PM', signal: 'SHORT', file: 'log2.json' },
      { date: '2026-04-10', local_date: '2026-04-10', candle_time: '2026-04-10-11-00-AM', time: '11:00 AM', signal: 'HOLD', file: 'log3.json' },
    ]);

    render(
      <MemoryRouter initialEntries={['/history/BTC-USD']}>
        <Routes>
          <Route path="/history/:ticker" element={<History />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('2026-04-11')).toBeInTheDocument();
    });

    // Should show "2 analyses" for 2026-04-11 group
    expect(screen.getByText('2 analyses')).toBeInTheDocument();
    // Should show "1 analysis" for 2026-04-10 group
    expect(screen.getByText('1 analysis')).toBeInTheDocument();
  });

  it('shows distinct times for multiple runs in same hour', async () => {
    mockFetchTickers.mockResolvedValue([{ ticker: 'BTC-USD', analysis_count: 2, latest_date: '2026-04-11' }]);
    mockFetchAnalyses.mockResolvedValue([
      { date: '2026-04-11', local_date: '2026-04-11', candle_time: '2026-04-11-02-15-PM', time: '2:15 PM', signal: 'BUY', file: 'log1.json' },
      { date: '2026-04-11', local_date: '2026-04-11', candle_time: '2026-04-11-02-45-PM', time: '2:45 PM', signal: 'SELL', file: 'log2.json' },
    ]);

    render(
      <MemoryRouter initialEntries={['/history/BTC-USD']}>
        <Routes>
          <Route path="/history/:ticker" element={<History />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('2:15 PM')).toBeInTheDocument();
    });

    expect(screen.getByText('2:45 PM')).toBeInTheDocument();
  });
});
