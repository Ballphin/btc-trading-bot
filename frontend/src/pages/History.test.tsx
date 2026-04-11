import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import History from './History';

vi.mock('../lib/api', () => ({
  fetchTickers: vi.fn(async () => [{ ticker: 'BTC-USD', analysis_count: 1, latest_date: '2026-01-01' }]),
  fetchAnalyses: vi.fn(async () => [{ date: '2026-01-01', signal: 'BUY', file: 'x.json' }]),
}));

describe('History page', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders ticker list and selected ticker analyses', async () => {
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
});
