import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Pulse from './Pulse';

const basePulseResponse = {
  pulses: [],
  total: 0,
  has_more: false,
};

const baseSchedulerResponse = {
  enabled: false,
  tickers: ['BTC-USD'],
  interval_minutes: 15,
  last_run: null,
  last_status: null,
};

const baseScorecardResponse = {
  ticker: 'BTC-USD',
  total: 0,
  scored: 0,
  hit_rates: {},
};

const backtestResultWithSignals = {
  ticker: 'BTC-USD',
  period: '2026-03-01 to 2026-03-02',
  total_signals: 1,
  signal_breakdown: { BUY: 1, SHORT: 0, NEUTRAL: 0 },
  hit_rates: {
    '+5m': { overall: 0.4, BUY: 0.6, SHORT: 0.0 },
    '+15m': { overall: 0.38, BUY: 0.58, SHORT: 0.02 },
    '+1h': { overall: 0.42, BUY: 0.63, SHORT: 0.05 },
  },
  sl_tp_win_rate: 0.0,
  outcomes: { tp_hit: 0, sl_hit: 0, timeout: 1, missing_sltp: 0 },
  sample_size_warning: true,
  sharpe_ratio: -1.2,
  max_drawdown_pct: 0.7,
  profitability_curve: [1, 0.99, 0.985],
  n_trades: 1,
  by_confidence_bucket: {},
  by_regime: {},
  gap_count: 0,
  n_excluded_warmup: 0,
  return_autocorr_lag1: 0,
  signals: [
    {
      ts: '2026-03-01T10:00:00Z',
      signal: 'BUY',
      price: 100,
      stop_loss: 95,
      take_profit: 108,
      hold_minutes: 60,
      'hit_+5m': true,
      'hit_+15m': false,
      'hit_+1h': true,
      'return_+5m': 0.01,
      'return_+15m': -0.005,
      'return_+1h': 0.025,
      'high_in_window_+5m': 105,
      'low_in_window_+5m': 99,
      'high_in_window_+15m': 110,
      'low_in_window_+15m': 96,
      'high_in_window_+1h': 120,
      'low_in_window_+1h': 94,
      'tp_hit_in_window_+5m': false,
      'tp_hit_in_window_+15m': true,
      'tp_hit_in_window_+1h': true,
      'sl_hit_in_window_+5m': false,
      'sl_hit_in_window_+15m': false,
      'sl_hit_in_window_+1h': true,
    },
  ],
};

const backtestResultWithoutSignals = {
  ...backtestResultWithSignals,
  signals: [],
};

function makeSseResponse(payload: unknown) {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode('event: progress\n'));
      controller.enqueue(encoder.encode('data: {"phase":"starting"}\n\n'));
      controller.enqueue(encoder.encode('event: result\n'));
      controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      controller.close();
    },
  });

  return {
    ok: true,
    body,
    json: () => Promise.resolve({}),
  } as Response;
}

function createFetchMock(backtestPayload = backtestResultWithSignals) {
  return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);

    if (url.includes('/api/pulse/backtest/') && init?.method === 'POST') {
      return Promise.resolve(makeSseResponse(backtestPayload));
    }

    if (url.includes('/api/pulse/scheduler/status')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(baseSchedulerResponse),
      } as Response);
    }

    if (url.includes('/api/pulse/scorecard/')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(baseScorecardResponse),
      } as Response);
    }

    if (url.includes('/api/pulse/')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(basePulseResponse),
      } as Response);
    }

    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({}),
    } as Response);
  });
}

function renderPulse() {
  return render(
    <MemoryRouter>
      <Pulse />
    </MemoryRouter>,
  );
}

describe('Pulse page backtest hit-rate drilldown', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', createFetchMock());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders hit-rate tiles as interactive buttons and auto-selects +5m after backtest results load', async () => {
    renderPulse();

    fireEvent.click(screen.getByRole('button', { name: /backtest/i }));
    fireEvent.click(screen.getByRole('button', { name: /run backtest/i }));

    const plus5 = await screen.findByRole('button', { name: /\+5m/i });
    const plus15 = await screen.findByRole('button', { name: /\+15m/i });
    const plus1h = await screen.findByRole('button', { name: /\+1h/i });

    expect(plus5).toHaveAttribute('aria-pressed', 'true');
    expect(plus15).toHaveAttribute('aria-pressed', 'false');
    expect(plus1h).toHaveAttribute('aria-pressed', 'false');

    expect(await screen.findByText('Signal Details +5m')).toBeInTheDocument();
    expect(screen.getByText('$105')).toBeInTheDocument();
    expect(screen.getByText('$99')).toBeInTheDocument();
  });

  it('swaps the detail rows when another horizon is clicked', async () => {
    renderPulse();

    fireEvent.click(screen.getByRole('button', { name: /backtest/i }));
    fireEvent.click(screen.getByRole('button', { name: /run backtest/i }));

    const plus15 = await screen.findByRole('button', { name: /\+15m/i });
    fireEvent.click(plus15);

    await waitFor(() => {
      expect(screen.getByText('Signal Details +15m')).toBeInTheDocument();
    });

    expect(plus15).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('$110')).toBeInTheDocument();
    expect(screen.getByText('$96')).toBeInTheDocument();
  });

  it('shows an empty-state detail panel when signals are absent', async () => {
    vi.stubGlobal('fetch', createFetchMock(backtestResultWithoutSignals));
    renderPulse();

    fireEvent.click(screen.getByRole('button', { name: /backtest/i }));
    fireEvent.click(screen.getByRole('button', { name: /run backtest/i }));

    await screen.findByRole('button', { name: /\+5m/i });
    expect(await screen.findByText('Signal Details +5m')).toBeInTheDocument();
    expect(screen.getByText('No horizon detail rows are available for this selection yet.')).toBeInTheDocument();
  });
});
