import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import HedgeFund from './HedgeFund';

const mockGetHedgeFundAgents = vi.fn();
const mockStartHedgeFundAnalysis = vi.fn();

vi.mock('../lib/api', () => ({
  API_BASE_URL: '/api',
  getHedgeFundAgents: (...args: any[]) => mockGetHedgeFundAgents(...args),
  startHedgeFundAnalysis: (...args: any[]) => mockStartHedgeFundAnalysis(...args),
}));

class FakeEventSource {
  static last: FakeEventSource | null = null;

  url: string;
  listeners: Record<string, Array<(e: any) => void>> = {};

  constructor(url: string) {
    this.url = url;
    FakeEventSource.last = this;
  }

  addEventListener(type: string, cb: (e: any) => void) {
    this.listeners[type] = this.listeners[type] || [];
    this.listeners[type].push(cb);
  }

  emit(type: string, data: string) {
    const cbs = this.listeners[type] || [];
    cbs.forEach((cb) => cb({ data }));
  }

  close() {}
}

describe('HedgeFund page', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockGetHedgeFundAgents.mockReset();
    mockStartHedgeFundAnalysis.mockReset();
    FakeEventSource.last = null;

    vi.stubGlobal('EventSource', FakeEventSource as any);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('selects first 13 analysts by default', async () => {
    mockGetHedgeFundAgents.mockResolvedValue(
      Array.from({ length: 15 }, (_, i) => ({
        key: `agent_${i + 1}`,
        display_name: `Agent ${i + 1}`,
        description: `Desc ${i + 1}`,
        investing_style: 'style',
        order: i,
      }))
    );

    render(<HedgeFund />);

    await waitFor(() => {
      expect(screen.getByText('13 selected')).toBeInTheDocument();
    });
  });

  it('shows cooldown message and disables run button on 429 error', async () => {
    mockGetHedgeFundAgents.mockResolvedValue([
      {
        key: 'warren_buffett',
        display_name: 'Warren Buffett',
        description: 'Oracle',
        investing_style: 'value',
        order: 1,
      },
    ]);
    mockStartHedgeFundAnalysis.mockRejectedValue(new Error('429 Too Many Requests'));

    render(<HedgeFund />);

    const runButton = await screen.findByRole('button', { name: /Run HedgeFund/i });
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText(/NVIDIA rate limit reached/i)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Retry in 45s/i })).toBeDisabled();
    });
  });

  it('renders object reasoning safely from done event', async () => {
    mockGetHedgeFundAgents.mockResolvedValue([
      {
        key: 'warren_buffett',
        display_name: 'Warren Buffett',
        description: 'Oracle',
        investing_style: 'value',
        order: 1,
      },
    ]);
    mockStartHedgeFundAnalysis.mockResolvedValue({ job_id: 'job-1' });

    render(<HedgeFund />);

    const runButton = await screen.findByRole('button', { name: /Run HedgeFund/i });
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(FakeEventSource.last).toBeTruthy();
    });

    await act(async () => {
      FakeEventSource.last!.emit(
        'done',
        JSON.stringify({
          decisions: {
            AAPL: {
              action: 'HOLD',
              quantity: 0,
              reasoning: {
                portfolio_value: 100000,
                remaining_limit: 12345,
              },
            },
          },
          analyst_signals: {
            risk_management_agent: {
              AAPL: {
                signal: 'neutral',
                confidence: 50,
                reasoning: { current_position_value: 0 },
              },
            },
          },
        })
      );
    });

    await waitFor(() => {
      expect(screen.getByText('Portfolio Manager Decision')).toBeInTheDocument();
    });

    expect(screen.getByText(/portfolio_value/)).toBeInTheDocument();
    expect(screen.getByText(/current_position_value/)).toBeInTheDocument();
  });
});
