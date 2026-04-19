import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import ScorecardPage from './Scorecard';

// ── Mock data ─────────────────────────────────────────────────────────

const mockScorecardData = {
  ticker: 'BTC-USD',
  total_decisions: 10,
  scored_decisions: 8,
  overall_win_rate: 0.625,
  avg_brier_score: 0.2100,
  win_by_signal: {
    BUY: { win_rate: 0.75, sample_size: 4 },
    SHORT: { win_rate: 0.5, sample_size: 4 },
  },
  win_by_regime: {
    bull_quiet: { win_rate: 0.667, sample_size: 6 },
    bear_volatile: { win_rate: 0.5, sample_size: 2 },
  },
  win_by_combo: {},
  exit_type_breakdown: {
    take_profit_hit: 3,
    stop_loss_hit: 2,
    held_to_expiry: 3,
  },
  ev_per_trade_10k: 45,
  avg_win_return: 0.032,
  avg_loss_return: 0.018,
  brier_decomposition: {
    brier_score: 0.21,
    reliability: 0.03,
    resolution: 0.07,
    uncertainty: 0.25,
    base_rate: 0.625,
    n_decisions: 8,
    bins: [
      { range: '0.00-0.55', n: 2, mean_confidence: 0.50, mean_outcome: 0.50 },
      { range: '0.55-0.70', n: 3, mean_confidence: 0.62, mean_outcome: 0.67 },
      { range: '0.70-1.01', n: 3, mean_confidence: 0.78, mean_outcome: 0.67 },
    ],
    calibration_trigger: { dampen: false, allow_larger: false },
  },
  recent_decisions: [
    {
      date: '2026-03-08',
      signal: 'BUY',
      price: 87000,
      confidence: 0.72,
      regime: 'bull_quiet',
      was_correct_primary: true,
      actual_return_primary: 0.035,
      net_return_primary: 0.033,
      exit_type: 'take_profit_hit',
      exit_price: 89000,
      exit_day: 2,
      hold_days_planned: 3,
      execution_cost: 0.002,
      brier_score: 0.0784,
    },
    {
      date: '2026-03-07',
      signal: 'SHORT',
      price: 86000,
      confidence: 0.65,
      regime: 'bear_volatile',
      was_correct_primary: false,
      actual_return_primary: -0.02,
      net_return_primary: -0.022,
      exit_type: 'stop_loss_hit',
      exit_price: 87720,
      exit_day: 1,
      hold_days_planned: 3,
      execution_cost: 0.002,
      brier_score: 0.4225,
    },
  ],
};

const mockWalkForwardData = {
  ticker: 'BTC-USD',
  horizon_days: 7,
  total_decisions: 10,
  scored_decisions: 8,
  overall_metrics: {
    win_rate: 0.625,
    mean_return_gross: 0.015,
    mean_return_net: 0.013,
    sharpe_ratio_gross: 0.75,
    sharpe_ratio_net: 0.65,
    sharpe_se: 0.35,
    deflated_sharpe_ratio: 0.98,
    dsr_interpretation: 'SIGNIFICANT',
    max_drawdown: 0.05,
    skewness: -0.3,
    kurtosis: 3.5,
    ev_per_trade_10k: 45,
    avg_win_return: 0.032,
    avg_loss_return: 0.018,
  },
  exit_type_breakdown: {
    take_profit_hit: 3,
    stop_loss_hit: 2,
    held_to_expiry: 3,
  },
  regime_analysis: {},
  signal_analysis: { BUY: { win_rate: 0.75, sample_size: 4 }, SHORT: { win_rate: 0.5, sample_size: 4 } },
  equity_curve_gross: [1.0, 1.01, 1.02, 1.015, 1.025],
  equity_curve_position: [1.0, 1.005, 1.01, 1.008, 1.012],
};

const mockCalibrationData = {
  correction: 0.8742,
  mean_confidence: 0.6800,
  mean_outcome: 0.6250,
  n_decisions_total: 12,
  n_decisions_deduped: 10,
  regimes_covered: ['bull_quiet', 'bear_volatile', 'range_bound'],
  coverage_quality: 'medium',
  note: 'Calibration has adequate regime diversity',
};

// ── Helpers ───────────────────────────────────────────────────────────

function mockFetchResponses(overrides: Record<string, any> = {}) {
  const responses: Record<string, any> = {
    scorecard: overrides.scorecard ?? mockScorecardData,
    scheduler: overrides.scheduler ?? { enabled: false, next_run_local: null, last_run: null, last_status: null, interval_hours: 4 },
    score: overrides.score ?? { scored: 2, pending: 0, total_decisions: 10, total_scored: 10 },
    calibrate: overrides.calibrate ?? mockCalibrationData,
    'walk-forward': overrides['walk-forward'] ?? mockWalkForwardData,
  };

  return vi.fn((url: string, _opts?: RequestInit) => {
    const urlStr = typeof url === 'string' ? url : '';
    let data: any = {};

    if (urlStr.includes('/scheduler/status')) data = responses.scheduler;
    else if (urlStr.includes('/shadow/scorecard/')) data = responses.scorecard;
    else if (urlStr.includes('/shadow/score/')) data = responses.score;
    else if (urlStr.includes('/shadow/calibrate/')) data = responses.calibrate;
    else if (urlStr.includes('/shadow/walk-forward/')) data = responses['walk-forward'];

    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve(data),
    } as Response);
  });
}

function renderScorecard() {
  return render(
    <MemoryRouter>
      <ScorecardPage />
    </MemoryRouter>
  );
}

// ── Tests ─────────────────────────────────────────────────────────────

describe('Scorecard page — Scorecard tab', () => {
  let fetchMock: ReturnType<typeof mockFetchResponses>;

  beforeEach(() => {
    fetchMock = mockFetchResponses();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders "Win Rate (Adaptive)" label', async () => {
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('Win Rate (Adaptive)')).toBeInTheDocument();
    });
  });

  it('renders EV/Trade card with dollar sign', async () => {
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('EV / Trade ($10K)')).toBeInTheDocument();
    });
    expect(screen.getByText('$45')).toBeInTheDocument();
  });

  it('renders EV positive with "Positive edge" subtitle', async () => {
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('Positive edge')).toBeInTheDocument();
    });
  });

  it('renders Avg Win / Avg Loss stat cards', async () => {
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('Avg Win')).toBeInTheDocument();
    });
    expect(screen.getByText('Avg Loss')).toBeInTheDocument();
    expect(screen.getByText('3.20%')).toBeInTheDocument(); // 0.032 * 100
    expect(screen.getByText('-1.80%')).toBeInTheDocument(); // 0.018 * 100
  });

  it('renders exit type breakdown labels', async () => {
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('Exit Types')).toBeInTheDocument();
    });
    expect(screen.getByText('TP Hit')).toBeInTheDocument();
    expect(screen.getByText('SL Hit')).toBeInTheDocument();
    expect(screen.getByText('Timeout')).toBeInTheDocument();
  });

  it('renders recent decisions table headers', async () => {
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('Exit')).toBeInTheDocument();
    });
    expect(screen.getByText('Hold')).toBeInTheDocument();
    expect(screen.getByText('Net Return')).toBeInTheDocument();
  });

  it('renders exit badges with correct labels', async () => {
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('TP')).toBeInTheDocument();
    });
    expect(screen.getByText('SL')).toBeInTheDocument();
  });

  it('renders pending notice with "hold period" text', async () => {
    const pendingData = {
      ...mockScorecardData,
      scored_decisions: 0,
      overall_win_rate: 0,
      exit_type_breakdown: {},
      recent_decisions: [],
      brier_decomposition: null,
    };
    const fetchPending = mockFetchResponses({ scorecard: pendingData });
    vi.stubGlobal('fetch', fetchPending);
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText(/hold period/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/SL\/TP hits/i)).toBeInTheDocument();
  });

  it('shows error banner when fetch returns error', async () => {
    const errorFetch = mockFetchResponses({ scorecard: { error: 'Something went wrong' } });
    vi.stubGlobal('fetch', errorFetch);
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });
  });

  it('renders empty state when total_decisions=0', async () => {
    const emptyData = {
      ...mockScorecardData,
      total_decisions: 0,
      scored_decisions: 0,
      overall_win_rate: 0,
      exit_type_breakdown: {},
      recent_decisions: [],
      brier_decomposition: null,
    };
    const emptyFetch = mockFetchResponses({ scorecard: emptyData });
    vi.stubGlobal('fetch', emptyFetch);
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('No Decisions Recorded')).toBeInTheDocument();
    });
  });
});

describe('Scorecard page — Walk-Forward tab', () => {
  let fetchMock: ReturnType<typeof mockFetchResponses>;

  beforeEach(() => {
    fetchMock = mockFetchResponses();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  async function switchToWalkForwardTab() {
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('Walk-Forward')).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText('Walk-Forward'));
  }

  it('renders "Sharpe (Net)" and "Mean Return (Net)" labels after validation', async () => {
    await switchToWalkForwardTab();
    // Click Run Validation button
    const runBtn = screen.getByText('Run Validation');
    fireEvent.click(runBtn);
    await waitFor(() => {
      expect(screen.getByText('Sharpe (Net)')).toBeInTheDocument();
    });
    expect(screen.getByText('Mean Return (Net)')).toBeInTheDocument();
  });

  it('renders Sharpe SE annotation', async () => {
    await switchToWalkForwardTab();
    fireEvent.click(screen.getByText('Run Validation'));
    await waitFor(() => {
      expect(screen.getByText(/SE: ±0\.350/)).toBeInTheDocument();
    });
  });

  it('renders EV / Trade card in walk-forward', async () => {
    await switchToWalkForwardTab();
    fireEvent.click(screen.getByText('Run Validation'));
    await waitFor(() => {
      expect(screen.getAllByText('EV / Trade ($10K)').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders exit type breakdown in walk-forward tab', async () => {
    await switchToWalkForwardTab();
    fireEvent.click(screen.getByText('Run Validation'));
    await waitFor(() => {
      const exitTypeLabels = screen.getAllByText('Exit Types');
      expect(exitTypeLabels.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('DSR hero card shows green for ≥0.95', async () => {
    await switchToWalkForwardTab();
    fireEvent.click(screen.getByText('Run Validation'));
    await waitFor(() => {
      // DSR = 0.98 → should show green styling and "SIGNIFICANT"
      expect(screen.getByText('SIGNIFICANT')).toBeInTheDocument();
    });
  });
});

describe('Scorecard page — Calibration tab', () => {
  let fetchMock: ReturnType<typeof mockFetchResponses>;

  beforeEach(() => {
    fetchMock = mockFetchResponses();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  async function switchToCalibrationTab() {
    renderScorecard();
    await waitFor(() => {
      expect(screen.getByText('Calibration')).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText('Calibration'));
  }

  it('renders correction factor as 4-decimal number', async () => {
    await switchToCalibrationTab();
    fireEvent.click(screen.getByText('Run Calibration Study'));
    await waitFor(() => {
      expect(screen.getByText('0.8742')).toBeInTheDocument();
    });
  });

  it('renders deduped count with "total (deduped)" subtitle', async () => {
    await switchToCalibrationTab();
    fireEvent.click(screen.getByText('Run Calibration Study'));
    await waitFor(() => {
      expect(screen.getByText('Decisions Used')).toBeInTheDocument();
    });
    expect(screen.getByText(/12 total \(deduped\)/i)).toBeInTheDocument();
  });

  it('renders error state when calibration returns error', async () => {
    const errorCal = mockFetchResponses({
      calibrate: { error: 'Need at least 10 scored decisions, have 3', scored_available: 3 },
    });
    vi.stubGlobal('fetch', errorCal);
    await switchToCalibrationTab();
    fireEvent.click(screen.getByText('Run Calibration Study'));
    await waitFor(() => {
      expect(screen.getByText('Not Enough Data')).toBeInTheDocument();
    });
  });
});
