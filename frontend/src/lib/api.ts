const _base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');
export const API_BASE_URL = _base ? `${_base}/api` : '/api';

export interface TickerInfo {
  ticker: string;
  analysis_count: number;
  latest_date: string | null;
}

export interface AnalysisSummary {
  date: string;
  local_date?: string;
  candle_time?: string;
  time?: string | null;
  signal: string;
  file: string;
}

export interface PriceRecord {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma50: number | null;
  sma200: number | null;
}

export interface AnalysisData {
  company_of_interest: string;
  trade_date: string;
  date_formatted?: string;  // Formatted local time with AM/PM
  market_report: string;
  sentiment_report: string;
  news_report: string;
  fundamentals_report: string;
  investment_debate_state: {
    bull_history: string;
    bear_history: string;
    history: string;
    current_response: string;
    judge_decision: string;
  };
  risk_debate_state: {
    aggressive_history: string;
    conservative_history: string;
    neutral_history: string;
    history: string;
    judge_decision: string;
  };
  trader_investment_decision: string;
  investment_plan: string;
  final_trade_decision: string;
  // Structured signal fields (optional, for new JSON-based signals)
  signal?: string;
  stop_loss_price?: number;
  take_profit_price?: number;
  confidence?: number;
  max_hold_days?: number;
  reasoning?: string;
  // Confidence scorer output
  position_size_pct?: number;
  conviction_label?: string;
  gated?: boolean;
  r_ratio?: number | null;
  r_ratio_warning?: boolean;
  hold_period_scalar?: number;
  hedge_penalty_applied?: number;
}

export interface BacktestMetrics {
  initial_capital: number;
  final_value: number;
  total_return_pct: number;
  total_pnl: number;
  annualized_return_pct: number;
  annualized_volatility_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  max_drawdown_start: string;
  max_drawdown_end: string;
  calmar_ratio: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate_pct: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number | null;  // null = ∞ (no losing trades)
  expectancy: number;
  start_date: string;
  end_date: string;
  n_periods: number;
  // Risk management metrics
  stops_hit?: number;
  takes_hit?: number;
  avg_hold_days?: number;
  avg_rr_ratio?: number;
  // Crypto-specific
  total_fees?: number;
  total_funding?: number;
  liquidations?: number;
  fee_impact_pct?: number;
  omega_ratio?: number;
  // Benchmark comparison
  benchmark_return_pct?: number | null;
  alpha_pct?: number | null;
  avg_leverage?: number;
  leverage_adjusted_return_pct?: number;
  tail_ratio?: number;
  funding_impact_pct?: number;
  alpha?: number;
  beta?: number;
  information_ratio?: number;
}

export interface BacktestDecision {
  date: string;
  signal: string;
  price: number;
  action: string;
  portfolio_value: number;
  position: string;
  stop_loss_price?: number | null;
  take_profit_price?: number | null;
  kelly_size?: number | null;
}

export interface TradeRecord {
  date: string;
  signal: string;
  price: number;
  action_taken: string;
  position_side: string;
  portfolio_value: number;
  cash: number;
  unrealized_pnl?: number;
  realized_pnl?: number;
  fees_paid?: number;
  funding_paid?: number;
  leverage?: number;
  liquidation_price?: number;
  // Risk management fields
  stop_loss?: number;
  take_profit?: number;
  hold_days?: number;
  exit_reason?: string;
  atr_at_entry?: number;
}

export interface BacktestConfig {
  ticker: string;
  start_date: string;
  end_date: string;
  mode: 'replay' | 'simulation';
  initial_capital: number;
  position_size_pct: number;
  frequency: string;
  // Crypto-specific
  leverage?: number;
  maker_fee?: number;
  taker_fee?: number;
  use_funding?: boolean;
  position_sizing?: string;
}

export interface BacktestResult {
  job_id: string;
  config: BacktestConfig;
  metrics: BacktestMetrics;
  decisions: BacktestDecision[];
  equity_curve: Array<{
    date: string;
    portfolio_value: number;
    cash?: number;
    position_side?: string;
  }>;
  trade_history: TradeRecord[];
  errors?: Array<{ date: string; error: string }>;
  created_at: string;
}

export async function fetchTickers(): Promise<TickerInfo[]> {
  const res = await fetch(`${API_BASE_URL}/history`);
  const data = await res.json();
  return data.tickers;
}

export async function fetchAnalyses(ticker: string): Promise<AnalysisSummary[]> {
  const res = await fetch(`${API_BASE_URL}/history/${ticker}`);
  const data = await res.json();
  return data.analyses;
}

export async function fetchAnalysis(ticker: string, date: string): Promise<AnalysisData> {
  const res = await fetch(`${API_BASE_URL}/history/${ticker}/${date}`);
  const data = await res.json();
  // Merge date_formatted from top level into data object
  return { ...data.data, date_formatted: data.date_formatted };
}

export async function fetchPrice(ticker: string, days = 90, interval = '1d'): Promise<PriceRecord[]> {
  const res = await fetch(`${API_BASE_URL}/price/${ticker}?days=${days}&interval=${interval}`);
  const data = await res.json();
  return data.data;
}

export async function startAnalysis(ticker: string, date?: string) {
  const res = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker, date }),
  });
  return res.json();
}

export interface SSEEvent {
  event: string;
  agent?: string;
  step?: number;
  total?: number;
  content?: string;
  report_key?: string;
  report?: string;
  signal?: string;
  message?: string;
  result?: Record<string, unknown>;
  elapsed?: number;
  // Structured signal fields
  stop_loss_price?: number;
  take_profit_price?: number;
  confidence?: number;
  max_hold_days?: number;
  reasoning?: string;
  // Confidence scorer output
  position_size_pct?: number;
  conviction_label?: string;
  gated?: boolean;
  r_ratio?: number | null;
  r_ratio_warning?: boolean;
  hold_period_scalar?: number;
  hedge_penalty_applied?: number;
}

const log = (...args: unknown[]) => { if (import.meta.env.DEV) console.log(...args); };

export function streamAnalysis(jobId: string, onEvent: (e: SSEEvent) => void): () => void {
  let closed = false;
  let terminal = false;
  let es: EventSource | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function connect() {
    if (closed) return;
    log('[SSE] Connecting to stream:', jobId);
    es = new EventSource(`${API_BASE_URL}/stream/${jobId}`);

    es.onopen = () => { log('[SSE] Connection opened'); };

    const handleMsg = (_eventName: string, msg: MessageEvent) => {
      try {
        const data = JSON.parse(msg.data);
        onEvent(data);
        if (data.event === 'done' || data.event === 'error') {
          terminal = true;
          es?.close();
        }
      } catch (err) {
        console.error('[SSE] Failed to parse event:', err);
      }
    };

    es.onmessage = (msg) => handleMsg('message', msg);
    for (const name of ['agent_start', 'agent_update', 'agent_report', 'decision', 'done', 'error', 'heartbeat']) {
      es.addEventListener(name, (msg: MessageEvent) => handleMsg(name, msg));
    }

    es.onerror = () => {
      es?.close();
      if (closed || terminal) return;
      log('[SSE] Connection lost, reconnecting in 2s...');
      reconnectTimer = setTimeout(connect, 2000);
    };
  }

  connect();

  return () => {
    closed = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    es?.close();
  };
}


// ── Pulse Explain Chart ────────────────────────────────────────────

export interface CandleBar {
  ts: number;     // unix seconds
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

export type PatternState =
  | 'forming'
  | 'completed'
  | 'confirmed'
  | 'retested'
  | 'invalidated';

export interface PatternAnchor {
  label: string;
  ts: string;       // ISO
  price: number;
  role: string;
  idx: number;
}

export interface PatternLine {
  from_idx: number;
  to_idx: number;
  role: string;
  style: 'solid' | 'dashed' | 'dotted';
  weight: 1 | 2 | 3;
  color_token: string;
}

export interface ChartPattern {
  name: string;
  display_name: string;
  bias: 'bullish' | 'bearish' | 'neutral';
  state: PatternState;
  fit_score: number;
  duration_score: number;
  volume_score: number;
  combined_score: number;
  timeframe: string;
  anchors: PatternAnchor[];
  lines: PatternLine[];
  bars_in_pattern: number;
  regime_aligned: boolean;
  description: string;
  color_token: string;
}

export interface PulseExplain {
  ticker: string;
  entry: {
    ts: string;
    price: number | null;
    signal: string;
    confidence: number;
    normalized_score: number | null;
  };
  levels: {
    stop_loss: number | null;
    take_profit: number | null;
    support: number | null;
    resistance: number | null;
    sr_source?: string | null;
    sr_near_side?: string | null;
  };
  tsmom: {
    direction: number | null;
    strength: number | null;
    gated_out: boolean | null;
  };
  timeframe_bias: string | null;
  regime_mode: string | null;
  breakdown_top3: { key: string; weight: number }[];
  breakdown: Record<string, number>;
  candles: Record<string, CandleBar[]>;
  chart_patterns: ChartPattern[];
  candlestick_patterns: { tf: string; name: string }[];
  reasoning_prose: string;
  detector_errors: { detector: string; tf: string; error: string }[];
  pattern_detection_degraded: boolean;
}

export async function fetchPulseExplain(
  ticker: string,
  ts: string,
): Promise<PulseExplain> {
  const url = `${API_BASE_URL}/pulse/explain/${encodeURIComponent(ticker)}/${encodeURIComponent(ts)}`;
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`PulseExplain ${res.status}: ${body || res.statusText}`);
  }
  return res.json();
}

