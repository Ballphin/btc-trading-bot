import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { Zap, Play, Pause, RefreshCw, TrendingUp, TrendingDown, Minus, Clock, BarChart3, AlertTriangle, Activity, Gauge, ShieldAlert, ChevronRight, Layers } from 'lucide-react';
import { clsx } from 'clsx';
import { API_BASE_URL } from '../lib/api';
import useDocumentTitle from '../hooks/useDocumentTitle';
import EnsembleTab from '../components/EnsembleTab';

// Pulse Explain Chart gating: only BUY/SHORT at >= 75% confidence are clickable.
const EXPLAIN_MIN_CONFIDENCE = 0.75;
function explainEligible(p: { signal: string; confidence: number }): boolean {
  return (
    (p.signal === 'BUY' || p.signal === 'SHORT') &&
    (p.confidence ?? 0) >= EXPLAIN_MIN_CONFIDENCE
  );
}

// ── Types ────────────────────────────────────────────────────────────

type FillModelName = 'best' | 'realistic' | 'maker_rejected' | 'maker_adverse';

interface FillModelRow {
  gross_return: number;
  cost_bps: number;
  net_return: number;
  notes?: string;
}

interface PulseEntry {
  ts: string;
  // v3 metadata
  engine_version?: number;
  config_hash?: string;
  // Core
  signal: 'BUY' | 'SHORT' | 'NEUTRAL';
  confidence: number;
  normalized_score: number;
  raw_normalized_score?: number;
  price: number | null;
  atr_1h_at_pulse?: number | null;
  partial_bar_flags?: Record<string, boolean>;
  stop_loss: number | null;
  take_profit: number | null;
  hold_minutes: number;
  timeframe_bias: string;
  funding_rate: number | null;
  premium_pct: number | null;
  reasoning: string;
  breakdown: Record<string, number>;
  volatility_flag: boolean;
  signal_threshold?: number;
  persistence_mul?: number;
  override_reason?: string | null;
  // v3 alpha layer
  tsmom_direction?: number | null;
  tsmom_strength?: number | null;
  tsmom_gated_out?: boolean;
  tsmom_gate_reason?: string | null;
  tsmom_gate_mode?: string | null;
  regime_mode?: string;
  // S/R (v3.1)
  support?: number | null;
  resistance?: number | null;
  sr_source?: 'pivot' | 'book' | 'both' | 'none';
  sr_near_side?: 'support' | 'resistance' | null;
  z_4h_return?: number | null;
  regime_snapshot?: {
    mode: string;
    vol_z_clipped?: number;
    returns_acf1?: number;
    directional_bias?: number;
  };
  book_imbalance?: number | null;
  liquidation_score?: number | null;
  day_volume_usd?: number | null;
  // Scoring
  scored: boolean;
  threshold_source?: string;
  'hit_+5m'?: boolean;
  'hit_+15m'?: boolean;
  'hit_+1h'?: boolean;
  'return_+5m'?: number;
  'return_+15m'?: number;
  'return_+1h'?: number;
  'threshold_+5m'?: number;
  'threshold_+15m'?: number;
  'threshold_+1h'?: number;
  'fills_+5m'?: Record<FillModelName, FillModelRow>;
  'fills_+15m'?: Record<FillModelName, FillModelRow>;
  'fills_+1h'?: Record<FillModelName, FillModelRow>;
}

interface SchedulerStatus {
  enabled: boolean;
  tickers: string[];
  interval_minutes: number;
  last_run: string | null;
  last_status: string | null;
}

interface HitRates {
  [horizon: string]: {
    overall: number;
    BUY?: number;
    SHORT?: number;
  };
}

interface FillSummaryEntry {
  count: number;
  mean_net_bps: number;
  win_rate: number;
}

interface ScorecardData {
  ticker: string;
  total: number;
  scored: number;
  hit_rates: HitRates;
  fill_summary?: Record<string, Record<FillModelName, FillSummaryEntry>>;
  engine_versions?: Array<string | number>;
}

interface BacktestResult {
  ticker: string;
  period: string;
  total_signals: number;
  signal_breakdown: { BUY: number; SHORT: number; NEUTRAL: number };
  hit_rates: HitRates;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  profitability_curve: number[];
  n_trades: number;
  by_confidence_bucket: Record<string, { range: string; n: number; hit_1h: number }>;
  by_regime: Record<string, { n: number; hit_1h: number; sharpe: number }>;
  gap_count: number;
  n_excluded_warmup: number;
  return_autocorr_lag1: number;
}

// ── Signal Badge ─────────────────────────────────────────────────────

function SignalBadge({ signal, size = 'sm' }: { signal: string; size?: 'sm' | 'lg' }) {
  const cfg = signal === 'BUY'
    ? { icon: TrendingUp, bg: 'bg-emerald-500/15', text: 'text-emerald-400', ring: 'ring-emerald-500/30' }
    : signal === 'SHORT'
      ? { icon: TrendingDown, bg: 'bg-red-500/15', text: 'text-red-400', ring: 'ring-red-500/30' }
      : { icon: Minus, bg: 'bg-slate-500/15', text: 'text-slate-400', ring: 'ring-slate-500/30' };
  const Icon = cfg.icon;
  const sz = size === 'lg' ? 'px-3 py-1.5 text-base gap-2' : 'px-2 py-0.5 text-xs gap-1';
  return (
    <span className={clsx('inline-flex items-center rounded-full ring-1 font-semibold', cfg.bg, cfg.text, cfg.ring, sz)}>
      <Icon className={size === 'lg' ? 'w-4 h-4' : 'w-3 h-3'} />
      {signal}
    </span>
  );
}

// ── Confidence Bar ───────────────────────────────────────────────────

function ConfBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? 'bg-emerald-500' : pct >= 50 ? 'bg-amber-400' : 'bg-slate-500';
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div className={clsx('h-full rounded-full transition-all', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-slate-400">{pct}%</span>
    </div>
  );
}

// ── v3 Status badges (TSMOM, Regime, Override, EngineVersion) ──────

function TsmomBadge({ direction, strength, gated, gateReason }: {
  direction?: number | null; strength?: number | null; gated?: boolean; gateReason?: string | null;
}) {
  if (direction === null || direction === undefined) {
    return <span className="text-[10px] text-slate-600 px-1.5 py-0.5 rounded bg-white/5" title="Trend (1–24h) momentum unavailable — insufficient history">Trend n/a</span>;
  }
  const color = direction > 0 ? 'text-emerald-400 bg-emerald-500/10 ring-emerald-500/30'
              : direction < 0 ? 'text-red-400 bg-red-500/10 ring-red-500/30'
              : 'text-slate-400 bg-slate-500/10 ring-slate-500/30';
  const label = direction > 0 ? 'up' : direction < 0 ? 'down' : 'flat';
  const arrow = direction > 0 ? '↑' : direction < 0 ? '↓' : '·';
  const strPct = strength != null ? ` ${(strength * 100).toFixed(0)}%` : '';
  const titleText = gated
    ? `Trend (1–24h) gate blocked this signal: ${gateReason || 'disagreement'}`
    : 'Trend (1–24h) — Time-Series Momentum. Primary alpha gate: a BUY/SHORT only fires if confluence agrees OR counter-trend confluence is strong enough (confidence_weighted mode).';
  return (
    <span
      title={titleText}
      className={clsx(
        'inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded ring-1 font-semibold',
        color,
        gated && 'line-through opacity-60',
      )}
    >
      <Activity className="w-2.5 h-2.5" /> Trend {arrow} {label}{strPct}
    </span>
  );
}

function HorizonBadge({ holdMinutes }: { holdMinutes?: number }) {
  if (holdMinutes == null) return null;
  const label = holdMinutes < 60
    ? `${holdMinutes}m`
    : holdMinutes < 1440
      ? `${Math.round(holdMinutes / 60)}h`
      : `${Math.round(holdMinutes / 1440)}d`;
  return (
    <span
      title={`Short-term hold: ~${label}. Pulse signals are intraday; horizons span 5 min to 8 h.`}
      className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-300 ring-1 ring-sky-500/30 font-semibold"
    >
      ⏱ {label}
    </span>
  );
}

function SRChip({ support, resistance, source, nearSide }: {
  support?: number | null;
  resistance?: number | null;
  source?: string;
  nearSide?: 'support' | 'resistance' | null;
}) {
  if ((support == null && resistance == null) || source === 'none') return null;
  const fmt = (v: number | null | undefined) =>
    v == null ? '—' : v >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : v.toFixed(2);
  const srcLabel = source === 'both' ? 'pivot+book' : source;
  return (
    <span
      title={`S/R source: ${srcLabel}. Price counts as "near a level" within 0.3 × ATR.${nearSide ? ` Currently near ${nearSide}.` : ''}`}
      className="inline-flex items-center gap-1.5 text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-slate-300 ring-1 ring-white/10 font-mono"
    >
      📍
      <span className={clsx('text-red-400', nearSide === 'support' && 'font-bold ring-1 ring-red-500/40 px-1 rounded')}>
        S {fmt(support)}
      </span>
      <span className="text-slate-600">/</span>
      <span className={clsx('text-emerald-400', nearSide === 'resistance' && 'font-bold ring-1 ring-emerald-500/40 px-1 rounded')}>
        R {fmt(resistance)}
      </span>
    </span>
  );
}

function RegimeBadge({ mode }: { mode?: string }) {
  if (!mode) return null;
  const cfg: Record<string, { bg: string; text: string }> = {
    trend: { bg: 'bg-emerald-500/10', text: 'text-emerald-400' },
    chop: { bg: 'bg-amber-500/10', text: 'text-amber-300' },
    high_vol_trend: { bg: 'bg-violet-500/10', text: 'text-violet-300' },
    mixed: { bg: 'bg-slate-500/10', text: 'text-slate-400' },
  };
  const c = cfg[mode] || cfg.mixed;
  return (
    <span className={clsx(
      'inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-semibold',
      c.bg, c.text,
    )}>
      <Gauge className="w-2.5 h-2.5" /> {mode}
    </span>
  );
}

function OverrideBadge({ reason }: { reason?: string | null }) {
  if (!reason) return null;
  return (
    <span
      title={`Override: ${reason}`}
      className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-semibold bg-rose-500/10 text-rose-300 ring-1 ring-rose-500/30"
    >
      <ShieldAlert className="w-2.5 h-2.5" /> {reason}
    </span>
  );
}

function EngineBadge({ version, hash }: { version?: number; hash?: string }) {
  if (version == null) return null;
  return (
    <span
      title={`engine v${version} · config ${hash || ''}`}
      className="inline-flex items-center text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-slate-500"
    >
      v{version}{hash ? `·${hash.slice(0, 6)}` : ''}
    </span>
  );
}

function PartialBarBadge({ flags }: { flags?: Record<string, boolean> }) {
  if (!flags) return null;
  const active = Object.entries(flags).filter(([, v]) => v).map(([k]) => k);
  if (active.length === 0) return null;
  return (
    <span
      title={`Partial bar on: ${active.join(', ')}`}
      className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 ring-1 ring-amber-500/30"
    >
      <AlertTriangle className="w-2.5 h-2.5" /> partial {active.join(',')}
    </span>
  );
}

// ── Mini Sparkline (SVG) ─────────────────────────────────────────────

function MiniSparkline({ data, width = 120, height = 32 }: { data: number[]; width?: number; height?: number }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(' ');
  const last = data[data.length - 1];
  const color = last >= data[0] ? '#10b981' : '#ef4444';
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="opacity-80">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Main Component ───────────────────────────────────────────────────

export default function Pulse() {
  useDocumentTitle('Pulse');

  // State
  const [ticker, setTicker] = useState('BTC-USD');
  const [pulses, setPulses] = useState<PulseEntry[]>([]);
  const [highConfOnly, setHighConfOnly] = useState<boolean>(() => {
    try { return localStorage.getItem('pulse.highConfOnly') === '1'; } catch { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem('pulse.highConfOnly', highConfOnly ? '1' : '0'); } catch { /* ignore */ }
  }, [highConfOnly]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [scheduler, setScheduler] = useState<SchedulerStatus | null>(null);
  const [scorecard, setScorecard] = useState<ScorecardData | null>(null);
  const [activeTab, setActiveTab] = useState<'signals' | 'scorecard' | 'backtest' | 'ensemble'>('signals');
  const [runningPulse, setRunningPulse] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  
  // Pagination state
  const [pulseOffset, setPulseOffset] = useState(0);
  const [pulseTotal, setPulseTotal] = useState(0);
  const [hasMorePulses, setHasMorePulses] = useState(true);
  const PAGE_SIZE = 50;
  
  // Refs for race condition prevention (BLOCKER fixes)
  const isFetchingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pulseOffsetRef = useRef(0);
  const fetchPulsesRef = useRef<(loadMore?: boolean) => Promise<void>>(async () => {});
  const fetchScorecardRef = useRef<() => Promise<void>>(async () => {});

  // Backtest form state
  const [btStartDate, setBtStartDate] = useState('');
  const [btEndDate, setBtEndDate] = useState('');
  const [btRunning, setBtRunning] = useState(false);
  const [btResult, setBtResult] = useState<BacktestResult | null>(null);
  const [btError, setBtError] = useState<string | null>(null);

  // Set default backtest dates (last 30 days)
  useEffect(() => {
    const end = new Date();
    const start = new Date(end);
    start.setDate(start.getDate() - 30);
    setBtStartDate(start.toISOString().split('T')[0]);
    setBtEndDate(end.toISOString().split('T')[0]);
  }, []);

  // Fetch data with race condition guards (BLOCKER: SSE fix)
  const fetchPulses = useCallback(async (loadMore = false) => {
    // Prevent concurrent fetches
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;
    
    if (loadMore) {
      setLoadingMore(true);
    } else {
      setLoading(true);
      setPulseOffset(0);
      pulseOffsetRef.current = 0;
    }
    
    // Abort any in-flight request
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();
    
    try {
      const offset = loadMore ? pulseOffsetRef.current + PAGE_SIZE : 0;
      const res = await fetch(
        `${API_BASE_URL}/pulse/${ticker}?limit=${PAGE_SIZE}&offset=${offset}`,
        { signal: abortControllerRef.current.signal }
      );
      
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      
      const data = await res.json();
      
      if (loadMore) {
        setPulses(prev => [...prev, ...(data.pulses || [])]);
        setPulseOffset(offset);
        pulseOffsetRef.current = offset;
      } else {
        setPulses(data.pulses || []);
        setPulseOffset(0);
        pulseOffsetRef.current = 0;
      }
      
      setPulseTotal(data.total || 0);
      setHasMorePulses(data.has_more || false);
    } catch (err) {
      // Ignore abort errors (expected when ticker changes)
      if (err instanceof Error && err.name === 'AbortError') return;
      if (!loadMore) setPulses([]);
    } finally {
      isFetchingRef.current = false;
      setLoading(false);
      setLoadingMore(false);
    }
  }, [ticker]); // Only depend on ticker, not pulseOffset

  // Keep offset ref in sync
  useEffect(() => {
    pulseOffsetRef.current = pulseOffset;
  }, [pulseOffset]);

  const fetchScheduler = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/pulse/scheduler/status`);
      setScheduler(await res.json());
    } catch {}
  }, []);

  const fetchScorecard = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/pulse/scorecard/${ticker}`, {
        signal: abortControllerRef.current?.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setScorecard(await res.json());
    } catch (err) {
      // Ignore abort errors
      if (err instanceof Error && err.name === 'AbortError') return;
    }
  }, [ticker]);

  // Keep refs updated with latest functions (HIGH: SSE fix for interval)
  fetchPulsesRef.current = fetchPulses;
  fetchScorecardRef.current = fetchScorecard;

  useEffect(() => { fetchPulses(); fetchScheduler(); fetchScorecard(); }, [fetchPulses, fetchScheduler, fetchScorecard]);

  // Auto-refresh every 60s using refs (HIGH: prevents stale closures)
  useEffect(() => {
    const iv = setInterval(() => { 
      fetchPulsesRef.current();
      fetchScorecardRef.current();
      setRefreshTrigger(n => n + 1);
    }, 60000);
    return () => clearInterval(iv);
  }, []); // Empty deps - uses refs instead

  // Actions
  const toggleScheduler = async () => {
    await fetch(`${API_BASE_URL}/pulse/scheduler/toggle`, { method: 'POST' });
    fetchScheduler();
    fetchPulses();
    fetchScorecard();
    setRefreshTrigger(n => n + 1);
  };

  const runManual = async () => {
    setRunningPulse(true);
    try {
      await fetch(`${API_BASE_URL}/pulse/run/${ticker}`, { method: 'POST' });
      fetchPulses();
      fetchScorecard();
      setRefreshTrigger(n => n + 1);
    } catch {}
    setRunningPulse(false);
  };

  const runBacktest = async () => {
    setBtRunning(true);
    setBtError(null);
    setBtResult(null);
    try {
      const res = await fetch(`${API_BASE_URL}/pulse/backtest/${ticker}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_date: btStartDate,
          end_date: btEndDate,
          interval_minutes: 15,
          threshold: 0.25,
        }),
      });

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const payload = JSON.parse(line.slice(6));
                if (payload.ticker || payload.hit_rates) {
                  setBtResult(payload);
                }
              } catch {}
            }
            if (line.startsWith('event: result')) {
              // next data line has the result
            }
          }
        }
      }
    } catch (e: any) {
      setBtError(e.message || 'Backtest failed');
    }
    setBtRunning(false);
  };

  const latestPulse = pulses.length > 0 ? pulses[pulses.length - 1] : null;
  
  // Price staleness check (BLOCKER: WCT fix)
  const priceStalenessMs = useMemo(() => {
    if (!latestPulse?.ts) return 0;
    return Date.now() - new Date(latestPulse.ts).getTime();
  }, [latestPulse?.ts]);
  const isPriceStale = priceStalenessMs > 30000; // 30 seconds

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center">
            <Zap className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">Quant Pulse</h1>
            <p className="text-xs text-slate-500">Intraday long/short signals — holds range from 5 min to 8 h, anchored to support/resistance.</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Ticker selector */}
          <select
            value={ticker}
            onChange={e => setTicker(e.target.value)}
            className="bg-navy-800 text-white text-sm rounded-lg px-3 py-2 border border-white/10 focus:outline-none focus:ring-1 focus:ring-accent-teal"
          >
            {['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD'].map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>

          {/* Manual run */}
          <button
            onClick={runManual}
            disabled={runningPulse}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-accent-teal/10 text-accent-teal text-sm font-medium hover:bg-accent-teal/20 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('w-3.5 h-3.5', runningPulse && 'animate-spin')} />
            Run Now
          </button>

          {/* Scheduler toggle */}
          <button
            onClick={toggleScheduler}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
              scheduler?.enabled
                ? 'bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25'
                : 'bg-slate-700/50 text-slate-400 hover:bg-slate-700/80',
            )}
          >
            {scheduler?.enabled ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {scheduler?.enabled ? 'Stop' : 'Start'} Auto
          </button>
        </div>
      </div>

      {/* Latest Pulse Hero Card */}
      {latestPulse && (
        <div className="rounded-2xl bg-gradient-to-br from-navy-900/80 to-navy-800/50 border border-white/5 p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-3 min-w-0 flex-1">
              <div className="flex items-center gap-3 flex-wrap">
                <SignalBadge signal={latestPulse.signal} size="lg" />
                <ConfBar value={latestPulse.confidence} />
                {latestPulse.volatility_flag && (
                  <span className="flex items-center gap-1 text-xs text-amber-400">
                    <AlertTriangle className="w-3 h-3" /> High Vol
                  </span>
                )}
                <TsmomBadge
                  direction={latestPulse.tsmom_direction}
                  strength={latestPulse.tsmom_strength}
                  gated={latestPulse.tsmom_gated_out}
                  gateReason={latestPulse.tsmom_gate_reason}
                />
                <HorizonBadge holdMinutes={latestPulse.hold_minutes} />
                <SRChip
                  support={latestPulse.support}
                  resistance={latestPulse.resistance}
                  source={latestPulse.sr_source}
                  nearSide={latestPulse.sr_near_side}
                />
                <RegimeBadge mode={latestPulse.regime_mode} />
                <OverrideBadge reason={latestPulse.override_reason} />
                <PartialBarBadge flags={latestPulse.partial_bar_flags} />
                <EngineBadge
                  version={latestPulse.engine_version}
                  hash={latestPulse.config_hash}
                />
                {isPriceStale && (
                  <span className="flex items-center gap-1 text-xs text-red-400 animate-pulse font-semibold">
                    <AlertTriangle className="w-3 h-3" /> PRICE STALE ({Math.round(priceStalenessMs / 1000)}s)
                  </span>
                )}
              </div>
              <div className="text-sm text-slate-400 max-w-xl">{latestPulse.reasoning}</div>
              <div className="flex items-center gap-4 text-xs text-slate-500 flex-wrap">
                <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {new Date(latestPulse.ts).toLocaleTimeString()}</span>
                <span>TF: {latestPulse.timeframe_bias}</span>
                <span>Hold: {latestPulse.hold_minutes}m</span>
                {latestPulse.normalized_score !== undefined && (
                  <span>Score: {latestPulse.normalized_score >= 0 ? '+' : ''}{latestPulse.normalized_score.toFixed(3)}</span>
                )}
                {latestPulse.signal_threshold !== undefined && (
                  <span>Thr: ±{latestPulse.signal_threshold.toFixed(2)}</span>
                )}
                {latestPulse.persistence_mul !== undefined && latestPulse.persistence_mul !== 1 && (
                  <span>Pers: ×{latestPulse.persistence_mul.toFixed(2)}</span>
                )}
                {latestPulse.book_imbalance !== undefined && latestPulse.book_imbalance !== null && (
                  <span>BookImb: {latestPulse.book_imbalance >= 0 ? '+' : ''}{latestPulse.book_imbalance.toFixed(2)}</span>
                )}
                {latestPulse.atr_1h_at_pulse && (
                  <span>ATR1h: ${latestPulse.atr_1h_at_pulse.toFixed(0)}</span>
                )}
              </div>
            </div>
            <div className="text-right space-y-1 shrink-0">
              {latestPulse.price && (
                <div className="text-2xl font-bold text-white">${latestPulse.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
              )}
              <div className="flex gap-3 text-xs justify-end">
                {latestPulse.stop_loss && (
                  <span className="text-red-400">SL ${latestPulse.stop_loss.toLocaleString()}</span>
                )}
                {latestPulse.take_profit && (
                  <span className="text-emerald-400">TP ${latestPulse.take_profit.toLocaleString()}</span>
                )}
              </div>
              {latestPulse.funding_rate !== null && latestPulse.funding_rate !== undefined && (
                <div className="text-[11px] text-slate-500 mt-1">
                  Funding: {(latestPulse.funding_rate * 24 * 365 * 100).toFixed(1)}% ann
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab Bar */}
      <div className="flex items-center gap-1 border-b border-white/5 pb-px">
        {[
          { key: 'signals', label: 'Signal History', icon: Zap },
          { key: 'ensemble', label: 'Ensemble', icon: Layers },
          { key: 'scorecard', label: 'Scorecard', icon: BarChart3 },
          { key: 'backtest', label: 'Backtest', icon: TrendingUp },
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as typeof activeTab)}
            className={clsx(
              'flex items-center gap-1.5 px-4 py-2 text-sm font-medium transition-colors rounded-t-lg',
              activeTab === tab.key
                ? 'text-accent-teal border-b-2 border-accent-teal bg-accent-teal/5'
                : 'text-slate-500 hover:text-slate-300',
            )}
          >
            <tab.icon className="w-3.5 h-3.5" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'signals' && (
        <div className="space-y-2">
          {/* Filter bar */}
          <div className="flex items-center justify-between gap-3 pb-1">
            <button
              onClick={() => setHighConfOnly((v) => !v)}
              aria-pressed={highConfOnly}
              className={clsx(
                'inline-flex items-center gap-2 px-2.5 py-1 text-[11px] rounded border transition-colors',
                highConfOnly
                  ? 'border-accent-teal/50 bg-accent-teal/10 text-accent-teal'
                  : 'border-white/10 bg-navy-900/40 text-slate-400 hover:text-slate-200 hover:border-white/20',
              )}
              title="Show only pulses with confidence ≥ 75%"
            >
              <span
                className={clsx(
                  'w-3 h-3 rounded-[3px] border flex items-center justify-center',
                  highConfOnly ? 'border-transparent bg-accent-teal' : 'border-slate-500/40',
                )}
                aria-hidden
              >
                {highConfOnly && (
                  <svg viewBox="0 0 10 10" className="w-2 h-2" fill="none" stroke="#0B1220" strokeWidth="2">
                    <path d="M2 5 L4.2 7 L8 3" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </span>
              High-confidence only (≥ 75%)
            </button>
            {(() => {
              const filteredCount = pulses.filter((p) => !highConfOnly || (p.confidence ?? 0) >= 0.75).length;
              return (
                <span className="text-[11px] text-slate-500">
                  Showing {filteredCount} of {pulseTotal} total signal{pulseTotal === 1 ? '' : 's'}
                  {hasMorePulses && <span className="text-slate-600 ml-1">(more available)</span>}
                </span>
              );
            })()}
          </div>

          {loading && <div className="text-sm text-slate-500 py-4 text-center">Loading...</div>}
          {!loading && pulses.length === 0 && (
            <div className="text-sm text-slate-500 py-12 text-center">No pulse signals yet. Click "Run Now" to generate your first signal.</div>
          )}
          {!loading && pulses.length > 0 && highConfOnly && pulses.every((p) => (p.confidence ?? 0) < 0.75) && (
            <div className="text-sm text-slate-500 py-8 text-center">
              No signals meet the ≥ 75% confidence threshold. <button onClick={() => setHighConfOnly(false)} className="text-accent-teal hover:underline">Show all</button>.
            </div>
          )}
          {[...pulses]
            .filter((p) => !highConfOnly || (p.confidence ?? 0) >= 0.75)
            .map((p, i) => {
            const eligible = explainEligible(p);
            const rowClass = clsx(
              'block rounded-xl bg-navy-900/50 border border-white/5 px-4 py-3 transition-colors',
              eligible && 'hover:border-accent-teal/30 hover:bg-navy-900/70 cursor-pointer',
            );
            const explainHref = `/pulse/explain/${encodeURIComponent(ticker)}/${encodeURIComponent(p.ts)}`;
            const inner = (<>
              <div className="flex items-center gap-4">
                <SignalBadge signal={p.signal} />
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-slate-400 truncate">{p.reasoning}</div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-slate-600 flex-wrap">
                    <span>{new Date(p.ts).toLocaleString()}</span>
                    <span>TF: {p.timeframe_bias}</span>
                    {p.price && <span>${p.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>}
                    {p.normalized_score !== undefined && (
                      <span className={p.normalized_score >= 0 ? 'text-emerald-500/60' : 'text-red-500/60'}>
                        {p.normalized_score >= 0 ? '+' : ''}{p.normalized_score.toFixed(3)}
                      </span>
                    )}
                  </div>
                </div>
                <ConfBar value={p.confidence} />
                {eligible && <ChevronRight className="w-4 h-4 text-slate-500 flex-shrink-0" />}
                {p.scored && (
                  <div className="flex gap-2 text-xs">
                    {p['hit_+5m'] !== undefined && (
                      <span className={p['hit_+5m'] ? 'text-emerald-400' : 'text-red-400'}>
                        5m {p['hit_+5m'] ? '✓' : '✗'}
                      </span>
                    )}
                    {p['hit_+15m'] !== undefined && (
                      <span className={p['hit_+15m'] ? 'text-emerald-400' : 'text-red-400'}>
                        15m {p['hit_+15m'] ? '✓' : '✗'}
                      </span>
                    )}
                    {p['hit_+1h'] !== undefined && (
                      <span className={p['hit_+1h'] ? 'text-emerald-400' : 'text-red-400'}>
                        1h {p['hit_+1h'] ? '✓' : '✗'}
                      </span>
                    )}
                  </div>
                )}
                {!p.scored && <span className="text-xs text-slate-600">Pending</span>}
              </div>
              {/* v3 meta row */}
              <div className="flex flex-wrap items-center gap-1.5 mt-2 pl-[68px]">
                <TsmomBadge
                  direction={p.tsmom_direction}
                  strength={p.tsmom_strength}
                  gated={p.tsmom_gated_out}
                  gateReason={p.tsmom_gate_reason}
                />
                <HorizonBadge holdMinutes={p.hold_minutes} />
                <SRChip
                  support={p.support}
                  resistance={p.resistance}
                  source={p.sr_source}
                  nearSide={p.sr_near_side}
                />
                <RegimeBadge mode={p.regime_mode} />
                <OverrideBadge reason={p.override_reason} />
                <PartialBarBadge flags={p.partial_bar_flags} />
                <EngineBadge version={p.engine_version} hash={p.config_hash} />
              </div>
            </>);
            return eligible ? (
              <Link key={i} to={explainHref} className={rowClass} title="Open chart + pattern explanation">
                {inner}
              </Link>
            ) : (
              <div key={i} className={rowClass}>{inner}</div>
            );
          })}
          
          {/* Load More Button */}
          {!loading && hasMorePulses && (
            <div className="pt-4 pb-2 text-center">
              <button
                onClick={() => fetchPulses(true)}
                disabled={loadingMore}
                className={clsx(
                  'inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                  loadingMore
                    ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                    : 'bg-accent-teal/10 text-accent-teal hover:bg-accent-teal/20 border border-accent-teal/30'
                )}
              >
                {loadingMore ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Loading...
                  </>
                ) : (
                  <>
                    <ChevronRight className="w-4 h-4 rotate-90" />
                    Load More
                    <span className="text-xs text-slate-500 ml-1">
                      ({pulseTotal - pulses.length} remaining)
                    </span>
                  </>
                )}
              </button>
            </div>
          )}
          
          {/* End of History Message */}
          {!loading && !hasMorePulses && pulses.length > 0 && (
            <div className="pt-4 pb-2 text-center text-xs text-slate-600">
              End of signal history ({pulseTotal} total signals)
            </div>
          )}
        </div>
      )}

      {activeTab === 'ensemble' && (
        <EnsembleTab ticker={ticker} refreshTrigger={refreshTrigger} />
      )}

      {activeTab === 'scorecard' && (
        <div className="space-y-4">
          {scorecard && scorecard.scored > 0 ? (
            <>
              {/* Engine versions banner */}
              {scorecard.engine_versions && scorecard.engine_versions.length > 0 && (
                <div className="rounded-lg bg-navy-900/40 border border-white/5 px-4 py-2 text-xs text-slate-500 flex items-center gap-3">
                  <span>Engine versions present:</span>
                  {scorecard.engine_versions.map(ev => (
                    <span key={String(ev)} className="px-1.5 py-0.5 rounded bg-white/5 text-slate-400 font-mono">
                      v{ev}
                    </span>
                  ))}
                </div>
              )}

              <div className="grid grid-cols-3 gap-4">
                {Object.entries(scorecard.hit_rates).map(([horizon, rates]) => (
                  <div key={horizon} className="rounded-xl bg-navy-900/50 border border-white/5 p-4">
                    <div className="text-xs text-slate-500 mb-2">Hit Rate {horizon}</div>
                    <div className="text-2xl font-bold text-white">{(rates.overall * 100).toFixed(1)}%</div>
                    <div className="flex gap-3 mt-2 text-xs">
                      <span className="text-emerald-400">BUY: {((rates.BUY ?? 0) * 100).toFixed(1)}%</span>
                      <span className="text-red-400">SHORT: {((rates.SHORT ?? 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* 4 fill models per horizon */}
              {scorecard.fill_summary && Object.keys(scorecard.fill_summary).length > 0 && (
                <div className="rounded-xl bg-navy-900/50 border border-white/5 p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-white">Fill-Model Net Returns</h3>
                    <span className="text-[10px] text-slate-600">best · realistic · maker_rejected · maker_adverse</span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-xs">
                      <thead>
                        <tr className="text-left text-slate-500 border-b border-white/5">
                          <th className="py-2 pr-3 font-medium">Horizon</th>
                          <th className="py-2 px-3 font-medium">Model</th>
                          <th className="py-2 px-3 font-medium">N</th>
                          <th className="py-2 px-3 font-medium text-right">Mean Net (bps)</th>
                          <th className="py-2 pl-3 font-medium text-right">Win Rate</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(scorecard.fill_summary).flatMap(([horizon, byModel]) =>
                          (['best', 'realistic', 'maker_rejected', 'maker_adverse'] as FillModelName[]).map(model => {
                            const v = byModel[model];
                            if (!v) return null;
                            return (
                              <tr key={`${horizon}-${model}`} className="border-b border-white/5 last:border-b-0">
                                <td className="py-2 pr-3 text-slate-400">{horizon}</td>
                                <td className="py-2 px-3 font-mono text-slate-300">{model}</td>
                                <td className="py-2 px-3 text-slate-500">{v.count}</td>
                                <td className={clsx(
                                  'py-2 px-3 text-right font-mono',
                                  v.mean_net_bps >= 0 ? 'text-emerald-400' : 'text-red-400',
                                )}>
                                  {v.mean_net_bps >= 0 ? '+' : ''}{v.mean_net_bps.toFixed(1)}
                                </td>
                                <td className={clsx(
                                  'py-2 pl-3 text-right',
                                  v.win_rate >= 0.5 ? 'text-emerald-400' : 'text-red-400',
                                )}>
                                  {(v.win_rate * 100).toFixed(1)}%
                                </td>
                              </tr>
                            );
                          })
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="text-xs text-slate-500">
                {scorecard.total} total signals / {scorecard.scored} scored
              </div>
            </>
          ) : (
            <div className="text-sm text-slate-500 py-12 text-center">
              No scored signals yet. Signals are scored automatically 1h after pulse timestamp.
            </div>
          )}
        </div>
      )}

      {activeTab === 'backtest' && (
        <div className="space-y-6">
          {/* Backtest Form */}
          <div className="rounded-xl bg-navy-900/50 border border-white/5 p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Pulse Backtest — Historical Replay</h3>
            <div className="flex items-end gap-4">
              <div>
                <label className="block text-xs text-slate-500 mb-1">Start Date</label>
                <input
                  type="date"
                  value={btStartDate}
                  onChange={e => setBtStartDate(e.target.value)}
                  className="bg-navy-800 text-white text-sm rounded-lg px-3 py-2 border border-white/10 focus:outline-none focus:ring-1 focus:ring-accent-teal"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-500 mb-1">End Date</label>
                <input
                  type="date"
                  value={btEndDate}
                  onChange={e => setBtEndDate(e.target.value)}
                  className="bg-navy-800 text-white text-sm rounded-lg px-3 py-2 border border-white/10 focus:outline-none focus:ring-1 focus:ring-accent-teal"
                />
              </div>
              <button
                onClick={runBacktest}
                disabled={btRunning}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-accent-teal text-navy-950 text-sm font-semibold hover:bg-accent-teal/90 transition-colors disabled:opacity-50"
              >
                {btRunning ? (
                  <><RefreshCw className="w-3.5 h-3.5 animate-spin" /> Running...</>
                ) : (
                  <><Play className="w-3.5 h-3.5" /> Run Backtest</>
                )}
              </button>
            </div>
            {btError && <div className="mt-3 text-sm text-red-400">{btError}</div>}
          </div>

          {/* Backtest Results */}
          {btResult && (
            <div className="space-y-4">
              {/* Summary cards */}
              <div className="grid grid-cols-4 gap-4">
                <div className="rounded-xl bg-navy-900/50 border border-white/5 p-4">
                  <div className="text-xs text-slate-500 mb-1">Signals</div>
                  <div className="text-xl font-bold text-white">{btResult.total_signals}</div>
                  <div className="flex gap-2 mt-1 text-xs">
                    <span className="text-emerald-400">{btResult.signal_breakdown.BUY} BUY</span>
                    <span className="text-red-400">{btResult.signal_breakdown.SHORT} SHORT</span>
                  </div>
                </div>
                <div className="rounded-xl bg-navy-900/50 border border-white/5 p-4">
                  <div className="text-xs text-slate-500 mb-1">Sharpe Ratio</div>
                  <div className={clsx('text-xl font-bold', btResult.sharpe_ratio >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                    {btResult.sharpe_ratio.toFixed(2)}
                  </div>
                </div>
                <div className="rounded-xl bg-navy-900/50 border border-white/5 p-4">
                  <div className="text-xs text-slate-500 mb-1">Max Drawdown</div>
                  <div className="text-xl font-bold text-red-400">{btResult.max_drawdown_pct.toFixed(1)}%</div>
                </div>
                <div className="rounded-xl bg-navy-900/50 border border-white/5 p-4">
                  <div className="text-xs text-slate-500 mb-1">Trades</div>
                  <div className="text-xl font-bold text-white">{btResult.n_trades}</div>
                  <div className="text-xs text-slate-500 mt-1">{btResult.gap_count} gaps, {btResult.n_excluded_warmup} warmup excluded</div>
                </div>
              </div>

              {/* Hit Rates */}
              <div className="rounded-xl bg-navy-900/50 border border-white/5 p-5">
                <h3 className="text-sm font-semibold text-white mb-3">Forward Return Hit Rates</h3>
                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(btResult.hit_rates).map(([horizon, rates]) => (
                    <div key={horizon} className="text-center">
                      <div className="text-xs text-slate-500 mb-1">{horizon}</div>
                      <div className="text-lg font-bold text-white">{(rates.overall * 100).toFixed(1)}%</div>
                      <div className="flex justify-center gap-3 mt-1 text-xs">
                        <span className="text-emerald-400">BUY {((rates.BUY ?? 0) * 100).toFixed(1)}%</span>
                        <span className="text-red-400">SHORT {((rates.SHORT ?? 0) * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Confidence Buckets */}
              {btResult.by_confidence_bucket && Object.keys(btResult.by_confidence_bucket).length > 0 && (
                <div className="rounded-xl bg-navy-900/50 border border-white/5 p-5">
                  <h3 className="text-sm font-semibold text-white mb-3">By Confidence Bucket</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {Object.entries(btResult.by_confidence_bucket).map(([label, bucket]) => (
                      <div key={label} className="text-center">
                        <div className="text-xs text-slate-500 mb-1 capitalize">{label}</div>
                        <div className="text-xs text-slate-600">{bucket.range}</div>
                        <div className="text-lg font-bold text-white mt-1">{(bucket.hit_1h * 100).toFixed(1)}%</div>
                        <div className="text-xs text-slate-500">n={bucket.n}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Regime Buckets */}
              {btResult.by_regime && Object.keys(btResult.by_regime).length > 0 && (
                <div className="rounded-xl bg-navy-900/50 border border-white/5 p-5">
                  <h3 className="text-sm font-semibold text-white mb-3">By Volatility Regime</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {Object.entries(btResult.by_regime).map(([regime, data]) => (
                      <div key={regime} className="text-center">
                        <div className="text-xs text-slate-500 mb-1 capitalize">{regime.replace('_', ' ')}</div>
                        <div className="text-lg font-bold text-white">{(data.hit_1h * 100).toFixed(1)}%</div>
                        <div className="text-xs text-slate-500">n={data.n} / Sharpe {data.sharpe.toFixed(2)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Equity Curve */}
              {btResult.profitability_curve && btResult.profitability_curve.length > 2 && (
                <div className="rounded-xl bg-navy-900/50 border border-white/5 p-5">
                  <h3 className="text-sm font-semibold text-white mb-3">Equity Curve</h3>
                  <MiniSparkline data={btResult.profitability_curve} width={600} height={120} />
                </div>
              )}

              {/* Diagnostics */}
              <div className="text-xs text-slate-600 flex gap-4">
                <span>Return autocorr (lag-1): {btResult.return_autocorr_lag1.toFixed(4)}</span>
                <span>Period: {btResult.period}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Scheduler info bar */}
      {scheduler && (
        <div className="flex items-center gap-4 text-xs text-slate-600 border-t border-white/5 pt-3">
          <span>Scheduler: {scheduler.enabled ? 'Running' : 'Stopped'}</span>
          {scheduler.last_run && <span>Last run: {new Date(scheduler.last_run).toLocaleTimeString()}</span>}
          {scheduler.last_status && <span>Status: {scheduler.last_status}</span>}
          <span>Interval: {scheduler.interval_minutes}m</span>
        </div>
      )}
    </div>
  );
}
