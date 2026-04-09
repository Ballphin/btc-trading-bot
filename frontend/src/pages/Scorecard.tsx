import { useState, useEffect, useCallback } from 'react';
import { Target, TrendingUp, TrendingDown, BarChart3, RefreshCw, AlertCircle, CheckCircle2, XCircle, Activity, Zap, Shield, Clock, PlayCircle, StopCircle } from 'lucide-react';
import { API_BASE_URL } from '../lib/api';
import useDocumentTitle from '../hooks/useDocumentTitle';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line, CartesianGrid, Legend } from 'recharts';

interface BrierBin {
  range: string;
  n: number;
  mean_confidence: number;
  mean_outcome: number;
}

interface WinRateEntry {
  win_rate: number;
  sample_size: number;
}

interface ScoredDecision {
  date: string;
  signal: string;
  price: number;
  confidence: number;
  regime: string;
  was_correct_7d?: boolean;
  actual_return_7d?: number;
  brier_score?: number;
}

interface Scorecard {
  ticker: string;
  total_decisions: number;
  scored_decisions: number;
  overall_win_rate: number;
  avg_brier_score: number | null;
  win_by_signal: Record<string, WinRateEntry>;
  win_by_regime: Record<string, WinRateEntry>;
  win_by_combo: Record<string, WinRateEntry>;
  brier_decomposition: {
    brier_score: number;
    reliability: number;
    resolution: number;
    uncertainty: number;
    base_rate: number;
    n_decisions: number;
    bins: BrierBin[];
    calibration_trigger: {
      dampen: boolean;
      allow_larger: boolean;
    };
  } | null;
  recent_decisions: ScoredDecision[];
}

interface WalkForwardResult {
  ticker: string;
  horizon_days: number;
  total_decisions: number;
  scored_decisions: number;
  overall_metrics: {
    win_rate: number;
    mean_return: number;
    sharpe_ratio: number;
    deflated_sharpe_ratio: number;
    dsr_interpretation: string;
    max_drawdown: number;
    skewness: number;
    kurtosis: number;
  };
  regime_analysis: Record<string, { win_rate: number; sample_size: number; mean_return: number }>;
  signal_analysis: Record<string, { win_rate: number; sample_size: number }>;
  equity_curve: number[];
}

interface CalibrationResult {
  correction: number;
  mean_confidence: number;
  mean_outcome: number;
  n_decisions: number;
  regimes_covered: string[];
  coverage_quality: string;
  note: string;
  error?: string;
}

const SIGNAL_COLORS: Record<string, string> = {
  BUY: '#06d6a0',
  SELL: '#ef4444',
  SHORT: '#f59e0b',
  COVER: '#a855f7',
  HOLD: '#64748b',
  OVERWEIGHT: '#0ea5e9',
  UNDERWEIGHT: '#f97316',
};

function StatCard({ label, value, sub, icon: Icon, color = 'text-white' }: {
  label: string; value: string | number; sub?: string; icon: any; color?: string;
}) {
  return (
    <div className="glass-static p-4 flex flex-col gap-1">
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <Icon className="w-3.5 h-3.5" />
        {label}
      </div>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

function SignalBadge({ signal }: { signal: string }) {
  const color = SIGNAL_COLORS[signal] || '#64748b';
  return (
    <span
      className="px-2 py-0.5 rounded-full text-xs font-semibold"
      style={{ background: `${color}20`, color, border: `1px solid ${color}40` }}
    >
      {signal}
    </span>
  );
}

export default function ScorecardPage() {
  useDocumentTitle('Forward-Test Scorecard');
  const [ticker, setTicker] = useState('BTC-USD');
  const [scorecard, setScorecard] = useState<Scorecard | null>(null);
  const [walkForward, setWalkForward] = useState<WalkForwardResult | null>(null);
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'scorecard' | 'walk-forward' | 'calibration'>('scorecard');

  // Scheduler state
  const [scheduler, setScheduler] = useState<{
    enabled: boolean;
    next_run_local: string | null;
    last_run: string | null;
    last_status: string | null;
    interval_hours: number;
  } | null>(null);
  const [schedulerToggling, setSchedulerToggling] = useState(false);

  const fetchSchedulerStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/scheduler/status`);
      const data = await res.json();
      setScheduler(data);
    } catch {
      // scheduler endpoint may not be available yet — fail silently
    }
  }, []);

  const handleToggleScheduler = async () => {
    setSchedulerToggling(true);
    try {
      const res = await fetch(`${API_BASE_URL}/scheduler/toggle`, { method: 'POST' });
      const data = await res.json();
      setScheduler(prev => prev ? { ...prev, enabled: data.enabled } : null);
      await fetchSchedulerStatus();
    } catch (err) {
      setError('Failed to toggle scheduler');
    } finally {
      setSchedulerToggling(false);
    }
  };

  useEffect(() => { fetchSchedulerStatus(); }, [fetchSchedulerStatus]);

  const fetchScorecard = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_BASE_URL}/shadow/scorecard/${ticker}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      // Normalize missing fields to safe defaults
      setScorecard({
        total_decisions: data.total_decisions ?? 0,
        scored_decisions: data.scored_decisions ?? 0,
        overall_win_rate: data.overall_win_rate ?? 0,
        avg_brier_score: data.avg_brier_score ?? null,
        win_by_signal: data.win_by_signal ?? {},
        win_by_regime: data.win_by_regime ?? {},
        win_by_combo: data.win_by_combo ?? {},
        brier_decomposition: data.brier_decomposition ?? null,
        recent_decisions: data.recent_decisions ?? [],
        ticker: data.ticker ?? ticker,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch scorecard');
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  useEffect(() => { fetchScorecard(); }, [fetchScorecard]);

  const handleScore = async () => {
    setScoring(true);
    try {
      const res = await fetch(`${API_BASE_URL}/shadow/score/${ticker}`, { method: 'POST' });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      await fetchScorecard();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scoring failed');
    } finally {
      setScoring(false);
    }
  };

  const handleCalibrate = async () => {
    setCalibrating(true);
    try {
      const res = await fetch(`${API_BASE_URL}/shadow/calibrate/${ticker}`, { method: 'POST' });
      const data = await res.json();
      setCalibration(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Calibration failed');
    } finally {
      setCalibrating(false);
    }
  };

  const handleWalkForward = async () => {
    setValidating(true);
    try {
      const res = await fetch(`${API_BASE_URL}/shadow/walk-forward/${ticker}`, { method: 'POST' });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setWalkForward(data);
      setActiveTab('walk-forward');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Walk-forward validation failed');
    } finally {
      setValidating(false);
    }
  };

  const tabs = [
    { id: 'scorecard' as const, label: 'Scorecard', icon: Target },
    { id: 'walk-forward' as const, label: 'Walk-Forward', icon: TrendingUp },
    { id: 'calibration' as const, label: 'Calibration', icon: Shield },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Target className="w-8 h-8 text-accent-teal" />
            Forward-Test Scorecard
          </h1>
          <p className="text-slate-400 mt-1 text-sm">Track live predictions, calibrate confidence, and validate edge</p>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="BTC-USD"
            className="w-36 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm font-mono focus:border-accent-teal/50 focus:outline-none"
          />
          <button
            onClick={fetchScorecard}
            disabled={loading}
            className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-300 hover:text-white hover:border-slate-600 transition-colors text-sm flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* 4H Scheduler Panel */}
      {scheduler !== null && (
        <div className={`mb-6 p-4 rounded-xl border flex flex-col sm:flex-row items-start sm:items-center gap-4 ${
          scheduler.enabled
            ? 'bg-emerald-500/10 border-emerald-500/25'
            : 'bg-slate-800/60 border-slate-700/50'
        }`}>
          <div className="flex items-center gap-3 flex-1">
            <div className={`p-2 rounded-lg ${
              scheduler.enabled ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700/60 text-slate-400'
            }`}>
              <Clock className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">
                4-Hour Auto-Analysis
                <span className={`ml-2 text-xs font-medium px-2 py-0.5 rounded-full ${
                  scheduler.enabled ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700 text-slate-400'
                }`}>
                  {scheduler.enabled ? 'ACTIVE' : 'INACTIVE'}
                </span>
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                {scheduler.enabled && scheduler.next_run_local
                  ? `Next run: ${new Date(scheduler.next_run_local).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })} local`
                  : scheduler.last_run
                  ? `Last run: ${scheduler.last_run}`
                  : 'Enable to run BTC-USD analysis every 4 hours'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {scheduler.enabled && (
              <button
                onClick={async () => {
                  try { await fetch(`${API_BASE_URL}/scheduler/run-now`, { method: 'POST' }); } catch {}
                }}
                className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-700 text-slate-300 hover:text-white hover:bg-slate-600 transition-colors"
              >
                Run Now
              </button>
            )}
            <button
              onClick={handleToggleScheduler}
              disabled={schedulerToggling}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
                scheduler.enabled
                  ? 'bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25'
                  : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/25'
              } disabled:opacity-50`}
            >
              {schedulerToggling
                ? <RefreshCw className="w-4 h-4 animate-spin" />
                : scheduler.enabled
                ? <StopCircle className="w-4 h-4" />
                : <PlayCircle className="w-4 h-4" />}
              {scheduler.enabled ? 'Disable Scheduler' : 'Enable Scheduler'}
            </button>
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex gap-1 mb-6 p-1 bg-slate-800/50 rounded-xl w-fit">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-accent-teal/15 text-accent-teal border border-accent-teal/30'
                : 'text-slate-400 hover:text-white hover:bg-white/5 border border-transparent'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 flex items-center gap-3 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          {error}
          <button onClick={() => setError('')} className="ml-auto text-red-400/60 hover:text-red-400">✕</button>
        </div>
      )}

      {/* ── SCORECARD TAB ── */}
      {activeTab === 'scorecard' && (
        <div className="space-y-6 animate-fade-in-up">
          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleScore}
              disabled={scoring}
              className="px-5 py-2.5 bg-gradient-to-r from-accent-teal to-accent-cyan text-navy-950 font-semibold rounded-xl hover:opacity-90 transition-opacity text-sm flex items-center gap-2 disabled:opacity-50"
            >
              {scoring ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
              Score Pending Decisions
            </button>
            <button
              onClick={handleWalkForward}
              disabled={validating}
              className="px-5 py-2.5 bg-slate-800 border border-slate-700 text-slate-300 rounded-xl hover:border-accent-teal/40 transition-colors text-sm flex items-center gap-2 disabled:opacity-50"
            >
              {validating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
              Run Walk-Forward
            </button>
            <button
              onClick={handleCalibrate}
              disabled={calibrating}
              className="px-5 py-2.5 bg-slate-800 border border-slate-700 text-slate-300 rounded-xl hover:border-accent-purple/40 transition-colors text-sm flex items-center gap-2 disabled:opacity-50"
            >
              {calibrating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Shield className="w-4 h-4" />}
              Run Calibration Study
            </button>
          </div>

          {scorecard && (
            <>
              {/* Stats Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard
                  label="Total Decisions"
                  value={scorecard.total_decisions}
                  sub={`${scorecard.scored_decisions} scored`}
                  icon={BarChart3}
                />
                <StatCard
                  label="Win Rate (T+7d)"
                  value={`${(scorecard.overall_win_rate * 100).toFixed(1)}%`}
                  sub={scorecard.overall_win_rate >= 0.5 ? 'Above breakeven' : 'Below breakeven'}
                  icon={scorecard.overall_win_rate >= 0.5 ? TrendingUp : TrendingDown}
                  color={scorecard.overall_win_rate >= 0.55 ? 'text-green-400' : scorecard.overall_win_rate >= 0.45 ? 'text-yellow-400' : 'text-red-400'}
                />
                <StatCard
                  label="Avg Brier Score"
                  value={scorecard.avg_brier_score !== null ? scorecard.avg_brier_score.toFixed(4) : '—'}
                  sub={scorecard.avg_brier_score !== null ? (scorecard.avg_brier_score < 0.25 ? 'Good calibration' : 'Needs improvement') : 'No data'}
                  icon={Target}
                  color={scorecard.avg_brier_score !== null && scorecard.avg_brier_score < 0.25 ? 'text-green-400' : 'text-yellow-400'}
                />
                <StatCard
                  label="Pending"
                  value={scorecard.total_decisions - scorecard.scored_decisions}
                  sub="Awaiting T+7d"
                  icon={RefreshCw}
                />
              </div>

              {/* Pending scoring notice */}
              {scorecard.scored_decisions === 0 && scorecard.total_decisions > 0 && (
                <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/25 flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-amber-300">Waiting for T+7 Day Scoring Window</p>
                    <p className="text-xs text-slate-400 mt-1">
                      You have <strong className="text-white">{scorecard.total_decisions}</strong> recorded decisions, but none have been scored yet.
                      Brier scoring requires <strong className="text-white">7 days</strong> to pass after the decision date so actual prices can be verified.
                      Click <strong className="text-white">Score Pending Decisions</strong> after 7+ days to see results.
                    </p>
                  </div>
                </div>
              )}

              {/* Win Rate Breakdown */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* By Signal */}
                <div className="glass-static p-5">
                  <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-accent-teal" />
                    Win Rate by Signal
                  </h3>
                  {Object.keys(scorecard.win_by_signal).length > 0 ? (
                    <div className="space-y-3">
                      {Object.entries(scorecard.win_by_signal).map(([signal, data]) => (
                        <div key={signal} className="flex items-center gap-3">
                          <SignalBadge signal={signal} />
                          <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${data.win_rate * 100}%`,
                                background: data.win_rate >= 0.5
                                  ? `linear-gradient(90deg, #06d6a0, #0ea5e9)`
                                  : `linear-gradient(90deg, #ef4444, #f59e0b)`,
                              }}
                            />
                          </div>
                          <span className="text-sm font-mono text-slate-300 w-20 text-right">
                            {(data.win_rate * 100).toFixed(1)}% <span className="text-slate-500 text-xs">({data.sample_size})</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-500 text-sm">No scored decisions yet</p>
                  )}
                </div>

                {/* By Regime */}
                <div className="glass-static p-5">
                  <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-accent-purple" />
                    Win Rate by Regime
                  </h3>
                  {Object.keys(scorecard.win_by_regime).length > 0 ? (
                    <div className="space-y-3">
                      {Object.entries(scorecard.win_by_regime).map(([regime, data]) => (
                        <div key={regime} className="flex items-center gap-3">
                          <span className="text-xs text-slate-400 w-24 truncate capitalize">
                            {regime.replace('_', ' ')}
                          </span>
                          <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${data.win_rate * 100}%`,
                                background: data.win_rate >= 0.5
                                  ? `linear-gradient(90deg, #06d6a0, #0ea5e9)`
                                  : `linear-gradient(90deg, #ef4444, #f59e0b)`,
                              }}
                            />
                          </div>
                          <span className="text-sm font-mono text-slate-300 w-20 text-right">
                            {(data.win_rate * 100).toFixed(1)}% <span className="text-slate-500 text-xs">({data.sample_size})</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-500 text-sm">No regime data yet</p>
                  )}
                </div>
              </div>

              {/* Brier Decomposition Chart */}
              {scorecard.brier_decomposition && (
                <div className="glass-static p-5">
                  <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                    <Target className="w-4 h-4 text-accent-cyan" />
                    Brier Score Decomposition
                    {scorecard.brier_decomposition.calibration_trigger.dampen && (
                      <span className="ml-2 text-xs px-2 py-0.5 bg-yellow-500/10 text-yellow-400 rounded-full border border-yellow-500/30">
                        Confidence dampening active
                      </span>
                    )}
                  </h3>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="text-center p-3 bg-slate-800/60 rounded-lg">
                      <div className="text-xs text-slate-400 mb-1">Reliability</div>
                      <div className={`text-xl font-bold ${scorecard.brier_decomposition.reliability <= 0.05 ? 'text-green-400' : 'text-yellow-400'}`}>
                        {scorecard.brier_decomposition.reliability.toFixed(4)}
                      </div>
                      <div className="text-xs text-slate-500">Lower = better</div>
                    </div>
                    <div className="text-center p-3 bg-slate-800/60 rounded-lg">
                      <div className="text-xs text-slate-400 mb-1">Resolution</div>
                      <div className={`text-xl font-bold ${scorecard.brier_decomposition.resolution >= 0.10 ? 'text-green-400' : 'text-slate-300'}`}>
                        {scorecard.brier_decomposition.resolution.toFixed(4)}
                      </div>
                      <div className="text-xs text-slate-500">Higher = better</div>
                    </div>
                    <div className="text-center p-3 bg-slate-800/60 rounded-lg">
                      <div className="text-xs text-slate-400 mb-1">Uncertainty</div>
                      <div className="text-xl font-bold text-slate-300">
                        {scorecard.brier_decomposition.uncertainty.toFixed(4)}
                      </div>
                      <div className="text-xs text-slate-500">Base rate var.</div>
                    </div>
                  </div>
                  {/* Calibration bins chart */}
                  {scorecard.brier_decomposition.bins.length > 0 && (
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={scorecard.brier_decomposition.bins} barGap={4}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="range" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                        <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0, 1]} />
                        <Tooltip
                          contentStyle={{ background: '#0f1629', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                          labelStyle={{ color: '#e2e8f0' }}
                        />
                        <Bar dataKey="mean_confidence" name="Mean Confidence" radius={[4, 4, 0, 0]}>
                          {scorecard.brier_decomposition.bins.map((_, i) => (
                            <Cell key={i} fill="#0ea5e9" fillOpacity={0.7} />
                          ))}
                        </Bar>
                        <Bar dataKey="mean_outcome" name="Mean Outcome" radius={[4, 4, 0, 0]}>
                          {scorecard.brier_decomposition.bins.map((_, i) => (
                            <Cell key={i} fill="#06d6a0" fillOpacity={0.7} />
                          ))}
                        </Bar>
                        <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </div>
              )}

              {/* Recent Decisions Table */}
              {scorecard.recent_decisions.length > 0 && (
                <div className="glass-static p-5 overflow-x-auto">
                  <h3 className="text-sm font-semibold text-white mb-4">Recent Scored Decisions</h3>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-xs text-slate-500 border-b border-slate-800">
                        <th className="text-left py-2 pr-4">Date</th>
                        <th className="text-left py-2 pr-4">Signal</th>
                        <th className="text-right py-2 pr-4">Price</th>
                        <th className="text-right py-2 pr-4">Conf.</th>
                        <th className="text-center py-2 pr-4">Correct?</th>
                        <th className="text-right py-2 pr-4">Return (7d)</th>
                        <th className="text-right py-2">Brier</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scorecard.recent_decisions.slice().reverse().map((d, i) => (
                        <tr key={i} className="border-b border-slate-800/50 hover:bg-white/[0.02]">
                          <td className="py-2 pr-4 text-slate-300 font-mono text-xs">{d.date?.split(' ')[0]}</td>
                          <td className="py-2 pr-4"><SignalBadge signal={d.signal} /></td>
                          <td className="py-2 pr-4 text-right text-slate-300 font-mono">${d.price?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                          <td className="py-2 pr-4 text-right text-slate-400 font-mono">{((d.confidence || 0) * 100).toFixed(0)}%</td>
                          <td className="py-2 pr-4 text-center">
                            {d.was_correct_7d === true ? (
                              <CheckCircle2 className="w-4 h-4 text-green-400 mx-auto" />
                            ) : d.was_correct_7d === false ? (
                              <XCircle className="w-4 h-4 text-red-400 mx-auto" />
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                          <td className={`py-2 pr-4 text-right font-mono ${(d.actual_return_7d || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {d.actual_return_7d !== undefined ? `${(d.actual_return_7d * 100).toFixed(2)}%` : '—'}
                          </td>
                          <td className="py-2 text-right font-mono text-slate-400 text-xs">
                            {d.brier_score !== undefined ? d.brier_score.toFixed(4) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Empty State */}
              {scorecard.total_decisions === 0 && (
                <div className="glass-static p-12 text-center">
                  <Target className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">No Decisions Recorded</h3>
                  <p className="text-slate-400 text-sm max-w-md mx-auto">
                    Run a live analysis on {ticker} to start recording decisions. The scorecard will automatically
                    track and score predictions at T+7d.
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ── WALK-FORWARD TAB ── */}
      {activeTab === 'walk-forward' && (
        <div className="space-y-6 animate-fade-in-up">
          {!walkForward ? (
            <div className="glass-static p-12 text-center">
              <TrendingUp className="w-12 h-12 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">Walk-Forward Validation</h3>
              <p className="text-slate-400 text-sm max-w-md mx-auto mb-6">
                Score all historical decisions against actual T+7d prices and compute the Deflated Sharpe Ratio.
              </p>
              <button
                onClick={handleWalkForward}
                disabled={validating}
                className="px-6 py-3 bg-gradient-to-r from-accent-teal to-accent-cyan text-navy-950 font-semibold rounded-xl hover:opacity-90 transition-opacity text-sm flex items-center gap-2 mx-auto disabled:opacity-50"
              >
                {validating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
                Run Validation
              </button>
            </div>
          ) : (
            <>
              {/* DSR Hero Card */}
              <div className={`glass-static p-6 border-l-4 ${
                walkForward.overall_metrics.deflated_sharpe_ratio >= 0.95 ? 'border-l-green-400' :
                walkForward.overall_metrics.deflated_sharpe_ratio >= 0.80 ? 'border-l-yellow-400' :
                'border-l-red-400'
              }`}>
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider">Deflated Sharpe Ratio</div>
                    <div className={`text-4xl font-bold ${
                      walkForward.overall_metrics.deflated_sharpe_ratio >= 0.95 ? 'text-green-400' :
                      walkForward.overall_metrics.deflated_sharpe_ratio >= 0.80 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {walkForward.overall_metrics.deflated_sharpe_ratio.toFixed(4)}
                    </div>
                    <div className="text-sm text-slate-400 mt-1">
                      {walkForward.overall_metrics.dsr_interpretation}
                    </div>
                  </div>
                  <div className="text-right space-y-1">
                    <div className="text-xs text-slate-500">n_strategies = 1</div>
                    <div className="text-xs text-slate-500">{walkForward.scored_decisions} decisions scored</div>
                    <div className="text-xs text-slate-500">T+{walkForward.horizon_days}d horizon</div>
                  </div>
                </div>
              </div>

              {/* Walk-Forward Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard label="Win Rate" value={`${(walkForward.overall_metrics.win_rate * 100).toFixed(1)}%`} icon={Target}
                  color={walkForward.overall_metrics.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'} />
                <StatCard label="Sharpe Ratio" value={walkForward.overall_metrics.sharpe_ratio.toFixed(3)} icon={TrendingUp}
                  color={walkForward.overall_metrics.sharpe_ratio > 0 ? 'text-green-400' : 'text-red-400'} />
                <StatCard label="Max Drawdown" value={`${(walkForward.overall_metrics.max_drawdown * 100).toFixed(1)}%`} icon={TrendingDown}
                  color="text-red-400" />
                <StatCard label="Mean Return" value={`${(walkForward.overall_metrics.mean_return * 100).toFixed(3)}%`} icon={Activity}
                  color={walkForward.overall_metrics.mean_return > 0 ? 'text-green-400' : 'text-red-400'} />
              </div>

              {/* Equity Curve */}
              {walkForward.equity_curve.length > 1 && (
                <div className="glass-static p-5">
                  <h3 className="text-sm font-semibold text-white mb-4">Walk-Forward Equity Curve</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={walkForward.equity_curve.map((v, i) => ({ period: i, value: v }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="period" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ background: '#0f1629', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                      <Line type="monotone" dataKey="value" stroke="#06d6a0" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Signal Analysis */}
              {Object.keys(walkForward.signal_analysis).length > 0 && (
                <div className="glass-static p-5">
                  <h3 className="text-sm font-semibold text-white mb-4">Signal Breakdown</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Object.entries(walkForward.signal_analysis).map(([sig, data]) => (
                      <div key={sig} className="p-3 bg-slate-800/60 rounded-lg text-center">
                        <SignalBadge signal={sig} />
                        <div className={`text-lg font-bold mt-2 ${data.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}`}>
                          {(data.win_rate * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-slate-500">{data.sample_size} trades</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ── CALIBRATION TAB ── */}
      {activeTab === 'calibration' && (
        <div className="space-y-6 animate-fade-in-up">
          {!calibration ? (
            <div className="glass-static p-12 text-center">
              <Shield className="w-12 h-12 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">Confidence Calibration Study</h3>
              <p className="text-slate-400 text-sm max-w-md mx-auto mb-6">
                Compute an overconfidence correction factor from scored decisions. Requires at least 10 scored predictions.
              </p>
              <button
                onClick={handleCalibrate}
                disabled={calibrating}
                className="px-6 py-3 bg-gradient-to-r from-accent-purple to-accent-teal text-navy-950 font-semibold rounded-xl hover:opacity-90 transition-opacity text-sm flex items-center gap-2 mx-auto disabled:opacity-50"
              >
                {calibrating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Shield className="w-4 h-4" />}
                Run Calibration Study
              </button>
            </div>
          ) : calibration.error ? (
            <div className="glass-static p-8 text-center">
              <AlertCircle className="w-10 h-10 text-yellow-400 mx-auto mb-3" />
              <h3 className="text-lg font-semibold text-white mb-2">Not Enough Data</h3>
              <p className="text-slate-400 text-sm">{calibration.error}</p>
            </div>
          ) : (
            <>
              {/* Correction Factor Hero */}
              <div className="glass-static p-6 border-l-4 border-l-accent-purple">
                <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider">Overconfidence Correction Factor</div>
                <div className="text-4xl font-bold text-accent-purple">{calibration.correction.toFixed(4)}</div>
                <div className="text-sm text-slate-400 mt-1">
                  LLM confidence is multiplied by this factor during cold-start (&lt;60 decisions)
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard label="Mean Confidence" value={`${(calibration.mean_confidence * 100).toFixed(1)}%`} icon={Target} color="text-accent-teal" />
                <StatCard label="Mean Outcome" value={`${(calibration.mean_outcome * 100).toFixed(1)}%`} icon={CheckCircle2}
                  color={calibration.mean_outcome >= calibration.mean_confidence ? 'text-green-400' : 'text-yellow-400'} />
                <StatCard label="Decisions Used" value={calibration.n_decisions} icon={BarChart3} />
                <StatCard label="Coverage Quality" value={calibration.coverage_quality.toUpperCase()} icon={Shield}
                  color={calibration.coverage_quality === 'high' ? 'text-green-400' : calibration.coverage_quality === 'medium' ? 'text-yellow-400' : 'text-red-400'} />
              </div>

              <div className="glass-static p-5">
                <h3 className="text-sm font-semibold text-white mb-3">Regimes Covered</h3>
                <div className="flex flex-wrap gap-2">
                  {calibration.regimes_covered.length > 0 ? calibration.regimes_covered.map(r => (
                    <span key={r} className="px-3 py-1 bg-slate-800 border border-slate-700 rounded-full text-xs text-slate-300 capitalize">
                      {r.replace('_', ' ')}
                    </span>
                  )) : (
                    <span className="text-slate-500 text-sm">No regime data</span>
                  )}
                </div>
                <p className="text-xs text-slate-500 mt-3">{calibration.note}</p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
