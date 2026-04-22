/**
 * Auto-Tune page — Stage 2 Phase A UI.
 *
 * Three tabs:
 *   1. New Tune         — form + SSE progress + verdict card
 *   2. Proposals        — list recent artifacts, detail modal, Apply flow
 *   3. Regime           — current regime status + per-regime metrics +
 *                         realized-vs-predicted calibration trend
 *
 * All backend calls live in `lib/api.ts` (streamAutoTune / listAutoTuneArtifacts
 * / getAutoTuneArtifact / applyAutoTuneArtifact / getAutoTuneJob).
 *
 * Design decisions:
 *   - Propose-only is enforced server-side; UI additionally greys the
 *     Apply button for non-PROPOSE verdicts so mis-clicks don't even
 *     round-trip.
 *   - The "expected_current_config_hash" field is populated automatically
 *     from the artifact's own `current_config_hash` so the server's
 *     drift check fires on stale artifacts.
 *   - Framer-motion is not imported — we keep the animation surface to
 *     CSS transitions for consistency with other pulse pages.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AlertTriangle, ArrowRight, CheckCircle, ChevronRight, Clock,
  FileText, Info, Loader2, Play, RefreshCw, Target, TrendingDown,
  TrendingUp, XCircle, Zap,
} from 'lucide-react';
import { clsx } from 'clsx';
import {
  applyAutoTuneArtifact,
  fetchCurrentRegime,
  getAutoTuneArtifact,
  listAutoTuneArtifacts,
  streamAutoTune,
  type AutoTuneArtifact,
  type AutoTuneArtifactSummary,
  type AutoTuneDiffEntry,
  type AutoTuneMetrics,
  type AutoTuneProgressEvent,
  type AutoTuneRegime,
  type AutoTuneResult,
  type AutoTuneVerdict,
  type CurrentDirectionalRegime,
} from '../lib/api';
import useDocumentTitle from '../hooks/useDocumentTitle';

// ── Types & helpers ──────────────────────────────────────────────────

type TabKey = 'new' | 'proposals' | 'regime';

// Stage 2 Commit G — 'range_bound' is the preferred name; 'sideways' is
// kept in the list for back-compat with existing YAML profiles.
const REGIMES: AutoTuneRegime[] = ['base', 'bull', 'bear', 'range_bound', 'sideways', 'ambiguous'];

const VERDICT_STYLE: Record<AutoTuneVerdict, { bg: string; text: string; icon: React.ReactNode }> = {
  PROPOSE: {
    bg: 'bg-emerald-500/10 border-emerald-500/30',
    text: 'text-emerald-400',
    icon: <CheckCircle className="w-4 h-4" />,
  },
  PROVISIONAL: {
    bg: 'bg-amber-500/10 border-amber-500/30',
    text: 'text-amber-400',
    icon: <AlertTriangle className="w-4 h-4" />,
  },
  REJECT: {
    bg: 'bg-rose-500/10 border-rose-500/30',
    text: 'text-rose-400',
    icon: <XCircle className="w-4 h-4" />,
  },
};

function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return v.toFixed(digits);
}
function daysAgo(iso: string | null | undefined): string {
  if (!iso) return '—';
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 0) return 'just now';
  const mins = Math.floor(ms / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

// ── Page ─────────────────────────────────────────────────────────────

export default function AutoTune() {
  useDocumentTitle('Auto-Tune');

  // Top-level tab selection. New Tune opens by default so first visit
  // drops the user directly into the primary action.
  const [tab, setTab] = useState<TabKey>('new');

  // Cross-tab state — when a tune completes we bump the artifact list
  // so the Proposals tab shows the new entry without a manual refresh.
  const [artifactsEpoch, setArtifactsEpoch] = useState(0);
  const bumpArtifacts = useCallback(() => setArtifactsEpoch((n) => n + 1), []);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <header className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Zap className="w-5 h-5 text-accent-teal" />
          <h1 className="text-2xl font-semibold text-white">Pulse Auto-Tune</h1>
        </div>
        <p className="text-slate-400 text-sm">
          Automatically tests dozens of parameter samples sequentially across historical data to find the safest, most robust configuration. Runs in "proposal mode" — no settings are changed without your approval.
        </p>
      </header>

      {/* Tabs */}
      <div
        role="tablist"
        aria-label="Auto-Tune tabs"
        className="flex items-center gap-1 mb-6 border-b border-white/5"
      >
        {([
          { key: 'new', label: 'New Tune', icon: Play },
          { key: 'proposals', label: 'Proposals', icon: FileText },
          { key: 'regime', label: 'Regime', icon: Activity },
        ] as const).map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => setTab(t.key)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 text-sm font-medium',
              'border-b-2 transition-colors -mb-px',
              tab === t.key
                ? 'border-accent-teal text-accent-teal'
                : 'border-transparent text-slate-400 hover:text-white',
            )}
          >
            <t.icon className="w-4 h-4" />
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'new' && <NewTuneTab onCompleted={bumpArtifacts} />}
      {tab === 'proposals' && <ProposalsTab epoch={artifactsEpoch} onApplied={bumpArtifacts} />}
      {tab === 'regime' && <RegimeTab epoch={artifactsEpoch} />}
    </div>
  );
}

// ── Tab 1: New Tune ──────────────────────────────────────────────────

function NewTuneTab({ onCompleted }: { onCompleted: () => void }) {
  const today = new Date().toISOString().slice(0, 10);
  const twoMonthsAgo = useMemo(() => {
    const d = new Date();
    d.setDate(d.getDate() - 60);
    return d.toISOString().slice(0, 10);
  }, []);

  const [ticker, setTicker] = useState('BTC-USD');
  const [startDate, setStartDate] = useState(twoMonthsAgo);
  const [endDate, setEndDate] = useState(today);
  const [nFolds, setNFolds] = useState(3);
  const [nConfigs, setNConfigs] = useState(30);
  const [regime, setRegime] = useState<AutoTuneRegime>('base');

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<AutoTuneProgressEvent | null>(null);
  const [result, setResult] = useState<AutoTuneResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  useEffect(() => () => cancelRef.current?.(), []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setProgress(null);
    setRunning(true);

    const { cancel, promise } = streamAutoTune(
      ticker,
      {
        start_date: startDate,
        end_date: endDate,
        n_folds: nFolds,
        n_configs: nConfigs,
        active_regime: regime,
      },
      {
        onProgress: (e) => setProgress(e),
        onResult: (r) => setResult(r),
        onError: (msg) => setError(msg),
      },
    );
    cancelRef.current = cancel;
    await promise;
    setRunning(false);
    cancelRef.current = null;
    onCompleted();
  };

  const stop = () => {
    cancelRef.current?.();
    cancelRef.current = null;
    setRunning(false);
    setError('Cancelled by user');
  };

  const pct = progress?.done && progress?.total
    ? Math.round((progress.done / progress.total) * 100)
    : null;

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
      {/* Left column — form */}
      <form
        onSubmit={submit}
        className="glass-static p-5 space-y-4 h-fit"
        aria-label="Auto-tune parameters"
      >
        <h2 className="text-sm font-semibold text-white flex items-center gap-2">
          <Target className="w-4 h-4 text-accent-teal" />
          Parameters
        </h2>

        <FormRow label="Ticker" hint="Crypto only (ends with -USD)">
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            disabled={running}
            className="form-input"
            pattern="[A-Z0-9]+-USD|[A-Z0-9]+USDT"
            required
          />
        </FormRow>

        <div className="grid grid-cols-2 gap-3">
          <FormRow label="Start date">
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              disabled={running}
              className="form-input"
              required
            />
          </FormRow>
          <FormRow label="End date">
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              disabled={running}
              className="form-input"
              required
            />
          </FormRow>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <FormRow label="Sequential Test Windows" hint="Number of distinct, forward-walking historical periods to test against (2–10)">
            <input
              type="number"
              value={nFolds}
              min={2}
              max={10}
              onChange={(e) => setNFolds(parseInt(e.target.value) || 3)}
              disabled={running}
              className="form-input"
            />
          </FormRow>
          <FormRow label="Parameter Samples" hint="Number of randomized parameter scenarios to evaluate (4–100)">
            <input
              type="number"
              value={nConfigs}
              min={4}
              max={100}
              onChange={(e) => setNConfigs(parseInt(e.target.value) || 30)}
              disabled={running}
              className="form-input"
            />
          </FormRow>
        </div>

        <FormRow label="Market Condition (Regime)" hint="Which market environment to optimize for (bull, bear, etc.)">
          <select
            value={regime}
            onChange={(e) => setRegime(e.target.value as AutoTuneRegime)}
            disabled={running}
            className="form-input"
          >
            {REGIMES.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </FormRow>

        <div className="pt-2 flex gap-2">
          {!running ? (
            <button
              type="submit"
              className={clsx(
                'flex-1 flex items-center justify-center gap-2',
                'px-4 py-2 rounded-lg bg-accent-teal/20 border border-accent-teal/40',
                'text-accent-teal hover:bg-accent-teal/30 transition-colors',
                'font-medium text-sm',
              )}
            >
              <Play className="w-4 h-4" />
              Start tune
            </button>
          ) : (
            <button
              type="button"
              onClick={stop}
              className={clsx(
                'flex-1 flex items-center justify-center gap-2',
                'px-4 py-2 rounded-lg bg-rose-500/10 border border-rose-500/30',
                'text-rose-400 hover:bg-rose-500/20 transition-colors',
                'font-medium text-sm',
              )}
            >
              <XCircle className="w-4 h-4" />
              Cancel
            </button>
          )}
        </div>

        <div className="pt-2 border-t border-white/5 text-xs text-slate-500 space-y-1">
          <div className="flex items-start gap-2">
            <Info className="w-3.5 h-3.5 text-slate-500 mt-0.5 shrink-0" />
            <span>
              6 parameters × {nConfigs} candidates × {nFolds} folds ={' '}
              <span className="text-slate-300 font-medium">
                {nConfigs * nFolds * 2}
              </span>{' '}
              backtests.
            </span>
          </div>
        </div>
      </form>

      {/* Right column — live progress + result */}
      <div className="space-y-4">
        {running && (
          <div className="glass-static p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Loader2 className="w-4 h-4 text-accent-teal animate-spin" />
                {progress?.phase === 'selection'
                  ? 'Finding the most robust settings…'
                  : progress?.phase === 'done'
                    ? 'Finalising…'
                    : 'Running backtests…'}
              </h3>
              {pct !== null && (
                <span className="text-xs text-slate-400">
                  {progress?.done} / {progress?.total} · {pct}%
                </span>
              )}
            </div>
            {pct !== null && (
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent-teal transition-all"
                  style={{ width: `${pct}%` }}
                />
              </div>
            )}
            {progress?.fold !== undefined && (
              <p className="text-xs text-slate-500 mt-3">
                fold={progress.fold} · config_idx={progress.config_idx}
              </p>
            )}
          </div>
        )}

        {error && (
          <div className="glass-static p-4 border border-rose-500/30 bg-rose-500/5">
            <div className="flex items-start gap-3">
              <XCircle className="w-5 h-5 text-rose-400 mt-0.5" />
              <div>
                <p className="text-rose-300 font-medium text-sm">Tune failed</p>
                <p className="text-xs text-slate-400 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {result && <ResultCard result={result} onApplied={onCompleted} />}

        {!running && !result && !error && (
          <div className="glass-static p-8 text-center text-slate-500 text-sm">
            Configure parameters and click <span className="text-slate-300">Start tune</span>.
            Progress + verdict will appear here.
          </div>
        )}
      </div>
    </div>
  );
}

// Helper form row.
function FormRow({
  label, hint, children,
}: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs text-slate-400 font-medium">{label}</span>
      {children}
      {hint && <span className="text-[11px] text-slate-600 mt-0.5 block">{hint}</span>}
    </label>
  );
}

// ── Result / Proposal card ───────────────────────────────────────────

function ResultCard({
  result,
  onApplied,
  // compact = inside the artifact detail view, hide the big action row
  compact = false,
}: {
  result: AutoTuneResult | AutoTuneArtifact;
  onApplied: () => void;
  compact?: boolean;
}) {
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);
  const [applied, setApplied] = useState(false);

  const verdict = result.verdict;
  const style = VERDICT_STYLE[verdict];
  const canApply = verdict === 'PROPOSE' && !applied;

  const artifactPath = 'artifact_path' in result ? result.artifact_path : null;

  const applyNow = async () => {
    if (!artifactPath) {
      setApplyError('No artifact_path — re-run the tune.');
      return;
    }
    if (!confirm(
      `Apply proposal to live config?\n\n` +
      `Regime: ${(result as AutoTuneArtifact).spec?.active_regime ?? 'base'}\n` +
      `Changes: ${result.diff.length}`,
    )) return;
    setApplying(true);
    setApplyError(null);
    try {
      await applyAutoTuneArtifact(artifactPath, result.current_config_hash);
      setApplied(true);
      onApplied();
    } catch (e) {
      setApplyError((e as Error).message);
    } finally {
      setApplying(false);
    }
  };

  return (
    <div className={clsx('glass-static p-5 border', style.bg)}>
      {/* Verdict row */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className={clsx(
            'inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-bold uppercase tracking-wide',
            style.text, style.bg,
          )}>
            {style.icon}
            {verdict}
          </div>
          {result.reasons?.length > 0 && (
            <p className="text-xs text-slate-400 mt-2 max-w-lg">
              {result.reasons[0]}
            </p>
          )}
        </div>
        {!compact && artifactPath && canApply && (
          <button
            onClick={applyNow}
            disabled={applying || applied}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium',
              applied
                ? 'bg-emerald-500/20 text-emerald-300 cursor-default'
                : 'bg-accent-teal/20 border border-accent-teal/40 text-accent-teal hover:bg-accent-teal/30',
            )}
          >
            {applying ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : applied ? (
              <><CheckCircle className="w-4 h-4" /> Applied</>
            ) : (
              <><ArrowRight className="w-4 h-4" /> Apply to config</>
            )}
          </button>
        )}
      </div>

      {applyError && (
        <div className="mb-4 px-3 py-2 rounded-md bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300">
          {applyError}
        </div>
      )}

      {/* Metrics grid */}
      <MetricsGrid metrics={result.metrics} />

      {/* Diff table */}
      {result.diff.length > 0 && (
        <div className="mt-5">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
            Proposed changes
          </h4>
          <DiffTable diff={result.diff} />
        </div>
      )}

      {result.diff.length === 0 && (
        <p className="mt-4 text-xs text-slate-500 italic">
          No parameter changes vs current config.
        </p>
      )}

      {/* Config hash row */}
      <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between text-[11px] text-slate-500 font-mono">
        <span>base {result.current_config_hash.slice(0, 12)}</span>
        <ChevronRight className="w-3 h-3" />
        <span>proposed {result.proposed_config_hash.slice(0, 12)}</span>
      </div>
    </div>
  );
}

function MetricsGrid({ metrics }: { metrics: AutoTuneMetrics }) {
  const items = [
    { label: 'Out-of-Sample Sharpe', value: fmtNum(metrics.oos_sharpe_point, 2), sub:
      metrics.oos_sharpe_ci_lower !== undefined && metrics.oos_sharpe_ci_upper !== undefined
        ? `[${fmtNum(metrics.oos_sharpe_ci_lower, 2)}, ${fmtNum(metrics.oos_sharpe_ci_upper, 2)}]`
        : 'Risk-adjusted performance on unseen data',
    },
    { label: 'Robustness Score (Deflated Sharpe)', value: fmtNum(metrics.deflated_oos_sharpe, 2),
      sub: 'Accounts for multiple testing bias' },
    { label: 'Risk of Overfitting (PBO)', value: fmtNum(metrics.pbo, 3),
      sub: 'Probability of Backtest Overfitting (lower is better)' },
    { label: 'Total Trades (OOS)', value: metrics.oos_n_trades_total?.toString() ?? '—',
      sub: metrics.n_folds_used ? `Trades made during test periods (${metrics.n_folds_used} folds)` : 'Trades made during test periods' },
  ];
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {items.map((i) => (
        <div key={i.label} className="bg-slate-900/40 border border-white/5 rounded-lg p-3">
          <div className="text-[11px] text-slate-500 uppercase tracking-wide">{i.label}</div>
          <div className="text-lg font-semibold text-white mt-0.5 tabular-nums">{i.value}</div>
          {i.sub && <div className="text-[10px] text-slate-600 mt-0.5">{i.sub}</div>}
        </div>
      ))}
    </div>
  );
}

function DiffTable({ diff }: { diff: AutoTuneDiffEntry[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 border-b border-white/5">
            <th className="text-left py-2 font-medium">Parameter</th>
            <th className="text-right py-2 font-medium">Current</th>
            <th className="text-right py-2 font-medium">Proposed</th>
            <th className="text-right py-2 font-medium">Δ</th>
          </tr>
        </thead>
        <tbody>
          {diff.map((row) => {
            const delta = row.delta ?? 0;
            const up = delta > 0;
            return (
              <tr key={row.path} className="border-b border-white/5 last:border-0">
                <td className="py-2 text-slate-300 font-mono text-[11px]">
                  {row.path.replace('confluence.', '')}
                </td>
                <td className="text-right text-slate-400 tabular-nums">{fmtNum(row.old, 4)}</td>
                <td className="text-right text-white tabular-nums">{fmtNum(row.new, 4)}</td>
                <td className={clsx(
                  'text-right tabular-nums flex items-center justify-end gap-1',
                  up ? 'text-emerald-400' : 'text-amber-400',
                )}>
                  {up ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  {delta > 0 ? '+' : ''}{fmtNum(delta, 4)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Tab 2: Proposals list + detail ───────────────────────────────────

function ProposalsTab({ epoch, onApplied }: { epoch: number; onApplied: () => void }) {
  const [list, setList] = useState<AutoTuneArtifactSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [verdictFilter, setVerdictFilter] = useState<AutoTuneVerdict | 'ALL'>('ALL');
  const [selected, setSelected] = useState<AutoTuneArtifactSummary | null>(null);

  useEffect(() => {
    let abort = false;
    setLoading(true);
    listAutoTuneArtifacts(50)
      .then((arts) => { if (!abort) { setList(arts); setError(null); } })
      .catch((e) => { if (!abort) setError((e as Error).message); })
      .finally(() => { if (!abort) setLoading(false); });
    return () => { abort = true; };
  }, [epoch]);

  const filtered = useMemo(
    () => verdictFilter === 'ALL' ? list : list.filter((a) => a.verdict === verdictFilter),
    [list, verdictFilter],
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Filter:</span>
          {(['ALL', 'PROPOSE', 'PROVISIONAL', 'REJECT'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setVerdictFilter(v)}
              className={clsx(
                'px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
                verdictFilter === v
                  ? 'bg-accent-teal/20 text-accent-teal border border-accent-teal/40'
                  : 'text-slate-400 hover:text-white border border-white/5',
              )}
            >
              {v}
            </button>
          ))}
        </div>
        <button
          onClick={() => {
            setLoading(true);
            listAutoTuneArtifacts(50).then(setList).finally(() => setLoading(false));
          }}
          className="text-xs text-slate-400 hover:text-white flex items-center gap-1"
        >
          <RefreshCw className="w-3 h-3" />
          Refresh
        </button>
      </div>

      {loading && <div className="text-sm text-slate-500">Loading proposals…</div>}
      {error && <div className="text-sm text-rose-400">{error}</div>}
      {!loading && !error && filtered.length === 0 && (
        <div className="glass-static p-8 text-center text-slate-500 text-sm">
          No proposals yet. Start a tune in the <span className="text-slate-300">New Tune</span> tab.
        </div>
      )}

      {filtered.length > 0 && (
        <div className="glass-static overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-900/60 text-xs text-slate-500 uppercase tracking-wide">
              <tr>
                <th className="text-left px-4 py-3 font-medium">When</th>
                <th className="text-left px-4 py-3 font-medium">Ticker</th>
                <th className="text-left px-4 py-3 font-medium">Regime</th>
                <th className="text-left px-4 py-3 font-medium">Verdict</th>
                <th className="text-right px-4 py-3 font-medium">Changes</th>
                <th className="text-right px-4 py-3 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((a) => (
                <tr
                  key={a.artifact}
                  onClick={() => setSelected(a)}
                  className="border-t border-white/5 hover:bg-white/[0.02] cursor-pointer transition-colors"
                >
                  <td className="px-4 py-3 text-slate-300">
                    <div className="flex items-center gap-2">
                      <Clock className="w-3.5 h-3.5 text-slate-500" />
                      {daysAgo(a.ran_at)}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-white font-medium">{a.ticker}</td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-0.5 bg-slate-800 rounded text-xs capitalize">
                      {a.active_regime}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <VerdictBadge verdict={a.verdict} />
                  </td>
                  <td className="px-4 py-3 text-right text-slate-300 tabular-nums">
                    {a.n_changes}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <ChevronRight className="w-4 h-4 text-slate-500 inline" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <ArtifactModal
          summary={selected}
          onClose={() => setSelected(null)}
          onApplied={() => { onApplied(); setSelected(null); }}
        />
      )}
    </div>
  );
}

function VerdictBadge({ verdict }: { verdict: AutoTuneVerdict }) {
  const s = VERDICT_STYLE[verdict];
  return (
    <span className={clsx(
      'inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wide',
      s.text, s.bg, 'border',
    )}>
      {s.icon}
      {verdict}
    </span>
  );
}

function ArtifactModal({
  summary, onClose, onApplied,
}: {
  summary: AutoTuneArtifactSummary;
  onClose: () => void;
  onApplied: () => void;
}) {
  const [art, setArt] = useState<AutoTuneArtifact | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let abort = false;
    getAutoTuneArtifact(summary.artifact)
      .then((a) => { if (!abort) setArt(a); })
      .catch((e) => { if (!abort) setErr((e as Error).message); });
    return () => { abort = true; };
  }, [summary.artifact]);

  const tuneResult = useMemo<AutoTuneResult | null>(() => {
    if (!art) return null;
    // Map artifact → AutoTuneResult shape (adds artifact_path + job_id).
    return {
      job_id: summary.artifact.replace(/\.json$/, ''),
      verdict: art.verdict,
      reasons: art.reasons,
      current_config_hash: art.current_config_hash,
      proposed_config: art.proposed_config,
      proposed_config_hash: art.proposed_config_hash,
      diff: art.diff,
      metrics: art.metrics,
      per_fold: art.per_fold,
      artifact_path: summary.path,
      ran_at: art.ran_at,
    };
  }, [art, summary]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-navy-950/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="bg-navy-900 border border-white/10 rounded-xl max-w-3xl w-full max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-navy-900 border-b border-white/5 px-5 py-3 flex items-center justify-between">
          <div>
            <h3 className="text-white font-semibold">{summary.artifact}</h3>
            <p className="text-xs text-slate-500">
              {summary.ticker} · regime={summary.active_regime} · {daysAgo(summary.ran_at)}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white text-sm"
          >
            Close ✕
          </button>
        </div>
        <div className="p-5">
          {err && <div className="text-rose-400 text-sm">{err}</div>}
          {!art && !err && <div className="text-slate-500 text-sm">Loading…</div>}
          {tuneResult && <ResultCard result={tuneResult} onApplied={onApplied} />}
        </div>
      </div>
    </div>
  );
}

// ── Tab 3: Regime status + calibration trend ─────────────────────────

function RegimeTab({ epoch }: { epoch: number }) {
  const [list, setList] = useState<AutoTuneArtifactSummary[]>([]);
  const [details, setDetails] = useState<AutoTuneArtifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // Fetch recent artifacts + their full payloads (cap to 30 to bound the
  // number of round-trips; more than that and the calibration trend is
  // visually uninformative anyway).
  useEffect(() => {
    let abort = false;
    setLoading(true);
    (async () => {
      try {
        const summaries = await listAutoTuneArtifacts(30);
        if (abort) return;
        setList(summaries);
        const full = await Promise.all(
          summaries.map((s) => getAutoTuneArtifact(s.artifact).catch(() => null)),
        );
        if (abort) return;
        setDetails(full.filter((x): x is AutoTuneArtifact => x !== null));
      } catch (e) {
        if (!abort) setErr((e as Error).message);
      } finally {
        if (!abort) setLoading(false);
      }
    })();
    return () => { abort = true; };
  }, [epoch]);

  // Group artifacts by regime to compute latest-per-regime stats.
  const byRegime = useMemo(() => {
    const m = new Map<AutoTuneRegime, AutoTuneArtifact[]>();
    for (const d of details) {
      const r = (d.spec?.active_regime ?? 'base') as AutoTuneRegime;
      if (!m.has(r)) m.set(r, []);
      m.get(r)!.push(d);
    }
    // Sort each bucket newest-first.
    for (const [, arr] of m) {
      arr.sort((a, b) => (b.ran_at || '').localeCompare(a.ran_at || ''));
    }
    return m;
  }, [details]);

  return (
    <div className="space-y-4">
      {loading && <div className="text-sm text-slate-500">Loading regime data…</div>}
      {err && <div className="text-sm text-rose-400">{err}</div>}

      {!loading && !err && (
        <>
          {/* Stage 2 Commit J — "currently detected regime" callout. */}
          <CurrentRegimeCard ticker="BTC-USD" />

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {REGIMES.map((r) => {
              const arr = byRegime.get(r) ?? [];
              const latest = arr[0] ?? null;
              return <RegimeStatusCard key={r} regime={r} latest={latest} tuneCount={arr.length} />;
            })}
          </div>

          <CalibrationTrend artifacts={details} />

          {list.length === 0 && (
            <div className="glass-static p-8 text-center text-slate-500 text-sm">
              Once tunes complete, their realized-vs-predicted Sharpe will plot here.
            </div>
          )}
        </>
      )}
    </div>
  );
}

/**
 * Stage 2 Commit J — "currently detected regime" card.
 *
 * Polls `/api/pulse/regime/current/{ticker}` every 5 minutes. The card
 * is informational only: clicking it doesn't switch the active profile.
 * If the detected label differs from the user's active regime (inferred
 * from the most recent artifact), a soft amber callout appears
 * suggesting a manual switch.
 */
function CurrentRegimeCard({ ticker }: { ticker: string }) {
  const [data, setData] = useState<CurrentDirectionalRegime | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let abort = false;
    const load = async () => {
      try {
        const r = await fetchCurrentRegime(ticker);
        if (!abort) { setData(r); setErr(null); }
      } catch (e) {
        if (!abort) setErr((e as Error).message);
      }
    };
    load();
    const id = setInterval(load, 5 * 60 * 1000);
    return () => { abort = true; clearInterval(id); };
  }, [ticker]);

  const labelStyle: Record<string, string> = {
    bull: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
    bear: 'text-rose-400 bg-rose-500/10 border-rose-500/30',
    range_bound: 'text-sky-400 bg-sky-500/10 border-sky-500/30',
    ambiguous: 'text-slate-400 bg-slate-500/10 border-slate-500/30',
  };

  return (
    <div className="glass-static p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">
            Currently detected regime · {ticker}
          </p>
          {data ? (
            <>
              <div className="flex items-center gap-2">
                <span className={clsx(
                  'inline-block px-3 py-1 rounded-md border text-sm font-semibold capitalize',
                  labelStyle[data.label] ?? labelStyle.ambiguous,
                )}>
                  {data.label.replace('_', ' ')}
                </span>
                {data.insufficient_history && (
                  <span className="text-xs text-amber-400">insufficient history</span>
                )}
              </div>
              <p className="text-xs text-slate-400 mt-2">{data.reason}</p>
            </>
          ) : err ? (
            <p className="text-xs text-rose-400">{err}</p>
          ) : (
            <p className="text-xs text-slate-500">Loading…</p>
          )}
        </div>
        {data && !data.insufficient_history && (
          <div className="grid grid-cols-2 gap-2 text-right text-xs text-slate-400 min-w-[12rem]">
            <span>90d return</span>
            <span className="text-white font-mono">
              {(data.return_90d * 100).toFixed(1)}%
            </span>
            <span>30d return</span>
            <span className="text-white font-mono">
              {(data.return_30d * 100).toFixed(1)}%
            </span>
            <span>% above SMA30</span>
            <span className="text-white font-mono">
              {(data.frac_above_sma30 * 100).toFixed(0)}%
            </span>
            <span>range/ATR</span>
            <span className="text-white font-mono">
              {data.range_atr_ratio.toFixed(2)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}


function RegimeStatusCard({
  regime, latest, tuneCount,
}: {
  regime: AutoTuneRegime;
  latest: AutoTuneArtifact | null;
  tuneCount: number;
}) {
  return (
    <div className="glass-static p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-white capitalize">{regime}</h4>
        <span className="text-xs text-slate-500">
          {tuneCount} {tuneCount === 1 ? 'tune' : 'tunes'}
        </span>
      </div>

      {!latest ? (
        <p className="text-xs text-slate-500">No tunes for this regime yet.</p>
      ) : (
        <>
          <div className="flex items-center justify-between mb-2">
            <VerdictBadge verdict={latest.verdict} />
            <span className="text-[11px] text-slate-500">{daysAgo(latest.ran_at)}</span>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-3">
            <Stat label="OOS Sharpe" value={fmtNum(latest.metrics.oos_sharpe_point, 2)} />
            <Stat label="Deflated" value={fmtNum(latest.metrics.deflated_oos_sharpe, 2)} />
            <Stat label="PBO" value={fmtNum(latest.metrics.pbo, 2)} />
            <Stat label="OOS trades" value={latest.metrics.oos_n_trades_total?.toString() ?? '—'} />
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-900/40 border border-white/5 rounded p-2">
      <div className="text-[10px] text-slate-500 uppercase">{label}</div>
      <div className="text-sm font-semibold text-white tabular-nums">{value}</div>
    </div>
  );
}

function CalibrationTrend({ artifacts }: { artifacts: AutoTuneArtifact[] }) {
  // Build (x = ran_at, predicted = CI-lower, realized = point_estimate)
  // ordered oldest→newest so the chart reads left-to-right chronologically.
  // Predicted = ci_lower because that's what the selection optimized on;
  // realized = the point estimate from the same OOS fold pool. In an
  // honest walk-forward they should track closely; systematic bias
  // surfaces as the two series diverging.
  //
  // Stage 2 Commit Q — colour-code points by the UNBIASED delta
  // `realized - point_estimate`. When we plumb the live-scorecard
  // Sharpe through (future), the colour flips red/green on sign. For
  // the current mono-source path (realized = point from same artifact)
  // the delta is zero by construction so we colour by
  // `(point - CI-lower)` magnitude as a liquidity-of-confidence proxy
  // instead; note this is a v1 stopgap and the original v1 spec's
  // `realized - CI_lower` was biased ≥0 → the plan rejected it.
  const rows = useMemo(() => {
    return artifacts
      .filter((a) => a.metrics.oos_sharpe_point !== undefined)
      .sort((a, b) => (a.ran_at || '').localeCompare(b.ran_at || ''))
      .map((a) => ({
        ts: a.ran_at,
        label: new Date(a.ran_at).toLocaleDateString(),
        regime: (a.spec?.active_regime ?? 'base') as AutoTuneRegime,
        predicted: a.metrics.oos_sharpe_ci_lower ?? 0,
        realized: a.metrics.oos_sharpe_point ?? 0,
        upper: a.metrics.oos_sharpe_ci_upper ?? 0,
      }));
  }, [artifacts]);

  if (rows.length < 2) return null;

  // Simple SVG sparkline — avoids a Recharts import in what's otherwise
  // a single-axis scatter. The two series share the same y-scale.
  const W = 700, H = 180, P = 24;
  const ys = rows.flatMap((r) => [r.predicted, r.realized, r.upper]);
  const yMin = Math.min(...ys, 0);
  const yMax = Math.max(...ys, 1);
  const xStep = (W - 2 * P) / Math.max(1, rows.length - 1);
  const yScale = (v: number) => H - P - ((v - yMin) / (yMax - yMin || 1)) * (H - 2 * P);

  const pathPred = rows.map((r, i) => `${i === 0 ? 'M' : 'L'}${P + i * xStep},${yScale(r.predicted)}`).join(' ');
  const pathReal = rows.map((r, i) => `${i === 0 ? 'M' : 'L'}${P + i * xStep},${yScale(r.realized)}`).join(' ');

  return (
    <div className="glass-static p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Activity className="w-4 h-4 text-accent-teal" />
          Calibration trend
        </h3>
        <div className="flex items-center gap-3 text-[11px]">
          <span className="flex items-center gap-1 text-slate-400">
            <span className="w-3 h-0.5 bg-accent-teal inline-block" /> Realized (point)
          </span>
          <span className="flex items-center gap-1 text-slate-400">
            <span className="w-3 h-0.5 bg-slate-500 inline-block" /> Predicted (CI-lower)
          </span>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-44" aria-label="Calibration trend">
        {/* zero line */}
        <line
          x1={P} x2={W - P}
          y1={yScale(0)} y2={yScale(0)}
          stroke="rgba(148,163,184,0.2)" strokeWidth={1} strokeDasharray="2 3"
        />
        <path d={pathPred} fill="none" stroke="rgb(100,116,139)" strokeWidth={1.5} strokeDasharray="4 3" />
        <path d={pathReal} fill="none" stroke="rgb(20,184,166)" strokeWidth={2} />
        {rows.map((r, i) => (
          <circle
            key={r.ts}
            cx={P + i * xStep}
            cy={yScale(r.realized)}
            r={3}
            // Stage 2 Commit Q colour: red when delta < -0.1 (point
            // materially below CI-lower — broken bootstrap), amber for
            // 0-0.1 (tight gap → high uncertainty), teal elsewhere.
            // Swap to `realized_live - point` once scorecard is wired.
            fill={(() => {
              const delta = r.realized - r.predicted;
              if (delta < -0.1) return 'rgb(239,68,68)';
              if (delta < 0.1) return 'rgb(245,158,11)';
              return 'rgb(20,184,166)';
            })()}
          >
            <title>
              {r.label} · regime={r.regime}{'\n'}
              realized={r.realized.toFixed(2)} · predicted={r.predicted.toFixed(2)}
              {'\n'}delta={(r.realized - r.predicted).toFixed(2)}
            </title>
          </circle>
        ))}
      </svg>
      <p className="text-[11px] text-slate-500 mt-2">
        Point = realized OOS Sharpe. Dashed = bootstrap CI-lower (what
        UCB selection optimized on). Persistent divergence suggests a
        bias in the cost / funding / regime assumptions.
      </p>
    </div>
  );
}
