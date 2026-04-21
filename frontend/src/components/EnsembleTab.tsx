// R.6 — Ensemble tab for the Pulse page.
//
// Consumes the read-only ``/api/pulse/ensemble/*`` endpoints shipped in
// R.5. Four panes stacked vertically:
//
//   1. Agreement indicator — ring that colour-codes how many variants
//      voted with the champion on the most recent tick.
//   2. Variant grid — one card per config with the latest signal, DSR,
//      n-signals, and a "Promote" action.
//   3. Per-config metrics table — overall / weekend / OOS split.
//   4. Disagreements log — rolling list of ticks where ≥2 distinct
//      signals fired, so the user can drill into where configs diverge.

import { useCallback, useEffect, useMemo, useState } from 'react';
import { clsx } from 'clsx';
import { Crown, Loader2, RefreshCw } from 'lucide-react';

import SignalBadge from './SignalBadge';
import type {
  EnsembleDisagreementsResponse,
  EnsembleLatest,
  EnsembleMetricsResponse,
} from '../lib/api';
import {
  fetchEnsembleDisagreements,
  fetchEnsembleLatest,
  fetchEnsembleMetrics,
  setEnsembleChampion,
} from '../lib/api';

interface EnsembleTabProps {
  ticker: string;
}

// ── Helpers ─────────────────────────────────────────────────────────

function agreementTone(score: number | null): string {
  // Keep the threshold mapping local so the SignalBadge colour system
  // isn't coupled to ensemble semantics. 0.8+ = emerald, 0.6+ = amber,
  // < 0.6 = red (configs actively fighting each other).
  if (score === null) return 'text-slate-500';
  if (score >= 0.8) return 'text-emerald-400';
  if (score >= 0.6) return 'text-amber-400';
  return 'text-red-400';
}

function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(digits);
}

// ── Main component ──────────────────────────────────────────────────

export default function EnsembleTab({ ticker }: EnsembleTabProps) {
  const [latest, setLatest] = useState<EnsembleLatest | null>(null);
  const [metrics, setMetrics] = useState<EnsembleMetricsResponse | null>(null);
  const [disagreements, setDisagreements] = useState<EnsembleDisagreementsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [promoting, setPromoting] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Parallelise — the three endpoints are independent and the
      // combined latency is dominated by the slowest (metrics, which
      // reads K json files).
      const [l, m, d] = await Promise.all([
        fetchEnsembleLatest(ticker),
        fetchEnsembleMetrics(ticker),
        fetchEnsembleDisagreements(ticker, 25),
      ]);
      setLatest(l);
      setMetrics(m);
      setDisagreements(d);
      setLastRefreshed(new Date());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  useEffect(() => {
    load();
  }, [load]);

  const handlePromote = useCallback(
    async (cfg: string) => {
      if (!latest || cfg === latest.champion) return;
      if (!confirm(
        `Promote "${cfg}" to champion for ${ticker}?\n\n` +
        `This swaps which variant's pulses drive the live UI + ` +
        `downstream risk consumers. Order execution remains gated ` +
        `by EXECUTE_TRADES.`,
      )) return;
      setPromoting(cfg);
      try {
        await setEnsembleChampion(ticker, cfg);
        await load();
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setPromoting(null);
      }
    },
    [latest, ticker, load],
  );

  const variantNames = useMemo(() => {
    if (!latest) return [];
    return Object.keys(latest.variants);
  }, [latest]);

  if (loading && !latest) {
    return (
      <div className="flex items-center justify-center py-16 text-slate-500">
        <Loader2 className="w-5 h-5 animate-spin mr-2" />
        Loading ensemble state…
      </div>
    );
  }

  if (error && !latest) {
    return (
      <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-4 text-sm text-red-300">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header: agreement + refresh */}
      <div className="flex items-center justify-between rounded-xl bg-navy-900/50 border border-white/5 p-4">
        <div className="flex items-center gap-4">
          <div className="flex flex-col">
            <span className="text-xs text-slate-500 uppercase tracking-wide">Agreement</span>
            <span className={clsx('text-2xl font-bold tabular-nums', agreementTone(latest?.agreement_score ?? null))}>
              {latest?.agreement_score !== null && latest?.agreement_score !== undefined
                ? `${Math.round((latest.agreement_score ?? 0) * 100)}%`
                : '—'}
            </span>
            <span className="text-xs text-slate-500">
              {latest?.n_variants ?? 0} variant{(latest?.n_variants ?? 0) === 1 ? '' : 's'}
            </span>
          </div>
          <div className="h-10 w-px bg-white/5" />
          <div className="flex flex-col">
            <span className="text-xs text-slate-500 uppercase tracking-wide">Champion</span>
            <div className="flex items-center gap-2">
              <Crown className="w-4 h-4 text-amber-400" />
              <span className="text-lg font-medium text-white">{latest?.champion ?? 'baseline'}</span>
              {latest?.champion_signal && (
                <SignalBadge signal={latest.champion_signal} size="sm" />
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {lastRefreshed && (
            <span className="text-xs text-slate-500">
              updated {lastRefreshed.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={load}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-300 bg-white/5 hover:bg-white/10 rounded-md disabled:opacity-50"
          >
            <RefreshCw className={clsx('w-3 h-3', loading && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-300">
          {error}
        </div>
      )}

      {/* Variant grid */}
      <div>
        <h3 className="text-sm font-semibold text-slate-300 mb-2">Variants</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {variantNames.map(cfg => {
            const entry = latest?.variants[cfg];
            const cfgMetrics = metrics?.metrics[cfg];
            const isChampion = cfg === latest?.champion;
            const oos = cfgMetrics?.oos_validation;
            const overall = cfgMetrics?.overall;
            return (
              <div
                key={cfg}
                className={clsx(
                  'rounded-xl border p-4 transition-colors',
                  isChampion
                    ? 'bg-amber-500/5 border-amber-500/30'
                    : 'bg-navy-900/40 border-white/5 hover:border-white/10',
                )}
                data-testid={`variant-card-${cfg}`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-1.5">
                      {isChampion && <Crown className="w-3.5 h-3.5 text-amber-400" />}
                      <span className="text-sm font-medium text-white">{cfg}</span>
                    </div>
                    {entry?.ts && (
                      <span className="text-xs text-slate-500">
                        {new Date(entry.ts).toLocaleString()}
                      </span>
                    )}
                  </div>
                  {entry?.signal ? (
                    <SignalBadge signal={entry.signal} size="sm" confidence={entry.confidence} />
                  ) : (
                    <span className="text-xs text-slate-600">no data</span>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-slate-500 block">OOS DSR</span>
                    <span className="text-slate-200 tabular-nums font-medium">
                      {fmtNum(oos?.deflated_sharpe, 3)}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">n signals</span>
                    <span className="text-slate-200 tabular-nums">
                      {overall?.n_signals ?? 0}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Mean net</span>
                    <span className={clsx(
                      'tabular-nums',
                      (overall?.mean_net_return ?? 0) > 0 ? 'text-emerald-400' : 'text-red-400',
                    )}>
                      {fmtPct(overall?.mean_net_return, 3)}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Weekend n</span>
                    <span className={clsx(
                      'tabular-nums',
                      cfgMetrics?.weekend.champion_eligible ? 'text-slate-200' : 'text-amber-400',
                    )}>
                      {cfgMetrics?.weekend.n_signals ?? 0}
                      {cfgMetrics && !cfgMetrics.weekend.champion_eligible && (
                        <span className="text-[10px] text-amber-500 ml-1">(thin)</span>
                      )}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => handlePromote(cfg)}
                  disabled={isChampion || promoting !== null}
                  className={clsx(
                    'mt-3 w-full text-xs py-1.5 rounded-md transition-colors',
                    isChampion
                      ? 'bg-amber-500/10 text-amber-400 cursor-default'
                      : 'bg-white/5 hover:bg-white/10 text-slate-300',
                    promoting !== null && promoting !== cfg && 'opacity-40',
                  )}
                >
                  {isChampion
                    ? 'Current champion'
                    : promoting === cfg
                      ? 'Promoting…'
                      : 'Promote'}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Disagreements log */}
      <div>
        <h3 className="text-sm font-semibold text-slate-300 mb-2">
          Recent Disagreements{' '}
          <span className="text-xs font-normal text-slate-500">
            ({disagreements?.count ?? 0} total)
          </span>
        </h3>
        {disagreements && disagreements.disagreements.length > 0 ? (
          <div className="rounded-xl bg-navy-900/40 border border-white/5 overflow-hidden">
            <table className="w-full text-xs">
              <thead className="bg-white/5 text-slate-400">
                <tr>
                  <th className="text-left px-3 py-2 font-medium">Time</th>
                  {variantNames.map(cfg => (
                    <th key={cfg} className="text-left px-3 py-2 font-medium">{cfg}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {disagreements.disagreements.slice(0, 10).map(d => (
                  <tr key={d.ensemble_tick_id} className="border-t border-white/5">
                    <td className="px-3 py-2 text-slate-400 whitespace-nowrap">
                      {d.ts ? new Date(d.ts).toLocaleTimeString() : '—'}
                    </td>
                    {variantNames.map(cfg => (
                      <td key={cfg} className="px-3 py-2">
                        {d.signals[cfg] ? (
                          <SignalBadge signal={d.signals[cfg]} size="sm" />
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="rounded-xl bg-navy-900/40 border border-white/5 p-6 text-center text-sm text-slate-500">
            No variant disagreements in the recent window.
          </div>
        )}
      </div>
    </div>
  );
}
