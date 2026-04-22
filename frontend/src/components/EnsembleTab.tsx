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
//   4. Recent Ensemble Results — rolling list of ALL ticks with every
//      variant's signal, confidence, and expandable reasoning.

import { useCallback, useEffect, useMemo, useState } from 'react';
import { clsx } from 'clsx';
import { ChevronDown, ChevronRight, Crown, Loader2, RefreshCw } from 'lucide-react';

import SignalBadge from './SignalBadge';
import type {
  EnsembleLatest,
  EnsembleMetricsResponse,
  EnsembleTicksResponse,
} from '../lib/api';
import {
  fetchEnsembleLatest,
  fetchEnsembleMetrics,
  fetchEnsembleTicks,
  setEnsembleChampion,
} from '../lib/api';

interface EnsembleTabProps {
  ticker: string;
  refreshTrigger?: number;
}

// ── Helpers ─────────────────────────────────────────────────────────

function agreementTone(score: number | null): string {
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

export default function EnsembleTab({ ticker, refreshTrigger }: EnsembleTabProps) {
  const [latest, setLatest] = useState<EnsembleLatest | null>(null);
  const [metrics, setMetrics] = useState<EnsembleMetricsResponse | null>(null);
  const [ticks, setTicks] = useState<EnsembleTicksResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [promoting, setPromoting] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const [expandedTicks, setExpandedTicks] = useState<Set<string>>(new Set());

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [l, m, t] = await Promise.all([
        fetchEnsembleLatest(ticker),
        fetchEnsembleMetrics(ticker),
        fetchEnsembleTicks(ticker, 25),
      ]);
      setLatest(l);
      setMetrics(m);
      setTicks(t);
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

  // Re-load when parent triggers a refresh (60s poll, manual run, scheduler toggle)
  useEffect(() => {
    if (refreshTrigger !== undefined && refreshTrigger > 0) {
      load();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshTrigger]);

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

  const toggleExpand = useCallback((tickId: string) => {
    setExpandedTicks(prev => {
      const next = new Set(prev);
      if (next.has(tickId)) next.delete(tickId);
      else next.add(tickId);
      return next;
    });
  }, []);

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

      {/* Recent Ensemble Results (replaces Disagreements) */}
      <div>
        <h3 className="text-sm font-semibold text-slate-300 mb-2">
          Recent Ensemble Results{' '}
          <span className="text-xs font-normal text-slate-500">
            ({ticks?.count ?? 0} ticks)
          </span>
        </h3>
        {ticks && ticks.ticks.length > 0 ? (
          <div className="rounded-xl bg-navy-900/40 border border-white/5 overflow-hidden">
            <table className="w-full text-xs">
              <thead className="bg-white/5 text-slate-400">
                <tr>
                  <th className="text-left px-3 py-2 font-medium w-8"></th>
                  <th className="text-left px-3 py-2 font-medium">Time</th>
                  <th className="text-right px-3 py-2 font-medium">Price</th>
                  {variantNames.map(cfg => (
                    <th key={cfg} className="text-center px-3 py-2 font-medium">{cfg}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ticks.ticks.map((tick, idx) => {
                  const isExpanded = expandedTicks.has(tick.ensemble_tick_id);
                  return (
                    <TickRow
                      key={tick.ensemble_tick_id}
                      tick={tick}
                      variantNames={variantNames}
                      isExpanded={isExpanded}
                      isFirst={idx === 0}
                      onToggle={() => toggleExpand(tick.ensemble_tick_id)}
                    />
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="rounded-xl bg-navy-900/40 border border-white/5 p-6 text-center text-sm text-slate-500">
            No ensemble tick data available yet. Run a pulse to generate variant results.
          </div>
        )}
      </div>
    </div>
  );
}

// ── Tick Row (with expandable reasoning) ────────────────────────────

interface TickRowProps {
  tick: {
    ensemble_tick_id: string;
    ts: string;
    price?: number;
    variants: Record<string, { signal: string; confidence: number; normalized_score?: number; price?: number; reasoning?: string } | null>;
  };
  variantNames: string[];
  isExpanded: boolean;
  isFirst: boolean;
  onToggle: () => void;
}

function TickRow({ tick, variantNames, isExpanded, isFirst, onToggle }: TickRowProps) {
  return (
    <>
      <tr
        className={clsx(
          'border-t border-white/5 cursor-pointer hover:bg-white/[0.02] transition-colors',
          isFirst && 'bg-accent-teal/[0.03]',
        )}
        onClick={onToggle}
      >
        <td className="px-3 py-2 text-slate-500">
          {isExpanded
            ? <ChevronDown className="w-3 h-3" />
            : <ChevronRight className="w-3 h-3" />}
        </td>
        <td className="px-3 py-2 text-slate-400 whitespace-nowrap">
          {tick.ts ? new Date(tick.ts).toLocaleString() : '—'}
        </td>
        <td className="px-3 py-2 text-slate-300 text-right tabular-nums whitespace-nowrap">
          {tick.price != null
            ? `$${tick.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
            : '—'}
        </td>
        {variantNames.map(cfg => {
          const v = tick.variants[cfg];
          return (
            <td key={cfg} className="px-3 py-2 text-center">
              {v ? (
                <div className="flex flex-col items-center gap-0.5">
                  <SignalBadge signal={v.signal} size="sm" />
                  <span className="text-[10px] text-slate-500 tabular-nums">
                    {Math.round((v.confidence ?? 0) * 100)}%
                  </span>
                </div>
              ) : (
                <span className="text-slate-600">—</span>
              )}
            </td>
          );
        })}
      </tr>
      {isExpanded && (
        <tr className="border-t border-white/5">
          <td colSpan={3 + variantNames.length} className="p-0">
            <div className="grid gap-2 p-3 bg-navy-950/50" style={{ gridTemplateColumns: `repeat(${variantNames.length}, 1fr)` }}>
              {variantNames.map(cfg => {
                const v = tick.variants[cfg];
                return (
                  <div
                    key={cfg}
                    className="rounded-lg bg-navy-900/60 border border-white/5 p-3 text-xs space-y-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-slate-300">{cfg}</span>
                      {v ? <SignalBadge signal={v.signal} size="sm" /> : <span className="text-slate-600">—</span>}
                    </div>
                    {v && (
                      <>
                        <div className="flex gap-3 text-[10px] text-slate-500">
                          <span>Conf: {Math.round((v.confidence ?? 0) * 100)}%</span>
                          {v.normalized_score != null && (
                            <span className={v.normalized_score >= 0 ? 'text-emerald-500/70' : 'text-red-500/70'}>
                              Score: {v.normalized_score >= 0 ? '+' : ''}{v.normalized_score.toFixed(3)}
                            </span>
                          )}
                          {v.price != null && (
                            <span>${v.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                          )}
                        </div>
                        {v.reasoning && (
                          <p className="text-[11px] text-slate-400 leading-relaxed max-h-24 overflow-y-auto">
                            {v.reasoning}
                          </p>
                        )}
                      </>
                    )}
                    {!v && (
                      <p className="text-[11px] text-slate-600 italic">No data for this tick</p>
                    )}
                  </div>
                );
              })}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
