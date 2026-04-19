import { useEffect, useMemo, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  Info,
  BarChart3,
} from 'lucide-react';
import { LineStyle, type UTCTimestamp, type SeriesMarker } from 'lightweight-charts';
import clsx from 'clsx';

import {
  fetchPulseExplain,
  type PulseExplain,
  type ChartPattern,
  type CandleBar,
} from '../lib/api';
import { SEMANTIC, STATE_OPACITY, resolveColor } from '../lib/colors';
import TradingViewCandles, {
  type PriceLineSpec,
  type TrendLineSpec,
} from '../components/TradingViewCandles';
import PatternLegend from '../components/PatternLegend';

type TfKey = '5m' | '15m' | '1h' | '4h';

function styleToEnum(style: string): LineStyle {
  if (style === 'dashed') return LineStyle.Dashed;
  if (style === 'dotted') return LineStyle.Dotted;
  return LineStyle.Solid;
}

function SignalBadge({ signal }: { signal: string }) {
  const isBull = signal === 'BUY' || signal === 'COVER';
  const isBear = signal === 'SHORT' || signal === 'SELL';
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-bold tracking-wide',
        isBull && 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30',
        isBear && 'bg-red-500/15 text-red-400 border border-red-500/30',
        !isBull && !isBear && 'bg-slate-500/15 text-slate-400 border border-slate-500/30',
      )}
    >
      {isBull ? <TrendingUp className="w-3 h-3" /> : isBear ? <TrendingDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
      {signal}
    </span>
  );
}

function TsmomBadge({ direction, strength }: { direction: number | null; strength: number | null }) {
  if (direction === null || direction === undefined) return null;
  const up = direction > 0;
  const color = up ? SEMANTIC.tsmom_up : direction < 0 ? SEMANTIC.tsmom_down : '#64748B';
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium border"
      style={{ color, borderColor: `${color}40`, backgroundColor: `${color}10` }}
    >
      TSMOM {up ? '↑' : direction < 0 ? '↓' : '→'}
      {strength !== null && strength !== undefined && (
        <span className="opacity-70">{Math.abs(strength).toFixed(2)}</span>
      )}
    </span>
  );
}

export default function PulseExplain() {
  const params = useParams<{ ticker: string; ts: string }>();
  const ticker = (params.ticker || '').toUpperCase();
  const ts = decodeURIComponent(params.ts || '');

  const [data, setData] = useState<PulseExplain | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tf, setTf] = useState<TfKey>('1h');
  const [highlighted, setHighlighted] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setErr(null);
    setData(null);
    fetchPulseExplain(ticker, ts)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        // Default TF = one with the most high-scoring patterns, or timeframe_bias, or 1h.
        if (d.timeframe_bias && (d.candles[d.timeframe_bias] ?? []).length > 0) {
          setTf(d.timeframe_bias as TfKey);
        } else if (d.chart_patterns.length > 0) {
          setTf(d.chart_patterns[0].timeframe as TfKey);
        }
      })
      .catch((e) => {
        if (!cancelled) setErr(String(e.message || e));
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, ts]);

  const candles = useMemo<CandleBar[]>(() => data?.candles[tf] ?? [], [data, tf]);

  // Patterns scoped to current TF, sorted by combined_score, invalidated hidden.
  const tfPatterns = useMemo(() => {
    if (!data) return [];
    return data.chart_patterns
      .filter((p) => p.timeframe === tf)
      .sort((a, b) => b.combined_score - a.combined_score);
  }, [data, tf]);

  // Overlay patterns: top 3 + any explicitly highlighted.
  const overlayPatterns = useMemo<ChartPattern[]>(() => {
    const base = tfPatterns.slice(0, 3);
    if (highlighted) {
      const extra = tfPatterns.find((p) => p.name === highlighted && !base.includes(p));
      return extra ? [...base, extra] : base;
    }
    return base;
  }, [tfPatterns, highlighted]);

  // Build TV inputs
  const priceLines = useMemo<PriceLineSpec[]>(() => {
    if (!data) return [];
    const lines: PriceLineSpec[] = [];
    const lv = data.levels;
    const entry = data.entry.price;
    if (entry) lines.push({ price: entry, color: SEMANTIC.entry, lineWidth: 2, lineStyle: LineStyle.Solid, title: `Entry ${entry.toFixed(2)}` });
    if (lv.stop_loss) lines.push({ price: lv.stop_loss, color: SEMANTIC.sl, lineWidth: 2, lineStyle: LineStyle.Dashed, title: `SL ${lv.stop_loss.toFixed(2)}` });
    if (lv.take_profit) lines.push({ price: lv.take_profit, color: SEMANTIC.tp, lineWidth: 2, lineStyle: LineStyle.Dashed, title: `TP ${lv.take_profit.toFixed(2)}` });
    if (lv.support) lines.push({ price: lv.support, color: SEMANTIC.support, lineWidth: 1, lineStyle: LineStyle.Solid, title: `S ${lv.support.toFixed(2)}` });
    if (lv.resistance) lines.push({ price: lv.resistance, color: SEMANTIC.resistance, lineWidth: 1, lineStyle: LineStyle.Solid, title: `R ${lv.resistance.toFixed(2)}` });
    return lines;
  }, [data]);

  const trendLines = useMemo<TrendLineSpec[]>(() => {
    if (candles.length === 0) return [];
    const out: TrendLineSpec[] = [];
    for (const p of overlayPatterns) {
      if (p.state === 'invalidated') continue;
      const dim = highlighted !== null && highlighted !== p.name;
      for (const ln of p.lines) {
        const from = candles[ln.from_idx];
        const to = candles[ln.to_idx];
        if (!from || !to) continue;
        const color = resolveColor(ln.color_token || p.color_token);
        const alpha = Math.round(255 * (STATE_OPACITY[p.state] ?? 1) * (dim ? 0.3 : 1))
          .toString(16)
          .padStart(2, '0');
        out.push({
          from: { ts: from.ts, price: fromIdxPrice(p, ln.from_idx, from) },
          to: { ts: to.ts, price: fromIdxPrice(p, ln.to_idx, to) },
          color: `${color}${alpha}`,
          lineWidth: ln.weight,
          lineStyle: styleToEnum(ln.style),
          title: ln.role,
        });
      }
    }
    return out;
  }, [candles, overlayPatterns, highlighted]);

  const markers = useMemo<SeriesMarker<UTCTimestamp>[]>(() => {
    if (candles.length === 0) return [];
    const out: SeriesMarker<UTCTimestamp>[] = [];
    for (const p of overlayPatterns) {
      if (p.state === 'invalidated') continue;
      const dim = highlighted !== null && highlighted !== p.name;
      const color = resolveColor(p.color_token);
      for (const a of p.anchors) {
        const bar = candles[a.idx];
        if (!bar) continue;
        out.push({
          time: bar.ts as UTCTimestamp,
          position: a.role === 'peak' ? 'aboveBar' : 'belowBar',
          color: dim ? `${color}66` : color,
          shape: a.role === 'peak' ? 'arrowDown' : a.role === 'trough' ? 'arrowUp' : 'circle',
          text: a.label,
        });
      }
    }
    // Entry marker
    if (data?.entry.ts) {
      const entryDt = Math.floor(new Date(data.entry.ts).getTime() / 1000);
      const nearest = candles.reduce<CandleBar | null>((acc, c) => {
        if (!acc) return c;
        return Math.abs(c.ts - entryDt) < Math.abs(acc.ts - entryDt) ? c : acc;
      }, null);
      if (nearest) {
        out.push({
          time: nearest.ts as UTCTimestamp,
          position: 'inBar',
          color: SEMANTIC.entry,
          shape: 'circle',
          text: 'Entry',
        });
      }
    }
    return out.sort((a, b) => (a.time as number) - (b.time as number));
  }, [candles, overlayPatterns, highlighted, data]);

  if (err) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-16">
        <Link to="/pulse" className="text-sm text-slate-400 hover:text-slate-200 inline-flex items-center gap-1">
          <ArrowLeft className="w-4 h-4" /> Back to Pulse
        </Link>
        <div className="mt-8 rounded-lg border border-red-500/30 bg-red-500/5 p-6 text-red-300 text-sm">
          <div className="font-semibold mb-2">Failed to load explain chart</div>
          <div className="text-red-200/70">{err}</div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="max-w-6xl mx-auto px-6 py-16 text-center">
        <div className="w-8 h-8 border-2 border-accent-teal/30 border-t-accent-teal rounded-full animate-spin mx-auto" />
        <div className="text-sm text-slate-500 mt-4">Loading explain chart…</div>
      </div>
    );
  }

  const conf = data.entry.confidence ?? 0;
  const entryDate = data.entry.ts ? new Date(data.entry.ts) : null;

  return (
    <div className="max-w-[1400px] mx-auto px-4 md:px-6 py-6">
      {/* Top bar */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <Link to="/pulse" className="text-slate-400 hover:text-slate-200 inline-flex items-center gap-1 text-sm">
            <ArrowLeft className="w-4 h-4" /> Pulse
          </Link>
          <span className="text-slate-600">/</span>
          <span className="text-slate-300 font-semibold">{ticker}</span>
          <span className="text-slate-600 text-xs">· {entryDate?.toLocaleString()}</span>
        </div>

        <div className="flex items-center gap-2 flex-wrap justify-end">
          <SignalBadge signal={data.entry.signal} />
          <span className="text-xs text-slate-400 bg-navy-900/70 px-2 py-1 rounded border border-white/5">
            Conf {(conf * 100).toFixed(0)}%
          </span>
          <TsmomBadge direction={data.tsmom.direction} strength={data.tsmom.strength} />
          {data.regime_mode && (
            <span className="text-[11px] text-slate-400 bg-navy-900/70 px-2 py-1 rounded border border-white/5">
              {data.regime_mode}
            </span>
          )}
        </div>
      </div>

      {/* Degraded banner */}
      {data.pattern_detection_degraded && (
        <div className="mb-3 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300 flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5" />
          Pattern detection degraded: {data.detector_errors.length} detector(s) failed on this window.
        </div>
      )}

      {/* TF toggle */}
      <div className="mb-3 flex items-center gap-2">
        <span className="text-xs text-slate-500">Timeframe:</span>
        {(['5m', '15m', '1h', '4h'] as TfKey[]).map((t) => (
          <button
            key={t}
            onClick={() => setTf(t)}
            disabled={!data.candles[t] || data.candles[t].length === 0}
            className={clsx(
              'px-2.5 py-1 text-xs rounded border transition-colors',
              tf === t
                ? 'border-accent-teal/50 bg-accent-teal/10 text-accent-teal'
                : 'border-white/10 text-slate-400 hover:border-white/20 hover:text-slate-200',
              (!data.candles[t] || data.candles[t].length === 0) && 'opacity-30 cursor-not-allowed',
            )}
          >
            {t}
          </button>
        ))}
        {data.timeframe_bias && (
          <span className="text-[10px] text-slate-500 ml-2">
            Bias: <span className="text-slate-300">{data.timeframe_bias}</span>
          </span>
        )}
      </div>

      {/* Main layout: chart + sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-4">
        {/* Chart */}
        <div className="rounded-xl border border-white/10 bg-navy-950 p-3">
          {candles.length === 0 ? (
            <div className="h-96 flex items-center justify-center text-sm text-slate-500">
              No candle data available for this timeframe.
            </div>
          ) : (
            <TradingViewCandles
              candles={candles}
              priceLines={priceLines}
              trendLines={trendLines}
              markers={markers}
              height={560}
            />
          )}

          {/* First-view disclaimer */}
          <div className="mt-3 flex items-start gap-2 text-[11px] text-slate-500">
            <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
            <span>
              Fit score measures pattern geometry match, not historical success rate.
              Patterns are detected for explanation only — they do not alter the signal.
            </span>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Top factors */}
          {data.breakdown_top3.length > 0 && (
            <div className="rounded-xl border border-white/10 bg-navy-900/50 p-3">
              <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-2 flex items-center gap-1">
                <BarChart3 className="w-3 h-3" /> Top factors
              </div>
              <div className="space-y-1.5">
                {data.breakdown_top3.map((f) => {
                  const mag = Math.min(Math.abs(f.weight), 1);
                  const up = f.weight >= 0;
                  return (
                    <div key={f.key} className="text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-300">{f.key}</span>
                        <span className={up ? 'text-emerald-400' : 'text-red-400'}>
                          {f.weight >= 0 ? '+' : ''}{f.weight.toFixed(3)}
                        </span>
                      </div>
                      <div className="h-1 mt-1 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className={clsx('h-full rounded-full', up ? 'bg-emerald-500/60' : 'bg-red-500/60')}
                          style={{ width: `${mag * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Pattern legend */}
          <div className="rounded-xl border border-white/10 bg-navy-900/50 p-3">
            <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-2 flex items-center justify-between">
              <span>Patterns on {tf}</span>
              <span className="text-slate-600">{tfPatterns.length}</span>
            </div>
            <PatternLegend
              patterns={tfPatterns}
              highlightedName={highlighted}
              onHighlight={setHighlighted}
            />
          </div>

          {/* Candlestick patterns */}
          {data.candlestick_patterns.length > 0 && (
            <div className="rounded-xl border border-white/10 bg-navy-900/50 p-3">
              <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-2">
                Candlestick patterns
              </div>
              <div className="flex flex-wrap gap-1.5">
                {data.candlestick_patterns.map((c, i) => (
                  <span
                    key={i}
                    className="text-[11px] px-2 py-0.5 rounded bg-navy-800 border border-white/5 text-slate-300"
                  >
                    {c.name.replace(/_/g, ' ')}
                    <span className="text-slate-500 ml-1">· {c.tf}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Reasoning */}
          {data.reasoning_prose && (
            <div className="rounded-xl border border-white/10 bg-navy-900/50 p-3">
              <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-2">Reasoning</div>
              <div className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">
                {data.reasoning_prose}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Resolve the anchor price for a line endpoint. We prefer the pattern's anchor
// price when the line endpoint matches an anchor idx; fall back to candle close.
function fromIdxPrice(p: ChartPattern, idx: number, fallback: CandleBar): number {
  const a = p.anchors.find((x) => x.idx === idx);
  return a ? a.price : fallback.c;
}
