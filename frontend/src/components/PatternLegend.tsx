import { TrendingDown, TrendingUp, Minus, Sparkles } from 'lucide-react';
import clsx from 'clsx';
import type { ChartPattern } from '../lib/api';
import { resolveColor } from '../lib/colors';

interface Props {
  patterns: ChartPattern[];
  highlightedName: string | null;
  onHighlight: (name: string | null) => void;
}

const STATE_LABEL: Record<string, string> = {
  forming: 'Forming',
  completed: 'Completed',
  confirmed: 'Confirmed',
  retested: 'Retested',
  invalidated: 'Invalidated',
};

const STATE_COLOR: Record<string, string> = {
  forming: 'text-amber-300/80 bg-amber-500/10',
  completed: 'text-sky-300/80 bg-sky-500/10',
  confirmed: 'text-emerald-300 bg-emerald-500/15',
  retested: 'text-emerald-400 bg-emerald-500/20',
  invalidated: 'text-slate-500 bg-slate-500/10 line-through',
};

function BiasIcon({ bias }: { bias: string }) {
  if (bias === 'bullish') return <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />;
  if (bias === 'bearish') return <TrendingDown className="w-3.5 h-3.5 text-red-400" />;
  return <Minus className="w-3.5 h-3.5 text-slate-500" />;
}

export default function PatternLegend({ patterns, highlightedName, onHighlight }: Props) {
  if (patterns.length === 0) {
    return (
      <div className="text-xs text-slate-500 py-6 text-center border border-dashed border-white/5 rounded-lg">
        No chart patterns detected in this window.
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {patterns.map((p) => {
        const isHL = highlightedName === p.name;
        const dim = highlightedName !== null && !isHL;
        const invalid = p.state === 'invalidated';

        return (
          <button
            key={`${p.name}-${p.timeframe}-${p.anchors[0]?.idx}`}
            onClick={() => onHighlight(isHL ? null : p.name)}
            className={clsx(
              'w-full text-left rounded-lg border transition-all px-3 py-2',
              isHL
                ? 'border-accent-teal/40 bg-accent-teal/5'
                : 'border-white/5 bg-navy-900/40 hover:border-white/15',
              dim && 'opacity-40',
            )}
          >
            <div className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{
                  backgroundColor: resolveColor(p.color_token),
                  boxShadow: p.regime_aligned
                    ? `0 0 6px ${resolveColor(p.color_token)}`
                    : undefined,
                }}
                aria-hidden
              />
              <BiasIcon bias={p.bias} />
              <span
                className={clsx(
                  'text-sm font-medium flex-1 truncate',
                  invalid ? 'text-slate-500 line-through' : 'text-slate-200',
                )}
              >
                {p.display_name}
              </span>
              {p.regime_aligned && (
                <Sparkles className="w-3 h-3 text-amber-300" aria-label="Aligned with liquidation regime" />
              )}
              <span className="text-[10px] text-slate-500 uppercase tracking-wide">{p.timeframe}</span>
            </div>

            <div className="flex items-center gap-2 mt-1.5 text-[10px]">
              <span className={clsx('px-1.5 py-0.5 rounded', STATE_COLOR[p.state])}>
                {STATE_LABEL[p.state]}
              </span>
              <span className="text-slate-500">Fit {(p.fit_score * 100).toFixed(0)}%</span>
              <span className="text-slate-600">·</span>
              <span className="text-slate-500">Score {(p.combined_score * 100).toFixed(0)}%</span>
              <div className="flex-1 ml-1 h-1 bg-white/5 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.max(2, p.combined_score * 100)}%`,
                    backgroundColor: resolveColor(p.color_token),
                  }}
                />
              </div>
            </div>

            {p.description && (
              <div className="text-[11px] text-slate-500 mt-1.5 line-clamp-2">{p.description}</div>
            )}
          </button>
        );
      })}
    </div>
  );
}
