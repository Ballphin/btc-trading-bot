import { clsx } from 'clsx';

const SIGNAL_STYLES: Record<string, string> = {
  BUY: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  SELL: 'bg-red-500/20 text-red-400 border-red-500/30',
  SHORT: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  COVER: 'bg-teal-500/20 text-teal-400 border-teal-500/30',
  HOLD: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  OVERWEIGHT: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  UNDERWEIGHT: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  UNKNOWN: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
};

interface SignalBadgeProps {
  signal: string;
  size?: 'sm' | 'md' | 'lg';
  confidence?: number;
}

export default function SignalBadge({ signal, size = 'md', confidence }: SignalBadgeProps) {
  const s = signal?.toUpperCase() || 'UNKNOWN';
  const style = SIGNAL_STYLES[s] || SIGNAL_STYLES.UNKNOWN;
  const sizeClass = size === 'lg' ? 'text-2xl px-6 py-2' : size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1';
  
  return (
    <div className="inline-flex flex-col items-center gap-1">
      <span className={clsx('inline-flex items-center font-bold rounded-full border', style, sizeClass)}>
        {s}
      </span>
      {confidence !== undefined && (
        <div className="flex items-center gap-1">
          <div className="w-16 h-1 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-accent-teal transition-all"
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
          <span className="text-xs text-slate-400">{(confidence * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
}
