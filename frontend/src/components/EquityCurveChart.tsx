import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';

interface EquityPoint {
  date: string;
  portfolio_value: number;
  cash?: number;
  position_side?: string;
}

interface EquityCurveChartProps {
  equityCurve: EquityPoint[];
}

export default function EquityCurveChart({ equityCurve }: EquityCurveChartProps) {
  const data = useMemo(() => {
    if (!equityCurve || equityCurve.length === 0) return [];
    
    return equityCurve.map((point) => ({
      date: point.date,
      value: point.portfolio_value,
      cash: point.cash || 0,
      position: point.position_side || 'FLAT',
    }));
  }, [equityCurve]);

  const initialValue = data[0]?.value || 0;
  const finalValue = data[data.length - 1]?.value || 0;
  const maxValue = Math.max(...data.map((d) => d.value));
  const minValue = Math.min(...data.map((d) => d.value));

  // Calculate drawdown periods
  const drawdownData = useMemo(() => {
    let peak = data[0]?.value || 0;
    let maxDrawdown = 0;
    let inDrawdown = false;
    const periods: { start: string; end: string; depth: number }[] = [];
    let currentPeriod: { start: string; depth: number } | null = null;

    data.forEach((point) => {
      if (point.value > peak) {
        peak = point.value;
        if (inDrawdown && currentPeriod) {
          periods.push({
            start: currentPeriod.start,
            end: point.date,
            depth: currentPeriod.depth,
          });
          inDrawdown = false;
          currentPeriod = null;
        }
      } else {
        const drawdown = (peak - point.value) / peak;
        if (drawdown > 0.05) {
          // Only consider >5% drawdowns
          if (!inDrawdown) {
            inDrawdown = true;
            currentPeriod = { start: point.date, depth: drawdown };
          } else if (currentPeriod && drawdown > currentPeriod.depth) {
            currentPeriod.depth = drawdown;
          }
        }
        maxDrawdown = Math.max(maxDrawdown, drawdown);
      }
    });

    return { maxDrawdown, periods };
  }, [data]);

  const formatCurrency = (value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `$${(value / 1000).toFixed(0)}k`;
    }
    return `$${value.toFixed(0)}`;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  if (data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-slate-400">
        No equity data available
      </div>
    );
  }

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="positiveGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="negativeGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />

          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            stroke="#64748b"
            tick={{ fontSize: 11 }}
            minTickGap={30}
          />

          <YAxis
            domain={['auto', 'auto']}
            tickFormatter={formatCurrency}
            stroke="#64748b"
            tick={{ fontSize: 11 }}
            width={60}
          />

          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const point = payload[0].payload;
                const pnl = point.value - initialValue;
                const pnlPct = (pnl / initialValue) * 100;

                return (
                  <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-lg">
                    <div className="text-sm text-slate-400 mb-1">{point.date}</div>
                    <div className="text-lg font-bold text-white">
                      {formatCurrency(point.value)}
                    </div>
                    <div
                      className={`text-sm font-medium ${
                        pnl >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {pnl >= 0 ? '+' : ''}${pnl.toLocaleString()} ({pnlPct >= 0 ? '+' : ''}
                      {pnlPct.toFixed(2)}%)
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      Position: {point.position}
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />

          {/* Initial capital reference line */}
          <ReferenceLine
            y={initialValue}
            stroke="#64748b"
            strokeDasharray="4 4"
            strokeWidth={1}
          />

          {/* Max drawdown highlight */}
          {drawdownData.periods.map((period, idx) => (
            <ReferenceLine
              key={idx}
              x={period.start}
              stroke="#ef4444"
              strokeDasharray="2 2"
              strokeWidth={1}
              opacity={0.5}
            />
          ))}

          <Area
            type="monotone"
            dataKey="value"
            stroke={finalValue >= initialValue ? '#10b981' : '#ef4444'}
            strokeWidth={2}
            fill={
              finalValue >= initialValue
                ? 'url(#positiveGradient)'
                : 'url(#negativeGradient)'
            }
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Summary stats */}
      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-xs text-slate-400">Peak</div>
          <div className="text-sm font-semibold text-white">{formatCurrency(maxValue)}</div>
        </div>
        <div>
          <div className="text-xs text-slate-400">Trough</div>
          <div className="text-sm font-semibold text-white">{formatCurrency(minValue)}</div>
        </div>
        <div>
          <div className="text-xs text-slate-400">Max Drawdown</div>
          <div className="text-sm font-semibold text-red-400">
            -{(drawdownData.maxDrawdown * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}
