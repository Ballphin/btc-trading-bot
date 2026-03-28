import { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceDot,
  ReferenceLine,
  Label,
} from 'recharts';
import type { PriceRecord } from '../lib/api';

interface SignalMarker {
  date: string;
  signal: string;
  price: number;
}

interface Props {
  data: PriceRecord[];
  signals?: SignalMarker[];
  height?: number;
}

const SIGNAL_COLORS: Record<string, string> = {
  BUY: '#10b981',
  SELL: '#ef4444',
  SHORT: '#a855f7',
  COVER: '#06b6d4',
  HOLD: '#f59e0b',
};

function formatPrice(value: number) {
  if (value >= 10000) return `$${(value / 1000).toFixed(1)}k`;
  if (value >= 1000) return `$${(value / 1000).toFixed(2)}k`;
  return `$${value.toFixed(2)}`;
}

function formatVol(value: number) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
  return `${value}`;
}

interface TooltipProps {
  active?: boolean;
  payload?: Array<{ payload: PriceRecord }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="glass p-3 text-xs space-y-1 min-w-[160px]">
      <div className="font-bold text-white mb-2">{label}</div>
      <div className="flex justify-between"><span className="text-slate-400">Open</span><span>${d.open.toLocaleString()}</span></div>
      <div className="flex justify-between"><span className="text-slate-400">High</span><span className="text-emerald-400">${d.high.toLocaleString()}</span></div>
      <div className="flex justify-between"><span className="text-slate-400">Low</span><span className="text-red-400">${d.low.toLocaleString()}</span></div>
      <div className="flex justify-between"><span className="text-slate-400">Close</span><span className="font-bold">${d.close.toLocaleString()}</span></div>
      <div className="flex justify-between"><span className="text-slate-400">Volume</span><span>{formatVol(d.volume)}</span></div>
      {d.sma50 != null && <div className="flex justify-between"><span className="text-amber-400">SMA50</span><span>${d.sma50.toLocaleString()}</span></div>}
      {d.sma200 != null && <div className="flex justify-between"><span className="text-purple-400">SMA200</span><span>${d.sma200.toLocaleString()}</span></div>}
    </div>
  );
}

export default function PriceChart({ data, signals = [], height = 400 }: Props) {
  const maxVol = useMemo(() => Math.max(...data.map(d => d.volume)), [data]);

  if (!data.length) {
    return (
      <div className="glass p-8 flex items-center justify-center text-slate-500">
        No price data available
      </div>
    );
  }

  return (
    <div className="glass p-4">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="date"
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
            tickFormatter={(v: string) => v.slice(5)}
          />
          <YAxis
            yAxisId="price"
            orientation="right"
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={formatPrice}
            domain={['auto', 'auto']}
          />
          <YAxis
            yAxisId="volume"
            orientation="left"
            tick={false}
            axisLine={false}
            domain={[0, maxVol * 4]}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Volume bars */}
          <Bar
            yAxisId="volume"
            dataKey="volume"
            fill="rgba(14,165,233,0.15)"
            radius={[2, 2, 0, 0]}
          />

          {/* Price area */}
          <Area
            yAxisId="price"
            type="monotone"
            dataKey="close"
            stroke="#0ea5e9"
            strokeWidth={2}
            fill="url(#priceGrad)"
          />

          {/* SMA lines */}
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="sma50"
            stroke="#f59e0b"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
            connectNulls
          />
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="sma200"
            stroke="#a855f7"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
            connectNulls
          />

          {/* Signal markers */}
          {signals.map((s, i) => (
            <g key={i}>
              {/* Horizontal reference line at price level */}
              <ReferenceLine
                yAxisId="price"
                y={s.price}
                stroke={SIGNAL_COLORS[s.signal] || '#94a3b8'}
                strokeDasharray="3 3"
                strokeWidth={1}
                opacity={0.6}
                ifOverflow="extendDomain"
              />
              {/* Dot marker at decision date and price */}
              <ReferenceDot
                yAxisId="price"
                x={s.date}
                y={s.price}
                r={6}
                fill={SIGNAL_COLORS[s.signal] || '#94a3b8'}
                stroke="white"
                strokeWidth={2}
              />
              {/* Label annotation */}
              <Label
                value={`${s.signal} @ $${s.price.toLocaleString()}`}
                position="top"
                fill={SIGNAL_COLORS[s.signal] || '#94a3b8'}
                fontSize={12}
                fontWeight={600}
              />
            </g>
          ))}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-3 text-xs text-slate-400">
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-[#0ea5e9] inline-block rounded" /> Price</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-[#f59e0b] inline-block rounded border-dashed" /> SMA 50</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-[#a855f7] inline-block rounded" /> SMA 200</span>
        {signals.length > 0 && signals.map((s, i) => (
          <span key={i} className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: SIGNAL_COLORS[s.signal] || '#94a3b8' }} />
            {s.signal} ({s.date.slice(5)})
          </span>
        ))}
      </div>
    </div>
  );
}
