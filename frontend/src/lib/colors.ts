// Okabe-Ito colorblind-safe palette (7 colors, yellow dropped for navy contrast).
// Reserved: red/green for SL/TP/candle bodies. Patterns use tokens below.

export const OKABE_ITO = {
  ok_blue: '#0072B2',    // H&S / Inverse H&S
  ok_orange: '#E69F00',  // Double Top/Bottom
  ok_green: '#009E73',   // Triple Top/Bottom
  ok_pink: '#CC79A7',    // Wedges / Cup-and-Handle
  ok_sky: '#56B4E9',     // Triangles
  ok_vermil: '#D55E00',  // Flags / Pennants
  ok_gray: '#BBBBBB',    // Channels, auto S/R trendlines
} as const;

export type ColorToken = keyof typeof OKABE_ITO;

export function resolveColor(token: string | undefined): string {
  if (!token) return OKABE_ITO.ok_gray;
  return (OKABE_ITO as Record<string, string>)[token] || OKABE_ITO.ok_gray;
}

// Semantic colors (reserved — do NOT assign to patterns)
export const SEMANTIC = {
  bull: '#26A69A',
  bear: '#EF5350',
  sl: '#EF4444',         // stop-loss (red)
  tp: '#10B981',         // take-profit (green)
  entry: '#F59E0B',      // entry (amber, contrasts with bull/bear)
  support: '#3B82F6',    // blue
  resistance: '#3B82F6',
  tsmom_up: '#10B981',
  tsmom_down: '#EF4444',
} as const;

// Line-style encoding (accessibility: distinguish without relying on color alone)
export const LINE_STYLE = {
  solid: 0,
  dotted: 1,
  dashed: 2,
  large_dashed: 3,
  sparse_dotted: 4,
} as const;

// Opacity per pattern state
export const STATE_OPACITY: Record<string, number> = {
  forming: 0.5,
  completed: 0.8,
  confirmed: 1.0,
  retested: 1.0,
  invalidated: 0.0,      // hidden from chart, visible in legend
};
