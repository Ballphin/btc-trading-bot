import React from 'react';
import ReactMarkdown from 'react-markdown';
import { CheckCircle2, AlertTriangle, XCircle, TrendingDown, TrendingUp, Minus } from 'lucide-react';
import SignalBadge from './SignalBadge';

interface StructuredDecision {
  signal: string;
  stop_loss_price?: number;
  take_profit_price?: number;
  confidence?: number;
  max_hold_days?: number;
  reasoning?: string;
}

function tryParseJson(text: string): StructuredDecision | null {
  if (!text) return null;
  let trimmed = text.trim();
  // Strip markdown code fences: ```json ... ``` or ``` ... ```
  const fenceMatch = trimmed.match(/^```(?:json)?\s*\n?([\s\S]*?)\n?```\s*$/);
  if (fenceMatch) {
    trimmed = fenceMatch[1].trim();
  }
  // Only try if it looks like JSON
  if (!trimmed.startsWith('{')) return null;
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === 'object' && parsed.signal) {
      return parsed as StructuredDecision;
    }
  } catch {
    // not valid JSON — fall through to markdown
  }
  return null;
}

interface Props {
  text: string;
  signal?: string;
  confidence?: number;
}

export default function FinalDecisionCard({ text, signal, confidence }: Props) {
  const parsed = tryParseJson(text);

  if (parsed) {
    const sig = (parsed.signal || signal || 'UNKNOWN').toUpperCase();
    const conf = parsed.confidence ?? confidence;
    const sl = parsed.stop_loss_price;
    const tp = parsed.take_profit_price;
    const rRatio = sl && tp && sl > 0 && tp > 0 ? Math.abs(tp - sl) / Math.abs(sl) : null;

    const signalColor =
      sig === 'BUY' || sig === 'COVER' ? 'text-green-400' :
      sig === 'SHORT' || sig === 'SELL' ? 'text-red-400' :
      'text-slate-300';

    const SignalIcon =
      sig === 'BUY' || sig === 'COVER' ? TrendingUp :
      sig === 'SHORT' || sig === 'SELL' ? TrendingDown :
      Minus;

    return (
      <div className="space-y-4">
        {/* Signal Hero */}
        <div className="flex items-center gap-3 p-4 rounded-xl bg-slate-800/60 border border-slate-700/50">
          <div className={`p-2.5 rounded-lg bg-slate-700/50 ${signalColor}`}>
            <SignalIcon className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wider">Final Signal</p>
            <p className={`text-xl font-bold ${signalColor}`}>{sig}</p>
          </div>
          {conf !== undefined && (
            <div className="ml-auto text-right">
              <p className="text-xs text-slate-500 mb-1">Confidence</p>
              <div className="flex items-center gap-2">
                <div className="w-20 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${conf >= 0.70 ? 'bg-emerald-400' : conf >= 0.50 ? 'bg-yellow-400' : 'bg-red-400'}`}
                    style={{ width: `${conf * 100}%` }}
                  />
                </div>
                <span className="text-sm font-semibold text-white">{Math.round(conf * 100)}%</span>
              </div>
            </div>
          )}
        </div>

        {/* Price Levels Grid */}
        {(sl || tp || parsed.max_hold_days) && (
          <div className="grid grid-cols-3 gap-3">
            {sl && sl > 0 && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                <p className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                  <XCircle className="w-3 h-3" /> Stop Loss
                </p>
                <p className="text-lg font-bold text-red-400">${sl.toLocaleString()}</p>
              </div>
            )}
            {tp && tp > 0 && (
              <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                <p className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" /> Take Profit
                </p>
                <p className="text-lg font-bold text-emerald-400">${tp.toLocaleString()}</p>
              </div>
            )}
            {parsed.max_hold_days && (
              <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/50">
                <p className="text-xs text-slate-500 mb-1">Max Hold</p>
                <p className="text-lg font-bold text-white">{parsed.max_hold_days}d</p>
              </div>
            )}
          </div>
        )}

        {/* R:R Ratio */}
        {rRatio !== null && (
          <div className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm ${
            rRatio >= 2.0 ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400' :
            rRatio >= 1.0 ? 'bg-yellow-500/10 border border-yellow-500/20 text-yellow-400' :
            'bg-red-500/10 border border-red-500/20 text-red-400'
          }`}>
            {rRatio >= 2.0 ? <CheckCircle2 className="w-4 h-4" /> :
             rRatio >= 1.0 ? <AlertTriangle className="w-4 h-4" /> :
             <XCircle className="w-4 h-4" />}
            <span className="font-medium">R:R Ratio — {rRatio.toFixed(2)}:1</span>
            {rRatio < 1.0 && <span className="text-xs ml-1 opacity-70">(unfavorable)</span>}
          </div>
        )}

        {/* Reasoning */}
        {parsed.reasoning && (
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/30">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Reasoning</p>
            <p className="text-sm text-slate-300 leading-relaxed">{parsed.reasoning}</p>
          </div>
        )}
      </div>
    );
  }

  // Fallback: prose/markdown text
  return (
    <div className="markdown-content text-sm max-h-[500px] overflow-y-auto leading-relaxed">
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  );
}
