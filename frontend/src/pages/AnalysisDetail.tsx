import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, BarChart3, MessageSquare, Newspaper, PieChart, CheckCircle2, AlertTriangle, XCircle, Briefcase } from 'lucide-react';
import SignalBadge from '../components/SignalBadge';
import AgentReportCard from '../components/AgentReportCard';
import DebatePanel from '../components/DebatePanel';
import PriceChart from '../components/PriceChart';
import ReactMarkdown from 'react-markdown';
import FinalDecisionCard from '../components/FinalDecisionCard';
import {
  fetchAnalysisEnvelope,
  fetchPrice,
  type AnalysisData,
  type AnalysisDetailEnvelope,
  type HedgeFundHistoryEntry,
  type PriceRecord,
} from '../lib/api';
import useDocumentTitle from '../hooks/useDocumentTitle';

function HedgeFundDetail({
  ticker,
  date,
  entry,
  onBack,
}: {
  ticker: string;
  date: string;
  entry: HedgeFundHistoryEntry;
  onBack: () => void;
}) {
  const action = (entry.action || 'hold').toUpperCase();
  const confPct = entry.confidence_0_1 != null
    ? Math.round(entry.confidence_0_1 * 100)
    : null;
  const fmtUsd = (v: number | null | undefined) =>
    v == null ? '—' : `$${v.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-center gap-4 mb-8">
        <button onClick={onBack} aria-label="Back to history" className="text-slate-400 hover:text-white transition-colors">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            {ticker}
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-semibold">
              <Briefcase className="w-3 h-3" />
              HF · {action}{entry.quantity ? ` ${entry.quantity}` : ''}
            </span>
          </h1>
          <p className="text-sm text-slate-500">HedgeFund run · {entry.ts_local || date}</p>
        </div>
        {confPct != null && (
          <div className="text-right">
            <div className="text-xs text-slate-500">Confidence</div>
            <div className="text-xl font-semibold text-white">{confPct}%</div>
          </div>
        )}
      </div>

      {entry.truncated && (
        <div className="mb-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-sm">
          Some analyst reasoning was truncated to keep this record under 256 KB.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass p-6 space-y-3">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wide">Decision Context</h2>
          <Row label="Action" value={action} />
          <Row label="Quantity" value={entry.quantity?.toString() ?? '—'} />
          <Row label="Price at decision" value={fmtUsd(entry.price_at_decision_usd)} />
          <Row label="Notional" value={fmtUsd(entry.notional_usd)} />
          <Row label="Initial cash" value={fmtUsd(entry.initial_cash)} />
          <Row label="Tickers in run" value={(entry.tickers_in_run || []).join(', ') || '—'} />
          <Row label="Model" value={`${entry.model_provider ?? '?'} / ${entry.model_name ?? '?'}`} />
          <Row label="Window" value={`${entry.start_date ?? '?'} → ${entry.end_date ?? '?'}`} />
          <Row label="Local time" value={entry.ts_local || '—'} />
          {entry.price_capture_error && (
            <div className="text-xs text-amber-400 mt-2">Price capture: {entry.price_capture_error}</div>
          )}
        </div>

        <div className="glass p-6 lg:col-span-2">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">Reasoning</h2>
          <div className="prose prose-invert max-w-none text-slate-300 text-sm leading-relaxed whitespace-pre-wrap break-words">
            <ReactMarkdown>{entry.reasoning || '_No reasoning recorded._'}</ReactMarkdown>
          </div>
        </div>
      </div>

      <div className="glass p-6 mt-6">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
          Analyst Signals
          {entry.analyst_signals_empty && (
            <span className="ml-2 text-xs font-normal text-slate-500">(no per-analyst signal recorded for this ticker)</span>
          )}
        </h2>
        {!entry.analyst_signals_empty && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {Object.values(entry.analyst_signals).map((sig) => {
              const sigColor =
                sig.signal === 'bullish' ? 'text-green-400 bg-green-500/10 border-green-500/30' :
                sig.signal === 'bearish' ? 'text-red-400 bg-red-500/10 border-red-500/30' :
                'text-slate-300 bg-slate-700/30 border-slate-600/40';
              const conf = sig.confidence_0_1 != null ? Math.round(sig.confidence_0_1 * 100) : null;
              return (
                <details key={sig.agent} className="rounded-lg bg-slate-900/40 border border-white/5 p-3">
                  <summary className="cursor-pointer flex items-center justify-between gap-3">
                    <span className="text-sm font-medium text-white capitalize">
                      {sig.agent.replace(/_agent$/, '').replaceAll('_', ' ')}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className={`text-xs px-2 py-0.5 rounded border font-medium ${sigColor}`}>
                        {(sig.signal || 'neutral').toUpperCase()}
                      </span>
                      {conf != null && (
                        <span className="text-xs text-slate-400 font-mono">{conf}%</span>
                      )}
                    </div>
                  </summary>
                  {sig.raw && (
                    <pre className="mt-2 text-xs text-slate-400 overflow-x-auto whitespace-pre-wrap break-words">
                      {JSON.stringify(sig.raw, null, 2)}
                    </pre>
                  )}
                </details>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-slate-500">{label}</span>
      <span className="text-white font-medium text-right">{value}</span>
    </div>
  );
}

function extractSignal(text: string): string {
  const upper = (text || '').toUpperCase();
  for (const s of ['SHORT', 'COVER', 'OVERWEIGHT', 'UNDERWEIGHT', 'BUY', 'SELL', 'HOLD']) {
    if (upper.includes(s)) return s;
  }
  return 'UNKNOWN';
}

function extractEntryPrice(text: string): number | null {
  // Look for price patterns like $67,000, $67000, 67000, entry at $67000
  const patterns = [
    /entry\s+(?:price\s+)?[:@\s]*\$?([\d,]+(?:\.\d+)?)/i,
    /(?:enter|entry)\s+(?:at\s+)?[:@\s]*\$?([\d,]+(?:\.\d+)?)/i,
    /(?:price|level)\s*[:@\s]*\$?([\d,]+(?:\.\d+)?)/i,
    /\$?([\d,]{3,}(?:\.\d+)?)/,  // fallback: any number with 3+ digits
  ];
  
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      const priceStr = match[1].replace(/,/g, '');
      const price = parseFloat(priceStr);
      if (!isNaN(price) && price > 0) {
        return price;
      }
    }
  }
  return null;
}

export default function AnalysisDetail() {
  const { ticker, date } = useParams<{ ticker: string; date: string }>();
  useDocumentTitle(`${ticker} — ${date}`);
  const navigate = useNavigate();
  const [envelope, setEnvelope] = useState<AnalysisDetailEnvelope | null>(null);
  const [priceData, setPriceData] = useState<PriceRecord[]>([]);
  const [analysisKey, setAnalysisKey] = useState('');
  const [activeTab, setActiveTab] = useState<'overview' | 'reports' | 'debates'>('overview');
  const requestKey = ticker && date ? `${ticker}:${date}` : '';
  const loading = Boolean(requestKey) && analysisKey !== requestKey;

  useEffect(() => {
    if (!ticker || !date || !requestKey) return;
    let cancelled = false;

    Promise.all([
      fetchAnalysisEnvelope(ticker, date).catch(() => null),
      fetchPrice(ticker, 30, '4h').catch(() => []),
    ]).then(([env, p]) => {
      if (cancelled) return;
      setEnvelope(env);
      setPriceData(p);
      setAnalysisKey(requestKey);
    });

    return () => {
      cancelled = true;
    };
  }, [ticker, date, requestKey]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-20 text-center text-slate-500">
        Loading analysis...
      </div>
    );
  }

  if (!envelope) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-20 text-center">
        <p className="text-slate-500">Analysis not found for {ticker} on {date}</p>
        <button onClick={() => navigate('/history')} className="mt-4 text-accent-teal text-sm hover:underline">
          Back to History
        </button>
      </div>
    );
  }

  // Dispatch on kind from the API envelope (not from the file body) — see
  // SSE rebuttal in the v2 plan. HedgeFund records have a fundamentally
  // different schema (no SL/TP/horizon/reports) and get their own renderer.
  if (envelope.kind === 'hedgefund') {
    return (
      <HedgeFundDetail
        ticker={ticker!}
        date={date!}
        entry={envelope.data as HedgeFundHistoryEntry}
        onBack={() => navigate(`/history/${ticker}`)}
      />
    );
  }

  // Main-analysis path: keep existing `AnalysisData` shape with the
  // date_formatted merged from the envelope top-level.
  const analysis: AnalysisData = {
    ...(envelope.data as AnalysisData),
    date_formatted: envelope.date_formatted || (envelope.data as AnalysisData).date_formatted,
  };

  // SSE FIX: Format date with error boundary - use API formatted date or fallback
  const displayDate = analysis.date_formatted || date;

  const signal = analysis.decision || extractSignal(analysis.final_trade_decision);

  // Try to extract entry price from trader investment decision or final decision
  const priceFromText = extractEntryPrice(analysis.trader_investment_decision || '') 
    || extractEntryPrice(analysis.final_trade_decision || '');
  
  // Get closing price on analysis date as fallback
  const closePrice = priceData.find(p => p.date === date)?.close 
    || priceData[priceData.length - 1]?.close 
    || 0;
  
  const entryPrice = priceFromText || closePrice;

  const signalMarkers = [{
    date: date!,
    signal,
    price: entryPrice,
  }];

  const TABS = [
    { key: 'overview', label: 'Overview' },
    { key: 'reports', label: 'Agent Reports' },
    { key: 'debates', label: 'Debates' },
  ] as const;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button onClick={() => navigate(`/history/${ticker}`)} aria-label="Back to history" className="text-slate-400 hover:text-white transition-colors">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">{ticker}</h1>
          <p className="text-sm text-slate-500">{displayDate}</p>
        </div>
        <SignalBadge signal={signal} size="lg" />
      </div>

      {/* Tabs */}
      <div role="tablist" aria-label="Analysis sections" className="flex gap-1 mb-6 bg-navy-800/50 p-1 rounded-lg w-fit">
        {TABS.map(tab => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={activeTab === tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 min-h-[44px] rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? 'bg-accent-teal/10 text-accent-teal'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          {/* Price chart */}
          {priceData.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-slate-400 mb-3">Price Chart (90d)</h2>
              <PriceChart data={priceData} signals={signalMarkers} height={350} />
            </div>
          )}

          {/* Decision card */}
          <div className="glass p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-white">Final Decision</h2>
              <SignalBadge signal={signal} size="md" />
            </div>
            <FinalDecisionCard
              text={analysis.trader_investment_decision || analysis.final_trade_decision}
              signal={signal}
              confidence={analysis.confidence}
              rRatio={analysis.r_ratio}
              stopLoss={analysis.stop_loss_price}
              takeProfit={analysis.take_profit_price}
              maxHoldDays={analysis.max_hold_days}
            />
          </div>

          {/* Risk Parameters (if any metrics exists) */}
          {(analysis.stop_loss_price !== undefined || analysis.take_profit_price !== undefined || analysis.confidence !== undefined || analysis.r_ratio != null || analysis.position_size_pct !== undefined) && (
            <div>
              <h3 className="text-sm font-medium text-slate-400 mb-3">Risk Parameters</h3>

              {/* Warnings - only show R:R warning */}
              {analysis.r_ratio_warning && (
                <div className="flex flex-col gap-2 mb-3">
                  <div className="px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30">
                    <span className="text-amber-400 text-xs font-medium">⚠ Unfavorable R:R — Risk exceeded potential reward on this trade.</span>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {analysis.stop_loss_price !== undefined && analysis.stop_loss_price > 0 && (
                  <div className="glass p-4">
                    <p className="text-xs text-slate-500 mb-1">Stop Loss</p>
                    <p className="text-sm font-medium text-red-400">${analysis.stop_loss_price.toLocaleString()}</p>
                  </div>
                )}
                {analysis.take_profit_price !== undefined && analysis.take_profit_price > 0 && (
                  <div className="glass p-4">
                    <p className="text-xs text-slate-500 mb-1">Take Profit</p>
                    <p className="text-sm font-medium text-green-400">${analysis.take_profit_price.toLocaleString()}</p>
                  </div>
                )}
                {analysis.confidence != null && (
                  <div className="glass p-4">
                    <p className="text-xs text-slate-500 mb-1">Conviction</p>
                    <div className="flex items-center gap-1.5 mb-1">
                      {analysis.conviction_label && (
                        <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${
                          analysis.conviction_label === 'VERY HIGH' ? 'bg-accent-teal/20 text-accent-teal' :
                          analysis.conviction_label === 'HIGH'      ? 'bg-green-500/20 text-green-300' :
                          analysis.conviction_label === 'MODERATE'  ? 'bg-yellow-500/20 text-yellow-300' :
                          'bg-red-500/20 text-red-300'
                        }`}>{analysis.conviction_label}</span>
                      )}
                      <span className="text-sm font-medium text-white">{(analysis.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          analysis.confidence >= 0.75 ? 'bg-accent-teal' :
                          analysis.confidence >= 0.60 ? 'bg-green-400' :
                          analysis.confidence >= 0.45 ? 'bg-yellow-400' :
                          'bg-red-400'
                        }`}
                        style={{ width: `${analysis.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                )}
                {analysis.position_size_pct != null && (
                  <div className="glass p-4">
                    <p className="text-xs text-slate-500 mb-1">Suggested Size</p>
                    <p className={`text-sm font-medium ${
                      analysis.position_size_pct > 0.5 ? 'text-green-400' :
                      analysis.position_size_pct > 0.2 ? 'text-yellow-400' :
                      'text-slate-300'
                    }`}>
                      {`${(analysis.position_size_pct * 100).toFixed(1)}%`}
                    </p>
                  </div>
                )}
                <div className="glass p-4">
                  <p className="text-xs text-slate-500 mb-1">R:R Ratio</p>
                  <div className="flex items-center gap-1">
                    {analysis.r_ratio != null ? (
                      <>
                        <p className={`text-sm font-medium ${
                          analysis.r_ratio >= 2.0 ? 'text-green-400' :
                          analysis.r_ratio >= 1.0 ? 'text-yellow-400' :
                          'text-red-400'
                        }`}>{analysis.r_ratio.toFixed(2)}:1</p>
                        {analysis.r_ratio >= 2.0 ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                        ) : analysis.r_ratio >= 1.0 ? (
                          <AlertTriangle className="w-3.5 h-3.5 text-yellow-400" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5 text-red-400" />
                        )}
                      </>
                    ) : (
                      <p className="text-sm font-medium text-slate-400">N/A</p>
                    )}
                  </div>
                </div>
                {analysis.max_hold_days != null && (
                  <div className="glass p-4">
                    <p className="text-xs text-slate-500 mb-1">Max Hold</p>
                    <p className="text-sm font-medium text-white">{analysis.max_hold_days} days</p>
                    {analysis.hold_period_scalar != null && analysis.hold_period_scalar < 1.0 && (
                      <p className="text-xs text-slate-500 mt-0.5">scaled ×{analysis.hold_period_scalar.toFixed(2)}</p>
                    )}
                  </div>
                )}
              </div>

              {/* Reasoning with hedge penalty */}
              {analysis.reasoning && (
                <div className="glass p-4 mt-3">
                  <p className="text-xs text-slate-500 mb-1">Reasoning</p>
                  <p className="text-sm text-slate-300">{analysis.reasoning}</p>
                  {analysis.hedge_penalty_applied != null && analysis.hedge_penalty_applied > 0 && (
                    <p className="text-xs text-amber-500/70 mt-1">
                      Hedge-word penalty: −{(analysis.hedge_penalty_applied * 100).toFixed(0)}% confidence
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}

      {activeTab === 'reports' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
          <AgentReportCard
            title="Market Analysis"
            icon={<BarChart3 className="w-4 h-4" />}
            content={analysis.market_report}
            defaultOpen={true}
          />
          <AgentReportCard
            title="Social & Sentiment"
            icon={<MessageSquare className="w-4 h-4" />}
            content={analysis.sentiment_report}
          />
          <AgentReportCard
            title="News Analysis"
            icon={<Newspaper className="w-4 h-4" />}
            content={analysis.news_report}
          />
          <AgentReportCard
            title="Fundamentals"
            icon={<PieChart className="w-4 h-4" />}
            content={analysis.fundamentals_report}
          />

          {/* Trader plan */}
          {analysis.trader_investment_decision && (
            <div className="glass p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Trader Investment Plan</h3>
              <div className="markdown-content text-sm max-h-64 overflow-y-auto">
                <ReactMarkdown>{analysis.trader_investment_decision}</ReactMarkdown>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {activeTab === 'debates' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div>
            <h2 className="text-sm font-medium text-slate-400 mb-3">Investment Debate (Bull vs Bear)</h2>
            <DebatePanel
              type="investment"
              data={{
                bull_history: analysis.investment_debate_state?.bull_history,
                bear_history: analysis.investment_debate_state?.bear_history,
                judge_decision: analysis.investment_debate_state?.judge_decision,
              }}
            />
          </div>
          <div>
            <h2 className="text-sm font-medium text-slate-400 mb-3">Risk Debate</h2>
            <DebatePanel
              type="risk"
              data={{
                aggressive_history: analysis.risk_debate_state?.aggressive_history,
                conservative_history: analysis.risk_debate_state?.conservative_history,
                neutral_history: analysis.risk_debate_state?.neutral_history,
                judge_decision: analysis.risk_debate_state?.judge_decision,
              }}
            />
          </div>

          {/* Investment Plan (Judge) */}
          {analysis.investment_plan && (
            <div className="glass p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Investment Plan (Research Manager)</h3>
              <div className="markdown-content text-sm max-h-64 overflow-y-auto">
                <ReactMarkdown>{analysis.investment_plan}</ReactMarkdown>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
