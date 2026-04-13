import { useState, useEffect, useReducer, useCallback, useRef } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { BarChart3, MessageSquare, Newspaper, PieChart, ArrowLeft, CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';
import ProgressStepper, { type StepState } from '../components/ProgressStepper';
import AgentReportCard from '../components/AgentReportCard';
import DebatePanel from '../components/DebatePanel';
import SignalBadge from '../components/SignalBadge';
import FinalDecisionCard from '../components/FinalDecisionCard';
import { startAnalysis, streamAnalysis, type SSEEvent } from '../lib/api';
import useDocumentTitle from '../hooks/useDocumentTitle';

const INITIAL_STEPS: StepState[] = [
  { key: 'market', label: 'Market Analyst', status: 'pending' },
  { key: 'social', label: 'Social Media Analyst', status: 'pending' },
  { key: 'news', label: 'News Analyst', status: 'pending' },
  { key: 'fundamentals', label: 'Fundamentals Analyst', status: 'pending' },
  { key: 'bull_bear', label: 'Bull vs Bear Debate', status: 'pending' },
  { key: 'research_manager', label: 'Research Manager', status: 'pending' },
  { key: 'trader', label: 'Trader', status: 'pending' },
  { key: 'risk_debate', label: 'Risk Debate', status: 'pending' },
  { key: 'portfolio_manager', label: 'Portfolio Manager', status: 'pending' },
];

interface State {
  status: 'idle' | 'running' | 'done' | 'error';
  steps: StepState[];
  currentStep: number;
  reports: Record<string, string>;
  debates: {
    investment: { bull_history: string; bear_history: string; judge_decision: string };
    risk: { aggressive_history: string; conservative_history: string; neutral_history: string; judge_decision: string };
  };
  decision: string;
  finalReport: string;
  error: string;
  // Structured signal fields
  stopLoss?: number;
  takeProfit?: number;
  confidence?: number;
  maxHoldDays?: number;
  reasoning?: string;
  // Confidence scorer output
  positionSizePct?: number;
  convictionLabel?: string;
  gated?: boolean;
  rRatio?: number | null;
  rRatioWarning?: boolean;
  holdPeriodScalar?: number;
  hedgePenaltyApplied?: number;
  lastHeartbeat: number;
  startTime: number;
}

type Action =
  | { type: 'START' }
  | { type: 'AGENT_START'; step: number; label: string }
  | { type: 'AGENT_UPDATE'; content: string; step: number }
  | { type: 'REPORT'; key: string; content: string }
  | { type: 'DECISION'; signal: string }
  | { type: 'DONE'; result: Record<string, unknown> }
  | { type: 'ERROR'; message: string }
  | { type: 'HEARTBEAT' };

const STEP_KEY_MAP: Record<number, string> = {
  1: 'market', 2: 'social', 3: 'news', 4: 'fundamentals',
  5: 'bull_bear', 6: 'research_manager', 7: 'trader', 8: 'risk_debate', 9: 'portfolio_manager',
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'START':
      return { ...state, status: 'running', steps: INITIAL_STEPS.map(s => ({ ...s, status: 'pending' as const })), currentStep: 0, startTime: Date.now(), lastHeartbeat: Date.now() };
    case 'AGENT_START': {
      const stepKey = STEP_KEY_MAP[action.step];
      const steps = state.steps.map(s => {
        if (s.key === stepKey) return { ...s, status: 'running' as const };
        if (s.status === 'running' && s.key !== stepKey) return { ...s, status: 'done' as const };
        return s;
      });
      return { ...state, steps, currentStep: action.step };
    }
    case 'AGENT_UPDATE': {
      const stepKey = STEP_KEY_MAP[action.step];
      const steps = state.steps.map(s =>
        s.key === stepKey ? { ...s, content: action.content } : s
      );
      return { ...state, steps };
    }
    case 'REPORT':
      return { ...state, reports: { ...state.reports, [action.key]: action.content } };
    case 'DECISION': {
      const steps = state.steps.map(s => s.status === 'running' ? { ...s, status: 'done' as const } : s);
      return { ...state, decision: action.signal, steps };
    }
    case 'DONE': {
      const steps = state.steps.map(s => ({ ...s, status: 'done' as const }));
      const result = action.result || {};
      return {
        ...state,
        status: 'done',
        steps,
        currentStep: 9,
        finalReport: (result.final_trade_decision as string) || '',
        reports: {
          ...state.reports,
          market_report: (result.market_report as string) || state.reports.market_report || '',
          sentiment_report: (result.sentiment_report as string) || state.reports.sentiment_report || '',
          news_report: (result.news_report as string) || state.reports.news_report || '',
          fundamentals_report: (result.fundamentals_report as string) || state.reports.fundamentals_report || '',
        },
        debates: {
          investment: {
            bull_history: ((result.investment_debate as Record<string, string>)?.bull_history) || '',
            bear_history: ((result.investment_debate as Record<string, string>)?.bear_history) || '',
            judge_decision: ((result.investment_debate as Record<string, string>)?.judge_decision) || '',
          },
          risk: {
            aggressive_history: ((result.risk_debate as Record<string, string>)?.aggressive) || '',
            conservative_history: ((result.risk_debate as Record<string, string>)?.conservative) || '',
            neutral_history: ((result.risk_debate as Record<string, string>)?.neutral) || '',
            judge_decision: ((result.risk_debate as Record<string, string>)?.judge_decision) || '',
          },
        },
        decision: (result.decision as string) || state.decision,
        stopLoss: result.stop_loss_price as number | undefined,
        takeProfit: result.take_profit_price as number | undefined,
        confidence: result.confidence as number | undefined,
        maxHoldDays: result.max_hold_days as number | undefined,
        reasoning: result.reasoning as string | undefined,
        positionSizePct: result.position_size_pct as number | undefined,
        convictionLabel: result.conviction_label as string | undefined,
        gated: result.gated as boolean | undefined,
        rRatio: result.r_ratio as number | null | undefined,
        rRatioWarning: result.r_ratio_warning as boolean | undefined,
        holdPeriodScalar: result.hold_period_scalar as number | undefined,
        hedgePenaltyApplied: result.hedge_penalty_applied as number | undefined,
      };
    }
    case 'ERROR':
      return { ...state, status: 'error', error: action.message };
    case 'HEARTBEAT':
      return { ...state, lastHeartbeat: Date.now() };
    default:
      return state;
  }
}

const initialState: State = {
  status: 'idle',
  steps: INITIAL_STEPS.map(s => ({ ...s })),
  currentStep: 0,
  reports: {},
  debates: {
    investment: { bull_history: '', bear_history: '', judge_decision: '' },
    risk: { aggressive_history: '', conservative_history: '', neutral_history: '', judge_decision: '' },
  },
  decision: '',
  finalReport: '',
  error: '',
  stopLoss: undefined,
  takeProfit: undefined,
  confidence: undefined,
  maxHoldDays: undefined,
  reasoning: undefined,
  positionSizePct: undefined,
  convictionLabel: undefined,
  gated: undefined,
  rRatio: undefined,
  rRatioWarning: undefined,
  holdPeriodScalar: undefined,
  hedgePenaltyApplied: undefined,
  lastHeartbeat: 0,
  startTime: 0,
};

export default function Analyze() {
  const { ticker } = useParams<{ ticker: string }>();
  useDocumentTitle(`${ticker} — Live Analysis`);
  const navigate = useNavigate();
  const location = useLocation();
  const [state, dispatch] = useReducer(reducer, initialState);
  const [jobId, setJobId] = useState<string | null>(null);
  const startedRef = useRef(false);

  const start = useCallback(async () => {
    if (!ticker || startedRef.current) return;
    startedRef.current = true;
    dispatch({ type: 'START' });

    // If navigated from "Run Now" with an existing job, reuse it
    const existingJobId = (location.state as { jobId?: string })?.jobId;
    if (existingJobId) {
      setJobId(existingJobId);
      return;
    }

    try {
      const res = await startAnalysis(ticker);
      setJobId(res.job_id);
    } catch (err) {
      dispatch({ type: 'ERROR', message: String(err) });
    }
  }, [ticker, location.state]);

  // Auto-start on mount (ref guard prevents Strict Mode double-fire)
  useEffect(() => {
    queueMicrotask(() => {
      void start();
    });
  }, [start]);

  // SSE stream
  useEffect(() => {
    if (!jobId) return;
    const close = streamAnalysis(jobId, (e: SSEEvent) => {
      // #region agent log
      fetch('http://127.0.0.1:7444/ingest/e6a1deb5-6f13-4c9b-a327-42294f68dcb9',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'f18c74'},body:JSON.stringify({sessionId:'f18c74',location:'Analyze.tsx:SSE',message:'SSE event received',data:{event:e.event,step:e.step,agent:e.agent},timestamp:Date.now()})}).catch(()=>{});
      // #endregion
      switch (e.event) {
        case 'agent_start':
          dispatch({ type: 'AGENT_START', step: e.step!, label: e.agent! });
          break;
        case 'agent_update':
          dispatch({ type: 'AGENT_UPDATE', content: e.content!, step: e.step! });
          break;
        case 'agent_report':
          dispatch({ type: 'REPORT', key: e.report_key!, content: e.report! });
          break;
        case 'decision':
          dispatch({ type: 'DECISION', signal: e.signal! });
          break;
        case 'done':
          dispatch({ type: 'DONE', result: e.result || {} });
          break;
        case 'error':
          dispatch({ type: 'ERROR', message: e.message || 'Unknown error' });
          break;
        case 'heartbeat':
          dispatch({ type: 'HEARTBEAT' });
          break;
      }
    });
    return close;
  }, [jobId]);

  const REPORT_CARDS = [
    { key: 'market_report', title: 'Market Analysis', icon: <BarChart3 className="w-4 h-4" /> },
    { key: 'sentiment_report', title: 'Social & Sentiment', icon: <MessageSquare className="w-4 h-4" /> },
    { key: 'news_report', title: 'News Analysis', icon: <Newspaper className="w-4 h-4" /> },
    { key: 'fundamentals_report', title: 'Fundamentals', icon: <PieChart className="w-4 h-4" /> },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <button onClick={() => navigate('/')} aria-label="Back to home" className="text-slate-400 hover:text-white transition-colors">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white">{ticker}</h1>
          <p className="text-sm text-slate-500">Live Analysis</p>
        </div>
        {state.decision && (
          <div className="ml-auto">
            <SignalBadge signal={state.decision} size="lg" confidence={state.confidence} />
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Progress */}
        <div className="lg:col-span-1">
          <ProgressStepper
            steps={state.steps}
            currentStep={state.currentStep}
            totalSteps={9}
            isRunning={state.status === 'running'}
            lastHeartbeat={state.lastHeartbeat}
            startTime={state.startTime}
          />
        </div>

        {/* Right: Reports */}
        <div className="lg:col-span-2 space-y-4">
          {state.status === 'error' && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass p-6 border-red-500/40 border rounded-xl"
            >
              <div className="flex items-start gap-3">
                <XCircle className="w-6 h-6 text-red-400 mt-0.5 shrink-0" />
                <div className="space-y-2">
                  <h3 className="text-red-400 font-semibold text-base">Analysis Failed</h3>
                  <p className="text-slate-300 text-sm leading-relaxed">{state.error}</p>
                  <button
                    onClick={() => window.location.reload()}
                    className="mt-3 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {/* Agent reports */}
          {REPORT_CARDS.map(rc => (
            <AgentReportCard
              key={rc.key}
              title={rc.title}
              icon={rc.icon}
              content={state.reports[rc.key] || ''}
              defaultOpen={false}
            />
          ))}

          {/* Debates */}
          {(state.debates.investment.bull_history || state.debates.investment.bear_history) && (
            <div>
              <h3 className="text-sm font-medium text-slate-400 mb-2">Investment Debate</h3>
              <DebatePanel type="investment" data={state.debates.investment} />
            </div>
          )}

          {(state.debates.risk.aggressive_history || state.debates.risk.conservative_history) && (
            <div>
              <h3 className="text-sm font-medium text-slate-400 mb-2">Risk Debate</h3>
              <DebatePanel type="risk" data={state.debates.risk} />
            </div>
          )}

          {/* Final Decision */}
          {state.finalReport && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="glass p-6"
            >
              <h3 className="text-lg font-bold text-white mb-4">Final Decision</h3>
              <FinalDecisionCard
                text={state.finalReport}
                signal={state.decision}
                confidence={state.confidence}
                rRatio={state.rRatio}
              />
            </motion.div>
          )}

          {/* Trade Parameters (if structured signal available) */}
          {state.stopLoss !== undefined && state.takeProfit !== undefined && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="glass p-6"
            >
              <h3 className="text-lg font-bold text-white mb-4">Trade Parameters</h3>

              {/* Warnings row - only show R:R warning, not gating */}
              {state.rRatioWarning && (
                <div className="flex flex-col gap-2 mb-4">
                  <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30">
                    <span className="text-amber-400 text-sm font-medium">⚠ Unfavorable R:R — You risk more than you stand to gain on this trade.</span>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {/* Stop Loss */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <p className="text-xs text-slate-400 mb-1">Stop Loss</p>
                  <p className="text-lg font-semibold text-red-400">
                    ${state.stopLoss?.toLocaleString() || 'N/A'}
                  </p>
                </div>

                {/* Take Profit */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <p className="text-xs text-slate-400 mb-1">Take Profit</p>
                  <p className="text-lg font-semibold text-green-400">
                    ${state.takeProfit?.toLocaleString() || 'N/A'}
                  </p>
                </div>

                {/* Conviction */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <p className="text-xs text-slate-400 mb-1">Conviction</p>
                  <div className="flex items-center gap-2">
                    {state.convictionLabel && (
                      <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                        state.convictionLabel === 'VERY HIGH' ? 'bg-accent-teal/20 text-accent-teal' :
                        state.convictionLabel === 'HIGH'      ? 'bg-green-500/20 text-green-300' :
                        state.convictionLabel === 'MODERATE'  ? 'bg-yellow-500/20 text-yellow-300' :
                        'bg-red-500/20 text-red-300'
                      }`}>{state.convictionLabel}</span>
                    )}
                    <span className="text-sm font-semibold text-white">
                      {((state.confidence || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="mt-1.5 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        (state.confidence || 0) >= 0.75 ? 'bg-accent-teal' :
                        (state.confidence || 0) >= 0.60 ? 'bg-green-400' :
                        (state.confidence || 0) >= 0.45 ? 'bg-yellow-400' :
                        'bg-red-400'
                      }`}
                      style={{ width: `${(state.confidence || 0) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Suggested Position Size */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <p className="text-xs text-slate-400 mb-1">Suggested Size</p>
                  <p className={`text-lg font-semibold ${
                    (state.positionSizePct || 0) > 0.5 ? 'text-green-400' :
                    (state.positionSizePct || 0) > 0.2 ? 'text-yellow-400' :
                    'text-slate-300'
                  }`}>
                    {`${((state.positionSizePct || 0) * 100).toFixed(1)}%`}
                  </p>
                  <div className="mt-1.5 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent-teal transition-all"
                      style={{ width: `${Math.min((state.positionSizePct || 0) * 100, 100)}%` }}
                    />
                  </div>
                </div>

                {/* R:R Ratio */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <p className="text-xs text-slate-400 mb-1">R:R Ratio</p>
                  {state.rRatio != null ? (
                    <div className="flex items-center gap-1.5">
                      <p className={`text-lg font-semibold ${
                        state.rRatio >= 2.0 ? 'text-green-400' :
                        state.rRatio >= 1.0 ? 'text-yellow-400' :
                        'text-red-400'
                      }`}>
                        {state.rRatio.toFixed(2)}:1
                      </p>
                      {state.rRatio >= 2.0 ? (
                        <CheckCircle2 className="w-4 h-4 text-green-400" />
                      ) : state.rRatio >= 1.0 ? (
                        <AlertTriangle className="w-4 h-4 text-yellow-400" />
                      ) : (
                        <XCircle className="w-4 h-4 text-red-400" />
                      )}
                    </div>
                  ) : (
                    <p className="text-slate-500 text-sm">N/A</p>
                  )}
                </div>

                {/* Hold Period */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <p className="text-xs text-slate-400 mb-1">Max Hold</p>
                  <p className="text-lg font-semibold text-white">
                    {state.maxHoldDays || 'N/A'} days
                  </p>
                  {state.holdPeriodScalar != null && state.holdPeriodScalar < 1.0 && (
                    <p className="text-xs text-slate-500 mt-0.5">
                      size scaled ×{state.holdPeriodScalar.toFixed(2)}
                    </p>
                  )}
                </div>

                {/* Reasoning */}
                {state.reasoning && (
                  <div className="bg-slate-800/50 rounded-lg p-4 col-span-2 md:col-span-3">
                    <p className="text-xs text-slate-400 mb-1">Reasoning</p>
                    <p className="text-sm text-slate-300">{state.reasoning}</p>
                    {state.hedgePenaltyApplied != null && state.hedgePenaltyApplied > 0 && (
                      <p className="text-xs text-amber-500/70 mt-1">
                        Hedge-word penalty: −{(state.hedgePenaltyApplied * 100).toFixed(0)}% confidence
                      </p>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Done message */}
          {state.status === 'done' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-4"
            >
              <button
                onClick={() => navigate(`/history/${ticker}`)}
                className="px-6 py-2.5 rounded-lg bg-accent-teal/10 border border-accent-teal/20 text-accent-teal text-sm font-medium hover:bg-accent-teal/20 transition-colors"
              >
                View in History
              </button>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
