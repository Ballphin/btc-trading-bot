import { useState, useEffect, useRef } from 'react';
import { Briefcase, AlertCircle, Play, CheckCircle2, Circle } from 'lucide-react';
import { clsx } from 'clsx';
import { getHedgeFundAgents, startHedgeFundAnalysis, API_BASE_URL } from '../lib/api';
import type { HedgeFundAgent, HedgeFundRequest, HedgeFundResult } from '../lib/api';

export default function HedgeFund() {
  const [agents, setAgents] = useState<HedgeFundAgent[]>([]);
  const [selectedAgents, setSelectedAgents] = useState<Set<string>>(new Set());
  const [tickerInput, setTickerInput] = useState('AAPL');
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ node: string }[]>([]);
  const [result, setResult] = useState<HedgeFundResult | null>(null);
  
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    getHedgeFundAgents()
      .then(data => {
        setAgents(data);
        // Pre-select a few common ones
        const defaultSet = new Set(['warren_buffett', 'michael_burry', 'technical_analyst']);
        const initialSelected = new Set(data.map(a => a.key).filter(k => defaultSet.has(k)));
        if (initialSelected.size === 0 && data.length > 0) {
            initialSelected.add(data[0].key);
        }
        setSelectedAgents(initialSelected);
      })
      .catch(err => console.error('Failed to load agents:', err));
  }, []);

  const toggleAgent = (key: string) => {
    const next = new Set(selectedAgents);
    if (next.has(key)) {
      next.delete(key);
    } else {
      next.add(key);
    }
    setSelectedAgents(next);
  };

  const handleRun = async () => {
    if (!tickerInput) {
      setError('Please enter a ticker');
      return;
    }
    if (selectedAgents.size === 0) {
      setError('Please select at least one analyst');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setProgress([]);
      setResult(null);

      const req: HedgeFundRequest = {
        tickers: tickerInput.split(',').map(t => t.trim()),
        selected_analysts: Array.from(selectedAgents),
        model_name: 'gpt-4o', // Default model
        model_provider: 'OpenAI',
      };

      const res = await startHedgeFundAnalysis(req);
      setJobId(res.job_id);
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;

    const source = new EventSource(`${API_BASE_URL}/hedgefund/stream/${jobId}`);
    eventSourceRef.current = source;

    source.addEventListener('progress', (e: any) => {
      const data = JSON.parse(e.data);
      setProgress(p => [...p, data]);
    });

    source.addEventListener('done', (e: any) => {
      const data = JSON.parse(e.data);
      setResult(data);
      setLoading(false);
      source.close();
    });

    source.addEventListener('error', (e: any) => {
      let msg = 'Stream error';
      try {
        const data = JSON.parse(e.data);
        if (data.error) msg = data.error;
      } catch {}
      setError(msg);
      setLoading(false);
      source.close();
    });

    return () => {
      source.close();
    };
  }, [jobId]);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-center gap-3 mb-8">
        <div className="p-2.5 bg-accent-blue/10 rounded-xl">
          <Briefcase className="w-6 h-6 text-accent-blue" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white">HedgeFund Analysis</h1>
          <p className="text-slate-400">Multi-agent LangGraph analysis powered by virattt/ai-hedge-fund</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Config Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
            <h2 className="text-lg font-medium text-white mb-4">Configuration</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Target Ticker</label>
                <input
                  type="text"
                  value={tickerInput}
                  onChange={(e) => setTickerInput(e.target.value)}
                  placeholder="e.g. AAPL, MSFT"
                  className="w-full bg-slate-950 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-accent-blue"
                  disabled={loading}
                />
                <p className="text-xs text-slate-500 mt-1">Note: Works for stocks only (financialdatasets.ai)</p>
              </div>

              <div className="pt-2">
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-slate-300">Select Analysts</label>
                  <span className="text-xs text-slate-500">{selectedAgents.size} selected</span>
                </div>
                <div className="h-96 overflow-y-auto pr-2 space-y-2 custom-scrollbar">
                  {agents.map((agent) => {
                    const isSelected = selectedAgents.has(agent.key);
                    return (
                      <button
                        key={agent.key}
                        onClick={() => toggleAgent(agent.key)}
                        disabled={loading}
                        className={clsx(
                          "w-full text-left p-3 rounded-xl border transition-all",
                          isSelected 
                            ? "bg-accent-blue/10 border-accent-blue/30 text-white" 
                            : "bg-slate-950 border-white/5 text-slate-400 hover:bg-slate-900"
                        )}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium text-sm">{agent.display_name}</span>
                          {isSelected ? <CheckCircle2 className="w-4 h-4 text-accent-blue" /> : <Circle className="w-4 h-4 opacity-30" />}
                        </div>
                        <p className="text-xs opacity-70 line-clamp-1">{agent.description}</p>
                      </button>
                    );
                  })}
                </div>
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 flex gap-2">
                  <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  <p className="text-sm text-red-400">{error}</p>
                </div>
              )}

              <button
                onClick={handleRun}
                disabled={loading || selectedAgents.size === 0 || !tickerInput}
                className="w-full py-3 rounded-xl bg-accent-blue text-white font-medium hover:bg-accent-blue/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Running Analysis...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Run HedgeFund
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Progress Stream */}
          {(loading || progress.length > 0) && !result && (
            <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
              <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-accent-blue animate-pulse" />
                Live Execution Progress
              </h3>
              <div className="space-y-3">
                {progress.map((p, i) => (
                  <div key={i} className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                    <span className="text-slate-300">
                      <span className="text-accent-blue/80 font-mono">[{p.node}]</span> completed analysis
                    </span>
                  </div>
                ))}
                {loading && (
                  <div className="flex items-center gap-3 text-sm text-slate-500">
                    <div className="w-4 h-4 border-2 border-slate-700 border-t-slate-500 rounded-full animate-spin" />
                    Waiting for next agent...
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Results Display */}
          {result && (
            <div className="space-y-6">
              <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
                <h3 className="text-xl font-bold text-white mb-2">Portfolio Manager Decision</h3>
                {Object.entries(result.decisions).map(([ticker, dec]: [string, any]) => (
                  <div key={ticker} className="mt-4 p-4 rounded-xl bg-slate-950 border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <div className="text-2xl font-bold text-white">{ticker}</div>
                      <div className={clsx(
                        "px-4 py-1.5 rounded-lg font-bold text-sm",
                        dec.action === 'BUY' ? "bg-green-500/10 text-green-400 border border-green-500/20" :
                        dec.action === 'SELL' ? "bg-red-500/10 text-red-400 border border-red-500/20" :
                        "bg-slate-500/10 text-slate-400 border border-slate-500/20"
                      )}>
                        {dec.action}
                      </div>
                    </div>
                    {dec.quantity !== undefined && (
                      <div className="text-slate-400 mb-2">Quantity: <span className="text-white">{dec.quantity}</span></div>
                    )}
                    {dec.reasoning && (
                      <div className="text-sm text-slate-300 mt-4 leading-relaxed">
                        {dec.reasoning}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
                <h3 className="text-lg font-medium text-white mb-4">Analyst Signals</h3>
                <div className="space-y-4">
                  {Object.entries(result.analyst_signals || {}).map(([analystName, data]: [string, any]) => {
                    const signalInfo = data[tickerInput] || data[Object.keys(data)[0]]; // fallback if ticker format differs
                    if (!signalInfo) return null;
                    return (
                      <div key={analystName} className="p-4 rounded-xl bg-slate-950 border border-white/5">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <div className="font-medium text-white">{analystName.replace('_agent', '').split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}</div>
                          </div>
                          <div className={clsx(
                            "px-2 py-0.5 rounded text-xs font-medium",
                            signalInfo.signal === 'bullish' ? "bg-green-500/20 text-green-400" :
                            signalInfo.signal === 'bearish' ? "bg-red-500/20 text-red-400" :
                            "bg-slate-500/20 text-slate-400"
                          )}>
                            {signalInfo.signal?.toUpperCase() || 'NEUTRAL'}
                          </div>
                        </div>
                        {signalInfo.confidence !== undefined && (
                          <div className="text-xs text-slate-500 mb-2">Confidence: {Math.round(signalInfo.confidence)}%</div>
                        )}
                        <p className="text-sm text-slate-400 mt-2 line-clamp-3 hover:line-clamp-none transition-all">
                          {signalInfo.reasoning}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
